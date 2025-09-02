import torch
import torch.nn as nn
import timm
import torchvision
import torchvision.transforms as transforms
import os
import time
import math
import argparse
import copy
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from timm.models.vision_transformer import VisionTransformer, Attention
from fvcore.nn import FlopCountAnalysis
from torch.jit import Final
from timm.layers import use_fused_attn

from dataset.datasets import build_dataset
from dataset.samplers import RASampler
from dataset.augment import new_data_aug_generator
from config import get_train_args
from utils import *
from engine import train_one_epoch, comp_evaluate

import config
from utils import *

import pruning
from pruning import *


class NoAttentionBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        # MLP 관련 구성 요소만 유지
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = original_block.ls2
        self.drop_path2 = original_block.drop_path2
    
    def forward(self, x):
        # 어텐션 부분을 완전히 건너뛰고 MLP 부분만 실행
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CustomTimmAttention(nn.Module):
    fused_attn: Final[bool]
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., qk_sizes=None, v_sizes=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.qk_sizes = qk_sizes if qk_sizes else [dim//num_heads] * num_heads
        self.v_sizes = v_sizes if v_sizes else [dim//num_heads] * num_heads
        
        self.scale = 1 / math.sqrt(64)
        self.fused_attn = use_fused_attn()
        
        # Calculate total dimensions for QKV matrix
        qkv_dim = sum(self.qk_sizes) * 2 + sum(self.v_sizes)
        self.qk_dim = sum(self.qk_sizes) * 2
        self.v_sum = sum(self.v_sizes)
        
        self.qkv = nn.Linear(dim, qkv_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.v_sum, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def transpose_for_scores(self, x, head_sizes):
        """Split into heads of different sizes"""
        B, N, _ = x.size()
        heads = []
        start_idx = 0
    
        for size in head_sizes:
            if size > 0:
                # 차원이 1인 경우 처리
                if size == 1:
                    head = x[:, :, start_idx:start_idx + size].reshape(B, N, 1, size)
                    # contiguous() 호출로 메모리 정렬 보장
                    head = head.permute(0, 2, 1, 3).contiguous()
                else:
                    head = x[:, :, start_idx:start_idx + size].reshape(B, N, 1, size)
                    head = head.permute(0, 2, 1, 3).contiguous()
                
                heads.append(head)
                start_idx += size
        return heads
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        # Split QKV
        q_proj = qkv[..., :sum(self.qk_sizes)]
        k_proj = qkv[..., sum(self.qk_sizes):self.qk_dim]
        v_proj = qkv[..., self.qk_dim:]
        
        # Split into heads with potentially different sizes
        q_heads = self.transpose_for_scores(q_proj, self.qk_sizes)
        k_heads = self.transpose_for_scores(k_proj, self.qk_sizes)
        v_heads = self.transpose_for_scores(v_proj, self.v_sizes)
        
        # Process each head individually - always use manual attention
        attn_outputs = []
        for i, (q_head, k_head, v_head) in enumerate(zip(q_heads, k_heads, v_heads)):
            # 항상 수동 방식으로 어텐션 계산
            q_scaled = q_head * self.scale
            attn = torch.matmul(q_scaled, k_head.transpose(-1, -2))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v_head)
            
            attn_outputs.append(out)
        
        # Combine outputs from all heads
        if attn_outputs:
            out = torch.cat(attn_outputs, dim=3)
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.v_sum)
        else:
            # Handle case where all heads are empty
            out = torch.zeros(B, N, self.v_sum, device=x.device)
        
        # Project back to original dimension
        x = self.proj(out)
        x = self.proj_drop(x)
        
        return x


def identify_pruned_heads_timm(block):
    qkv_weight = block.qkv.weight.data
    num_heads = block.num_heads
    head_dim = block.head_dim
    
    pruned_heads = set()
    qkv_dim = qkv_weight.size(0) // 3
    
    for head in range(num_heads):
        start_idx = head * head_dim
        end_idx = start_idx + head_dim
        
        q_head = qkv_weight[start_idx:end_idx]
        k_head = qkv_weight[qkv_dim+start_idx:qkv_dim+end_idx]
        v_head = qkv_weight[2*qkv_dim+start_idx:2*qkv_dim+end_idx]
        
        if torch.all(q_head == 0) or torch.all(k_head == 0) or torch.all(v_head == 0):
            pruned_heads.add(head)
    
    if len(pruned_heads) == num_heads:
        pruned_heads.remove(num_heads - 1)
    
    return pruned_heads


def get_pruned_head_config_timm(model: VisionTransformer):
    head_configs = {}
    
    for block_idx, block in enumerate(model.blocks):
        if hasattr(block, 'attn'):
            pruned_heads = identify_pruned_heads_timm(block.attn)
            head_dim = block.attn.head_dim
            num_heads = block.attn.num_heads
            
            qk_sizes = []
            v_sizes = []
            q_indices = []
            k_indices = []
            v_indices = []
            
            qkv_weight = block.attn.qkv.weight.data
            qkv_dim = qkv_weight.size(0) // 3
            
            for head in range(num_heads):
                if head in pruned_heads:
                    # 완전히 프루닝된 헤드는 0 차원으로 유지
                    qk_sizes.append(0)
                    v_sizes.append(0)
                    q_indices.append([])
                    k_indices.append([])
                    v_indices.append([])
                    continue
                    
                start_idx = head * head_dim
                end_idx = start_idx + head_dim
                
                # Q 헤드 처리
                q_head = qkv_weight[start_idx:end_idx]
                output_mask_q = q_head.any(dim=1)
                q_nonzero = torch.nonzero(output_mask_q).squeeze().tolist()
                
                # 단일 값이나 빈 리스트 처리
                if isinstance(q_nonzero, int):
                    q_nonzero = [q_nonzero]
                elif not isinstance(q_nonzero, list):
                    q_nonzero = []
                
                q_size = len(q_nonzero)
                q_indices.append(q_nonzero)

                # K 헤드 처리
                k_head = qkv_weight[qkv_dim+start_idx:qkv_dim+end_idx]
                output_mask_k = k_head.any(dim=1)
                k_nonzero = torch.nonzero(output_mask_k).squeeze().tolist()
                
                if isinstance(k_nonzero, int):
                    k_nonzero = [k_nonzero]
                elif not isinstance(k_nonzero, list):
                    k_nonzero = []
                
                k_size = len(k_nonzero)
                k_indices.append(k_nonzero)
                
                # Q와 K의 최소 크기를 사용 (둘 다 0일 수 있음)
                qk_size = min(q_size, k_size)
                qk_sizes.append(qk_size)
                
                # V 헤드 처리
                v_head = qkv_weight[2*qkv_dim+start_idx:2*qkv_dim+end_idx]
                output_mask_v = v_head.any(dim=1)
                v_nonzero = torch.nonzero(output_mask_v).squeeze().tolist()
                
                if isinstance(v_nonzero, int):
                    v_nonzero = [v_nonzero]
                elif not isinstance(v_nonzero, list):
                    v_nonzero = []
                
                v_size = len(v_nonzero)
                v_indices.append(v_nonzero)
                v_sizes.append(v_size)

            head_configs[block_idx] = {
                'qk_sizes': qk_sizes,
                'v_sizes': v_sizes,
                'num_heads': num_heads - len(pruned_heads),
                'q_indices': q_indices,
                'k_indices': k_indices,
                'v_indices': v_indices,
                'head_dim': head_dim
            }
    
    return head_configs

def compress_timm_model(model: VisionTransformer, head_configs: Dict):
    compressed_model = copy.deepcopy(model)
    
    for block_idx, config in head_configs.items():
        if hasattr(compressed_model.blocks[block_idx], 'attn'):
            block = compressed_model.blocks[block_idx]
            
            # 모든 헤드가 프루닝되었는지 또는 모든 차원이 0인지 확인
            all_heads_pruned = all(size == 0 for size in config['qk_sizes']) and all(size == 0 for size in config['v_sizes'])
            
            if all_heads_pruned:
                # 어텐션이 완전히 프루닝된 경우, NoAttentionBlock으로 대체
                compressed_model.blocks[block_idx] = NoAttentionBlock(block)
                print(f"Block {block_idx}: All attention heads pruned, replaced with NoAttentionBlock")
                continue
            
            # 0이 아닌 헤드 크기만 필터링
            filtered_qk_sizes = []
            filtered_v_sizes = []
            active_head_indices = []
            
            for i, (qk_size, v_size) in enumerate(zip(config['qk_sizes'], config['v_sizes'])):
                if qk_size > 0 and v_size > 0:
                    filtered_qk_sizes.append(qk_size)
                    filtered_v_sizes.append(v_size)
                    active_head_indices.append(i)
            
            # 활성 헤드가 없으면 NoAttentionBlock으로 대체
            if not active_head_indices:
                compressed_model.blocks[block_idx] = NoAttentionBlock(block)
                print(f"Block {block_idx}: No active attention heads, replaced with NoAttentionBlock")
                continue
                
            new_attn = CustomTimmAttention(
                dim=block.attn.qkv.in_features,
                num_heads=len(filtered_qk_sizes),  
                qk_sizes=filtered_qk_sizes,
                v_sizes=filtered_v_sizes,
                qkv_bias=block.attn.qkv.bias is not None,
                attn_drop=block.attn.attn_drop.p,
                proj_drop=block.attn.proj_drop.p
            )
            
            # 가중치 전송
            with torch.no_grad():
                old_qkv = block.attn.qkv.weight.data
                old_proj = block.attn.proj.weight.data
                qkv_dim = old_qkv.size(0) // 3
                head_dim = config['head_dim']
                
                # 새 가중치 수집
                new_q_weights = []
                new_k_weights = []
                new_v_weights = []
                
                for head_idx in active_head_indices:
                    start_idx = head_idx * head_dim
                    
                    # 인덱스 가져오기
                    q_indices = config['q_indices'][head_idx]
                    k_indices = config['k_indices'][head_idx]
                    v_indices = config['v_indices'][head_idx]
                    
                    # 가중치 추출 및 압축
                    q_head = old_qkv[start_idx:start_idx+head_dim]
                    k_head = old_qkv[qkv_dim+start_idx:qkv_dim+start_idx+head_dim]
                    v_head = old_qkv[2*qkv_dim+start_idx:2*qkv_dim+start_idx+head_dim]
                    
                    q_nonzero = q_head[q_indices]
                    k_nonzero = k_head[k_indices]
                    v_nonzero = v_head[v_indices]
                    
                    new_q_weights.append(q_nonzero)
                    new_k_weights.append(k_nonzero)
                    new_v_weights.append(v_nonzero)
                
                # 가중치 결합
                if new_q_weights:
                    # Q, K, V 가중치를 각각 먼저 결합한 후 전체 결합
                    new_qkv_weight = torch.cat([
                        torch.cat(new_q_weights, dim=0),
                        torch.cat(new_k_weights, dim=0),
                        torch.cat(new_v_weights, dim=0)
                    ], dim=0)
                    
                    new_attn.qkv.weight.data = new_qkv_weight
                
                # 바이어스 처리
                if block.attn.qkv.bias is not None:
                    old_qkv_bias = block.attn.qkv.bias.data
                    new_q_bias = []
                    new_k_bias = []
                    new_v_bias = []
                    
                    for head_idx in active_head_indices:
                        start_idx = head_idx * head_dim
                        
                        q_indices = config['q_indices'][head_idx]
                        k_indices = config['k_indices'][head_idx]
                        v_indices = config['v_indices'][head_idx]
                        
                        q_bias = old_qkv_bias[start_idx:start_idx+head_dim][q_indices]
                        k_bias = old_qkv_bias[qkv_dim+start_idx:qkv_dim+start_idx+head_dim][k_indices]
                        v_bias = old_qkv_bias[2*qkv_dim+start_idx:2*qkv_dim+start_idx+head_dim][v_indices]
                        
                        new_q_bias.append(q_bias)
                        new_k_bias.append(k_bias)
                        new_v_bias.append(v_bias)
                    
                    if new_q_bias:
                        new_qkv_bias = torch.cat([
                            torch.cat(new_q_bias, dim=0),
                            torch.cat(new_k_bias, dim=0),
                            torch.cat(new_v_bias, dim=0)
                        ], dim=0)
                        
                        new_attn.qkv.bias.data = new_qkv_bias
                
                # Projection 레이어 처리
                if hasattr(block.attn, 'proj'):
                    v_indices_flat = []
                    
                    for head_idx in active_head_indices:
                        start = head_idx * head_dim
                        v_ind = config['v_indices'][head_idx]
                        v_indices_flat.extend(start + idx for idx in v_ind)
                    
                    new_proj_weight = old_proj[:, v_indices_flat]
                    new_attn.proj.weight.data = new_proj_weight
                    
                    if block.attn.proj.bias is not None:
                        new_attn.proj.bias.data = block.attn.proj.bias.data
            
            # 원래 어텐션을 압축된 어텐션으로 교체
            block.attn = new_attn
            print(f"Block {block_idx}: Compressed attention with {len(filtered_qk_sizes)} heads of varying dimensions")
    
    return compressed_model

def compress_ffn_timm(model: VisionTransformer):
    compressed_model = copy.deepcopy(model)
    
    for block in compressed_model.blocks:  # .model 추가
        if hasattr(block, 'mlp'):
            # FC1 압축
            if hasattr(block.mlp, 'fc1'):
                fc1_weight = block.mlp.fc1.weight.data
                fc1_mask = (fc1_weight != 0)
                fc1_rows = fc1_mask.any(dim=1)
                fc1_cols = fc1_mask.any(dim=0)
                
                if fc1_rows.any() and fc1_cols.any():
                    new_fc1 = nn.Linear(
                        fc1_cols.sum().item(),
                        fc1_rows.sum().item(),
                        bias=block.mlp.fc1.bias is not None
                    )
                    new_fc1.weight.data = fc1_weight[fc1_rows][:, fc1_cols]
                    if block.mlp.fc1.bias is not None:
                        new_fc1.bias.data = block.mlp.fc1.bias.data[fc1_rows]
                    block.mlp.fc1 = new_fc1
            
            # FC2 압축
            if hasattr(block.mlp, 'fc2'):
                fc2_weight = block.mlp.fc2.weight.data
                if fc1_rows.any():  # fc1에서 계산된 rows 사용
                    new_fc2 = nn.Linear(
                        fc1_rows.sum().item(),
                        fc2_weight.size(0),
                        bias=block.mlp.fc2.bias is not None
                    )
                    new_fc2.weight.data = fc2_weight[:, fc1_rows]
                    if block.mlp.fc2.bias is not None:
                        new_fc2.bias.data = block.mlp.fc2.bias.data
                    block.mlp.fc2 = new_fc2
    
    return compressed_model

def measure_performance(model, device):
   model.eval()
   batch_size = 64
   num_iterations = 1000  
   warm_up = 200  
   
   total_params = sum(p.numel() for p in model.parameters()) / 1e6  # M
   
   dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
   
   with torch.no_grad():
       for _ in range(warm_up):
           _ = model(dummy_input)
           if device == "cuda":
               torch.cuda.synchronize()
   
   total_time = 0
   latencies = []
   
   with torch.no_grad():
       for _ in range(num_iterations):
           if device == "cuda":
               torch.cuda.synchronize()
           start = time.time()
           
           _ = model(dummy_input)
           
           if device == "cuda":
               torch.cuda.synchronize()
           end = time.time()
           
           batch_time = end - start
           latencies.append(batch_time * 1000)  # ms로 변환
           total_time += batch_time
   
   # 최종 메트릭 계산
   avg_latency = np.mean(latencies)
   throughput = (batch_size * num_iterations) / total_time  # images/second
   
   # # GFLOPs 계산
   input_sample = torch.randn(1, 3, 224, 224).to(device)
   flops = FlopCountAnalysis(model, input_sample)
   gflops = flops.total() / 1e9
   
   return {
       'latency': avg_latency,
       'throughput': throughput,
       'gflops': gflops,
       'parameters': total_params
   }

def replace_all_linear_layers(module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Linear):
            new_module = MaskLinear(sub_module.in_features, sub_module.out_features)
            with torch.no_grad():
                new_module.weight.copy_(sub_module.weight)
                if sub_module.bias is not None:
                    new_module.bias.copy_(sub_module.bias)
            setattr(module, name, new_module)
        else:
            replace_all_linear_layers(sub_module) 
            
def reverse_replace_all_linear_layers(module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, MaskLinear):
            new_module = nn.Linear(sub_module.in_features, sub_module.out_features)
            with torch.no_grad():
                new_module.weight.copy_(sub_module.weight)
                if sub_module.bias is not None:
                    new_module.bias.copy_(sub_module.bias)
            setattr(module, name, new_module)
        else:
            reverse_replace_all_linear_layers(sub_module) 


def remove_module_prefix(state_dict):
    return {k.replace('module.model.', ''): v for k, v in state_dict.items()}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def hyperparam():
    args = config.get_train_args()
    return args

def calculate_pruning_stats(model):
    """
    모델의 각 컴포넌트(어텐션, FFN)에 대한 프루닝 비율을 계산합니다.
    
    Args:
        model: 평가할 모델
        
    Returns:
        stats: 프루닝 통계가 담긴 딕셔너리
    """
    total_stats = {
        'attn': {'total': 0, 'pruned': 0},
        'ffn': {'total': 0, 'pruned': 0},
        'overall': {'total': 0, 'pruned': 0}
    }
    
    # 블록별 통계
    block_stats = []
    
    for i, block in enumerate(model.blocks):
        block_stat = {
            'block_idx': i,
            'attn': {'total': 0, 'pruned': 0, 'ratio': 0},
            'ffn': {'total': 0, 'pruned': 0, 'ratio': 0},
            'overall': {'total': 0, 'pruned': 0, 'ratio': 0}
        }
        
        # 어텐션 분석
        if hasattr(block, 'attn') and not isinstance(block, NoAttentionBlock):
            # QKV 가중치 분석
            if hasattr(block.attn, 'qkv'):
                qkv_weight = block.attn.qkv.weight.data
                qkv_total = qkv_weight.numel()
                qkv_pruned = (qkv_weight == 0).sum().item()
                
                block_stat['attn']['total'] = qkv_total
                block_stat['attn']['pruned'] = qkv_pruned
                block_stat['attn']['ratio'] = qkv_pruned / qkv_total * 100 if qkv_total > 0 else 0
                
                total_stats['attn']['total'] += qkv_total
                total_stats['attn']['pruned'] += qkv_pruned
            
            # Proj 가중치 분석 (일반적으로 어텐션에 포함)
            if hasattr(block.attn, 'proj'):
                proj_weight = block.attn.proj.weight.data
                proj_total = proj_weight.numel()
                proj_pruned = (proj_weight == 0).sum().item()
                
                block_stat['attn']['total'] += proj_total
                block_stat['attn']['pruned'] += proj_pruned
                block_stat['attn']['ratio'] = block_stat['attn']['pruned'] / block_stat['attn']['total'] * 100 if block_stat['attn']['total'] > 0 else 0
                
                total_stats['attn']['total'] += proj_total
                total_stats['attn']['pruned'] += proj_pruned
        elif isinstance(block, NoAttentionBlock):
            # 어텐션이 완전히 프루닝된 경우 (NoAttentionBlock으로 대체됨)
            # 원래 어텐션 레이어의 크기를 추정하기는 어려우므로, 이 경우 100% 프루닝으로 간주
            block_stat['attn']['ratio'] = 100.0
        
        # FFN 분석
        if hasattr(block, 'mlp'):
            # FC1 분석
            if hasattr(block.mlp, 'fc1'):
                fc1_weight = block.mlp.fc1.weight.data
                fc1_total = fc1_weight.numel()
                fc1_pruned = (fc1_weight == 0).sum().item()
                
                block_stat['ffn']['total'] = fc1_total
                block_stat['ffn']['pruned'] = fc1_pruned
                
                total_stats['ffn']['total'] += fc1_total
                total_stats['ffn']['pruned'] += fc1_pruned
            
            # FC2 분석
            if hasattr(block.mlp, 'fc2'):
                fc2_weight = block.mlp.fc2.weight.data
                fc2_total = fc2_weight.numel()
                fc2_pruned = (fc2_weight == 0).sum().item()
                
                block_stat['ffn']['total'] += fc2_total
                block_stat['ffn']['pruned'] += fc2_pruned
                
                total_stats['ffn']['total'] += fc2_total
                total_stats['ffn']['pruned'] += fc2_pruned
            
            # FFN 프루닝 비율 계산
            block_stat['ffn']['ratio'] = block_stat['ffn']['pruned'] / block_stat['ffn']['total'] * 100 if block_stat['ffn']['total'] > 0 else 0
        
        # 블록 전체 통계 계산
        block_stat['overall']['total'] = block_stat['attn']['total'] + block_stat['ffn']['total']
        block_stat['overall']['pruned'] = block_stat['attn']['pruned'] + block_stat['ffn']['pruned']
        block_stat['overall']['ratio'] = block_stat['overall']['pruned'] / block_stat['overall']['total'] * 100 if block_stat['overall']['total'] > 0 else 0
        
        block_stats.append(block_stat)
    
    # 전체 통계 계산
    total_stats['overall']['total'] = total_stats['attn']['total'] + total_stats['ffn']['total']
    total_stats['overall']['pruned'] = total_stats['attn']['pruned'] + total_stats['ffn']['pruned']
    
    # 프루닝 비율 계산
    for component in ['attn', 'ffn', 'overall']:
        if total_stats[component]['total'] > 0:
            total_stats[component]['ratio'] = total_stats[component]['pruned'] / total_stats[component]['total'] * 100
        else:
            total_stats[component]['ratio'] = 0
    
    return {
        'total': total_stats,
        'blocks': block_stats
    }

def print_pruning_stats(stats):
    """
    계산된 프루닝 통계를 콘솔에 출력합니다.
    
    Args:
        stats: calculate_pruning_stats 함수로부터 반환된 통계
    """
    print("\n" + "="*70)
    print("프루닝 통계 요약")
    print("="*70)
    
    # 전체 통계 출력
    total_stats = stats['total']
    print(f"전체 프루닝 비율: {total_stats['overall']['ratio']:.2f}%")
    print(f"어텐션 프루닝 비율: {total_stats['attn']['ratio']:.2f}% ({total_stats['attn']['pruned']} / {total_stats['attn']['total']} 매개변수)")
    print(f"FFN 프루닝 비율: {total_stats['ffn']['ratio']:.2f}% ({total_stats['ffn']['pruned']} / {total_stats['ffn']['total']} 매개변수)")
    print("-"*70)
    
    # 블록별 통계 헤더
    print(f"{'블록':^5} | {'어텐션 프루닝':^20} | {'FFN 프루닝':^20} | {'전체 프루닝':^20}")
    print("-"*70)
    
    # 각 블록 통계 출력
    for block in stats['blocks']:
        print(f"{block['block_idx']:^5} | {block['attn']['ratio']:^20.2f}% | {block['ffn']['ratio']:^20.2f}% | {block['overall']['ratio']:^20.2f}%")
    
    print("="*70)

def create_pruning_ratio_table(models_stats):
    """
    여러 모델의 프루닝 비율을 비교하는 테이블을 출력합니다.
    
    Args:
        models_stats: 모델 이름을 키로, 프루닝 통계를 값으로 가지는 딕셔너리
    """
    print("\n" + "="*90)
    print("모델 프루닝 비율 비교")
    print("="*90)
    
    print(f"{'모델':^15} | {'어텐션 프루닝':^20} | {'FFN 프루닝':^20} | {'전체 프루닝':^20}")
    print("-"*90)
    
    for model_name, stats in models_stats.items():
        total_stats = stats['total']
        attn_ratio = total_stats['attn']['ratio']
        ffn_ratio = total_stats['ffn']['ratio']
        overall_ratio = total_stats['overall']['ratio']
        
        print(f"{model_name:^15} | {attn_ratio:^20.2f}% | {ffn_ratio:^20.2f}% | {overall_ratio:^20.2f}%")
    
    print("="*90)


def main(args):    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    device = torch.device(args.device)

    # 데이터셋 및 데이터로더 설정
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    test_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * 512),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 모델 이름 및 체크포인트 경로 설정
    # Create model using timm
    model_configs = {
       'tiny': {
           'model': 'vit_tiny_patch16_224',
           'checkpoint': 'checkpoint/deit_tiny_patch16_224-a1311bcf.pth',
           'prompt_dim': 192
       },
       'small': {
           'model': 'deit_small_patch16_224.fb_in1k', 
           'checkpoint': 'checkpoint/deit_small_patch16_224-cd65a155.pth',
           'prompt_dim': 384
       },
       'base': {
           'model': 'vit_base_patch16_224',
           'checkpoint': 'checkpoint/deit_base_patch16_224-b5f2ef4d.pth',
           'prompt_dim': 768
       }
    }
    
    model_name = model_configs[args.model_size]['model']
    checkpoint_path = model_configs[args.model_size]['checkpoint']
    
    # 모델 로드
    original_model = timm.create_model(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    original_model.load_state_dict(checkpoint['model'])
    original_model = original_model.to(device)

    # test_stats1 = comp_evaluate(test_loader, original_model, device)
    # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats1['acc1']:.1f}%")
    
    pruned_model = timm.create_model(model_name, pretrained=True)
    zeroing_model = timm.create_model(model_name, pretrained=True)
    ffn_compression_model = timm.create_model(model_name, pretrained=True)
    attn_compression_model = timm.create_model(model_name, pretrained=True)

    # MaskLinear 교체
    for model in [pruned_model, zeroing_model, ffn_compression_model, attn_compression_model]:
        replace_all_linear_layers(model)
    
    # 체크포인트 로드
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = remove_module_prefix(state_dict)
        
        for model in [pruned_model, zeroing_model, ffn_compression_model, attn_compression_model]:
            model.load_state_dict(state_dict)


    models = [original_model, pruned_model, zeroing_model, ffn_compression_model, attn_compression_model]
    
    for model in models :
       for name, module in model.named_modules():
           if hasattr(module, 'mask'):
               # 가중치 출력 차원 마스크
               module.weight.data *= module.mask
               output_mask = module.mask.any(dim=1)
               weight_indices = torch.nonzero(output_mask).squeeze().tolist()
               
               # 바이어스 마스크
               bias_indices = []
               if hasattr(module, 'bias') and module.bias is not None:
                   if hasattr(module, 'bias_mask'):
                       bias_indices = torch.nonzero(module.bias_mask).squeeze().tolist()
        
    # MaskLinear를 일반 Linear로 다시 변환
    for model in [pruned_model, zeroing_model, ffn_compression_model, attn_compression_model]:
        reverse_replace_all_linear_layers(model)
    
    # 모델 압축 적용
    # FFN만 압축
    ffn_compression_model = compress_ffn_timm(ffn_compression_model)
    pruned_model = compress_ffn_timm(pruned_model)
    
    # Attention만 압축
    head_configs_attn = get_pruned_head_config_timm(attn_compression_model)
    attn_compression_model = compress_timm_model(attn_compression_model, head_configs_attn)
    
    # 전체 압축
    head_configs = get_pruned_head_config_timm(pruned_model)
    print(head_configs)
    pruned_model = compress_timm_model(pruned_model, head_configs)
    pruned_model = compress_ffn_timm(pruned_model)
    
    # GPU로 이동
    models = [original_model, zeroing_model, ffn_compression_model, attn_compression_model, pruned_model]
    
    for model in models:
        model.to(device)
    
    print(pruned_model)
    
    # 각 모델에 대한 프루닝 통계 계산
    models_stats = {}
    model_names = ['original', 'zeroing', 'ffn_compressed', 'attn_compressed', 'fully_compressed']
    
    for name, model in zip(model_names, models):
        # 원본 모델은 프루닝되지 않았으므로 건너뜀
        if name == 'original':
            continue
        stats = calculate_pruning_stats(model)
        models_stats[name] = stats
    
    # 프루닝 비율 비교 테이블 출력
    create_pruning_ratio_table(models_stats)
    
    # 상세 프루닝 통계 출력 - 완전히 압축된 모델
    zeroing_stats = calculate_pruning_stats(zeroing_model)
    print("\nZeroing 모델(압축 전)의 프루닝 통계:")
    print_pruning_stats(zeroing_stats)
    
    # 압축된 모델의 정확도 평가
    test_stats = comp_evaluate(test_loader, pruned_model, device)
    print(f"\nAccuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    # 성능 측정
    print("\nMeasuring performance...")
    torch.cuda.empty_cache()
    
    performances = {}
    for model_name, model in zip(['original', 'zeroing', 'ffn', 'attn', 'full'], models):
        torch.cuda.empty_cache()
        performances[model_name] = measure_performance(model, device)
    
    # 결과 테이블 출력
    create_table_header()
    
    metrics = ['Throughput', 'Latency', 'GFLOPs', 'Parameters']
    for metric in metrics:
       metric_key = metric.lower()
       print(format_metric_line(
           metric,
           performances['original'][metric_key],
           performances['zeroing'][metric_key],
           performances['ffn'][metric_key],
           performances['attn'][metric_key], 
           performances['full'][metric_key]
       ))
    
    print("-" * 94)
    
    # 최종 프루닝 및 성능 요약
    print("\n" + "="*90)
    print("최종 프루닝 및 성능 요약")
    print("="*90)
    print(f"모델: {model_name}, 크기: {args.model_size}")
    print(f"체크포인트: {args.checkpoint}")
    print("\n프루닝 비율:")
    print(f"어텐션 프루닝: {models_stats['fully_compressed']['total']['attn']['ratio']:.2f}%")
    print(f"FFN 프루닝: {models_stats['fully_compressed']['total']['ffn']['ratio']:.2f}%")
    print(f"전체 프루닝: {models_stats['fully_compressed']['total']['overall']['ratio']:.2f}%")
    print("\n성능 향상:")
    print(f"Throughput 향상: {performances['full']['throughput']/performances['original']['throughput']:.2f}x")
    print(f"Latency 감소: {performances['original']['latency']/performances['full']['latency']:.2f}x")
    print(f"GFLOPs 감소: {performances['original']['gflops']/performances['full']['gflops']:.2f}x")
    print(f"Parameters 감소: {performances['original']['parameters']/performances['full']['parameters']:.2f}x")
    print(f"정확도: {test_stats['acc1']:.2f}%")
    print("="*90)
    
if __name__ == '__main__':
    args = get_train_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of visible GPUs according to PyTorch: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
             print(f"Name of the first visible GPU: {torch.cuda.get_device_name(0)}") # PyTorch 내부 인덱스 0번 장치 이름
             print(f"Current device index: {torch.cuda.current_device()}")
        else:
             print("PyTorch sees 0 CUDA devices.")
    else:
        print("PyTorch cannot see any CUDA devices.")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    main(args)
