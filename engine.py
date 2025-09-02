"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from utils import *
# from backpack import backpack, extend
# from backpack.extensions import BatchGrad
import pruning
from pruning import *

import torch.nn.functional as F
import torch.func

#from torch.func import vmap, grad as functional_grad
def compute_ce_loss_per_sample(params, buffers, model, criterion, x_sample, y_sample):
    output = torch.func.functional_call(model, (params, buffers), (x_sample.unsqueeze(0),))
    return criterion(output, y_sample.unsqueeze(0))

# DML Loss만을 위한 배치 단위 손실 함수 (새로 추가)
# def compute_dml_loss_value(model, pre_samples, pre_out, args):
#     """섭동된 모델과 이전 스텝의 출력을 사용해 DML Loss 값을 계산합니다."""
#     # out_pre는 이미 .clone().detach()가 적용되었을 겁니다. (이전 수정)
#     out_pre = model(pre_samples[:, 1, ...]).clone().detach()

#     # pre_out 역시 뷰일 가능성이 있으므로 안전하게 clone().detach() 적용
#     pre_out_safe = pre_out.clone().detach() # <-- 새로운 변수 이름으로 안전하게

#     # F.log_softmax와 F.softmax의 결과를 완전히 분리된 텐서로 만듭니다.
#     # .float()은 필요하다면 유지 (이전 AMP 비활성화로 이미 float32일 가능성 높음)
#     log_probs = F.log_softmax(out_pre / args.T, dim=1).float().clone().detach()
#     target_probs = F.softmax(pre_out_safe / args.T, dim=1).float().clone().detach() # pre_out_safe 사용

#     # KL Div Loss 계산
#     dml_loss = F.kl_div(
#         log_probs, # 수정된 변수 사용
#         target_probs, # 수정된 변수 사용
#         reduction="batchmean",
#     ) * (args.T * args.T)


#     return args.alpha * dml_loss
def calculate_revival_loss(model):
    """
    모델의 죽은 가중치에 대한 Revival Loss 계산
    """
    total_revival_loss = 0.0
    total_dead_weights = 0
    diversity_weight=0.005
    revival_strength=0.01
    min_strength=0.001
    adaptive_decay=0.99
        
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight = module.weight
            mask = module.mask
                
            # 죽은 가중치 추출
            dead_mask = (1-mask)
            dead_weights = weight * dead_mask
            # print(torch.sum(dead_mask),"============================")
                
            if torch.sum(dead_mask) > 0:
                    # L2 Revival Loss
                dead_weights_norm = torch.norm(dead_weights, p=2)
                if dead_weights_norm > 1e-8:  # 0 방지
                    revival_term = revival_strength * dead_weights_norm
                    total_revival_loss += revival_term
                    total_dead_weights += torch.sum(dead_mask).item()
                        
                        # Diversity Loss (선택적)
                    # if self.diversity_weight > 0 and weight.size(0) > 1:
                    #     diversity_loss = self._calculate_diversity_loss(
                    #         weight, dead_mask, dead_weights_norm
                    #     )
                        # total_revival_loss += diversity_loss
        
    return total_revival_loss

def kldiv( logits, targets, T=2.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=2

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss
        
criterion_kl = KLLoss().cuda()

def get_current_masks(model):
    """모델의 현재 마스크 상태를 딕셔너리로 반환"""
    current_masks = {}
    count = 1
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            current_masks[name] = module.mask.clone().detach()

    return current_masks

def calculate_mask_changes(prev_masks, current_masks):
    """이전 마스크와 현재 마스크의 차이를 계산 (0에서 1이 된 것만 카운트)"""
    if prev_masks is None:
        return 0
    count = 0
    total_changes = 0
    for name in current_masks:
        if name in prev_masks:
            
            # 이전 마스크가 0이고 현재 마스크가 1인 경우만 카운트
            changes = torch.logical_and(
                ~prev_masks[name].bool(),  # 이전 마스크가 0인 위치
                current_masks[name].bool()  # 현재 마스크가 1인 위치
            )
            total_changes += torch.sum(changes).item()
    
    return total_changes


def calculate_mask_changes_real(prev_masks, current_masks):
    """이전 마스크와 현재 마스크의 모든 변화(0↔1)를 계산"""
    if prev_masks is None:
        return 0
    total_changes = 0
    for name in current_masks:
        if name in prev_masks:
            # 이전 마스크와 현재 마스크가 다른 모든 위치 카운트
            changes = torch.logical_xor(
                prev_masks[name].bool(),  # 이전 마스크
                current_masks[name].bool()  # 현재 마스크
            )
            total_changes += torch.sum(changes).item()
    return total_changes


def fine_train_one_epoch(model: torch.nn.Module, criterion:torch.nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None ):
    
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)

        optimizer.first_step(zero_grad=True) # 
        with torch.cuda.amp.autocast():
           outputs = model(samples[:, 0, ...]) 
           if not args.cosub:
               cls_loss = criterion(outputs, targets)
           else:
               outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
               cls_loss = 0.25 * criterion(outputs[0], targets)
               cls_loss = cls_loss + 0.25 * criterion(outputs[1], targets)
               cls_loss = cls_loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
               cls_loss = cls_loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())
               
           # prompt_loss = compute_prompt_loss(layer_outputs, model, args.prompt_length)

           loss = cls_loss 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # optimizer.zero_grad()
        ##############################
        # 수동으로 역전파 수행 (그래디언트 계산만, 가중치 업데이트 없음)
        scaler = loss_scaler._scaler  # 내부 GradScaler 객체 접근
        
        # 스케일링된 손실을 사용하여 역전파
        scaled_loss = scaler.scale(loss)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        scaled_loss.backward(create_graph=is_second_order)
        # 그래디언트 클리핑 (필요한 경우)
        scaler.unscale_(optimizer)

            
        optimizer.second_step(zero_grad=False) #원래 여기원래 여기 내부에서 가중치 업데이트도 하는데 수정해서 안해서 grad = false로 둠 
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def fine_evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    model.module.set_all_type_values(0)
    print("et_all_type_values == 0 으로ㅓ 성ㄹ정")
    
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)  
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    model.module.set_all_type_values(0)
    print("et_all_type_values == 0 으로ㅓ 성ㄹ정")
    
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # output, _ = model(images)  # 두 번째 반환값 무시
            output = model(images)  # 두 번째 반환값 무시
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def comp_evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)  # 두 번째 반환값 무시
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}







def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                   max_norm=0, model_ema=None, mixup_fn=None, set_training_mode=True,
                   args=None, iteration=0, prev_masks=None, mask_changes=0):
   
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    attn_target_sparsity = 0
    ffn_target_sparsity = 0
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
      
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
          
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
          
        # 타겟 스파시티 계산 (어텐션과 FFN에 대해 별도로)
        if (epoch + 1) < args.target_epoch:
            progression = iteration / (args.target_epoch * len(data_loader))
            cubic_decay = (1 - progression) ** 3
            
            attn_target_sparsity = args.attn_prune_rate - args.attn_prune_rate * cubic_decay
            ffn_target_sparsity = args.ffn_prune_rate - args.ffn_prune_rate * cubic_decay
        else:
            attn_target_sparsity = args.attn_prune_rate
            ffn_target_sparsity = args.ffn_prune_rate

        if epoch == args.target_epoch:
            print("epoch == args.target_epoch이라 0으로 변경함 타겟 에포크:50 이하일땐 나오면 안됨")
            model.module.set_all_type_values(0)



        optimizer.first_step(zero_grad=True) # 
        #print("first step 실 행됨")
        # 순방향 전파 및 손실 계산
        # with torch.cuda.amp.autocast():

        
        with torch.amp.autocast('cuda'):
            outputs = model(samples[:, 0, ...]) 
            if not args.cosub:
                cls_loss = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                cls_loss = 0.25 * criterion(outputs[0], targets)
                cls_loss = cls_loss + 0.25 * criterion(outputs[1], targets)
                cls_loss = cls_loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                cls_loss = cls_loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())
               
            loss = cls_loss

            #### 이 바로 아래가 dynamic부분 
            revival_loss = 0.0
            if args.mask_type == 1:
                revival_loss = calculate_revival_loss(model)

            loss = loss + revival_loss
            
        
        loss_value = loss.item()
        #print("losssssss",  loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        

        
        
        # 수동으로 역전파 수행 (그래디언트 계산만,loss_scaler 가중치 업데이트 없음)
        
        scaler = loss_scaler._scaler  # 내부 GradScaler 객체 접근
        
        # # 스케일링된 손실을 사용하여 역전파
        scaled_loss = scaler.scale(loss)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        scaled_loss.backward(create_graph=is_second_order)
        
        #loss.backward(create_graph=is_second_order)

        


        # 그래디언트 클리핑 (필요한 경우)
        scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        #print("second step 실 행")
        
        if args.second_step_after_mask == 1: #세컨스텝 후 마스크 
            optimizer.second_step(zero_grad=False)
        else: #마스크를
            pass #원래 여기원래 여기 내부에서 가중치 업데이트도 하는데 수정해서 안해서 grad = false로 둠 

        if epoch >= args.warmup:
            for name, module in model.named_modules():
                if isinstance(module, MaskLinear):
                    weight = module.weight
                    grad = module.weight.grad 

        
        if epoch >= args.warmup and epoch < args.target_epoch:
            if i % args.prune_freq == 0:
                # 이제 별도의 프루닝 비율로 마스크 적용
                if args.method == 'ours':
                    masks = pruning.get_vit_masks_with_two_stage_grouping(
                        model, 
                        attn_target_sparsity, 
                        ffn_target_sparsity, 
                        args.prune_imp, 
                        args.mag_type, 
                        args.group_size
                    )
                    pruning.apply_vit_masks(model, masks) ##여기에서 직접 weight에 0이 되는것이 아닌. mask만 0이 됨 
                
                elif args.method == 'unified_qkv':
                    masks = pruning.get_vit_masks_with_two_stage_grouping_unified_qkv(
                        model,
                        attn_target_sparsity,
                        ffn_target_sparsity,
                        args.prune_imp,
                        args.mag_type,
                        args.group_size
                    )
                    pruning.apply_vit_masks(model, masks)

                elif args.method == 'neuron':
                    masks = pruning.get_simple_vit_masks(
                        model,
                        attn_target_sparsity,
                        ffn_target_sparsity,
                        args.prune_imp,
                        args.mag_type
                    )
                    pruning.apply_vit_masks(model, masks)
                    
                # 마스크 변경 추적                  

                current_masks = get_current_masks(model)
                now_mask_change = calculate_mask_changes(prev_masks, current_masks)
                
                if args.random_mask_change == 1: 
                    if prev_masks is not None and now_mask_change == 0:
                        #print(f"\n[INFO] Masks unchanged at iteration {i}. Forcing changes...")
                        if args.method == 'unified_qkv':
                            # 새로 만든 'change_zero' 함수를 호출
                            perturbed_masks = pruning.get_vit_masks_with_two_stage_grouping_unified_qkv_change_zero(
                                model,
                                attn_target_sparsity,
                                ffn_target_sparsity,
                                args.prune_imp,
                                args.mag_type,
                                args.group_size
                            )
                            pruning.apply_vit_masks(model, perturbed_masks)
                            current_masks = get_current_masks(model)
                            print("mash change 0이라 섭동 랜덤하게 추가했음")
                            mask_changes += calculate_mask_changes(prev_masks, current_masks)
                    else:
                        mask_changes += now_mask_change
                else:
                    mask_changes += now_mask_change
                        
                prev_masks = {name: mask.clone() for name, mask in current_masks.items()}
                metric_logger.update(mask_changes=mask_changes)
                
            iteration += 1            
        
        # 이제 가중치 업데이트 수행
        
        if args.second_step_after_mask == 1: #세컨스텝 후 마스크 (이미 위에서 세컨스텝 했음ㅂ)
            pass
        else: #마스크를
            optimizer.second_step(zero_grad=False)
        # optimizer.step()
        scaler.step(optimizer)
        
        scaler.update()


        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 현재 프루닝 비율 로깅
        metric_logger.update(attn_sparsity=attn_target_sparsity)
        metric_logger.update(ffn_sparsity=ffn_target_sparsity)
      
    metric_logger.synchronize_between_processes()
    print('Mask changes:', mask_changes) 
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, attn_target_sparsity, ffn_target_sparsity, iteration, prev_masks, mask_changes







# compute_dml_loss_value 함수는 기존과 동일하게 사용합니다.
def compute_dml_loss_value(model, pre_samples, pre_out, args):
    """섭동된 모델과 이전 스텝의 출력을 사용해 DML Loss 값을 계산합니다."""
    
    # autocast의 영향을 받아 out_pre는 float16일 수 있습니다.
    out_pre = model(pre_samples[:, 1, ...])
    
    pre_out_safe = pre_out.clone().detach()

    # ‼️ [수정] 수치 안정성을 위해 연산 전에 명시적으로 float32로 캐스팅합니다.
    log_probs = F.log_softmax((out_pre / args.T).float(), dim=1)
    target_probs = F.softmax((pre_out_safe / args.T).float(), dim=1)

    # 이제 log_probs와 target_probs는 안정적인 float32 텐서입니다.
    dml_loss = F.kl_div(
        log_probs,
        target_probs,
        reduction="batchmean",
    ) * (args.T * args.T)

    return args.alpha * dml_loss

def train_one_epoch_DLB_test(model,criterion, data_loader, optimizer, device, epoch, loss_scaler,
                   max_norm=0, model_ema=None, mixup_fn=None, set_training_mode=True,
                   args=None, iteration=0, prev_masks=None, mask_changes=0):

    
    model.train(set_training_mode)
    #get_per_sample_grads = vmap(functional_grad(compute_ce_loss_per_sample, argnums=(0, 1)), in_dims=(None, None, None, None, 0, 0), randomness='same')
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    attn_target_sparsity = 0
    ffn_target_sparsity = 0
    

    metric_logger = MetricLogger(delimiter="  ")

    pre_data = None
    pre_out = None


    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = samples.to(device), targets.to(device)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
          
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)


        if args.mask_finding_cos == 0:
            # --- 프루닝 비율 계산 (기존과 동일) ---
            if (epoch + 1) < args.target_epoch:
                progression = iteration / (args.target_epoch * len(data_loader))
                attn_target_sparsity = args.attn_prune_rate * (1 - (1 - progression) ** 3)
                ffn_target_sparsity = args.ffn_prune_rate * (1 - (1 - progression) ** 3)
            else:
                attn_target_sparsity = args.attn_prune_rate
                ffn_target_sparsity = args.ffn_prune_rate
        elif args.mask_finding_cos == 1:
            if (epoch + 1) < args.target_epoch:
                total_iterations = args.target_epoch * len(data_loader)
                progression = min(1.0, iteration / total_iterations)

                  # 1. 선형적으로 증가하는 기본 추세
                base_sparsity_attn = args.attn_prune_rate * progression
                base_sparsity_ffn = args.ffn_prune_rate * progression

                  # 2. 주기를 가지는 진동 부분 (Sine 함수 사용)
                  # progression이 0에서 1로 갈 때, sin(cycles * 2 * pi * progression)는 'cycles' 횟수만큼 진동합니다.
                oscillation = args.prune_amplitude * math.sin(args.prune_cycles * 2 * math.pi * progression)

                  # 3. 기본 추세와 진동을 더해 최종 비율 계산
                attn_target_sparsity = base_sparsity_attn + oscillation
                ffn_target_sparsity = base_sparsity_ffn + oscillation

                  # 프루닝 비율이 0 미만 또는 1 초과가 되지 않도록 범위 제한 (매우 중요)
                attn_target_sparsity = max(0.0, min(1.0, attn_target_sparsity))
                ffn_target_sparsity = max(0.0, min(1.0, ffn_target_sparsity))

            else:
                  # 목표 에폭 도달 시 최종 목표 비율로 고정
                attn_target_sparsity = args.attn_prune_rate
                ffn_target_sparsity = args.ffn_prune_rate
            

        if epoch == args.target_epoch: model.module.set_all_type_values(0)

        # ========================================================================
        # 1. 첫 번째 스텝 (Bootstrapping)
        # ========================================================================
        # ========================================================================
        # 1. 첫 번째 스텝 (Bootstrapping)
        # ========================================================================
        if i == 0:
            print("첫 이터레이션 준비 (FP32 Full Precision)")
            optimizer.zero_grad()
            optimizer.first_step(zero_grad=True)
            
            outputs = model(samples[:, 0, ...])
            loss = criterion(outputs, targets)

            # ❗ loss_scaler 없이 직접 backward 호출
            loss.backward()

            optimizer.second_step(zero_grad=True)

            # ❗ loss_scaler 없이 직접 optimizer step 호출
            optimizer.step()
            optimizer.zero_grad()

            pre_data = (samples.clone(), targets.clone())
            pre_out = outputs.clone().detach()
            continue # 다음 루프로

        # ========================================================================
        # 2. SAM의 first_step
        # ========================================================================
        # --- SAM의 first_step ---
        optimizer.zero_grad()

        optimizer.first_step(zero_grad=True) # 가중치 섭동

        with torch.cuda.amp.autocast():
            #####################이제 전체 배치에 대해(이전 배치의 새로운 뷰 + 현재 배치의 뷰 0
            pre_samples, pre_targets = pre_data
            all_samples = torch.cat((pre_samples[:, 1, ...], samples[:, 0, ...]), dim=0) #이전샘플의 현재 view 에서 
            all_targets = torch.cat((pre_targets, targets), dim=0)
            
            all_outputs = model(all_samples)
            total_loss = criterion(all_outputs, all_targets)

            if args.DLB_loss == 1:
                dml_loss = compute_dml_loss_value(model, pre_samples, pre_out, args)
                total_loss = total_loss + dml_loss
                

            loss_value = total_loss.item()
    
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
                
            loss_scaler._scaler.scale(total_loss).backward()
            
            loss_scaler._scaler.unscale_(optimizer)
        #if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)



        # ========================================================================
        # 4. 상태 업데이트
        # ========================================================================
        pre_data = (samples.clone(), targets.clone()) #둘다 현재 배치 
        # all_outputs에서 현재 배치에 해당하는 부분만 저장
        pre_out = all_outputs[len(pre_targets):].clone().detach() #이것도 현재 배치의 opuitput 

        ###mask finding for save e
        if args.prune_with_e == 1 :
            if epoch >= args.warmup and i % args.prune_freq == 0:
                info_for_pruning = {}
                for name, p in model.named_parameters():
                    if p.requires_grad and "e_w" in optimizer.state[p]:
                        info_for_pruning[name] = {"e_w": optimizer.state[p]["e_w"].clone()}
                        
        optimizer.second_step(zero_grad=False)
            

            
        if epoch >= args.warmup:
            for name, module in model.named_modules():
                if isinstance(module, MaskLinear):
                    weight = module.weight
                    grad = module.weight.grad 

        # ==========================================================
        if epoch >= args.warmup and epoch < args.target_epoch:
            if i % args.prune_freq == 0:
                if args.method == 'unified_qkv':
                    if args.prune_with_e == 0 :
                        masks = pruning.get_vit_masks_with_two_stage_grouping_unified_qkv(
                            model,
                            attn_target_sparsity,
                            ffn_target_sparsity,
                            args.prune_imp,
                            args.mag_type,
                            args.group_size,
                        )
                    elif  args.prune_with_e == 1 :
                        masks = pruning.get_vit_masks_with_two_stage_grouping_unified_qkv_with_e(
                            info_for_pruning,
                            model,
                            attn_target_sparsity,
                            ffn_target_sparsity,
                            args.prune_imp,
                            args.mag_type,
                            args.group_size,
                        )
                        
                    pruning.apply_vit_masks(model, masks)

                current_masks = get_current_masks(model)
                now_mask_change = calculate_mask_changes(prev_masks, current_masks)
                mask_changes += now_mask_change
                        
                prev_masks = {name: mask.clone() for name, mask in current_masks.items()}
                metric_logger.update(mask_changes=mask_changes)
                
            iteration += 1            
        
        # 이제 가중치 업데이트 수행
        loss_scaler._scaler.step(optimizer)  # 또는 SAM의 optimizer.second_step()
        loss_scaler._scaler.update()


        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 현재 프루닝 비율 로깅
        metric_logger.update(attn_sparsity=attn_target_sparsity)
        metric_logger.update(ffn_sparsity=ffn_target_sparsity)
      
    metric_logger.synchronize_between_processes()
    print('Mask changes:', mask_changes) 
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, attn_target_sparsity, ffn_target_sparsity, iteration, prev_masks, mask_changes


def fine_train_one_epoch_DLB_test(model,criterion, data_loader, optimizer, device, epoch, loss_scaler,
                           max_norm=0, model_ema=None, mixup_fn=None, set_training_mode=True,
                           args=None):

    
    model.train(set_training_mode)
    #get_per_sample_grads = vmap(functional_grad(compute_ce_loss_per_sample, argnums=(0, 1)), in_dims=(None, None, None, None, 0, 0), randomness='same')
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    
    
    metric_logger = MetricLogger(delimiter="  ")
    # ... (이하 metric_logger, header 등 초기 설정은 기존과 동일) ...
    pre_data = None
    pre_out = None

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = samples.to(device), targets.to(device)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
          
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)



        # ========================================================================
        # 1. 첫 번째 스텝 (Bootstrapping)
        # ========================================================================
        if i == 0:
            print("첫 이터레이션 준비 (FP32 Full Precision)")
            optimizer.zero_grad()
            optimizer.first_step(zero_grad=True)
            
            outputs = model(samples[:, 0, ...])
            loss = criterion(outputs, targets)

            # ❗ loss_scaler 없이 직접 backward 호출
            loss.backward()

            optimizer.second_step(zero_grad=False)

            # ❗ loss_scaler 없이 직접 optimizer step 호출
            optimizer.step()
            optimizer.zero_grad()

            pre_data = (samples.clone(), targets.clone())
            pre_out = outputs.clone().detach()
            continue # 다음 루프로
       
        # ========================================================================
        # 2. SAM의 first_step
        # ========================================================================
        # --- SAM의 first_step ---
        optimizer.zero_grad()

        optimizer.first_step(zero_grad=True) # 가중치 섭동
        
        with torch.cuda.amp.autocast():
            #####################이제 전체 배치에 대해(이전 배치의 새로운 뷰 + 현재 배치의 뷰 0
            pre_samples, pre_targets = pre_data
            all_samples = torch.cat((pre_samples[:, 1, ...], samples[:, 0, ...]), dim=0) #이전샘플의 현재 view 에서 
            all_targets = torch.cat((pre_targets, targets), dim=0)
            
            all_outputs = model(all_samples)
            total_loss = criterion(all_outputs, all_targets)
    
            if args.DLB_loss == 1:
                dml_loss = compute_dml_loss_value(model, pre_samples, pre_out, args)
                total_loss = total_loss + dml_loss
                
            # 💡 [핵심 변경 1] 이 backward()는 오직 가중치 업데이트만을 위한 것입니다.
            
            loss_value = total_loss.item()
    
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
                
            loss_scaler._scaler.scale(total_loss).backward()
            
            loss_scaler._scaler.unscale_(optimizer)
        #if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


        # ========================================================================
        # 4. 상태 업데이트
        # ========================================================================
        pre_data = (samples.clone(), targets.clone()) #둘다 현재 배치 
        # all_outputs에서 현재 배치에 해당하는 부분만 저장
        pre_out = all_outputs[len(pre_targets):].clone().detach() #이것도 현재 배치의 opuitput 
        
        optimizer.second_step(zero_grad=False)


        # 이제 가중치 업데이트 수행
        loss_scaler._scaler.step(optimizer)  # 또는 SAM의 optimizer.second_step()
        loss_scaler._scaler.update()
        
        optimizer.zero_grad()
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}