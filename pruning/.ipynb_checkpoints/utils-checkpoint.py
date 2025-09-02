import torch
import torch.nn as nn
import numpy as np
from .dpf.mnn import MaskLinear

def get_importance(weight, imp_type, grad=None, activation=None):
    """기본 중요도 계산 함수"""
    if imp_type == 'L1':
        return weight.abs().mean(dim=1).detach().cpu().numpy()
    elif imp_type == 'L2':
        return weight.pow(2).mean(dim=1).detach().cpu().numpy()
    elif imp_type == 'taylor1':
        if grad is None:
            raise ValueError("Gradient required for Taylor expansion methods")
        return (weight * grad).abs().mean(dim=1).detach().cpu().numpy()
    elif imp_type == 'act_taylor':
        if grad is None or activation is None:
            raise ValueError("Both gradient and activation required for act_taylor method")
        # 배치 차원에 대해 활성화 값의 평균 계산
        mean_activation = activation.mean(dim=0)  # [in_features]
        # 활성화, 가중치, 그래디언트의 곱으로 중요도 계산
        return (weight * mean_activation.unsqueeze(0) * grad).abs().mean(dim=1).detach().cpu().numpy()
    elif imp_type == 'grad':
        return grad.abs().mean(dim=1).detach().cpu().numpy()
        
    else:
        raise ValueError("Invalid importance type. Choose 'L1', 'L2', 'taylor1', or 'act_taylor'.")

def taylor_importance(weight, grad, imp_type):
    if imp_type == 'taylor1':
        return (weight * grad).abs().mean(dim=1).detach().cpu().numpy()

def get_threshold(importance_all, rate):
    """임계값 계산 함수"""
    return np.percentile(importance_all, rate * 100)

def expand_mask(mask, in_features):
    """마스크 확장 함수"""
    return np.repeat(mask[:, np.newaxis], in_features, axis=1)

def get_independent_head_importance(weight_or_grad, num_heads, imp_type):
    """독립적인 헤드별 중요도 계산"""
    # weight shape: [out_dim, in_dim] where out_dim = embed_dim
    # For timm ViT, qkv is combined into one linear layer
    # out_dim = 3 * num_heads * head_dim
    out_dim, in_dim = weight_or_grad.shape
    head_dim = out_dim // (3 * num_heads)
    
    # Reshape to [3(qkv), num_heads, head_dim, in_dim]
    weight_reshaped = weight_or_grad.reshape(3, num_heads, head_dim, in_dim)
    
    if imp_type == 'L1':
        return weight_reshaped.abs().mean(dim=-1).detach().cpu().numpy()
    elif imp_type == 'L2':
        return weight_reshaped.pow(2).mean(dim=-1).detach().cpu().numpy()
    

def get_gradient_importance(grad: torch.Tensor, num_heads: int):
    """
    오직 그래디언트의 크기만을 사용하여 헤드별 중요도를 계산합니다.

    Args:
        grad (torch.Tensor): 그래디언트 텐서. 
                             예상 shape: [out_dim, in_dim]
        num_heads (int): 어텐션 헤드의 수.

    Returns:
        numpy.ndarray: 헤드별 그래디언트 중요도.
                       shape: [3(qkv), num_heads, head_dim]
    """
    # 그래디언트가 제대로 전달되었는지 확인
    if grad is None:
        raise ValueError("Gradient tensor is required, but got None.")
        
    # 그래디언트 텐서의 shape 가져오기
    out_dim, in_dim = grad.shape
    
    # Q, K, V를 포함한 각 헤드의 차원 계산
    # out_dim은 3 * num_heads * head_dim과 같아야 함
    if out_dim % (3 * num_heads) != 0:
        raise ValueError(f"out_dim ({out_dim}) is not divisible by 3 * num_heads ({3 * num_heads})")
    head_dim = out_dim // (3 * num_heads)
    
    # 그래디언트를 [3(qkv), num_heads, head_dim, in_dim] 형태로 재구성
    grad_reshaped = grad.reshape(3, num_heads, head_dim, in_dim)
    
    # 각 헤드 차원의 평균 그래디언트 절대값 크기를 중요도로 계산
    # 입력 차원(in_dim)에 대해 평균을 냅니다.
    importance = grad_reshaped.abs().mean(dim=-1)
    
    # 결과를 numpy 배열로 변환하여 반환
    return importance.detach().cpu().numpy()
def get_taylot_importance(weight, grad, num_heads, imp_type):
    out_dim, in_dim = weight.shape
    head_dim = out_dim // (3 * num_heads)
    
    # Reshape to [3(qkv), num_heads, head_dim, in_dim]
    weight_reshaped = weight.reshape(3, num_heads, head_dim, in_dim)
    grad_reshaped = grad.reshape(3, num_heads, head_dim, in_dim)
    
    if imp_type == 'taylor1':
        if grad is None:
            raise ValueError("Gradient required for Taylor expansion methods")

        return (weight_reshaped * grad_reshaped).abs().mean(dim=-1).detach().cpu().numpy()


def get_activation_taylor_importance(weight, grad, activation, num_heads):
    """
    가중치, 그래디언트, 활성화 값을 결합한 테일러 중요도 계산
    
    Args:
        weight: 가중치 텐서 [out_dim, in_dim]
        grad: 그래디언트 텐서 [out_dim, in_dim]
        activation: 활성화 값 텐서 [batch_size, seq_length, in_dim]
        num_heads: 어텐션 헤드 수
    
    Returns:
        헤드별 테일러 중요도 [3(qkv), num_heads, head_dim]
    """
    out_dim, in_dim = weight.shape
    head_dim = out_dim // (3 * num_heads)
    
    # 활성화 값을 배치와 시퀀스 차원 모두에 대해 평균 계산
    # [batch_size, seq_length, in_dim] -> [in_dim]
    mean_activation = activation.mean(dim=[0, 1])  # 배치와 시퀀스 모두에 대해 평균
    
    # [3(qkv), num_heads, head_dim, in_dim] 형태로 재구성
    weight_reshaped = weight.reshape(3, num_heads, head_dim, in_dim)
    grad_reshaped = grad.reshape(3, num_heads, head_dim, in_dim)
    
    # 브로드캐스팅을 위한 활성화 값 재구성
    mean_activation = mean_activation.view(1, 1, 1, in_dim)
    
    # 가중치 × 활성화 × 그래디언트의 곱으로 중요도 계산
    importance = (weight_reshaped * mean_activation * grad_reshaped).abs().mean(dim=-1)
    
    return importance.detach().cpu().numpy()


def generate_head_masks_with_min_dims(importance_per_head, threshold):
    """최소 차원을 보장하는 헤드별 마스크 생성"""
    num_heads, head_dim = importance_per_head.shape
    masks = []
    kept_dims = []
    
    # 각 헤드별로 최소 하나의 차원은 유지하고 threshold 기반 마스크 생성
    for head_idx in range(num_heads):
        head_mask = importance_per_head[head_idx] > threshold
        kept_dims.append(max(1, np.sum(head_mask)))
        masks.append(head_mask)
    
    # 평균 유지 차원 계산 (8의 배수로 조정하지 않음)
    target_dims = max(1, int(np.mean(kept_dims)))
    
    # 최종 마스크 생성 - 중요도 기반으로 상위 차원 선택
    final_masks = np.zeros((num_heads, head_dim), dtype=bool)
    for head_idx in range(num_heads):
        head_importance = importance_per_head[head_idx]
        sorted_idx = np.argsort(-head_importance)  # 중요도 높은 순으로 정렬
        final_masks[head_idx, sorted_idx[:target_dims]] = True
    
    return final_masks

def get_vit_masks_with_independent_heads(model, pruning_rate, imp_type, mag_type):
    """TimM ViT용 독립적 헤드 프루닝 마스크 생성"""
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    all_importance = []
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight

    # Get num_heads from model structure
    num_heads = None
    for name, module in model.named_modules():
        if "attn" in name.lower() and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            break
    
    if num_heads is None:
        raise ValueError("Could not find number of attention heads in model")

    # 중요도 점수 수집
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight_or_grad = get_weight_or_grad(module, mag_type)
            
            if "qkv" in name.lower():
                # Split QKV weight and calculate importance
                importance_scores = get_independent_head_importance(weight_or_grad, num_heads, imp_type)
                
                # QKV 중요도 분리 [3, num_heads, head_dim]
                q_importance = importance_scores[0]  # [num_heads, head_dim]
                k_importance = importance_scores[1]
                v_importance = importance_scores[2]
                
                # QK 평균 중요도 계산
                qk_importance_avg = (q_importance + k_importance) / 2
                importance_dict['qk'].append(qk_importance_avg)
                importance_dict['v'].append(v_importance)
                
                all_importance.extend(qk_importance_avg.flatten())
                all_importance.extend(v_importance.flatten())
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = get_importance(weight_or_grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.extend(ffn_importance.flatten())
    
    # 임계값 계산
    all_importance = np.array(all_importance)
    threshold = get_threshold(all_importance, pruning_rate)
    
    # 마스크 생성
    masks = {}
    pruning_ratios = {'qk': [], 'v': [], 'ffn1': []}
    dimension_stats = {'qk': [], 'v': [], 'ffn1': []}
    
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "qkv" in name.lower():
                qk_importance = importance_dict['qk'].pop(0)
                v_importance = importance_dict['v'].pop(0)
                
                # 헤드별 독립적인 마스크 생성
                qk_masks = generate_head_masks_with_min_dims(qk_importance, threshold)
                v_masks = generate_head_masks_with_min_dims(v_importance, threshold)
                
                kept_dims_qk = np.sum(qk_masks, axis=1).mean()
                kept_dims_v = np.sum(v_masks, axis=1).mean()
                
                # QKV 마스크 결합
                combined_mask = np.concatenate([
                    np.tile(qk_masks, (2, 1)),  # Q와 K에 동일한 마스크 적용
                    v_masks
                ], axis=0).reshape(-1)
                
                # QKV 레이어 마스크 생성
                masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
                
                # Projection layer 마스크 - 수정된 부분
                proj_name = name.replace('qkv', 'proj')
                proj_module = next(m for n, m in model.named_modules() if n == proj_name)
                head_dim = proj_module.in_features // num_heads
                proj_mask = v_masks.reshape(-1)  # V 마스크를 기반으로 projection 마스크 생성
                masks[proj_name] = np.tile(proj_mask[:, np.newaxis], (1, proj_module.out_features)).T
                
                # 통계 계산
                pruning_ratio = 1 - (np.sum(combined_mask) / combined_mask.size)
                pruning_ratios['qk'].extend([pruning_ratio] * 2)
                pruning_ratios['v'].append(pruning_ratio)
                dimension_stats['qk'].append(kept_dims_qk)
                dimension_stats['v'].append(kept_dims_v)
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 마스크 생성 (8의 배수로 정렬)
                initial_mask = ffn_importance > threshold
                kept_dims = max(8, ((np.sum(initial_mask) + 7) // 8) * 8)
                
                sorted_idx = np.argsort(-ffn_importance)
                final_mask = np.zeros_like(ffn_importance, dtype=bool)
                final_mask[sorted_idx[:kept_dims]] = True
                
                # FC1과 FC2 마스크 생성
                masks[name] = expand_mask(final_mask, module.in_features)
                
                fc2_name = name.replace('fc1', 'fc2')
                output_module = next(m for n, m in model.named_modules() if n == fc2_name)
                masks[fc2_name] = expand_mask(final_mask, output_module.out_features).T
                
                # 통계 계산
                pruning_ratio = 1 - (np.sum(final_mask) / final_mask.size)
                pruning_ratios['ffn1'].append(pruning_ratio)
                dimension_stats['ffn1'].append(kept_dims)
    
    return masks

def apply_vit_masks(model, masks):
    """마스크 적용 함수 (가중치 및 바이어스)"""
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            with torch.no_grad():
                if name in masks:
                    # Apply weight mask
                    module.mask.data = torch.from_numpy(masks[name]).float().to(module.mask.device)
                    
                    # Create bias mask based on output dimension
                    if module.bias is not None:
                        # Calculate output dimension mask by taking OR across input dimension
                        bias_mask = np.any(masks[name], axis=1)
                        module.bias_mask.data = torch.from_numpy(bias_mask).float().to(module.bias.device)



def group_head_dimensions(importance_per_head, group_size):
    """
    헤드 차원을 그룹으로 나누고 그룹별 중요도 계산
    
    Args:
        importance_per_head: 헤드별 중요도 [num_heads, head_dim]
        group_size: 그룹 크기 (1, 8, 16 등)
    
    Returns:
        grouped_importance: 그룹별 중요도 [num_heads, num_groups]
        group_indices: 각 그룹의 인덱스 [num_heads, num_groups, elements_per_group]
    """
    num_heads, head_dim = importance_per_head.shape
    
    # 그룹 크기가 헤드 차원의 약수인지 확인
    if head_dim % group_size != 0:
        raise ValueError(f"Head dimension ({head_dim}) must be divisible by group size ({group_size})")
    
    num_groups = head_dim // group_size
    grouped_importance = np.zeros((num_heads, num_groups))
    group_indices = np.zeros((num_heads, num_groups, group_size), dtype=int)
    
    # 각 헤드별로 차원을 그룹화
    for head_idx in range(num_heads):
        # 중요도 기준으로 인덱스 정렬 (오름차순)
        sorted_indices = np.argsort(importance_per_head[head_idx])
        
        # 그룹별로 인덱스 저장 및 평균 중요도 계산
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            group_indices[head_idx, group_idx] = sorted_indices[start_idx:end_idx]
            
            # 해당 그룹의 평균 중요도 계산
            grouped_importance[head_idx, group_idx] = np.mean(
                importance_per_head[head_idx, group_indices[head_idx, group_idx]]
            )
    
    return grouped_importance, group_indices

def group_head_dimensions_two_stage(importance_per_head, dims_per_group):
    """
    2단계 그룹화: 
    1) 각 헤드 내에서 중요도 기준으로 차원을 정렬하고 각 그룹에 dims_per_group 차원씩 할당
    2) 같은 그룹 인덱스를 가진 헤드 간 그룹을 결합
    
    Args:
        importance_per_head: 헤드별 중요도 [num_heads, head_dim]
        dims_per_group: 각 그룹에 포함될 차원의 수 (기본값: 1)
    """
    num_heads, head_dim = importance_per_head.shape
    
    # 그룹 수 계산 (각 그룹이 dims_per_group 차원을 포함)
    num_groups = head_dim // dims_per_group
    
    # 1단계: 각 헤드 내에서 차원을 중요도 기준으로 정렬하고 그룹화
    per_head_grouped_importance = np.zeros((num_heads, num_groups))
    group_indices = {}
    
    for head_idx in range(num_heads):
        group_indices[head_idx] = {}
        
        # 중요도에 따라 차원 인덱스 정렬 (내림차순)
        sorted_indices = np.argsort(-importance_per_head[head_idx])
        
        for group_idx in range(num_groups):
            start_idx = group_idx * dims_per_group
            end_idx = min(start_idx + dims_per_group, head_dim)
            
            # 정렬된 인덱스에서 해당 그룹에 포함될 차원 선택
            selected_indices = sorted_indices[start_idx:end_idx]
            group_indices[head_idx][group_idx] = selected_indices
            
            # 선택된 차원들의 평균 중요도 계산
            group_importance = np.mean(importance_per_head[head_idx, selected_indices])
            per_head_grouped_importance[head_idx, group_idx] = group_importance
    
    # 2단계: 헤드 간 동일 그룹 인덱스 결합
    cross_head_importance = np.zeros(num_groups)
    
    for group_idx in range(num_groups):
        # 모든 헤드에서 동일한 그룹 인덱스의 중요도 평균
        group_values = per_head_grouped_importance[:, group_idx]
        cross_head_importance[group_idx] = np.mean(group_values)
    
    return per_head_grouped_importance, cross_head_importance, group_indices


def generate_masks_with_two_stage_grouping(importance_per_head, global_threshold, group_size):
    """
    2단계 그룹화를 통한 마스크 생성
    
    Args:
        importance_per_head: 헤드별 중요도 [num_heads, head_dim]
        global_threshold: 글로벌 임계값
        group_size: 각 그룹에 포함될 차원의 수
    
    Returns:
        final_masks: 헤드별 프루닝 마스크 [num_heads, head_dim]
    """
    num_heads, head_dim = importance_per_head.shape
    
    # 2단계 그룹화 수행
    _, cross_head_importance, group_indices = group_head_dimensions_two_stage(
        importance_per_head, group_size
    )
    
    # 임계값 계산이 아닌 외부에서 받은 임계값 사용
    # threshold = np.percentile(cross_head_importance, sparsity * 100)
    
    # 최종 마스크 초기화 (모두 False로 시작)
    final_masks = np.zeros((num_heads, head_dim), dtype=bool)
    
    # 헤드 간 그룹 중요도가 임계값보다 큰 그룹만 보존
    for group_idx in range((head_dim//group_size)):
        if cross_head_importance[group_idx] > global_threshold:
            # 해당 그룹 인덱스를 가진 모든 헤드의 해당 그룹 차원을 True로 설정
            for head_idx in range(num_heads):
                indices = group_indices[head_idx][group_idx]
                final_masks[head_idx, indices] = True
    
    return final_masks

def get_vit_masks_with_two_stage_grouping(model, attn_pruning_rate, ffn_pruning_rate, imp_type, mag_type, group_size=16):
    """
    2단계 헤드 그룹화를 적용한 ViT 마스크 생성 - 어텐션과 FFN에 대해 별도의 프루닝 비율 적용
    FFN은 group_size만을 사용하여 그룹화
    
    Args:
        model: ViT 모델
        attn_pruning_rate: 어텐션(QK, V)에 대한 프루닝 비율 (0.0-1.0)
        ffn_pruning_rate: FFN에 대한 프루닝 비율 (0.0-1.0)
        imp_type: 중요도 계산 방식 ('L1', 'L2', 'taylor1', 'act_taylor')
        mag_type: 중요도 계산 대상 ('weight' 또는 'grad')
        group_size: 각 그룹에 포함될 차원의 수 (기본값: 16)
    
    Returns:
        masks: 레이어별 마스크 딕셔너리
    """
    # 각 컴포넌트별 중요도 저장
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    
    # 각 컴포넌트별 헤드 간 그룹 중요도 저장
    attn_cross_head_importances = []  # QK와 V에 대한 중요도
    ffn_cross_head_importances = []   # FFN에 대한 중요도
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight

    # 모델 구조에서 num_heads 가져오기
    num_heads = None
    for name, module in model.named_modules():
        if "attn" in name.lower() and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            break
    
    if num_heads is None:
        raise ValueError("Could not find number of attention heads in model")

        
    total_weights_zero =0
    total_grads_zero = 0
    total_masked_weights_zero = 0
    total_masked_grads_zero = 0
    # 중요도 점수 수집
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight = module.weight
            grad = module.weight.grad if mag_type == 'grad' else None
            masked_weight = module.weight * module.mask
            masked_grad = module.weight.grad * module.mask
            
            total_masked_weights_zero += (masked_weight == 0).sum().item()
            total_masked_grads_zero += (masked_grad == 0).sum().item()
            total_weights_zero += (weight ==0).sum().item()
            total_grads_zero += (grad ==0).sum().item()
            
            
            if "qkv" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                
                if imp_type == 'taylor1':
                    # 테일러 중요도 계산
                    importance_scores = get_taylot_importance(weight, grad, num_heads, imp_type)
                elif imp_type == 'act_taylor':
                    # 활성화 기반 테일러 중요도 계산
                    importance_scores = get_activation_taylor_importance(weight, grad, activation, num_heads)
                else:
                    # L1/L2 중요도 계산
                    importance_scores = get_independent_head_importance(
                        get_weight_or_grad(module, mag_type), num_heads, imp_type
                    )
                
                # QKV 중요도 분리 [3, num_heads, head_dim]
                q_importance = importance_scores[0]  # [num_heads, head_dim]
                k_importance = importance_scores[1]
                v_importance = importance_scores[2]
                
                # QK 평균 중요도 계산
                qk_importance_avg = (q_importance + k_importance) / 2
                
                # 헤드별 그룹화 수행 후 헤드 간 그룹 중요도 계산
                _, qk_cross_head_importance, _ = group_head_dimensions_two_stage(
                    qk_importance_avg, group_size
                )
                _, v_cross_head_importance, _ = group_head_dimensions_two_stage(
                    v_importance, group_size
                )
                
                importance_dict['qk'].append(qk_importance_avg)
                importance_dict['v'].append(v_importance)
                
                # 어텐션 컴포넌트의 중요도를 별도로 저장
                attn_cross_head_importances.extend(qk_cross_head_importance)
                attn_cross_head_importances.extend(v_cross_head_importance)

            elif "mlp.fc1" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                
                if imp_type == 'taylor1':
                    ffn_importance = taylor_importance(weight, grad, imp_type)
                elif imp_type == 'act_taylor':
                    if activation is not None:
                        # 배치와 시퀀스 차원 모두에 대해 평균 계산
                        mean_activation = activation.mean(dim=[0, 1])
                        
                        # 가중치와 곱하기 위한 차원 추가
                        ffn_importance = (weight * mean_activation.unsqueeze(0) * grad).abs().mean(dim=1).detach().cpu().numpy()
                    else:
                        # 활성화 값이 없는 경우 기본 테일러 중요도 사용
                        ffn_importance = taylor_importance(weight, grad, imp_type)
                else:
                    # L1, L2 등 다른 중요도 측정 방식 사용
                    ffn_importance = get_importance(get_weight_or_grad(module, mag_type), imp_type)
                
                importance_dict['ffn1'].append(ffn_importance)
                
                # 수정: FFN 레이어는 그룹당 차원 수로 group_size만 사용
                out_features = len(ffn_importance)
                dims_per_group = group_size  # 변경된 부분: num_heads를 곱하지 않음
                num_groups = out_features // dims_per_group
                
                # FFN 차원을 중요도에 따라 정렬하고 그룹화
                sorted_indices = np.argsort(-ffn_importance)
                ffn_cross_group_importance = np.zeros(num_groups)
                ffn_group_indices = {}
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    selected_indices = sorted_indices[start_idx:end_idx]
                    ffn_group_indices[i] = selected_indices
                    
                    if start_idx < end_idx:
                        ffn_cross_group_importance[i] = np.mean(ffn_importance[selected_indices])
                
                # FFN 컴포넌트의 중요도를 별도로 저장
                ffn_cross_head_importances.extend(ffn_cross_group_importance)
                

    print("total_weights_zero =0total_grads_zero = 0total_masked_weights_zero = 0 total_masked_grads_zero = 0")
    print("total_weights_zero :", total_weights_zero)
    print("total_grads_zero",total_grads_zero)
    print("total_mased wieght zero : ", total_masked_weights_zero)
    print("total_masked_Grad_zero :", total_masked_grads_zero)

    
    # 마스크 생성
    masks = {}
    
    # 어텐션과 FFN에 대해 별도의 임계값 계산
    attn_cross_head_importances_array = np.array(attn_cross_head_importances)
    ffn_cross_head_importances_array = np.array(ffn_cross_head_importances)
    
    attn_threshold = np.percentile(attn_cross_head_importances_array, attn_pruning_rate * 100)
    ffn_threshold = np.percentile(ffn_cross_head_importances_array, ffn_pruning_rate * 100)
    
    # 각 컴포넌트에 대해 별도의 임계값을 사용하여 마스크 생성
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "qkv" in name.lower():
                qk_importance = importance_dict['qk'].pop(0)
                v_importance = importance_dict['v'].pop(0)
                
                # 어텐션 임계값을 사용하여 마스크 생성
                qk_masks = generate_masks_with_two_stage_grouping(
                    qk_importance, attn_threshold, group_size
                )
                v_masks = generate_masks_with_two_stage_grouping(
                    v_importance, attn_threshold, group_size
                )
                                
                # QKV 마스크 결합
                combined_mask = np.concatenate([
                    np.tile(qk_masks, (2, 1)),  # Q와 K에 동일한 마스크 적용
                    v_masks
                ], axis=0).reshape(-1)
                
                # QKV 레이어 마스크 생성
                masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
                
                # Projection layer 마스크
                proj_name = name.replace('qkv', 'proj')
                proj_module = next(m for n, m in model.named_modules() if n == proj_name)
                head_dim = proj_module.in_features // num_heads
                
                # V 마스크를 기반으로 projection 마스크 생성
                proj_mask = v_masks.reshape(-1)
                masks[proj_name] = np.tile(proj_mask[:, np.newaxis], (1, proj_module.out_features)).T
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 레이어 차원을 중요도에 따라 정렬
                sorted_indices = np.argsort(-ffn_importance)
                out_features = len(ffn_importance)
                
                # 수정: 그룹 크기로 group_size만 사용
                dims_per_group = group_size
                num_groups = out_features // dims_per_group
                
                # 그룹 마스크 생성 (FFN 임계값 사용)
                ffn_mask = np.zeros(out_features, dtype=bool)
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    # 정렬된 인덱스에서 해당 그룹에 포함될 차원 선택
                    selected_indices = sorted_indices[start_idx:end_idx]
                    
                    # 그룹 중요도 계산
                    group_importance = np.mean(ffn_importance[selected_indices])
                    
                    # FFN 임계값과 비교
                    if group_importance > ffn_threshold:
                        ffn_mask[selected_indices] = True
                
                # FC1과 FC2 마스크 생성
                masks[name] = np.tile(ffn_mask[:, np.newaxis], (1, module.in_features))
                
                fc2_name = name.replace('fc1', 'fc2')
                output_module = next(m for n, m in model.named_modules() if n == fc2_name)
                masks[fc2_name] = np.tile(ffn_mask[:, np.newaxis], (1, output_module.out_features)).T
    
    return masks



def get_vit_masks_with_two_stage_grouping_global(model, pruning_rate, imp_type, mag_type, group_size=1):
    """
    2단계 헤드 그룹화를 적용한 ViT 마스크 생성
    
    Args:
        model: ViT 모델
        pruning_rate: 프루닝 비율 (0.0-1.0)
        imp_type: 중요도 계산 방식 ('L1', 'L2', 'taylor1', 'act_taylor')
        mag_type: 중요도 계산 대상 ('weight' 또는 'grad')
        group_size: 각 헤드를 몇 개의 그룹으로 나눌지 (기본값: 1)
    
    Returns:
        masks: 레이어별 마스크 딕셔너리
    """
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    all_cross_head_importances = []
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight

    # 모델 구조에서 num_heads 가져오기
    num_heads = None
    for name, module in model.named_modules():
        if "attn" in name.lower() and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            break
    
    if num_heads is None:
        raise ValueError("Could not find number of attention heads in model")

    # 중요도 점수 수집
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight = module.weight
            grad = module.weight.grad if mag_type == 'grad' else None
            
            if "qkv" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                
                if imp_type == 'taylor1':
                    # 기존 테일러 중요도 계산 방식 유지
                    importance_scores = get_taylot_importance(weight, grad, num_heads, imp_type)
                elif imp_type == 'act_taylor':
                    # 새로운 활성화 기반 테일러 중요도 계산
                    importance_scores = get_activation_taylor_importance(weight, grad, activation, num_heads)
                else:
                    # 기존 L1/L2 중요도 계산
                    importance_scores = get_independent_head_importance(
                        get_weight_or_grad(module, mag_type), num_heads, imp_type
                    )
                
                # QKV 중요도 분리 [3, num_heads, head_dim]
                q_importance = importance_scores[0]  # [num_heads, head_dim]
                k_importance = importance_scores[1]
                v_importance = importance_scores[2]
                
                # QK 평균 중요도 계산
                qk_importance_avg = (q_importance + k_importance) / 2
                
                # 헤드별 그룹화 수행 후 헤드 간 그룹 중요도 계산
                _, qk_cross_head_importance, _ = group_head_dimensions_two_stage(
                    qk_importance_avg, group_size
                )
                _, v_cross_head_importance, _ = group_head_dimensions_two_stage(
                    v_importance, group_size
                )
                
                importance_dict['qk'].append(qk_importance_avg)
                importance_dict['v'].append(v_importance)
                
                # 모든 헤드 간 그룹 중요도를 리스트에 추가
                all_cross_head_importances.extend(qk_cross_head_importance)
                all_cross_head_importances.extend(v_cross_head_importance)


            elif "mlp.fc1" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                
                if imp_type == 'taylor1':
                    ffn_importance = taylor_importance(weight, grad, imp_type)
                elif imp_type == 'act_taylor':
                    if activation is not None:
                        # 활성화 형태 출력 (디버깅용)
                        # print(f"FFN 활성화 값 형태: {activation.shape}")
                        
                        # 배치와 시퀀스 차원 모두에 대해 평균 계산
                        # [배치_크기, 시퀀스_길이, 은닉_차원] -> [은닉_차원]
                        mean_activation = activation.mean(dim=[0, 1])
                        
                        # 가중치와 곱하기 위한 차원 추가
                        ffn_importance = (weight * mean_activation.unsqueeze(0) * grad).abs().mean(dim=1).detach().cpu().numpy()
                    else:
                        # 활성화 값이 없는 경우 기본 테일러 중요도 사용
                        # print("경고: FFN 레이어의 활성화 값이 없습니다. 기본 테일러 중요도를 사용합니다.")
                        ffn_importance = taylor_importance(weight, grad, imp_type)
                else:
                    # L1, L2 등 다른 중요도 측정 방식 사용
                    ffn_importance = get_importance(get_weight_or_grad(module, mag_type), imp_type)
                
                importance_dict['ffn1'].append(ffn_importance)
                
                # FFN 레이어는 그룹당 차원 수를 group_size * num_heads로 설정
                out_features = len(ffn_importance)
                dims_per_group = group_size * num_heads  # 그룹당 차원 수 (예: 8*3=24)
                num_groups = out_features // dims_per_group  # 전체 그룹 수
                
                # FFN 차원을 중요도에 따라 정렬하고 그룹화
                sorted_indices = np.argsort(-ffn_importance)
                ffn_cross_group_importance = np.zeros(num_groups)
                ffn_group_indices = {}
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    selected_indices = sorted_indices[start_idx:end_idx]
                    ffn_group_indices[i] = selected_indices
                    
                    if start_idx < end_idx:
                        ffn_cross_group_importance[i] = np.mean(ffn_importance[selected_indices])
                
                all_cross_head_importances.extend(ffn_cross_group_importance)
    
    # 마스크 생성
    masks = {}
    
    # 전체 헤드 간 그룹 중요도 배열에서 임계값 계산
    all_cross_head_importances_array = np.array(all_cross_head_importances)
    global_threshold = np.percentile(all_cross_head_importances_array, pruning_rate * 100)
    
    
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "qkv" in name.lower():
                qk_importance = importance_dict['qk'].pop(0)
                v_importance = importance_dict['v'].pop(0)
                
                # 2단계 그룹화를 적용한 마스크 생성
                qk_masks = generate_masks_with_two_stage_grouping(
                    qk_importance, global_threshold, group_size  # pruning_rate 대신 global_threshold 전달
                )
                v_masks = generate_masks_with_two_stage_grouping(
                    v_importance, global_threshold, group_size  # pruning_rate 대신 global_threshold 전달
                )
                                
                # QKV 마스크 결합
                # Q와 K에 동일한 마스크 적용, V에는 별도 마스크
                combined_mask = np.concatenate([
                    np.tile(qk_masks, (2, 1)),  # Q와 K에 동일한 마스크 적용
                    v_masks
                ], axis=0).reshape(-1)
                
                # QKV 레이어 마스크 생성
                masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
                
                # Projection layer 마스크
                proj_name = name.replace('qkv', 'proj')
                proj_module = next(m for n, m in model.named_modules() if n == proj_name)
                head_dim = proj_module.in_features // num_heads
                
                # V 마스크를 기반으로 projection 마스크 생성
                proj_mask = v_masks.reshape(-1)
                masks[proj_name] = np.tile(proj_mask[:, np.newaxis], (1, proj_module.out_features)).T
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 레이어 차원을 중요도에 따라 정렬
                sorted_indices = np.argsort(-ffn_importance)
                out_features = len(ffn_importance)
                dims_per_group = group_size * num_heads  # 그룹당 차원 수 (예: 8*3=24)
                num_groups = out_features // dims_per_group  # 전체 그룹 수
                
                # 그룹 마스크 생성
                ffn_mask = np.zeros(out_features, dtype=bool)
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    # 정렬된 인덱스에서 해당 그룹에 포함될 차원 선택
                    selected_indices = sorted_indices[start_idx:end_idx]
                    
                    # 그룹 중요도 계산
                    group_importance = np.mean(ffn_importance[selected_indices])
                    
                    # 글로벌 임계값과 비교
                    if group_importance > global_threshold:
                        ffn_mask[selected_indices] = True
                
                # FC1과 FC2 마스크 생성
                masks[name] = np.tile(ffn_mask[:, np.newaxis], (1, module.in_features))
                
                fc2_name = name.replace('fc1', 'fc2')
                output_module = next(m for n, m in model.named_modules() if n == fc2_name)
                masks[fc2_name] = np.tile(ffn_mask[:, np.newaxis], (1, output_module.out_features)).T
    
    return masks


def get_simple_vit_masks(model, attn_pruning_rate, ffn_pruning_rate, imp_type, mag_type):
    """
    ViT 마스크 생성 함수 - 어텐션과 FFN에 대해 별도의 프루닝 비율 적용
    
    Args:
        model: ViT 모델
        attn_pruning_rate: 어텐션(QK, V)에 대한 프루닝 비율 (0.0-1.0)
        ffn_pruning_rate: FFN에 대한 프루닝 비율 (0.0-1.0)
        imp_type: 중요도 계산 방식 ('L1', 'L2', 'taylor1', 'act_taylor')
        mag_type: 중요도 계산 대상 ('weight' 또는 'grad')
    
    Returns:
        masks: 레이어별 마스크 딕셔너리
    """
    # 각 컴포넌트별 중요도 저장
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    
    # 각 컴포넌트별 중요도 점수 수집
    attn_importance_scores = []  # QK와 V에 대한 중요도
    ffn_importance_scores = []   # FFN에 대한 중요도
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight

    # 모델 구조에서 num_heads 가져오기
    num_heads = None
    for name, module in model.named_modules():
        if "attn" in name.lower() and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            break
    
    if num_heads is None:
        raise ValueError("Could not find number of attention heads in model")

    # 중요도 점수 수집
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight = module.weight
            grad = module.weight.grad if mag_type == 'grad' else None
            weight_or_grad = get_weight_or_grad(module, mag_type)
            
            if "qkv" in name.lower():
                if imp_type == 'taylor1':
                    # 테일러 중요도 계산
                    importance_scores = get_taylot_importance(weight, grad, num_heads, imp_type)
                else:
                    # L1/L2 중요도 계산
                    importance_scores = get_independent_head_importance(
                        get_weight_or_grad(module, mag_type), num_heads, imp_type
                    )
                
                q_importance = importance_scores[0]  # [num_heads, head_dim]
                k_importance = importance_scores[1]  
                v_importance = importance_scores[2]  
                
                # QK 평균 중요도 계산
                qk_importance_avg = (q_importance + k_importance) / 2
                importance_dict['qk'].append(qk_importance_avg)
                importance_dict['v'].append(v_importance)
                
                # 어텐션 컴포넌트의 중요도 저장
                attn_importance_scores.extend(qk_importance_avg.flatten())
                attn_importance_scores.extend(v_importance.flatten())
                
            elif "mlp.fc1" in name.lower():
                if imp_type == 'taylor1':
                    ffn_importance = taylor_importance(weight, grad, imp_type)
                else:
                    # L1, L2 등 다른 중요도 측정 방식 사용
                    ffn_importance = get_importance(get_weight_or_grad(module, mag_type), imp_type)

                importance_dict['ffn1'].append(ffn_importance)
                
                # FFN 컴포넌트의 중요도 저장
                ffn_importance_scores.extend(ffn_importance.flatten())
    
    # 어텐션과 FFN에 대해 별도의 임계값 계산
    attn_importance_array = np.array(attn_importance_scores)
    ffn_importance_array = np.array(ffn_importance_scores)
    
    attn_threshold = np.percentile(attn_importance_array, attn_pruning_rate * 100)
    ffn_threshold = np.percentile(ffn_importance_array, ffn_pruning_rate * 100)
    
    # 마스크 생성
    masks = {}
    
    # 각 컴포넌트에 대해 별도의 임계값을 사용하여 마스크 생성
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "qkv" in name.lower():
                qk_importance = importance_dict['qk'].pop(0)
                v_importance = importance_dict['v'].pop(0)
                
                # 어텐션 임계값을 사용한 단순 임계화
                qk_masks = qk_importance > attn_threshold
                v_masks = v_importance > attn_threshold
                
                # QKV 마스크 결합
                combined_mask = np.concatenate([
                    np.tile(qk_masks, (2, 1)),  # Q와 K에 동일한 마스크 적용
                    v_masks
                ], axis=0).reshape(-1)
                
                # QKV 레이어 마스크 생성
                masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
                
                # Projection layer 마스크 생성
                proj_name = name.replace('qkv', 'proj')
                proj_module = next(m for n, m in model.named_modules() if n == proj_name)
                head_dim = proj_module.in_features // num_heads
                proj_mask = v_masks.reshape(-1)
                masks[proj_name] = np.tile(proj_mask[:, np.newaxis], (1, proj_module.out_features)).T
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 임계값을 사용한 단순 임계화
                ffn_mask = ffn_importance > ffn_threshold
                
                # FC1과 FC2 마스크 생성
                masks[name] = expand_mask(ffn_mask, module.in_features)
                
                fc2_name = name.replace('fc1', 'fc2')
                output_module = next(m for n, m in model.named_modules() if n == fc2_name)
                masks[fc2_name] = expand_mask(ffn_mask, output_module.out_features).T
    
    return masks


def get_vit_masks_with_two_stage_grouping_unified_qkv(model, attn_pruning_rate, ffn_pruning_rate, imp_type, mag_type, group_size=1):
    """
    2단계 헤드 그룹화를 적용한 ViT 마스크 생성 - QKV를 통합된 하나의 컴포넌트로 처리
    어텐션과 FFN에 대해 별도의 프루닝 비율 적용
    
    Args:
        model: ViT 모델
        attn_pruning_rate: 어텐션(QKV)에 대한 프루닝 비율 (0.0-1.0)
        ffn_pruning_rate: FFN에 대한 프루닝 비율 (0.0-1.0)
        imp_type: 중요도 계산 방식 ('L1', 'L2', 'taylor1', 'act_taylor')
        mag_type: 중요도 계산 대상 ('weight' 또는 'grad')
        group_size: 각 그룹에 포함될 차원의 수 (기본값: 16)
    
    Returns:
        masks: 레이어별 마스크 딕셔너리
    """
    # 각 컴포넌트별 중요도 저장
    importance_dict = {'qkv': [], 'ffn1': []}
    
    # 각 컴포넌트별 헤드 간 그룹 중요도 저장
    attn_cross_head_importances = []  # QKV에 대한 중요도
    ffn_cross_head_importances = []   # FFN에 대한 중요도
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight

    # 모델 구조에서 num_heads 가져오기
    num_heads = None
    for name, module in model.named_modules():
        if "attn" in name.lower() and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            break
    
    if num_heads is None:
        raise ValueError("Could not find number of attention heads in model")


        
    # 중요도 점수 수집
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight = module.weight
            grad = module.weight.grad if mag_type == 'grad' else None

            
            if "qkv" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                
                if imp_type == 'taylor1':
                    # 테일러 중요도 계산
                    importance_scores = get_taylot_importance(weight, grad, num_heads, imp_type)
                elif imp_type == 'act_taylor':
                    # 활성화 기반 테일러 중요도 계산
                    importance_scores = get_activation_taylor_importance(weight, grad, activation, num_heads)
                elif imp_type == 'grad':
                    importance_scores = get_gradient_importance(grad, num_heads)
                else:
                    # L1/L2 중요도 계산
                    importance_scores = get_independent_head_importance(
                        get_weight_or_grad(module, mag_type), num_heads, imp_type
                    )
                
                # QKV 중요도 분리 [3, num_heads, head_dim]
                q_importance = importance_scores[0]  # [num_heads, head_dim]
                k_importance = importance_scores[1]
                v_importance = importance_scores[2]
                
                # QKV 통합 중요도 계산 - Q, K, V의 평균
                qkv_importance_avg = (q_importance + k_importance + v_importance) / 3
                
                # 헤드별 그룹화 수행 후 헤드 간 그룹 중요도 계산
                _, qkv_cross_head_importance, _ = group_head_dimensions_two_stage(
                    qkv_importance_avg, group_size
                )
                
                importance_dict['qkv'].append(qkv_importance_avg)
                
                # 어텐션 컴포넌트의 중요도를 별도로 저장
                attn_cross_head_importances.extend(qkv_cross_head_importance)

            elif "mlp.fc1" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                
                if imp_type == 'taylor1':
                    ffn_importance = taylor_importance(weight, grad, imp_type)
                elif imp_type == 'act_taylor':
                    if activation is not None:
                        # 배치와 시퀀스 차원 모두에 대해 평균 계산
                        mean_activation = activation.mean(dim=[0, 1])
                        
                        # 가중치와 곱하기 위한 차원 추가
                        ffn_importance = (weight * mean_activation.unsqueeze(0) * grad).abs().mean(dim=1).detach().cpu().numpy()
                    else:
                        # 활성화 값이 없는 경우 기본 테일러 중요도 사용
                        ffn_importance = taylor_importance(weight, grad, imp_type)
                elif imp_type == 'grad':
                    ffn_importance = grad.abs().mean(dim=1).detach().cpu().numpy()
                else:
                    # L1, L2 등 다른 중요도 측정 방식 사용
                    ffn_importance = get_importance(get_weight_or_grad(module, mag_type), imp_type)
                
                importance_dict['ffn1'].append(ffn_importance)
                
                # 수정: FFN 레이어는 그룹당 차원 수로 group_size만 사용
                out_features = len(ffn_importance)
                dims_per_group = group_size  # 변경된 부분: num_heads를 곱하지 않음
                num_groups = out_features // dims_per_group
                
                # FFN 차원을 중요도에 따라 정렬하고 그룹화
                sorted_indices = np.argsort(-ffn_importance)
                ffn_cross_group_importance = np.zeros(num_groups)
                ffn_group_indices = {}
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    selected_indices = sorted_indices[start_idx:end_idx]
                    ffn_group_indices[i] = selected_indices
                    
                    if start_idx < end_idx:
                        ffn_cross_group_importance[i] = np.mean(ffn_importance[selected_indices])
                
                # FFN 컴포넌트의 중요도를 별도로 저장
                ffn_cross_head_importances.extend(ffn_cross_group_importance)
    # print("total_weights_zero =0total_grads_zero = 0total_masked_weights_zero = 0 total_masked_grads_zero = 0")
    # print("total_weights_zero :", total_weights_zero)
    # print("total_grads_zero",total_grads_zero)
    # print("total_mased wieght zero : ", total_masked_weights_zero)
    # print("total_masked_Grad_zero :", total_masked_grads_zero)
    # 마스크 생성
    masks = {}
    
    # 어텐션과 FFN에 대해 별도의 임계값 계산
    attn_cross_head_importances_array = np.array(attn_cross_head_importances)
    ffn_cross_head_importances_array = np.array(ffn_cross_head_importances)
    
    attn_threshold = np.percentile(attn_cross_head_importances_array, attn_pruning_rate * 100)
    ffn_threshold = np.percentile(ffn_cross_head_importances_array, ffn_pruning_rate * 100)
    
    # 각 컴포넌트에 대해 별도의 임계값을 사용하여 마스크 생성
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "qkv" in name.lower():
                qkv_importance = importance_dict['qkv'].pop(0)
                
                # 어텐션 임계값을 사용하여 마스크 생성
                qkv_masks = generate_masks_with_two_stage_grouping(
                    qkv_importance, attn_threshold, group_size
                )
                
                # 통합된 QKV 마스크를 Q, K, V 모두에 적용
                combined_mask = np.tile(qkv_masks, (3, 1)).reshape(-1)
                
                # QKV 레이어 마스크 생성
                masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
                
                # Projection layer 마스크
                proj_name = name.replace('qkv', 'proj')
                proj_module = next(m for n, m in model.named_modules() if n == proj_name)
                head_dim = proj_module.in_features // num_heads
                
                # QKV 마스크를 기반으로 projection 마스크 생성
                proj_mask = qkv_masks.reshape(-1)
                masks[proj_name] = np.tile(proj_mask[:, np.newaxis], (1, proj_module.out_features)).T
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 레이어 차원을 중요도에 따라 정렬
                sorted_indices = np.argsort(-ffn_importance)
                out_features = len(ffn_importance)
                
                # 수정: 그룹 크기로 group_size만 사용
                dims_per_group = group_size
                num_groups = out_features // dims_per_group
                
                # 그룹 마스크 생성 (FFN 임계값 사용)
                ffn_mask = np.zeros(out_features, dtype=bool)
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    # 정렬된 인덱스에서 해당 그룹에 포함될 차원 선택
                    selected_indices = sorted_indices[start_idx:end_idx]
                    
                    # 그룹 중요도 계산
                    group_importance = np.mean(ffn_importance[selected_indices])
                    
                    # FFN 임계값과 비교
                    if group_importance > ffn_threshold:
                        ffn_mask[selected_indices] = True
                
                # FC1과 FC2 마스크 생성
                masks[name] = np.tile(ffn_mask[:, np.newaxis], (1, module.in_features))
                
                fc2_name = name.replace('fc1', 'fc2')
                output_module = next(m for n, m in model.named_modules() if n == fc2_name)
                masks[fc2_name] = np.tile(ffn_mask[:, np.newaxis], (1, output_module.out_features)).T
    
    return masks
import numpy as np


import numpy as np

# 이 함수는 group_head_dimensions_two_stage, MaskLinear 및
# get_taylot_importance, get_independent_head_importance, get_importance와 같은
# 보조 함수들이 이미 정의되어 있다고 가정합니다.
import numpy as np

# 이 함수는 필요한 모든 보조 함수들(group_head_dimensions_two_stage, 
# get_taylot_importance, get_activation_taylor_importance, get_gradient_importance,
# get_independent_head_importance, get_importance 등)이 정의되어 있다고 가정합니다.

# def get_vit_masks_with_two_stage_grouping_unified_qkv_change_zero(
#     model, attn_pruning_rate, ffn_pruning_rate, imp_type, mag_type, group_size=1,
#     candidate_pool_ratio=0.10,
#     revival_ratio=0.05
# ):

    
#     """
#     [최종 수정] 상세 중요도 계산 로직을 완벽히 복원하여 'NoneType' 오류를 해결한 버전.
#     """
#     print(f"--- Running Final Corrected Revival Function with Full Importance Logic ---")
#     print(f"attn_Candidate Pool: Top {candidate_pool_ratio * attn_pruning_rate *100}% of losers. Revival Count: {revival_ratio*attn_pruning_rate*100}% of losers.")
#     print(f"attn_Candidate Pool: Top {candidate_pool_ratio * ffn_pruning_rate *100}% of losers. Revival Count: {revival_ratio*ffn_pruning_rate*100}% of losers.")
    
#     # 1단계: 데이터 구조 초기화
#     importance_dict = {'qkv': [], 'ffn1': []}
#     attn_cross_head_importances = []
#     ffn_cross_head_importances = []
#     layer_attn_info = []

#     def get_weight_or_grad(module, mag_type):
#         return module.weight.grad if mag_type == 'grad' else module.weight

#     num_heads = None
#     head_dim = None
#     for name, module in model.named_modules():
#         if "attn" in name.lower() and hasattr(module, 'num_heads'):
#             num_heads = module.num_heads
#             if hasattr(module, 'qkv') and module.qkv is not None:
#                  head_dim = module.qkv.weight.shape[0] // (3 * num_heads)
#             elif hasattr(module, 'in_proj_weight'):
#                  head_dim = module.in_proj_weight.shape[0] // (3 * num_heads)
#             if head_dim is not None:
#                 break
            
#     if num_heads is None or head_dim is None:
#         raise ValueError("Could not find number of attention heads and head_dim in model")

#     # 중요도 점수 수집
#     for name, module in model.named_modules():
#         if not isinstance(module, MaskLinear):
#             continue

#         weight = module.weight
#         grad = module.weight.grad if mag_type == 'grad' else None
        
#         if "qkv" in name.lower():
#             # [오류 해결] 사용자님의 원래 상세 로직으로 완벽하게 복원
#             activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
            
#             if imp_type == 'taylor1':
#                 importance_scores = get_taylot_importance(weight, grad, num_heads, imp_type)
#             elif imp_type == 'act_taylor':
#                 importance_scores = get_activation_taylor_importance(weight, grad, activation, num_heads)
#             elif imp_type == 'grad':
#                 importance_scores = get_gradient_importance(grad, num_heads)
#             else:
#                 importance_scores = get_independent_head_importance(
#                     get_weight_or_grad(module, mag_type), num_heads, imp_type
#                 )
            
#             # 여기서부터는 importance_scores가 정상 값이므로 오류가 발생하지 않습니다.
#             q_importance, k_importance, v_importance = importance_scores[0], importance_scores[1], importance_scores[2]
#             qkv_importance_avg = (q_importance + k_importance + v_importance) / 3

#             _, qkv_grouped_importance, qkv_group_indices = group_head_dimensions_two_stage(
#                 qkv_importance_avg, group_size
#             )
            
#             importance_dict['qkv'].append(qkv_importance_avg)
#             attn_cross_head_importances.extend(qkv_grouped_importance)
#             layer_attn_info.append({
#                 'name': name,
#                 'group_indices': qkv_group_indices,
#                 'num_groups': len(qkv_grouped_importance)
#             })

#         elif "mlp.fc1" in name.lower():
#             # [오류 해결] FFN도 사용자님의 원래 상세 로직으로 완벽하게 복원
#             activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
            
#             if imp_type == 'taylor1':
#                 ffn_importance = taylor_importance(weight, grad, imp_type)
#             elif imp_type == 'act_taylor':
#                 if activation is not None:
#                     mean_activation = activation.mean(dim=[0, 1])
#                     ffn_importance = (weight * mean_activation.unsqueeze(0) * grad).abs().mean(dim=1).detach().cpu().numpy()
#                 else:
#                     ffn_importance = taylor_importance(weight, grad, imp_type)
#             elif imp_type == 'grad':
#                 ffn_importance = grad.abs().mean(dim=1).detach().cpu().numpy()
#             else:
#                 ffn_importance = get_importance(get_weight_or_grad(module, mag_type), imp_type)

#             importance_dict['ffn1'].append(ffn_importance)
            
#             out_features = len(ffn_importance)
#             dims_per_group = group_size
#             if dims_per_group == 0: continue
#             num_groups = out_features // dims_per_group
            
#             sorted_indices = np.argsort(-ffn_importance)
#             ffn_cross_group_importance = np.zeros(num_groups)
            
#             for i in range(num_groups):
#                 start_idx = i * dims_per_group
#                 end_idx = min(start_idx + dims_per_group, out_features)
#                 selected_indices = sorted_indices[start_idx:end_idx]
#                 if start_idx < end_idx:
#                     ffn_cross_group_importance[i] = np.mean(ffn_importance[selected_indices])
            
#             ffn_cross_head_importances.extend(ffn_cross_group_importance)

#     # --- 2, 3, 4단계: 전역 점수 기반 커트라인 계산 및 패자부활 ---
#     # (이 부분은 이전 답변과 동일하게 올바른 로직입니다)
#     attn_scores = np.array(attn_cross_head_importances)
#     ffn_scores = np.array(ffn_cross_head_importances)
    
#     if len(attn_scores) == 0 or len(ffn_scores) == 0:
#         return {}

#     initial_attn_threshold = np.percentile(attn_scores, attn_pruning_rate * 100)
#     initial_ffn_threshold = np.percentile(ffn_scores, ffn_pruning_rate * 100)

#     attn_candidate_pool_ratio = candidate_pool_ratio * attn_pruning_rate
#     attn_revival_ratio = revival_ratio * attn_pruning_rate
    
#     ffn_candidate_pool_ratio = candidate_pool_ratio *  ffn_pruning_rate
#     ffn_revival_ratio = revival_ratio *  ffn_pruning_rate
    
#     # Attention 패자부활
#     loser_indices_attn = np.where(attn_scores < initial_attn_threshold)[0]
#     if len(loser_indices_attn) > 0:
#         strongest_losers_indices = loser_indices_attn[np.argsort(-attn_scores[loser_indices_attn])]
#         pool_size = int(len(loser_indices_attn) * attn_candidate_pool_ratio)
#         candidate_pool_indices = strongest_losers_indices[:pool_size]
#         num_to_revive = int(len(loser_indices_attn) * attn_revival_ratio)
#         if len(candidate_pool_indices) > 0 and num_to_revive > 0:
#             num_to_actually_revive = min(num_to_revive, len(candidate_pool_indices))
#             indices_to_revive = np.random.choice(candidate_pool_indices, size=num_to_actually_revive, replace=False)
#             attn_scores[indices_to_revive] = initial_attn_threshold + 1e-6

#     # FFN 패자부활
#     loser_indices_ffn = np.where(ffn_scores < initial_ffn_threshold)[0]
#     if len(loser_indices_ffn) > 0:
#         strongest_losers_indices_ffn = loser_indices_ffn[np.argsort(-ffn_scores[loser_indices_ffn])]
#         pool_size_ffn = int(len(loser_indices_ffn) * ffn_candidate_pool_ratio)
#         candidate_pool_indices_ffn = strongest_losers_indices_ffn[:pool_size_ffn]
#         num_to_revive_ffn = int(len(loser_indices_ffn) * ffn_revival_ratio)
#         if len(candidate_pool_indices_ffn) > 0 and num_to_revive_ffn > 0:
#             num_to_actually_revive = min(num_to_revive_ffn, len(candidate_pool_indices_ffn))
#             indices_to_revive = np.random.choice(candidate_pool_indices_ffn, size=num_to_actually_revive, replace=False)
#             ffn_scores[indices_to_revive] = initial_ffn_threshold + 1e-6
            
#     final_attn_threshold = np.percentile(attn_scores, attn_pruning_rate * 100)
#     final_ffn_threshold = np.percentile(ffn_scores, ffn_pruning_rate * 100)
#     print("  - Recalculated global thresholds to perform swap.")

#     # --- 6단계: 최종 전역 임계값을 이용한 마스크 생성 ---
#     # (이 부분은 이전 답변과 동일하게 올바른 로직입니다)
#     masks = {}
#     attn_scores_pointer = 0
#     ffn_pointer = 0
    
#     # Attention 마스크 생성
#     for info in layer_attn_info:
#         name = info['name']
#         group_indices = info['group_indices']
#         num_groups = info['num_groups']
        
#         layer_grouped_scores = attn_scores[attn_scores_pointer : attn_scores_pointer + num_groups]
#         attn_scores_pointer += num_groups
        
#         group_mask = layer_grouped_scores >= final_attn_threshold
        
#         head_dim_mask = np.zeros((num_heads, head_dim), dtype=bool)
        
#         for head_idx in range(num_heads):
#             for group_idx in range(num_groups):
#                 if group_mask[group_idx]:
#                     original_indices = group_indices[head_idx][group_idx]
#                     head_dim_mask[head_idx, original_indices] = True
        
#         qkv_mask_1d = head_dim_mask.flatten()
#         module = next(m for n, m in model.named_modules() if n == name)
#         combined_mask = np.tile(qkv_mask_1d, 3)
#         masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
        
#         proj_name = name.replace('qkv', 'proj')
#         proj_module = next(m for n, m in model.named_modules() if n == proj_name)
#         masks[proj_name] = np.tile(qkv_mask_1d[:, np.newaxis], (1, proj_module.out_features)).T

#     # FFN 마스크 생성
#     ffn_fc1_names = [name for name, module in model.named_modules() if isinstance(module, MaskLinear) and "mlp.fc1" in name.lower()]
#     for name in ffn_fc1_names:
#         module = next(m for n, m in model.named_modules() if n == name)
#         ffn_importance_original = importance_dict['ffn1'].pop(0)
#         out_features = ffn_importance_original.size
        
#         if group_size == 0: continue
#         num_groups = out_features // group_size
        
#         layer_ffn_scores = ffn_scores[ffn_pointer : ffn_pointer + num_groups]
#         ffn_pointer += num_groups

#         ffn_mask = np.zeros(out_features, dtype=bool)
#         sorted_indices = np.argsort(-ffn_importance_original.flatten())

#         for i in range(num_groups):
#             if layer_ffn_scores[i] >= final_ffn_threshold:
#                 start_idx = i * group_size
#                 end_idx = start_idx + group_size
#                 original_indices = sorted_indices[start_idx:end_idx]
#                 ffn_mask[original_indices] = True
        
#         masks[name] = np.tile(ffn_mask[:, np.newaxis], (1, module.in_features))
#         fc2_name = name.replace('fc1', 'fc2')
#         output_module = next(m for n, m in model.named_modules() if n == fc2_name)
#         masks[fc2_name] = np.tile(ffn_mask[:, np.newaxis], (1, output_module.out_features)).T
            
#     return masks


def get_taylot_importance(weight, grad, num_heads, imp_type):
    out_dim, in_dim = weight.shape
    head_dim = out_dim // (3 * num_heads)
    
    # Reshape to [3(qkv), num_heads, head_dim, in_dim]
    weight_reshaped = weight.reshape(3, num_heads, head_dim, in_dim)
    grad_reshaped = grad.reshape(3, num_heads, head_dim, in_dim)
    
    if imp_type == 'taylor1':
        if grad is None:
            raise ValueError("Gradient required for Taylor expansion methods")

        return (weight_reshaped * grad_reshaped).abs().mean(dim=-1).detach().cpu().numpy()
    
    

def get_temporal_importance_with_e(weight, g_current, e_previous, num_heads):
    """
    이전 스텝의 섭동 벡터(e_w)와의 방향 일관성을 반영한
    Temporal Taylor 중요도를 계산합니다.
    e_previous: 이전 스텝에서 저장해 둔 e_w 텐서
    """
    # 기본 테일러 중요도 계산 (|W * g|)
    base_taylor_importance = (weight * g_current).abs()
    
    d_model = weight.shape[1]
    head_dim = d_model // num_heads
    
    # Q, K, V 별로 중요도 분리 및 헤드별 평균
    base_taylor_q, base_taylor_k, base_taylor_v = torch.chunk(base_taylor_importance, 3, dim=0)
    final_scores = torch.stack([
        base_taylor_q.reshape(num_heads, head_dim, d_model).mean(dim=-1), 
        base_taylor_k.reshape(num_heads, head_dim, d_model).mean(dim=-1), 
        base_taylor_v.reshape(num_heads, head_dim, d_model).mean(dim=-1)
    ])
    
    # 이전 e_w가 없으면, 방향성 보정 없이 기본 점수만 반환
    if e_previous is None:
        return final_scores.detach().cpu().numpy()

    # --- 방향성 보정 계수 (alpha) 계산 ---
    g_curr_reshaped = g_current.reshape(3, num_heads, head_dim, d_model)
    e_prev_reshaped = e_previous.reshape(3, num_heads, head_dim, d_model)
    
    epsilon = 0
    dot_product = (g_curr_reshaped * e_prev_reshaped).sum(dim=(-1, -2))
    norm_curr = g_curr_reshaped.norm(p=2, dim=(-1, -2))
    norm_prev = e_prev_reshaped.norm(p=2, dim=(-1, -2))
    
    cosine_similarity = dot_product / (norm_curr * norm_prev + epsilon)
    alpha = (1 + torch.clamp(cosine_similarity, -1.0, 1.0)) / 2
    
    # 최종 중요도 = 기존 중요도 * alpha
    final_scores_corrected = final_scores * alpha.unsqueeze(-1)
    
    return final_scores_corrected.detach().cpu().numpy()


def get_ffn_temporal_importance_with_e(weight, g_current, e_previous):
    """
    FFN 레이어에 대해 이전 스텝의 섭동 벡터(e_w)와의 방향 일관성을
    반영한 Temporal Taylor 중요도를 계산합니다.
    """
    # 기본 테일러 중요도 계산 (|W * g|)
    # FFN의 경우, 출력 뉴런(차원)별로 중요도를 계산합니다.
    base_importance = (weight * g_current).abs().mean(dim=1)
    
    # 이전 e_w가 없으면, 방향성 보정 없이 기본 점수만 반환
    if e_previous is None:
        return base_importance.detach().cpu().numpy()

    # --- 방향성 보정 계수 (alpha) 계산 ---
    # FFN 가중치 텐서 전체를 하나의 벡터로 보고 코사인 유사도 계산
    g_flat = g_current.flatten()
    e_flat = e_previous.flatten()
    
    epsilon = 0
    dot_product = torch.dot(g_flat, e_flat)
    norm_curr = torch.norm(g_flat, p=2)
    norm_prev = torch.norm(e_flat, p=2)
    
    cosine_similarity = dot_product / (norm_curr * norm_prev + epsilon)
    
    # alpha는 단일 스칼라 값이 됨
    alpha = (1 + torch.clamp(cosine_similarity, -1.0, 1.0)) / 2
    
    # 최종 중요도 = 기존 중요도 * alpha (스칼라 값이 브로드캐스팅됨)
    final_importance = base_importance * alpha
    
    return final_importance.detach().cpu().numpy()

def get_vit_masks_with_two_stage_grouping_unified_qkv_with_e(info_for_pruning,model, attn_pruning_rate, ffn_pruning_rate, imp_type, mag_type, group_size=1):
    """
    2단계 헤드 그룹화를 적용한 ViT 마스크 생성 - QKV를 통합된 하나의 컴포넌트로 처리
    어텐션과 FFN에 대해 별도의 프루닝 비율 적용
    
    Args:
        model: ViT 모델
        attn_pruning_rate: 어텐션(QKV)에 대한 프루닝 비율 (0.0-1.0)
        ffn_pruning_rate: FFN에 대한 프루닝 비율 (0.0-1.0)
        imp_type: 중요도 계산 방식 ('L1', 'L2', 'taylor1', 'act_taylor')
        mag_type: 중요도 계산 대상 ('weight' 또는 'grad')
        group_size: 각 그룹에 포함될 차원의 수 (기본값: 16)
    
    Returns:
        masks: 레이어별 마스크 딕셔너리
    """
    # 각 컴포넌트별 중요도 저장
    importance_dict = {'qkv': [], 'ffn1': []}
    
    # 각 컴포넌트별 헤드 간 그룹 중요도 저장
    attn_cross_head_importances = []  # QKV에 대한 중요도
    ffn_cross_head_importances = []   # FFN에 대한 중요도
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight

    # 모델 구조에서 num_heads 가져오기
    num_heads = None
    for name, module in model.named_modules():
        if "attn" in name.lower() and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            break
    
    if num_heads is None:
        raise ValueError("Could not find number of attention heads in model")


        
    # 중요도 점수 수집
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight = module.weight
            grad = module.weight.grad if mag_type == 'grad' else None

            if "qkv" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                e_previous = info_for_pruning.get(name, {}).get("e_w")

                importance_scores = get_temporal_importance_with_e(weight, grad, e_previous, num_heads)

                # QKV 중요도 분리 [3, num_heads, head_dim]
                q_importance = importance_scores[0]  # [num_heads, head_dim]
                k_importance = importance_scores[1]
                v_importance = importance_scores[2]
                
                # QKV 통합 중요도 계산 - Q, K, V의 평균
                qkv_importance_avg = (q_importance + k_importance + v_importance) / 3
                
                # 헤드별 그룹화 수행 후 헤드 간 그룹 중요도 계산
                _, qkv_cross_head_importance, _ = group_head_dimensions_two_stage(
                    qkv_importance_avg, group_size
                )
                
                importance_dict['qkv'].append(qkv_importance_avg)
                
                # 어텐션 컴포넌트의 중요도를 별도로 저장
                attn_cross_head_importances.extend(qkv_cross_head_importance)

            elif "mlp.fc1" in name.lower():
                # 활성화 값 가져오기 (act_taylor에만 필요)
                activation = model.module.get_activation(name) if imp_type == 'act_taylor' else None
                

                ffn_importance = get_ffn_temporal_importance_with_e(weight, grad, e_previous)

                
                importance_dict['ffn1'].append(ffn_importance)
                
                # 수정: FFN 레이어는 그룹당 차원 수로 group_size만 사용
                out_features = len(ffn_importance)
                dims_per_group = group_size  # 변경된 부분: num_heads를 곱하지 않음
                num_groups = out_features // dims_per_group
                
                # FFN 차원을 중요도에 따라 정렬하고 그룹화
                sorted_indices = np.argsort(-ffn_importance)
                ffn_cross_group_importance = np.zeros(num_groups)
                ffn_group_indices = {}
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    selected_indices = sorted_indices[start_idx:end_idx]
                    ffn_group_indices[i] = selected_indices
                    
                    if start_idx < end_idx:
                        ffn_cross_group_importance[i] = np.mean(ffn_importance[selected_indices])
                
                # FFN 컴포넌트의 중요도를 별도로 저장
                ffn_cross_head_importances.extend(ffn_cross_group_importance)

    # 마스크 생성
    masks = {}
    
    # 어텐션과 FFN에 대해 별도의 임계값 계산
    attn_cross_head_importances_array = np.array(attn_cross_head_importances)
    ffn_cross_head_importances_array = np.array(ffn_cross_head_importances)
    
    attn_threshold = np.percentile(attn_cross_head_importances_array, attn_pruning_rate * 100)
    ffn_threshold = np.percentile(ffn_cross_head_importances_array, ffn_pruning_rate * 100)
    
    # 각 컴포넌트에 대해 별도의 임계값을 사용하여 마스크 생성
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "qkv" in name.lower():
                qkv_importance = importance_dict['qkv'].pop(0)
                
                # 어텐션 임계값을 사용하여 마스크 생성
                qkv_masks = generate_masks_with_two_stage_grouping(
                    qkv_importance, attn_threshold, group_size
                )
                
                # 통합된 QKV 마스크를 Q, K, V 모두에 적용
                combined_mask = np.tile(qkv_masks, (3, 1)).reshape(-1)
                
                # QKV 레이어 마스크 생성
                masks[name] = np.tile(combined_mask[:, np.newaxis], (1, module.in_features))
                
                # Projection layer 마스크
                proj_name = name.replace('qkv', 'proj')
                proj_module = next(m for n, m in model.named_modules() if n == proj_name)
                head_dim = proj_module.in_features // num_heads
                
                # QKV 마스크를 기반으로 projection 마스크 생성
                proj_mask = qkv_masks.reshape(-1)
                masks[proj_name] = np.tile(proj_mask[:, np.newaxis], (1, proj_module.out_features)).T
                
            elif "mlp.fc1" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 레이어 차원을 중요도에 따라 정렬
                sorted_indices = np.argsort(-ffn_importance)
                out_features = len(ffn_importance)
                
                # 수정: 그룹 크기로 group_size만 사용
                dims_per_group = group_size
                num_groups = out_features // dims_per_group
                
                # 그룹 마스크 생성 (FFN 임계값 사용)
                ffn_mask = np.zeros(out_features, dtype=bool)
                
                for i in range(num_groups):
                    start_idx = i * dims_per_group
                    end_idx = min(start_idx + dims_per_group, out_features)
                    
                    # 정렬된 인덱스에서 해당 그룹에 포함될 차원 선택
                    selected_indices = sorted_indices[start_idx:end_idx]
                    
                    # 그룹 중요도 계산
                    group_importance = np.mean(ffn_importance[selected_indices])
                    
                    # FFN 임계값과 비교
                    if group_importance > ffn_threshold:
                        ffn_mask[selected_indices] = True
                
                # FC1과 FC2 마스크 생성
                masks[name] = np.tile(ffn_mask[:, np.newaxis], (1, module.in_features))
                
                fc2_name = name.replace('fc1', 'fc2')
                output_module = next(m for n, m in model.named_modules() if n == fc2_name)
                masks[fc2_name] = np.tile(ffn_mask[:, np.newaxis], (1, output_module.out_features)).T
    
    return masks