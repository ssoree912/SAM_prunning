import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x 
 
    @staticmethod
    def backward(ctx, grad):
        return grad


    
    #마스크가 0인 부분은 0으로 만든다. 일반적인 정적 프루닝 마스크 
class Static_Masker(torch.autograd.Function):
    @staticmethod
    def forward(weight, mask):
        return weight * mask

    @staticmethod
    def setup_context(ctx, inputs, output):
        weight, mask = inputs
        ctx.save_for_backward(mask)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_weight = grad_output * mask
        return grad_weight, None

    # vmap을 위한 정적 메서드 추가
    @staticmethod
    def vmap(info, in_dims, weight, mask):
        # weight는 배치 차원이 있고 (0), mask는 없다고 (None) 가정
        # 실제 사용 사례에 따라 in_dims_mask를 조정해야 할 수 있습니다.
        in_dims_weight, in_dims_mask = in_dims
        
        # in_dims에 따라 입력을 배치 단위로 변환
        batch_weight = torch.movedim(weight, in_dims_weight, 0)
        
        # Static_Masker.apply를 배치에 대해 호출
        masked_batch = Static_Masker.apply(batch_weight, mask)
        
        # 출력 차원을 원래대로 복원
        return torch.movedim(masked_batch, 0, in_dims_weight), None
    
    
    #살아남은 프루닝에 조금 더 큰 그래디언트를 흘려줌 
class Static_scale_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        x, mask = ctx.saved_tensors
        
        return x.abs() * grad * mask  , None
    
class scale_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        x, mask = ctx.saved_tensors
        
        return (x ** 2) * grad   , None
    
class Dynamic_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask + x * (1 - mask) * 0.0001

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        # 마스크가 0인 위치: 1000배 증폭, 1인 위치: 원본 유지
        scaling_factor = mask + (1 - mask) * 100   # [1 or 1000] 형태의 텐서
        return grad * scaling_factor, None  # x에 대한 gradient만 반환
    
class ReviveGrad_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, revival_factor=0.05): # revival_factor를 인자로 추가
        ctx.save_for_backward(x, mask)
        ctx.revival_factor = revival_factor # revival_factor 저장
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        x, mask = ctx.saved_tensors
        revival_factor = ctx.revival_factor

        grad_alive = grad * mask

        grad_revived_component = (1 - mask) * x.abs() * grad * revival_factor

        final_grad_x = grad_alive + grad_revived_component

        return final_grad_x, None, None 

# class MaskerScaling(torch.autograd.Function):
#     # mask에 따라서 다르게 주기
#     @staticmethod
#     def forward(ctx, x, mask, beta, threshold):
#         ctx.save_for_backward(mask)
#         ctx.beta = beta
#         ctx.th = torch.abs(torch.abs(x) - threshold)
#         ctx.th2 = torch.abs(x)

#         return x * mask

#     @staticmethod
#     def backward(ctx, grad):
#         (mask,) = ctx.saved_tensors

#         # TODO : 양쪽 beta 튜닝
#         # 살아있는애는 0으로, 죽은애는 threshold로
#         return (
#             (grad * (1 - mask) * ctx.th * ctx.beta)
#             + (grad * mask * ctx.th2 * ctx.beta),
#             None,
#             None,
#             None,
#         )


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',scale_it=None):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.type_value = 1

    def forward(self, input):
        if self.type_value == 0:
            masked_weight = Static_Masker.apply(self.weight, self.mask)

        elif self.type_value == 1:
            masked_weight = Dynamic_Masker.apply(self.weight, self.mask)
        elif self.type_value == 2:
            masked_weight = dense.apply(self.weight)
        elif self.type_value == 5:
            masked_weight = ReviveGrad_Masker.apply(self.wegith, self.mask)

        return super(MaskConv2d, self)._conv_forward(input, masked_weight,self.bias)

    
class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias)
        self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.type_value = 0
        if bias:
            self.bias_mask = Parameter(torch.ones(out_features), requires_grad=False)

    def forward(self, input):

        if self.type_value == 0:
            masked_weight = Static_Masker.apply(self.weight, self.mask)
        elif self.type_value == 1:
            masked_weight = Dynamic_Masker.apply(self.weight, self.mask)
        elif self.type_value == 2:
            masked_weight = dense.apply(self.weight)
        elif self.type_value == 3:
            masked_weight = Static_scale_Masker.apply(self.weight, self.mask)
        elif self.type_value == 4:
            masked_weight = scale_Masker.apply(self.weight, self.mask)
        elif self.type_value == 5:
            masked_weight = ReviveGrad_Masker.apply(self.wegith, self.mask)

        return F.linear(input, masked_weight, self.bias * self.bias_mask)

    def set_type_value(self, type_value):
        self.type_value = type_value