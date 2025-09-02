export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

#!/bin/bash


ms="small"

python compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/OXO_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs1_unified_qkv_grad_taylor18_imagnet.pth\
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/OXO_${ms}_eval_imagenet.txt

python diff_head_v_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/XXO_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs1_ours_grad_taylor18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/XXO_${ms}_eval_imagenet.txt

python diff_head_v_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/XOO_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs16_ours_grad_taylor18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/XOO_${ms}_eval_imagenet.txt

python neuron_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/XXX_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs1_neuron_grad_taylor18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/XXX_${ms}_eval_imagenet.txt

python neuron_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/OOX_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs16_unified_qkv16_diff_heads_grad_taylor18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/OOX_${ms}_eval_imagenet.txt

python neuron_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/OXX_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs1_unified_qkv16_diff_heads_grad_taylor18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/OXX_${ms}_eval_imagenet.txt

python neuron_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/XOX_timm_pretrain50_scale_0.0002small_attn0.6_ffn0.7_gs16_qk_v16_diff_heads_grad_taylor18_imagnet.pth\
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 3 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./ablation_log/XOX_${ms}_eval_imagenet.txt
