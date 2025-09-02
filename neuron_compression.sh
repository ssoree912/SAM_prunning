#!/bin/bash

# model_size
ms="small"

python neuron_compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/timm_pretrain50_taylor_0.0002small_attn0.3_ffn0.4_gs16_neuron_weight_L18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 1 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./log/${ms}_eval_imagenet.txt

