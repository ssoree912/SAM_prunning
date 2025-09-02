#!/bin/bash

# model_size
ms="small"

python compression.py --data-set IMNET \
    --data-path /workspace/data/ImageNet \
    --checkpoint ./output/Best_orin_pre50_fine150_DSP_static0.0002small0.75_ours_grad_taylor18_imagnet.pth \
    --epochs 1 \
    --gpuids 0 \
    --group_size 16 \
    --cu_num 1 \
    --prompt_length 1 \
    --model_size ${ms} \
    --lr 5e-5 \
    > ./log/${ms}_eval_imagenet.txt