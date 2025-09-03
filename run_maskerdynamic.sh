#!/bin/bash

# MaskerDynamic (type 6) 테스트 실행 스크립트
# 새로운 "full gradient" DPF 방식

echo "Running with MaskerDynamic (type 6) - Full Gradient DPF"

python cifar.py \
    --model vit_tiny_patch16_224 \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.0005 \
    --mask_type 6 \
    --attn-prune-rate 0.3 \
    --ffn-prune-rate 0.3 \
    --prune-freq 10 \
    --warmup 5 \
    --target-epoch 80 \
    --method ours \
    --wandb \
    --project-name "prunning" \
    --run-name "maskerdynamic_type6_test"

echo "Completed MaskerDynamic test"