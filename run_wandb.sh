#!/bin/bash

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0

# Wandb 로그인 확인
echo "Checking wandb login status..."
if ! wandb status | grep -q "Logged in"; then
    echo "Please login to wandb first:"
    echo "wandb login"
    exit 1
fi

# 실험 설정
MODEL_SIZE=${1:-small}
PRUNE_RATE=${2:-0.3}
ATTN_PRUNE_RATE=${3:-0.3}
FFN_PRUNE_RATE=${4:-0.3}
EXPERIMENT_NAME=${5:-"sam_pruning_${MODEL_SIZE}_${PRUNE_RATE}"}

echo "Starting training with wandb logging..."
echo "Model size: $MODEL_SIZE"
echo "Prune rate: $PRUNE_RATE"
echo "Attention prune rate: $ATTN_PRUNE_RATE"
echo "FFN prune rate: $FFN_PRUNE_RATE"
echo "Experiment name: $EXPERIMENT_NAME"

python cifar.py \
  --cu_num 0 \
  --wandb \
  --wandb-project "pruning" \
  --wandb-name "$EXPERIMENT_NAME" \
  --model_size "$MODEL_SIZE" \
  --prune-rate "$PRUNE_RATE" \
  --attn-prune-rate "$ATTN_PRUNE_RATE" \
  --ffn-prune-rate "$FFN_PRUNE_RATE" \
  --data-set CIFAR \
  --epochs 100 \
  --batch-size 256 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --prune-freq 8 \
  --method ours \
  --warmup 1 \
  --target_epoch 70 \
  --prune-imp L1 \
  --mag_type weight

echo "Training completed!"