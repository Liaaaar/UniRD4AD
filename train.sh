#!/usr/bin/env bash

GPU=3
IMG_SIZE=256

# Set your own experiment name and paths here
EXP_NAME=your_own_exp_name
SAVE_DIR=./your_own_path/${EXP_NAME}
LOG_FILE=./your_own_path/${EXP_NAME}.log

mkdir -p "$SAVE_DIR"

CUDA_VISIBLE_DEVICES=$GPU nohup python -u train.py \
  --seed 42 \
  --dataset mvtec \
  --img_size "$IMG_SIZE" \
  --backbone wide_resnet50_2 \
  --rd_loss cosine \
  --cluster_loss cosine \
  --epochs 200 \
  --lr 0.01 \
  --batch_size 32 \
  --save_path "$SAVE_DIR" \
  > "$LOG_FILE" 2>&1 &
