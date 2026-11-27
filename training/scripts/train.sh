#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/../.."

python training/scripts/train.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --train training/datasets/output2/train.jsonl \
  --validation training/datasets/output2/validation.jsonl \
  --output training/outputs/fgs-output2 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 5e-7 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-length 8192