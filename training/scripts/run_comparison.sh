#!/usr/bin/env bash
#
# 1) output2 만 사용한 데이터셋과 output1+output2 데이터셋을 만든다.
# 2) 각 데이터셋으로 LoRA 모델을 학습한다.
# 3) 세 모델(베이스 / output2 / output1+output2)을 같은 output2 검증셋으로 비교한다.
#
# 사용:
#   bash training/scripts/run_comparison.sh
#   FORCE=1 bash training/scripts/run_comparison.sh   # 기존 어댑터도 다시 학습
#
# 환경 변수로 학습 설정을 바꿀 수 있다:
#   FETCH, MODEL, EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, LORA_R, LORA_ALPHA,
#   LORA_DROPOUT, MAX_LENGTH, OUT2_DIR, OUT12_DIR, EVAL_DIR, EVAL_ARGS

set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
OUT2_DIR="${OUT2_DIR:-training/outputs/fgs-output2}"
OUT12_DIR="${OUT12_DIR:-training/outputs/fgs-output1-output2}"
EVAL_DIR="${EVAL_DIR:-training/eval_results}"
FORCE="${FORCE:-0}"

# ---------- 1) 데이터셋 ----------
# fetch_figma.py 가 만든 training/datasets/index.json 을 사용한다.
# index.json 이 없거나 새로 받고 싶으면 FETCH=1 로 실행한다.
# (training/datasets 에 figma_links.csv 또는 figma_links_2.csv 가 있어야 한다)

if [ "${FETCH:-0}" = "1" ]; then
  echo "[FETCH] fetch_figma.py 로 Figma layout/image/index.json 갱신"
  (cd training/datasets && python fetch_figma.py --outdir .)
fi

python training/scripts/prepare_dataset.py \
  --answer-keys output2 \
  --name output2

python training/scripts/prepare_dataset.py \
  --answer-keys output output2 \
  --name output1_output2

# ---------- 2) 학습 ----------
train_one () {
  local train_file="$1"
  local validation_file="$2"
  local output_dir="$3"

  if [ -f "$output_dir/adapter_config.json" ] && [ "$FORCE" != "1" ]; then
    echo "[SKIP] 이미 학습된 어댑터가 있습니다: $output_dir (FORCE=1 이면 다시 학습)"
    return
  fi

  echo "[TRAIN] $train_file -> $output_dir"

  python training/scripts/train.py \
    --model "$MODEL" \
    --train "$train_file" \
    --validation "$validation_file" \
    --output "$output_dir" \
    --epochs "${EPOCHS:-3}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --gradient-accumulation "${GRAD_ACCUM:-8}" \
    --learning-rate "${LR:-5e-7}" \
    --lora-r "${LORA_R:-16}" \
    --lora-alpha "${LORA_ALPHA:-32}" \
    --lora-dropout "${LORA_DROPOUT:-0.05}" \
    --max-length "${MAX_LENGTH:-8192}"
}

train_one \
  training/datasets/output2/train.jsonl \
  training/datasets/output2/validation.jsonl \
  "$OUT2_DIR"

train_one \
  training/datasets/output1_output2/train.jsonl \
  training/datasets/output1_output2/validation.jsonl \
  "$OUT12_DIR"

# ---------- 3) 비교 평가 ----------
# 세 모델 모두 같은 output2 검증셋으로 평가한다.
python training/scripts/evaluate.py \
  --model "$MODEL" \
  --validation training/datasets/output2/validation.jsonl \
  --adapter "output2=$OUT2_DIR" \
  --adapter "output1_output2=$OUT12_DIR" \
  --output "$EVAL_DIR" \
  ${EVAL_ARGS:-}
