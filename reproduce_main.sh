#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/af3_checkpoint}"
TRAIN_JSONL="${TRAIN_JSONL:-/path/to/castella_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/path/to/castella_val.jsonl}"
TEST_JSONL="${TEST_JSONL:-/path/to/castella_test.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-./outputs/features}"
STAGE_A_DIR="${STAGE_A_DIR:-./outputs/stage_a}"
STAGE_B_DIR="${STAGE_B_DIR:-./outputs/stage_b}"
EVAL_DIR="${EVAL_DIR:-./outputs/eval}"

python scripts/extract_af3_features.py \
  --model-path "${MODEL_PATH}" \
  --input-jsonl "${TRAIN_JSONL}" "${VAL_JSONL}" "${TEST_JSONL}" \
  --output-root "${FEATURE_ROOT}"

python scripts/train_tokenizer_stage.py \
  --config configs/stage_a_castella.yaml \
  --train-jsonl "${TRAIN_JSONL}" \
  --feature-root "${FEATURE_ROOT}" \
  --output-dir "${STAGE_A_DIR}"

python scripts/train_amr_stage.py \
  --config configs/stage_b_castella.yaml \
  --model-path "${MODEL_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --feature-root "${FEATURE_ROOT}" \
  --stage-a-dir "${STAGE_A_DIR}" \
  --output-dir "${STAGE_B_DIR}"

python scripts/eval_amr.py \
  --model-path "${MODEL_PATH}" \
  --model-ckpt "${STAGE_B_DIR}/best.pt" \
  --input-jsonl "${TEST_JSONL}" \
  --feature-root "${FEATURE_ROOT}" \
  --stage-a-dir "${STAGE_A_DIR}" \
  --output-dir "${EVAL_DIR}"
