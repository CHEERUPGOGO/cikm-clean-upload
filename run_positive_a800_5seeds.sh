#!/usr/bin/env bash
#
# Positive-control A800 run.
#
# This keeps the negative/main experiment protocol unchanged except for the
# training prompt: all trained positive-control variants receive gold evidence
# during training.
#
# Counterpart mapping:
#   oracle_sft  -> oracle_ctx
#   hard_kd     -> context_hard_kd
#   logit_kd    -> context_logit_kd
#
# It runs Qwen2.5-3B and Qwen2.5-7B for five seeds by default.
# Outputs are isolated under:
#   model_runs/<RUN_TAG_PREFIX>_<seed>/<model_slug>/
#   outputs/<RUN_TAG_PREFIX>_<seed>_<model_slug>/
#
# Default RUN_TAG_PREFIX is exp5000_v4_positive, so seed 42 writes to:
#   model_runs/exp5000_v4_positive_42/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45 46}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-exp5000_v4_positive}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_GPU="${REQUIRE_GPU:-true}"
export TEACHER_DEVICE_MAP="${TEACHER_DEVICE_MAP:-auto}"
export TEACHER_LOAD_IN_4BIT="${TEACHER_LOAD_IN_4BIT:-false}"

export INSTALL_DEPS="${INSTALL_DEPS:-0}"
export DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
# Keep REBUILD_DATA=0 by default so this positive run can reuse the same
# data.json and 32B teacher cache as the negative/main run.
export REBUILD_DATA="${REBUILD_DATA:-0}"
export FORCE_TEACHER_CACHE="${FORCE_TEACHER_CACHE:-0}"
export REQUIRE_TEACHER_CACHE="${REQUIRE_TEACHER_CACHE:-1}"

export NUM_FACTS="${NUM_FACTS:-5000}"
export FACT_TEST_RATIO="${FACT_TEST_RATIO:-0.2}"
export TRAIN_PARAPHRASES_PER_FACT="${TRAIN_PARAPHRASES_PER_FACT:-2}"
export EVAL_PARAPHRASES_PER_FACT="${EVAL_PARAPHRASES_PER_FACT:-3}"

export BASE_TRAIN_MODES="${BASE_TRAIN_MODES:-oracle_ctx,context_hard_kd,context_logit_kd}"
# Keep this false because context_logit_kd is explicitly listed above.
# Setting ENABLE_LOGIT_KD=true would also add the answer-only logit_kd mode.
export ENABLE_LOGIT_KD="${ENABLE_LOGIT_KD:-false}"
export EVAL_CONTEXT_VARIANTS="${EVAL_CONTEXT_VARIANTS:-gold,retrieved_bm25,retrieved_dense,random,noisy}"
export RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
export RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cpu}"
export FACT_SCORE_METRIC="${FACT_SCORE_METRIC:-answer_acc}"

export TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"

export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"

export TEACHER_ANSWER_MODE="${TEACHER_ANSWER_MODE:-generate}"
export TEACHER_INVALID_ACTION="${TEACHER_INVALID_ACTION:-keep}"
export KD_TOP_K="${KD_TOP_K:-128}"
export KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
export KD_ALPHA="${KD_ALPHA:-0.5}"

run_step() {
  echo
  echo "===== $* ====="
  "$@"
}

if [[ "$INSTALL_DEPS" == "1" ]]; then
  run_step "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  run_step "$PYTHON_BIN" download_models.py \
    --qwen_qwen2.5-32b \
    --Qwen \
    --qwen_qwen2.5-7b \
    --dense-retriever
fi

teacher_cache_valid() {
  "$PYTHON_BIN" - <<'PY'
from run_all import valid_teacher_cache
raise SystemExit(0 if valid_teacher_cache() else 1)
PY
}

for seed in $SEEDS; do
  export SEED="$seed"
  export RUN_TAG="${RUN_TAG_PREFIX}_${seed}"

  echo
  echo "============================================================"
  echo "Positive-control 3B + 7B A800 run"
  echo "RUN_TAG:           $RUN_TAG"
  echo "SEED:              $SEED"
  echo "CUDA_VISIBLE:      $CUDA_VISIBLE_DEVICES"
  echo "TRAIN_MODES:       $BASE_TRAIN_MODES"
  echo "ENABLE_LOGIT_KD:   $ENABLE_LOGIT_KD"
  echo "REBUILD_DATA:      $REBUILD_DATA"
  echo "FORCE_CACHE:       $FORCE_TEACHER_CACHE"
  echo "REQUIRE_CACHE:     $REQUIRE_TEACHER_CACHE"
  echo "NUM_FACTS:         $NUM_FACTS"
  echo "INFER_BATCH:       $INFERENCE_BATCH_SIZE"
  echo "============================================================"

  if [[ "$REBUILD_DATA" == "1" || ! -f data.json ]]; then
    run_step "$PYTHON_BIN" build_dataset.py
  fi

  if [[ "$FORCE_TEACHER_CACHE" == "1" ]]; then
    run_step "$PYTHON_BIN" precompute_teacher_cache.py
  elif teacher_cache_valid; then
    echo
    echo "===== teacher cache is valid; skip precompute_teacher_cache.py ====="
  elif [[ "$REQUIRE_TEACHER_CACHE" == "1" ]]; then
    echo
    echo "ERROR: teacher cache is missing or does not match data.json."
    echo "This positive run defaults to reusing the negative/main run cache."
    echo "Check that REBUILD_DATA=0 and teacher_cache/ matches the current data.json."
    echo "If you intentionally want to rebuild it, set REQUIRE_TEACHER_CACHE=0 FORCE_TEACHER_CACHE=1."
    exit 1
  else
    run_step "$PYTHON_BIN" precompute_teacher_cache.py
  fi

  run_step "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b
done

echo
AGGREGATE_RUN_PREFIX="$RUN_TAG_PREFIX" "$PYTHON_BIN" aggregate_seeds.py
echo "Positive-control A800 seeds finished."
echo "Per-seed summaries: model_runs/${RUN_TAG_PREFIX}_<seed>/"
echo "Mean/std summary:   model_runs/${RUN_TAG_PREFIX}_aggregate/internalization_mean_std.csv"
