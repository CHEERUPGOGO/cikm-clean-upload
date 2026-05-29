#!/usr/bin/env bash
#
# Multi-Document Noisy Positive-control A800 run.
#
# This explicitly tests if training with noisy context (1 gold + 4 distractors)
# solves the massive variance in BM25 retrieval by eliminating the positional shortcut.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45 46}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-exp5000_v4_noisy_positive}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_GPU="${REQUIRE_GPU:-true}"
export TEACHER_DEVICE_MAP="${TEACHER_DEVICE_MAP:-auto}"
export TEACHER_LOAD_IN_4BIT="${TEACHER_LOAD_IN_4BIT:-false}"

export INSTALL_DEPS="${INSTALL_DEPS:-0}"
export DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"

# We always rebuild the data once before the loop to generate the new noisy_context field.
export REBUILD_DATA="0"
export REQUIRE_TEACHER_CACHE="${REQUIRE_TEACHER_CACHE:-0}"

export NUM_FACTS="${NUM_FACTS:-5000}"
export FACT_TEST_RATIO="${FACT_TEST_RATIO:-0.2}"
export TRAIN_PARAPHRASES_PER_FACT="${TRAIN_PARAPHRASES_PER_FACT:-2}"
export EVAL_PARAPHRASES_PER_FACT="${EVAL_PARAPHRASES_PER_FACT:-3}"

# These are the new modes
export BASE_TRAIN_MODES="${BASE_TRAIN_MODES:-noisy_oracle_ctx,noisy_context_hard_kd,noisy_context_logit_kd}"
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

# Rebuild data.json once before the seed loop to ensure noisy_context is present
echo "Rebuilding data.json with SEED=42 to inject noisy_context..."
export SEED="42"
run_step "$PYTHON_BIN" build_dataset.py

# Force teacher cache generation once on all user-specified GPUs
echo "Precomputing teacher cache on GPUs: $CUDA_VISIBLE_DEVICES ..."
run_step "$PYTHON_BIN" precompute_teacher_cache.py

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
if [ ${#GPU_ARRAY[@]} -lt 5 ]; then
  echo "ERROR: This parallel script expects at least 5 GPUs specified in CUDA_VISIBLE_DEVICES."
  exit 1
fi

GPU0="${GPU_ARRAY[0]}"
GPU1="${GPU_ARRAY[1]}"
GPU2="${GPU_ARRAY[2]}"
GPU3="${GPU_ARRAY[3]}"
GPU4="${GPU_ARRAY[4]}"

# Now run seeds in parallel using 1 GPU per seed
echo
echo "============================================================"
echo "Launching Seed 42, 43, 44, 45, 46 in parallel on GPUs $GPU0, $GPU1, $GPU2, $GPU3, $GPU4..."
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU0 SEED=42 RUN_TAG="${RUN_TAG_PREFIX}_42" "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b &
CUDA_VISIBLE_DEVICES=$GPU1 SEED=43 RUN_TAG="${RUN_TAG_PREFIX}_43" "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b &
CUDA_VISIBLE_DEVICES=$GPU2 SEED=44 RUN_TAG="${RUN_TAG_PREFIX}_44" "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b &
CUDA_VISIBLE_DEVICES=$GPU3 SEED=45 RUN_TAG="${RUN_TAG_PREFIX}_45" "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b &
CUDA_VISIBLE_DEVICES=$GPU4 SEED=46 RUN_TAG="${RUN_TAG_PREFIX}_46" "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b &
wait

echo
AGGREGATE_RUN_PREFIX="$RUN_TAG_PREFIX" "$PYTHON_BIN" aggregate_seeds.py
echo "Noisy Positive-control A800 seeds finished."
echo "Per-seed summaries: model_runs/${RUN_TAG_PREFIX}_<seed>/"
echo "Mean/std summary:   model_runs/${RUN_TAG_PREFIX}_aggregate/internalization_mean_std.csv"
