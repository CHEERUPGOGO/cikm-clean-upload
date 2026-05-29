#!/usr/bin/env bash
#
# Run the FPFT (Full Parameter Fine-Tuning) pipeline for five seeds.
# Parallelized version aligned with Noisy LoRA experiments.
# Maps 5 seeds across physical GPUs designated by AVAILABLE_GPUS concurrently.
#
# Each seed writes to:
#   model_runs/seed_<seed>/<model_slug>/
#   outputs/seed_<seed>_<model_slug>/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Core pipeline configuration
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_GPU="${REQUIRE_GPU:-true}"
export NUM_FACTS="${NUM_FACTS:-5000}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
export MAX_LENGTH="${MAX_LENGTH:-256}"

# FPFT Critical Overrides
# 1. BATCH_SIZE=1 to prevent OOM
# 2. LEARNING_RATE=2e-5 for faster convergence over 3 epochs
# 3. USE_LORA=false to enable full-parameter updating
export BATCH_SIZE=1
export LEARNING_RATE=2e-5
export USE_LORA="false"

# Experiment settings
SEEDS=(${SEEDS:-42 43 44 45 46})
AVAILABLE_GPUS=(${AVAILABLE_GPUS:-0 1 2 3 4})
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-fpft5000_noisy}"

# Aligned with noisy LoRA distillation settings
export BASE_TRAIN_MODES="${BASE_TRAIN_MODES:-noisy_oracle_ctx,noisy_context_hard_kd}"
export ENABLE_LOGIT_KD="${ENABLE_LOGIT_KD:-false}"
export INCLUDE_7B="${INCLUDE_7B:-1}"
export EVAL_CONTEXT_VARIANTS="${EVAL_CONTEXT_VARIANTS:-gold,retrieved_bm25,retrieved_dense,random,noisy}"
export RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
export RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cpu}"

# Teacher Cache Settings
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

echo "============================================================"
echo "Starting PARALLEL FPFT for seeds: ${SEEDS[*]}"
echo "Using Physical GPUs: ${AVAILABLE_GPUS[*]}"
echo "Models included: 3B, 7B (7B included: $INCLUDE_7B)"
echo "Train Modes: $BASE_TRAIN_MODES"
echo "============================================================"

# PRE-REQUISITES (MUST RUN BEFORE PARALLELIZATION)
# 1. Rebuild data.json once before the seed loop to ensure noisy_context is present
echo "Rebuilding data.json with SEED=42 to inject noisy_context..."
export SEED="42"
export REBUILD_DATA="1"
run_step "$PYTHON_BIN" build_dataset.py
export REBUILD_DATA="0"

# 2. Force teacher cache generation once on all available GPUs
echo "Precomputing teacher cache on GPUs: ${AVAILABLE_GPUS[*]} ..."
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")
run_step "$PYTHON_BIN" precompute_teacher_cache.py

# Launch each seed in the background on a specific GPU
echo
echo "============================================================"
echo "Launching seeds concurrently on designated physical GPUs..."
echo "============================================================"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  run_tag="${RUN_TAG_PREFIX}_${seed}"
  
  # Pick the corresponding physical GPU from the array
  gpu_id="${AVAILABLE_GPUS[$i]}"
  
  echo "=> Launching seed $seed on physical GPU $gpu_id (RUN_TAG=$run_tag)..."
  
  # Run in a subshell in the background
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export SEED="$seed"
    export RUN_TAG="$run_tag"
    
    if [ "$INCLUDE_7B" = "1" ]; then
      "$PYTHON_BIN" resume_run.py --Qwen --qwen_qwen2.5-7b
    else
      "$PYTHON_BIN" resume_run.py --Qwen
    fi
  ) &
done

echo "All 5 seeds have been dispatched to the GPUs."
echo "Waiting for all background training jobs to finish..."
wait
echo "All parallel training jobs completed successfully!"

echo
echo "Aggregating results..."
AGGREGATE_RUN_PREFIX="$RUN_TAG_PREFIX" "$PYTHON_BIN" aggregate_seeds.py
echo "All A800 FPFT seeds finished. Per-seed summaries are under model_runs/<run_tag>/."
echo "Mean/std summary: model_runs/${RUN_TAG_PREFIX}_aggregate/internalization_mean_std.csv"
