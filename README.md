# Retrieval-Augmented Generation (RAG) Distillation Diagnostic

This repository contains the code, dataset generators, and diagnostic evaluation framework for analyzing how small language models internalize facts into parameters versus utilizing external retrieval evidence.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download required models**:
```bash
python download_models.py
```

3. **Run the complete diagnostic pipeline** (Dataset generation, teacher cache precomputation, distillation training, and evaluation):
```bash
bash run_full_distill.sh
```

## Repository Structure

- `build_dataset.py`: Generates the synthetic controlled QA dataset with various context configurations.
- `precompute_teacher_cache.py`: Generates the teacher model's outputs (answers and logits) to be used as distillation targets.
- `train.py`: Handles model training across various distillation modes (e.g., oracle SFT, hard-label KD, logit KD).
- `retrieval.py`: Implements BM25 and dense retrieval evidence variants.
- `inference.py` & `eval.py`: Runs generation inference and computes robust evaluation metrics (e.g., FactAcc, Internalization Efficiency).
- `run_all.py` / `resume_run.py` / `aggregate_seeds.py`: Python scripts for pipeline execution, resuming, and multi-seed result aggregation.
- `*.sh`: Bash entrypoints for launching full multi-gpu experiments across multiple seeds.

## Outputs and Logging

Experiment results and logs are saved to the `model_runs/` directory. Summary statistics are compiled into structured JSON and CSV formats detailing example-level metrics, fact-level metrics, and retrieval-conditioned evaluations.
