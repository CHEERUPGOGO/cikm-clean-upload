import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

from config import (
    DATA_PATH,
    KD_TOP_K,
    PROJECT_DIR,
    RETRIEVAL_TOP_K,
    RUN_TAG,
    TEACHER_CACHE_VERSION,
    TEACHER_CACHE_PATH,
    TEACHER_MODEL_NAME,
)
from eval import (
    compute_internalization_report,
    fact_scores_by_mode_condition,
    load_json,
    merge_prediction_groups,
    summarize_retrieval_quality,
    summarize_retrieval_quality_effect,
    summarize_predictions,
)
from experiment import (
    get_inference_modes,
    get_train_examples,
    get_train_modes,
    iter_jsonl,
    load_experiment_data,
    parse_inference_mode,
    slugify_model_name,
    train_examples_fingerprint,
)


KD_ALIGNMENT_VERSION = "causal_shift_v2"
LOGIT_KD_MODES = {"logit_kd", "context_logit_kd"}
TEACHER_CACHE_MODES = {"hard_kd", "logit_kd", "context_hard_kd", "context_logit_kd"}


MODEL_PRESETS = {
    "distilgpt2": "distilgpt2",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen": "Qwen/Qwen2.5-3B-Instruct",
    "facebook_opt_350m": "facebook/opt-350m",
    "qwen_qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
FORCE_INFERENCE = os.getenv("FORCE_INFERENCE", "0").lower() in {"1", "true", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume fact-internalization experiments.")
    for key in MODEL_PRESETS:
        parser.add_argument(f"--{key}", dest=key.replace(".", "_").replace("-", "_"), action="store_true")
    parser.add_argument("--model-name", type=str, default="", help="Run one custom Hugging Face model.")
    return parser.parse_args()


def resolve_target_models(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.model_name.strip():
        custom = args.model_name.strip()
        return [(slugify_model_name(custom), custom)]

    selected = []
    for key, model_name in MODEL_PRESETS.items():
        attr = key.replace(".", "_").replace("-", "_")
        if getattr(args, attr, False):
            selected.append((key, model_name))
    if not selected:
        selected = [
            ("qwen_qwen2.5-0.5b", MODEL_PRESETS["qwen_qwen2.5-0.5b"]),
            ("qwen_qwen2.5-1.5b", MODEL_PRESETS["qwen_qwen2.5-1.5b"]),
            ("Qwen", MODEL_PRESETS["Qwen"]),
        ]
    return selected


def run_script(script_name: str, env_overrides: Dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [sys.executable, os.path.join(PROJECT_DIR, script_name)]
    print(f"\n===== Running {script_name} {env_overrides} =====")
    subprocess.run(cmd, env=env, check=True)


def valid_adapter(output_root: str, mode: str) -> bool:
    mode_dir = os.path.join(output_root, mode)
    is_lora = os.path.exists(os.path.join(mode_dir, "adapter_config.json"))

    if is_lora:
        required = [
            os.path.join(mode_dir, "adapter_config.json"),
            os.path.join(mode_dir, "tokenizer_config.json"),
        ]
        has_weights = any(
            os.path.exists(os.path.join(mode_dir, name))
            for name in ["adapter_model.safetensors", "adapter_model.bin"]
        )
    else:
        required = [
            os.path.join(mode_dir, "config.json"),
            os.path.join(mode_dir, "tokenizer_config.json"),
        ]
        has_weights = any(
            os.path.exists(os.path.join(mode_dir, name))
            for name in ["model.safetensors", "pytorch_model.bin", "model-00001-of-00002.safetensors", "model.safetensors.index.json"]
        )

    if not (all(os.path.exists(path) for path in required) and has_weights):
        return False
    if mode in LOGIT_KD_MODES:
        metadata_path = os.path.join(mode_dir, "experiment_metadata.json")
        if not os.path.exists(metadata_path):
            return False
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return False
        return metadata.get("kd_alignment_version") == KD_ALIGNMENT_VERSION
    return True


def valid_predictions(path: str, expected_len: int, mode: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        rows = load_json(path)
    except Exception:
        return False
    if not isinstance(rows, list) or len(rows) != expected_len:
        return False
    if not all(row.get("mode") == mode for row in rows):
        return False
    if mode.endswith("_ctx"):
        return all(
            "context_variant" in row
            and "retrieval_hit" in row
            and int(row.get("retrieval_top_k", -1)) == RETRIEVAL_TOP_K
            for row in rows
        )
    return True


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_dataset() -> int:
    try:
        payload = load_experiment_data(DATA_PATH)
    except Exception:
        run_script("build_dataset.py", {})
        payload = load_experiment_data(DATA_PATH)
    return len(payload["eval"])


def valid_teacher_cache() -> bool:
    if not os.path.exists(TEACHER_CACHE_PATH):
        return False
    meta_path = TEACHER_CACHE_PATH + ".meta.json"
    payload = load_experiment_data(DATA_PATH)
    train_examples = get_train_examples(payload)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("teacher_model") != TEACHER_MODEL_NAME:
            return False
        if meta.get("cache_version") != TEACHER_CACHE_VERSION:
            return False
        if int(meta.get("top_k", 0)) < KD_TOP_K:
            return False
        if meta.get("train_fingerprint") != train_examples_fingerprint(train_examples):
            return False
    except Exception:
        return False
    expected_ids = {str(example["example_id"]) for example in train_examples}
    cached_ids = set()
    try:
        for row in iter_jsonl(TEACHER_CACHE_PATH):
            cached_ids.add(str(row.get("example_id", "")))
    except Exception:
        return False
    return expected_ids == cached_ids


def maybe_precompute_teacher_cache() -> None:
    if not (set(get_train_modes()) & TEACHER_CACHE_MODES):
        return
    if valid_teacher_cache():
        print(f"Local teacher cache exists, skip: {TEACHER_CACHE_PATH}")
        return
    run_script("precompute_teacher_cache.py", {})


def summarize_model(run_root: str, prediction_files: Dict[str, str]) -> Dict[str, object]:
    all_predictions_by_mode = {mode: load_json(path) for mode, path in prediction_files.items()}
    merged = merge_prediction_groups(all_predictions_by_mode.values())
    example_rows = summarize_predictions(merged)
    fact_rows, _ = fact_scores_by_mode_condition(merged)
    internalization_rows = compute_internalization_report(merged)
    retrieval_quality_rows = summarize_retrieval_quality(merged)
    retrieval_effect_rows = summarize_retrieval_quality_effect(merged)

    with open(os.path.join(run_root, "all_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(all_predictions_by_mode, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_root, "example_summary.json"), "w", encoding="utf-8") as f:
        json.dump(example_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_root, "fact_summary.json"), "w", encoding="utf-8") as f:
        json.dump(fact_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_root, "internalization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(internalization_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_root, "retrieval_quality_summary.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_quality_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_root, "retrieval_quality_effect.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_effect_rows, f, ensure_ascii=False, indent=2)
    save_csv(os.path.join(run_root, "example_summary.csv"), example_rows)
    save_csv(os.path.join(run_root, "fact_summary.csv"), fact_rows)
    save_csv(os.path.join(run_root, "internalization_summary.csv"), internalization_rows)
    save_csv(os.path.join(run_root, "retrieval_quality_summary.csv"), retrieval_quality_rows)
    save_csv(os.path.join(run_root, "retrieval_quality_effect.csv"), retrieval_effect_rows)
    return {
        "example": example_rows,
        "fact": fact_rows,
        "internalization": internalization_rows,
        "retrieval_quality": retrieval_quality_rows,
        "retrieval_quality_effect": retrieval_effect_rows,
    }


def main() -> None:
    args = parse_args()
    targets = resolve_target_models(args)
    expected_eval_len = ensure_dataset()
    maybe_precompute_teacher_cache()

    aggregate = []
    model_runs_root = os.path.join(PROJECT_DIR, "model_runs", RUN_TAG) if RUN_TAG else os.path.join(PROJECT_DIR, "model_runs")
    for _, model_name in targets:
        slug = slugify_model_name(model_name)
        artifact_slug = f"{RUN_TAG}_{slug}" if RUN_TAG else slug
        run_root = os.path.join(model_runs_root, slug)
        output_root = os.path.join(PROJECT_DIR, "outputs", artifact_slug)
        os.makedirs(run_root, exist_ok=True)
        os.makedirs(output_root, exist_ok=True)
        shutil.copyfile(DATA_PATH, os.path.join(run_root, "data.json"))

        env_base = {"MODEL_NAME": model_name, "MODEL_ARTIFACT_SUBDIR": artifact_slug}
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")
        if RUN_TAG:
            print(f"Run tag: {RUN_TAG}")
        if FORCE_RETRAIN:
            print("FORCE_RETRAIN=1, existing checkpoints will be ignored.")
        if FORCE_INFERENCE:
            print("FORCE_INFERENCE=1, existing predictions will be ignored.")

        retrained_modes = set()
        for mode in get_train_modes():
            if not FORCE_RETRAIN and valid_adapter(output_root, mode):
                print(f"Checkpoint exists, skip training: {mode}")
            else:
                run_script("train.py", {**env_base, "MODE": mode})
                retrained_modes.add(mode)

        prediction_files: Dict[str, str] = {}
        for mode in get_inference_modes():
            path = os.path.join(run_root, f"predictions_{mode}.json")
            train_mode, _ = parse_inference_mode(mode)
            adapter_was_retrained = train_mode in retrained_modes
            if (
                not FORCE_INFERENCE
                and not adapter_was_retrained
                and valid_predictions(path, expected_eval_len, mode)
            ):
                print(f"Predictions exist, skip inference: {mode}")
            else:
                run_script("inference.py", {**env_base, "MODE": mode})
                shutil.copyfile(os.path.join(PROJECT_DIR, "predictions.json"), path)
            prediction_files[mode] = path

        summary = summarize_model(run_root, prediction_files)
        aggregate.append({"model_name": model_name, "model_slug": slug, **summary})

        print("\nInternalization")
        for row in summary["internalization"]:
            ie = row["internalization_efficiency"]
            ie_text = "nan" if ie is None else f"{ie:.4f}"
            print(f"{row['base_mode']} {row['context_variant']}: IE={ie_text}")

    aggregate_path = os.path.join(model_runs_root, "all_models_summary.json")
    os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"\nSaved aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
