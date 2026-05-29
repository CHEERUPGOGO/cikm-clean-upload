import csv
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List

from config import (
    DATA_PATH,
    KD_TOP_K,
    OUTPUT_DIR,
    PREDICTIONS_PATH,
    PROJECT_DIR,
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
    train_examples_fingerprint,
)


SUMMARY_DIR = os.path.join(PROJECT_DIR, "model_runs", "current")
SUMMARY_JSON_PATH = os.path.join(SUMMARY_DIR, "example_summary.json")
FACT_SUMMARY_JSON_PATH = os.path.join(SUMMARY_DIR, "fact_summary.json")
INTERNALIZATION_JSON_PATH = os.path.join(SUMMARY_DIR, "internalization_summary.json")
RETRIEVAL_QUALITY_JSON_PATH = os.path.join(SUMMARY_DIR, "retrieval_quality_summary.json")
RETRIEVAL_EFFECT_JSON_PATH = os.path.join(SUMMARY_DIR, "retrieval_quality_effect.json")
ALL_PREDICTIONS_PATH = os.path.join(SUMMARY_DIR, "all_predictions.json")
TEACHER_CACHE_MODES = {"hard_kd", "logit_kd", "context_hard_kd", "context_logit_kd"}


def run_script(script_name: str, mode: str = "") -> None:
    env = os.environ.copy()
    if mode:
        env["MODE"] = mode
    cmd = [sys.executable, os.path.join(PROJECT_DIR, script_name)]
    label = f" (MODE={mode})" if mode else ""
    print(f"\n===== Running {script_name}{label} =====")
    subprocess.run(cmd, env=env, check=True)


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


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
    run_script("precompute_teacher_cache.py")


def main() -> None:
    ensure_dirs()
    run_script("build_dataset.py")
    maybe_precompute_teacher_cache()

    for mode in get_train_modes():
        run_script("train.py", mode=mode)

    all_predictions_by_mode: Dict[str, List[Dict[str, object]]] = {}
    for mode in get_inference_modes():
        run_script("inference.py", mode=mode)
        predictions = load_json(PREDICTIONS_PATH)
        all_predictions_by_mode[mode] = predictions
        backup_path = os.path.join(SUMMARY_DIR, f"predictions_{mode}.json")
        shutil.copyfile(PREDICTIONS_PATH, backup_path)

    merged = merge_prediction_groups(all_predictions_by_mode.values())
    example_rows = summarize_predictions(merged)
    fact_rows, _ = fact_scores_by_mode_condition(merged)
    internalization_rows = compute_internalization_report(merged)
    retrieval_quality_rows = summarize_retrieval_quality(merged)
    retrieval_effect_rows = summarize_retrieval_quality_effect(merged)

    with open(ALL_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_predictions_by_mode, f, ensure_ascii=False, indent=2)
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(example_rows, f, ensure_ascii=False, indent=2)
    with open(FACT_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(fact_rows, f, ensure_ascii=False, indent=2)
    with open(INTERNALIZATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(internalization_rows, f, ensure_ascii=False, indent=2)
    with open(RETRIEVAL_QUALITY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(retrieval_quality_rows, f, ensure_ascii=False, indent=2)
    with open(RETRIEVAL_EFFECT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(retrieval_effect_rows, f, ensure_ascii=False, indent=2)

    save_csv(os.path.join(SUMMARY_DIR, "example_summary.csv"), example_rows)
    save_csv(os.path.join(SUMMARY_DIR, "fact_summary.csv"), fact_rows)
    save_csv(os.path.join(SUMMARY_DIR, "internalization_summary.csv"), internalization_rows)
    save_csv(os.path.join(SUMMARY_DIR, "retrieval_quality_summary.csv"), retrieval_quality_rows)
    save_csv(os.path.join(SUMMARY_DIR, "retrieval_quality_effect.csv"), retrieval_effect_rows)

    print("\n===== Internalization Summary =====")
    for row in internalization_rows:
        ie = row["internalization_efficiency"]
        ie_text = "nan" if ie is None else f"{ie:.4f}"
        print(
            f"{row['base_mode']}: IE={ie_text} "
            f"ctx={row['context_variant']} "
            f"seen_noctx={row['seen_noctx_fact_acc']:.4f} "
            f"unseen_noctx={row['unseen_noctx_fact_acc']:.4f} "
            f"seen_ctx={row['seen_ctx_fact_acc']:.4f}"
        )

    print(f"\nSaved summaries under {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
