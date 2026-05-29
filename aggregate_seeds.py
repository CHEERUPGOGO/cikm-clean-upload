import csv
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

NUMERIC_COLUMNS = [
    "seen_noctx_fact_acc",
    "unseen_noctx_fact_acc",
    "seen_ctx_fact_acc",
    "unseen_ctx_fact_acc",
    "fact_internalization_gain",
    "internalization_efficiency",
    "context_dependence_gap",
    "seen_ctx_retrieval_hit_rate",
    "seen_ctx_retrieval_mrr",
    "seen_ctx_retrieval_gold_fraction",
]

T_CRIT_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_float(value: str):
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return float(value)


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ci95_half_width(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    std = statistics.stdev(values)
    tcrit = T_CRIT_95.get(len(values), 1.96)
    return tcrit * std / (len(values) ** 0.5)


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    model_runs = os.path.join(root, "model_runs")
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    run_prefix = os.getenv("AGGREGATE_RUN_PREFIX", os.getenv("RUN_TAG_PREFIX", "seed")).strip()
    if not run_prefix:
        run_prefix = "seed"

    run_tag_pattern = re.compile(rf"{re.escape(run_prefix)}_\d+$")
    for seed_name in sorted(os.listdir(model_runs)) if os.path.exists(model_runs) else []:
        if not run_tag_pattern.match(seed_name):
            continue
        seed_root = os.path.join(model_runs, seed_name)
        if not os.path.isdir(seed_root):
            continue
        for model_slug in sorted(os.listdir(seed_root)):
            summary_path = os.path.join(seed_root, model_slug, "internalization_summary.csv")
            if not os.path.exists(summary_path):
                continue
            for row in read_rows(summary_path):
                key = (model_slug, row["base_mode"], row["context_variant"])
                item = dict(row)
                item["seed"] = seed_name
                grouped[key].append(item)

    output_rows: List[Dict[str, object]] = []
    for (model_slug, base_mode, context_variant), rows in sorted(grouped.items()):
        out: Dict[str, object] = {
            "model_slug": model_slug,
            "base_mode": base_mode,
            "context_variant": context_variant,
            "num_seeds": len(rows),
        }
        for col in NUMERIC_COLUMNS:
            values = [parse_float(row.get(col, "")) for row in rows]
            values = [value for value in values if value is not None]
            if not values:
                out[f"{col}_mean"] = ""
                out[f"{col}_std"] = ""
                out[f"{col}_ci95_low"] = ""
                out[f"{col}_ci95_high"] = ""
                continue
            mean = statistics.fmean(values)
            half_width = ci95_half_width(values)
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            out[f"{col}_ci95_low"] = mean - half_width
            out[f"{col}_ci95_high"] = mean + half_width
        output_rows.append(out)

    out_path = os.path.join(model_runs, f"{run_prefix}_aggregate", "internalization_mean_std.csv")
    if not output_rows:
        print(
            "No seed summaries found. Expected "
            f"model_runs/{run_prefix}_<seed>/<model>/internalization_summary.csv."
        )
        return
    write_csv(out_path, output_rows)
    print(f"Wrote {out_path}")
    print(f"Aggregated {len(output_rows)} model/mode/context rows for prefix '{run_prefix}'.")


if __name__ == "__main__":
    main()
