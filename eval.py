import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from config import FACT_SCORE_METRIC, FACT_SUCCESS_THRESHOLD, PREDICTIONS_PATH
from retrieval import (
    CONTEXT_VARIANTS,
    context_source_for_variant,
    normalize_context_variant,
    retriever_name_for_variant,
)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: str, answer: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(answer) else 0.0


def contains_match(prediction: str, answer: str) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(answer)
    if not pred or not gold:
        return 1.0 if pred == gold else 0.0
    return 1.0 if gold in pred else 0.0


def answer_match(prediction: str, answer: str) -> float:
    return max(exact_match(prediction, answer), contains_match(prediction, answer))


def f1_score(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_mode_name(mode: str) -> Tuple[str, Optional[str]]:
    mode = str(mode)
    if mode.endswith("_noctx"):
        return mode[: -len("_noctx")], None
    if mode.endswith("_ctx"):
        body = mode[: -len("_ctx")]
        for variant in sorted(CONTEXT_VARIANTS, key=len, reverse=True):
            suffix = f"_{variant}"
            if body.endswith(suffix):
                return body[: -len(suffix)], variant
        return body, "gold"
    return mode, None


def _prediction_context_variant(pred: Dict[str, object]) -> str:
    raw = str(pred.get("context_variant", "")).strip()
    if raw and raw != "noctx":
        return normalize_context_variant(raw)
    _, variant = _split_mode_name(str(pred.get("mode", "")))
    return variant or "noctx"


def _prediction_context_source(pred: Dict[str, object]) -> str:
    variant = _prediction_context_variant(pred)
    if variant == "noctx":
        return "none"
    return str(pred.get("context_source") or context_source_for_variant(variant))


def _prediction_retriever(pred: Dict[str, object]) -> str:
    variant = _prediction_context_variant(pred)
    if variant == "noctx":
        return "none"
    return str(pred.get("retriever") or retriever_name_for_variant(variant))


def _prediction_top_k(pred: Dict[str, object]) -> int:
    try:
        return int(pred.get("retrieval_top_k", 0))
    except (TypeError, ValueError):
        return 0


def _condition_candidates(fact_split: str, variant: Optional[str]) -> List[str]:
    if variant is None:
        return [f"{fact_split}_noctx"]
    variant = normalize_context_variant(variant)
    if variant == "gold":
        return [f"{fact_split}_gold_ctx", f"{fact_split}_ctx"]
    return [f"{fact_split}_{variant}_ctx"]


def _mode_candidates(base_mode: str, variant: Optional[str]) -> List[str]:
    if variant is None:
        return [f"{base_mode}_noctx"]
    variant = normalize_context_variant(variant)
    if variant == "gold":
        return [f"{base_mode}_ctx", f"{base_mode}_gold_ctx"]
    return [f"{base_mode}_{variant}_ctx"]


def _fact_map(
    fact_values: Dict[Tuple[str, str], Dict[str, float]],
    mode_candidates: List[str],
    condition_candidates: List[str],
) -> Dict[str, float]:
    for mode in mode_candidates:
        for condition in condition_candidates:
            values = fact_values.get((mode, condition))
            if values:
                return values
    return {}


def metric_tuple(pred: Dict[str, object]) -> Dict[str, float]:
    prediction = str(pred.get("prediction", ""))
    gold = str(pred.get("gold_answer", ""))
    return {
        "em": exact_match(prediction, gold),
        "f1": f1_score(prediction, gold),
        "contains": contains_match(prediction, gold),
        "answer_acc": answer_match(prediction, gold),
    }


def summarize_predictions(predictions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for pred in predictions:
        key = (str(pred["mode"]), str(pred["eval_condition"]))
        grouped[key].append(pred)

    rows = []
    for (mode, condition), group in sorted(grouped.items()):
        n = len(group)
        values = [metric_tuple(pred) for pred in group]
        first = group[0]
        rows.append(
            {
                "mode": mode,
                "eval_condition": condition,
                "context_variant": _prediction_context_variant(first),
                "context_source": _prediction_context_source(first),
                "retriever": _prediction_retriever(first),
                "retrieval_top_k": _prediction_top_k(first),
                "n": n,
                "em": sum(v["em"] for v in values) / n,
                "f1": sum(v["f1"] for v in values) / n,
                "contains": sum(v["contains"] for v in values) / n,
                "answer_acc": sum(v["answer_acc"] for v in values) / n,
            }
        )
    return rows


def fact_scores_by_mode_condition(
    predictions: List[Dict[str, object]],
    threshold: float = FACT_SUCCESS_THRESHOLD,
    metric_name: str = FACT_SCORE_METRIC,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], Dict[str, float]]]:
    valid_metrics = {"em", "f1", "contains", "answer_acc"}
    if metric_name not in valid_metrics:
        raise ValueError(f"metric_name must be one of {sorted(valid_metrics)}, got {metric_name}")

    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for pred in predictions:
        metrics = metric_tuple(pred)
        grouped[
            (
                str(pred["mode"]),
                str(pred["eval_condition"]),
                str(pred["fact_id"]),
            )
        ].append(metrics[metric_name])

    fact_values: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (mode, condition, fact_id), values in grouped.items():
        fact_values[(mode, condition)][fact_id] = sum(values) / len(values)

    rows = []
    for (mode, condition), values_by_fact in sorted(fact_values.items()):
        n_facts = len(values_by_fact)
        binary = [1.0 if v >= threshold else 0.0 for v in values_by_fact.values()]
        rows.append(
            {
                "mode": mode,
                "eval_condition": condition,
                "n_facts": n_facts,
                "fact_acc": sum(binary) / n_facts if n_facts else 0.0,
                "mean_paraphrase_acc": sum(values_by_fact.values()) / n_facts if n_facts else 0.0,
                "success_threshold": threshold,
                "metric": metric_name,
            }
        )
    return rows, dict(fact_values)


def paired_permutation_test(
    xs: List[float],
    ys: List[float],
    trials: int = 10000,
    seed: int = 42,
) -> float:
    if len(xs) != len(ys) or not xs:
        return 1.0
    diffs = [x - y for x, y in zip(xs, ys)]
    observed = abs(sum(diffs) / len(diffs))
    if observed == 0:
        return 1.0

    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        signed_mean = 0.0
        for d in diffs:
            signed_mean += d if rng.random() < 0.5 else -d
        signed_mean = abs(signed_mean / len(diffs))
        if signed_mean >= observed - 1e-12:
            extreme += 1
    return (extreme + 1.0) / (trials + 1.0)


def paired_p_from_fact_maps(
    xs_by_fact: Dict[str, float],
    ys_by_fact: Dict[str, float],
    trials: int = 10000,
) -> float:
    common = sorted(set(xs_by_fact) & set(ys_by_fact))
    if not common:
        return 1.0
    xs = [xs_by_fact[fact_id] for fact_id in common]
    ys = [ys_by_fact[fact_id] for fact_id in common]
    return paired_permutation_test(xs, ys, trials=trials)


def _fact_acc(values: Dict[str, float], threshold: float) -> Optional[float]:
    if not values:
        return None
    binary = [1.0 if v >= threshold else 0.0 for v in values.values()]
    return sum(binary) / len(binary)


def _mean_prediction_field(
    predictions: List[Dict[str, object]],
    mode_candidates: List[str],
    condition_candidates: List[str],
    field: str,
) -> Optional[float]:
    mode_set = set(mode_candidates)
    condition_set = set(condition_candidates)
    values = []
    for pred in predictions:
        if str(pred.get("mode")) not in mode_set:
            continue
        if str(pred.get("eval_condition")) not in condition_set:
            continue
        value = pred.get(field)
        if value is None:
            continue
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return sum(values) / len(values) if values else None


def compute_internalization_report(
    predictions: List[Dict[str, object]],
    threshold: float = FACT_SUCCESS_THRESHOLD,
    metric_name: str = FACT_SCORE_METRIC,
    trials: int = 10000,
) -> List[Dict[str, object]]:
    _, fact_values = fact_scores_by_mode_condition(
        predictions,
        threshold=threshold,
        metric_name=metric_name,
    )
    modes = sorted({str(pred["mode"]) for pred in predictions})
    split_modes = {mode: _split_mode_name(mode) for mode in modes}
    noctx_bases = {base for base, variant in split_modes.values() if variant is None}
    context_by_base: Dict[str, List[str]] = defaultdict(list)
    for _, (base, variant) in split_modes.items():
        if variant is not None:
            context_by_base[base].append(variant)
    base_modes = sorted(base for base in noctx_bases if base != "baseline")

    rows = []
    for base in base_modes:
        noctx_modes = _mode_candidates(base, None)
        seen_noctx_values = _fact_map(
            fact_values,
            noctx_modes,
            _condition_candidates("seen_fact", None),
        )
        unseen_noctx_values = _fact_map(
            fact_values,
            noctx_modes,
            _condition_candidates("unseen_fact", None),
        )
        seen_noctx = _fact_acc(seen_noctx_values, threshold)
        unseen_noctx = _fact_acc(unseen_noctx_values, threshold)
        if seen_noctx is None or unseen_noctx is None:
            continue

        for variant in sorted(set(context_by_base.get(base, []))):
            ctx_modes = _mode_candidates(base, variant)
            seen_ctx_conditions = _condition_candidates("seen_fact", variant)
            unseen_ctx_conditions = _condition_candidates("unseen_fact", variant)
            seen_ctx_values = _fact_map(fact_values, ctx_modes, seen_ctx_conditions)
            unseen_ctx_values = _fact_map(fact_values, ctx_modes, unseen_ctx_conditions)
            seen_ctx = _fact_acc(seen_ctx_values, threshold)
            unseen_ctx = _fact_acc(unseen_ctx_values, threshold)
            if seen_ctx is None:
                continue

            denom = seen_ctx - unseen_noctx
            ie = (seen_noctx - unseen_noctx) / denom if denom > 1e-12 else None
            p_ctx_vs_noctx = paired_p_from_fact_maps(
                seen_ctx_values,
                seen_noctx_values,
                trials=trials,
            )
            seen_hit_rate = _mean_prediction_field(
                predictions,
                ctx_modes,
                seen_ctx_conditions,
                "retrieval_hit",
            )
            seen_mrr = _mean_prediction_field(
                predictions,
                ctx_modes,
                seen_ctx_conditions,
                "retrieval_mrr",
            )
            seen_gold_fraction = _mean_prediction_field(
                predictions,
                ctx_modes,
                seen_ctx_conditions,
                "retrieval_gold_fraction",
            )
            unseen_hit_rate = _mean_prediction_field(
                predictions,
                ctx_modes,
                unseen_ctx_conditions,
                "retrieval_hit",
            )

            rows.append(
                {
                    "base_mode": base,
                    "context_variant": variant,
                    "context_source": context_source_for_variant(variant),
                    "retriever": retriever_name_for_variant(variant),
                    "retrieval_top_k": _mean_prediction_field(
                        predictions,
                        ctx_modes,
                        seen_ctx_conditions + unseen_ctx_conditions,
                        "retrieval_top_k",
                    ),
                    "seen_noctx_fact_acc": seen_noctx,
                    "unseen_noctx_fact_acc": unseen_noctx,
                    "seen_ctx_fact_acc": seen_ctx,
                    "unseen_ctx_fact_acc": unseen_ctx,
                    "fact_internalization_gain": seen_noctx - unseen_noctx,
                    "internalization_efficiency": ie,
                    "context_dependence_gap": seen_ctx - seen_noctx,
                    "seen_ctx_retrieval_hit_rate": seen_hit_rate,
                    "seen_ctx_retrieval_mrr": seen_mrr,
                    "seen_ctx_retrieval_gold_fraction": seen_gold_fraction,
                    "unseen_ctx_retrieval_hit_rate": unseen_hit_rate,
                    "ctx_vs_noctx_p_value": p_ctx_vs_noctx,
                    "success_threshold": threshold,
                    "metric": metric_name,
                }
            )
    return rows


def summarize_retrieval_quality(predictions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for pred in predictions:
        if not bool(pred.get("prompt_context", False)):
            continue
        variant = _prediction_context_variant(pred)
        if variant == "noctx":
            continue
        key = (
            str(pred["mode"]),
            str(pred["eval_condition"]),
            variant,
            _prediction_context_source(pred),
            _prediction_retriever(pred),
            _prediction_top_k(pred),
            str(pred.get("fact_split", "")),
        )
        grouped[key].append(pred)

    rows = []
    for (
        mode,
        condition,
        variant,
        source,
        retriever,
        top_k,
        fact_split,
    ), group in sorted(grouped.items()):
        n = len(group)
        metrics = [metric_tuple(pred) for pred in group]
        rows.append(
            {
                "mode": mode,
                "eval_condition": condition,
                "context_variant": variant,
                "context_source": source,
                "retriever": retriever,
                "retrieval_top_k": top_k,
                "fact_split": fact_split,
                "n": n,
                "retrieval_hit_rate": sum(
                    1.0 if pred.get("retrieval_hit") else 0.0 for pred in group
                )
                / n,
                "mean_retrieval_mrr": sum(float(pred.get("retrieval_mrr") or 0.0) for pred in group)
                / n,
                "mean_gold_count": sum(
                    float(pred.get("retrieval_gold_count") or 0.0) for pred in group
                )
                / n,
                "mean_gold_fraction": sum(
                    float(pred.get("retrieval_gold_fraction") or 0.0) for pred in group
                )
                / n,
                "em": sum(v["em"] for v in metrics) / n,
                "f1": sum(v["f1"] for v in metrics) / n,
                "contains": sum(v["contains"] for v in metrics) / n,
                "answer_acc": sum(v["answer_acc"] for v in metrics) / n,
            }
        )
    return rows


def summarize_retrieval_quality_effect(
    predictions: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for pred in predictions:
        if not bool(pred.get("prompt_context", False)):
            continue
        variant = _prediction_context_variant(pred)
        if variant == "noctx":
            continue
        hit = "hit" if pred.get("retrieval_hit") else "miss"
        key = (
            str(pred["mode"]),
            variant,
            _prediction_context_source(pred),
            _prediction_retriever(pred),
            _prediction_top_k(pred),
            str(pred.get("fact_split", "")),
            hit,
        )
        grouped[key].append(pred)

    rows = []
    for (mode, variant, source, retriever, top_k, fact_split, hit), group in sorted(grouped.items()):
        n = len(group)
        metrics = [metric_tuple(pred) for pred in group]
        rows.append(
            {
                "mode": mode,
                "context_variant": variant,
                "context_source": source,
                "retriever": retriever,
                "retrieval_top_k": top_k,
                "fact_split": fact_split,
                "retrieval_quality_bucket": hit,
                "n": n,
                "em": sum(v["em"] for v in metrics) / n,
                "f1": sum(v["f1"] for v in metrics) / n,
                "contains": sum(v["contains"] for v in metrics) / n,
                "answer_acc": sum(v["answer_acc"] for v in metrics) / n,
            }
        )
    return rows


def merge_prediction_groups(groups: Iterable[List[Dict[str, object]]]) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    for group in groups:
        merged.extend(group)
    return merged


def main() -> None:
    predictions = load_json(PREDICTIONS_PATH)
    if not isinstance(predictions, list) or (
        predictions and "eval_condition" not in predictions[0]
    ):
        raise RuntimeError(
            "predictions.json is not in the new fact-level format. "
            "Run inference.py or run_all.py after rebuilding data.json."
        )
    summary = summarize_predictions(predictions)
    fact_summary, _ = fact_scores_by_mode_condition(predictions)
    internalization = compute_internalization_report(predictions)
    retrieval_quality = summarize_retrieval_quality(predictions)

    print("Example-level metrics")
    for row in summary:
        print(
            f"{row['mode']} {row['eval_condition']}: "
            f"EM={row['em']:.4f} F1={row['f1']:.4f} "
            f"Contains={row['contains']:.4f} AnswerAcc={row['answer_acc']:.4f}"
        )

    print("\nFact-level metrics")
    for row in fact_summary:
        print(
            f"{row['mode']} {row['eval_condition']}: "
            f"FactAcc={row['fact_acc']:.4f} MeanParaAcc={row['mean_paraphrase_acc']:.4f}"
        )

    if internalization:
        print("\nInternalization")
        for row in internalization:
            ie = row["internalization_efficiency"]
            ie_text = "nan" if ie is None else f"{ie:.4f}"
            hit = row["seen_ctx_retrieval_hit_rate"]
            hit_text = "nan" if hit is None else f"{hit:.4f}"
            print(
                f"{row['base_mode']} {row['context_variant']}: IE={ie_text} "
                f"Gain={row['fact_internalization_gain']:.4f} "
                f"CtxGap={row['context_dependence_gap']:.4f} "
                f"Hit={hit_text} "
                f"p={row['ctx_vs_noctx_p_value']:.4g}"
            )

    if retrieval_quality:
        print("\nRetrieval quality")
        for row in retrieval_quality:
            print(
                f"{row['mode']} {row['fact_split']}: "
                f"Hit@{row['retrieval_top_k']}={row['retrieval_hit_rate']:.4f} "
                f"MRR={row['mean_retrieval_mrr']:.4f} "
                f"AnswerAcc={row['answer_acc']:.4f}"
            )


if __name__ == "__main__":
    main()
