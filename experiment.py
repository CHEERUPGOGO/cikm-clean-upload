import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from config import ENABLE_LOGIT_KD, EVAL_CONTEXT_VARIANTS
from retrieval import CONTEXT_VARIANTS, normalize_context_variant


def _csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


BASE_TRAIN_MODES = _csv_env("BASE_TRAIN_MODES", "oracle_sft,oracle_ctx,hard_kd")
OPTIONAL_TRAIN_MODES = ["logit_kd"] if ENABLE_LOGIT_KD else []
TRAIN_MODES = BASE_TRAIN_MODES + OPTIONAL_TRAIN_MODES
BASELINE_MODES = ["baseline_noctx", "baseline_ctx"]
CONTEXT_TRAIN_MODES = {"oracle_ctx", "context_hard_kd", "context_logit_kd", "noisy_oracle_ctx", "noisy_context_hard_kd", "noisy_context_logit_kd"}
GOLD_TARGET_TRAIN_MODES = {"oracle_sft", "oracle_ctx", "noisy_oracle_ctx"}
TEACHER_TARGET_TRAIN_MODES = {
    "hard_kd",
    "logit_kd",
    "context_hard_kd",
    "noisy_context_hard_kd",
    "noisy_context_logit_kd",
    "context_logit_kd",
}


def get_train_modes() -> List[str]:
    return list(TRAIN_MODES)


def get_eval_context_variants() -> List[str]:
    variants = []
    seen = set()
    for raw in EVAL_CONTEXT_VARIANTS:
        variant = normalize_context_variant(raw)
        if variant not in seen:
            variants.append(variant)
            seen.add(variant)
    return variants or ["gold"]


def _context_mode(base_mode: str, variant: str) -> str:
    variant = normalize_context_variant(variant)
    if variant == "gold":
        return f"{base_mode}_ctx"
    return f"{base_mode}_{variant}_ctx"


def get_inference_modes() -> List[str]:
    modes = ["baseline_noctx"]
    for variant in get_eval_context_variants():
        modes.append(_context_mode("baseline", variant))
    for mode in get_train_modes():
        modes.append(f"{mode}_noctx")
        for variant in get_eval_context_variants():
            modes.append(_context_mode(mode, variant))
    return modes


def slugify_model_name(model_name: str) -> str:
    slug = model_name.lower().strip()
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug)
    slug = slug.strip("._-")
    return slug or "model"


def load_experiment_data(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        raise ValueError(
            "This project now expects controlled fact data with top-level "
            "'train' and 'eval' keys. Run build_dataset.py to rebuild data.json."
        )
    for key in ["facts", "train", "eval"]:
        if key not in payload:
            raise ValueError(f"Controlled dataset is missing key: {key}")
    return payload


def get_train_examples(payload: Dict[str, object]) -> List[Dict[str, object]]:
    return list(payload.get("train", []))


def get_eval_examples(payload: Dict[str, object]) -> List[Dict[str, object]]:
    return list(payload.get("eval", []))


def train_examples_fingerprint(examples: List[Dict[str, object]]) -> str:
    payload = [
        {
            "example_id": example.get("example_id"),
            "fact_id": example.get("fact_id"),
            "question": example.get("question"),
            "context": example.get("context"),
            "answer": example.get("answer"),
        }
        for example in sorted(examples, key=lambda item: str(item.get("example_id", "")))
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_base_mode(base_mode: str) -> Optional[str]:
    if base_mode == "baseline":
        return None
    if base_mode in get_train_modes():
        return base_mode
    raise ValueError(
        f"Unsupported inference base mode '{base_mode}'. "
        f"Expected baseline or one of: {', '.join(get_train_modes())}"
    )


def parse_inference_mode(mode: str) -> Tuple[Optional[str], Optional[str]]:
    if mode.endswith("_noctx"):
        return _parse_base_mode(mode[: -len("_noctx")]), None

    if mode.endswith("_ctx"):
        body = mode[: -len("_ctx")]
        for variant in sorted(CONTEXT_VARIANTS, key=len, reverse=True):
            suffix = f"_{variant}"
            if body.endswith(suffix):
                return _parse_base_mode(body[: -len(suffix)]), variant
        return _parse_base_mode(body), "gold"

    raise ValueError(
        f"Unsupported MODE={mode}. Expected one of: {', '.join(get_inference_modes())}"
    )


def training_uses_context(train_mode: str) -> bool:
    return train_mode in CONTEXT_TRAIN_MODES


def build_prompt(
    example: Dict[str, object],
    use_context: bool,
    context_text: Optional[str] = None,
) -> str:
    question = str(example["question"]).strip()
    instruction = (
        "Instruction: Return only the shortest canonical answer string, "
        "with no explanation."
    )
    if use_context:
        context = str(example["context"] if context_text is None else context_text).strip()
        return f"{instruction}\nQuestion: {question}\nContext: {context}\nAnswer:"
    return f"{instruction}\nQuestion: {question}\nAnswer:"


def build_training_prompt(example: Dict[str, object], train_mode: str) -> str:
    use_ctx = training_uses_context(train_mode)
    ctx_text = example.get("noisy_context") if train_mode.startswith("noisy_") else None
    return build_prompt(example, use_context=use_ctx, context_text=ctx_text)


def target_for_train_mode(example: Dict[str, object], train_mode: str) -> str:
    if train_mode in GOLD_TARGET_TRAIN_MODES:
        return str(example["answer"])
    if train_mode in TEACHER_TARGET_TRAIN_MODES:
        teacher_answer = str(example.get("teacher_answer") or "").strip()
        if not teacher_answer:
            raise ValueError(
                f"{train_mode} requires teacher_answer from the local teacher cache. "
                "Run precompute_teacher_cache.py first."
            )
        return teacher_answer
    raise ValueError(f"Unsupported train mode: {train_mode}")


def eval_condition(
    example: Dict[str, object],
    use_context: bool,
    context_variant: Optional[str] = None,
) -> str:
    fact_split = str(example["fact_split"])
    if not use_context:
        return f"{fact_split}_noctx"
    variant = normalize_context_variant(context_variant or "gold")
    return f"{fact_split}_{variant}_ctx"


def iter_jsonl(path: str) -> Iterable[Dict[str, object]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
