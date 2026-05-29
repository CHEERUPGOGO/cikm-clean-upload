import json
import os

from dotenv import load_dotenv
import torch


load_dotenv()


# Base paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_TAG = os.getenv("RUN_TAG", "").strip()
DATA_PATH = os.path.join(PROJECT_DIR, "data.json")
PREDICTIONS_PATH = os.path.join(PROJECT_DIR, "predictions.json")
MODEL_PATHS_PATH = os.getenv("MODEL_PATHS_PATH", os.path.join(PROJECT_DIR, "model_paths.json"))
OUTPUT_ROOT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_ARTIFACT_SUBDIR = os.getenv("MODEL_ARTIFACT_SUBDIR", "").strip()
if MODEL_ARTIFACT_SUBDIR:
    _safe_subdir = MODEL_ARTIFACT_SUBDIR.replace("\\", "_").replace("/", "_")
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, _safe_subdir)
else:
    OUTPUT_DIR = OUTPUT_ROOT_DIR

# Model / training
MODEL_PRESETS = {
    "distilgpt2": "distilgpt2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "facebook_opt_350m": "facebook/opt-350m",
    "qwen_qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
}


def load_model_path_map() -> dict[str, str]:
    if not os.path.exists(MODEL_PATHS_PATH):
        return {}
    try:
        with open(MODEL_PATHS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    paths = payload.get("paths", payload)
    if not isinstance(paths, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in paths.items()
        if value and os.path.exists(str(value))
    }


MODEL_PATHS = load_model_path_map()


def resolve_model_name(raw_name: str) -> str:
    name = (raw_name or "").strip()
    if not name:
        name = MODEL_PRESETS["distilgpt2"]
    resolved = MODEL_PRESETS.get(name.lower(), name)
    return MODEL_PATHS.get(name, MODEL_PATHS.get(resolved, resolved))


MODEL_NAME = resolve_model_name(os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"))
MODE = os.getenv("MODE", "baseline_noctx")
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "3"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "false").lower() == "true"
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
SEED = int(os.getenv("SEED", "42"))

# Controlled fact dataset
NUM_FACTS = int(os.getenv("NUM_FACTS", os.getenv("NUM_SAMPLES", "6000")))
FACT_TEST_RATIO = float(os.getenv("FACT_TEST_RATIO", "0.2"))
TRAIN_PARAPHRASES_PER_FACT = int(os.getenv("TRAIN_PARAPHRASES_PER_FACT", "2"))
EVAL_PARAPHRASES_PER_FACT = int(os.getenv("EVAL_PARAPHRASES_PER_FACT", "3"))
FACT_SUCCESS_THRESHOLD = float(os.getenv("FACT_SUCCESS_THRESHOLD", "0.5"))
FACT_SCORE_METRIC = os.getenv("FACT_SCORE_METRIC", "answer_acc").strip().lower()


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Retrieval-conditioned evaluation. Keep the default close to the original
# oracle-context setting; set EVAL_CONTEXT_VARIANTS to include retrieval modes.
EVAL_CONTEXT_VARIANTS = _csv_env("EVAL_CONTEXT_VARIANTS", "gold")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "1"))
RETRIEVAL_QUERY_FIELD = os.getenv("RETRIEVAL_QUERY_FIELD", "question").strip().lower()
RETRIEVAL_SAVE_CONTEXT = os.getenv("RETRIEVAL_SAVE_CONTEXT", "false").lower() == "true"
DENSE_RETRIEVER_MODEL = os.getenv(
    "DENSE_RETRIEVER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DENSE_RETRIEVER_MODEL = MODEL_PATHS.get(DENSE_RETRIEVER_MODEL, DENSE_RETRIEVER_MODEL)
DENSE_RETRIEVER_BATCH_SIZE = int(os.getenv("DENSE_RETRIEVER_BATCH_SIZE", "64"))
RETRIEVER_DEVICE = os.getenv("RETRIEVER_DEVICE", "cpu").strip().lower()


def _resolve_torch_device() -> str:
    # DEVICE supports: auto / cuda / mps / cpu
    requested = os.getenv("DEVICE", "auto").strip().lower()

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "DEVICE=cuda was requested, but torch.cuda.is_available() is False. "
            "Install CUDA-enabled PyTorch and verify GPU drivers/CUDA runtime."
        )

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        raise RuntimeError(
            "DEVICE=mps was requested, but Apple MPS backend is unavailable."
        )

    if requested == "cpu":
        return "cpu"

    if requested != "auto":
        raise ValueError("DEVICE must be one of: auto, cuda, mps, cpu")

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


TORCH_DEVICE = _resolve_torch_device()
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "false").lower() == "true"
if REQUIRE_GPU and TORCH_DEVICE == "cpu":
    raise RuntimeError(
        "No GPU backend available. REQUIRE_GPU=true forbids CPU fallback. "
        "Set DEVICE=cuda for NVIDIA GPU or install CUDA-enabled PyTorch."
    )

USE_GPU = TORCH_DEVICE != "cpu"
USE_FP16 = TORCH_DEVICE == "cuda"


def _resolve_dtype(raw: str):
    value = (raw or "auto").strip().lower()
    if value in {"", "auto"}:
        if TORCH_DEVICE == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype '{raw}'. Use auto, bf16, fp16, or fp32.")


STUDENT_TORCH_DTYPE = _resolve_dtype(os.getenv("STUDENT_TORCH_DTYPE", "auto"))
TEACHER_TORCH_DTYPE = _resolve_dtype(os.getenv("TEACHER_TORCH_DTYPE", "auto"))
TEACHER_DEVICE_MAP = os.getenv("TEACHER_DEVICE_MAP", "auto").strip() or "auto"
TEACHER_LOAD_IN_4BIT = os.getenv("TEACHER_LOAD_IN_4BIT", "false").lower() == "true"

# LoRA
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
USE_LORA = os.getenv("USE_LORA", "true").lower() == "true"

# Generation
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "32"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
INFERENCE_BATCH_SIZE = max(1, int(os.getenv("INFERENCE_BATCH_SIZE", "16")))

# Local teacher distillation
TEACHER_MODEL_NAME = resolve_model_name(os.getenv("TEACHER_MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct"))
TEACHER_MAX_NEW_TOKENS = int(os.getenv("TEACHER_MAX_NEW_TOKENS", "32"))
TEACHER_CACHE_VERSION = "teacher_cache_v6_generated_keep_answeronly"
TEACHER_ANSWER_MODE = os.getenv("TEACHER_ANSWER_MODE", "generate").strip().lower()
if TEACHER_ANSWER_MODE not in {"canonical", "generate"}:
    raise ValueError("TEACHER_ANSWER_MODE must be one of: canonical, generate")
TEACHER_INVALID_ACTION = os.getenv("TEACHER_INVALID_ACTION", "").strip().lower()
if not TEACHER_INVALID_ACTION:
    legacy_fallback = os.getenv("TEACHER_GOLD_FALLBACK", "").strip().lower()
    TEACHER_INVALID_ACTION = "gold_fallback" if legacy_fallback == "true" else "keep"
if TEACHER_INVALID_ACTION not in {"fail", "gold_fallback", "keep"}:
    raise ValueError("TEACHER_INVALID_ACTION must be one of: fail, gold_fallback, keep")
TEACHER_CACHE_DIR = os.path.join(PROJECT_DIR, "teacher_cache")
TEACHER_CACHE_PATH = os.getenv(
    "TEACHER_CACHE_PATH",
    os.path.join(
        TEACHER_CACHE_DIR,
        f"{TEACHER_MODEL_NAME.lower().replace('/', '_').replace('-', '_')}.jsonl",
    ),
)

# White-box logit KD. Hard KD reads the same local teacher cache.
ENABLE_LOGIT_KD = os.getenv("ENABLE_LOGIT_KD", "false").lower() == "true"
KD_TEMPERATURE = float(os.getenv("KD_TEMPERATURE", "2.0"))
KD_ALPHA = float(os.getenv("KD_ALPHA", "0.5"))
KD_TOP_K = int(os.getenv("KD_TOP_K", "128"))
KD_LOGITS_PATH = os.getenv("KD_LOGITS_PATH", TEACHER_CACHE_PATH)

def get_train_mode(mode: str) -> str:
    return mode


def get_checkpoint_dir(mode: str) -> str:
    return os.path.join(OUTPUT_DIR, get_train_mode(mode))
