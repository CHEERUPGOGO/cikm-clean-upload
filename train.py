import inspect
import json
import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config import (
    BATCH_SIZE,
    DATA_PATH,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    KD_ALPHA,
    KD_TEMPERATURE,
    KD_TOP_K,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_LENGTH,
    MODEL_NAME,
    MODE,
    SEED,
    STUDENT_TORCH_DTYPE,
    TEACHER_CACHE_PATH,
    TORCH_DEVICE,
    TRAIN_EPOCHS,
    USE_FP16,
    USE_LORA,
    get_checkpoint_dir,
)
from experiment import (
    build_training_prompt,
    get_train_examples,
    get_train_modes,
    iter_jsonl,
    load_experiment_data,
    target_for_train_mode,
    training_uses_context,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


TEACHER_CACHE_MODES = {"hard_kd", "logit_kd", "context_hard_kd", "context_logit_kd", "noisy_context_hard_kd", "noisy_context_logit_kd"}
LOGIT_KD_MODES = {"logit_kd", "context_logit_kd", "noisy_context_logit_kd"}
KD_ALIGNMENT_VERSION = "causal_shift_v2"


def load_teacher_cache(path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Local teacher cache not found: {path}. Run precompute_teacher_cache.py first."
        )
    cache = {}
    for row in iter_jsonl(path):
        cache[str(row["example_id"])] = row
    return cache


def attach_teacher_answers(
    data: List[Dict[str, object]],
    teacher_cache: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    enriched = []
    for example in data:
        row = teacher_cache.get(str(example["example_id"]))
        if not row:
            raise KeyError(f"Missing teacher cache for example_id={example['example_id']}")
        item = dict(example)
        item["teacher_answer"] = str(row.get("teacher_answer", "")).strip()
        item["teacher_source"] = str(row.get("teacher_model", "local_teacher"))
        if not item["teacher_answer"]:
            raise ValueError(f"Empty teacher answer for example_id={example['example_id']}")
        enriched.append(item)
    return enriched


class QADataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, object]],
        tokenizer,
        mode: str,
        teacher_cache: Optional[Dict[str, Dict[str, object]]] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.teacher_cache = teacher_cache

    def __len__(self) -> int:
        return len(self.data)

    def _encode_example(self, example: Dict[str, object]) -> Dict[str, torch.Tensor]:
        source = build_training_prompt(example, self.mode)
        target = target_for_train_mode(example, self.mode)
        source_with_space = f"{source} "
        full_text = f"{source_with_space}{target}{self.tokenizer.eos_token}"

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        source_enc = self.tokenizer(
            source_with_space,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in full_enc.items()}
        labels = item["input_ids"].clone()
        source_len = min(source_enc["input_ids"].shape[1], MAX_LENGTH)
        labels[:source_len] = -100
        labels[item["attention_mask"] == 0] = -100
        item["labels"] = labels
        return item

    def _attach_kd_tensors(self, item: Dict[str, torch.Tensor], example: Dict[str, object]) -> None:
        top_ids = torch.zeros((MAX_LENGTH, KD_TOP_K), dtype=torch.long)
        top_logits = torch.full((MAX_LENGTH, KD_TOP_K), -1e9, dtype=torch.float)
        kd_mask = torch.zeros(MAX_LENGTH, dtype=torch.bool)

        if self.teacher_cache is None:
            item["kd_top_ids"] = top_ids
            item["kd_top_logits"] = top_logits
            item["kd_mask"] = kd_mask
            return

        row = self.teacher_cache.get(str(example["example_id"]))
        if not row:
            raise KeyError(f"Missing KD logits for example_id={example['example_id']}")

        label_positions = torch.nonzero(item["labels"] != -100, as_tuple=False).flatten().tolist()
        row_top_ids = row.get("top_ids", [])
        row_top_logits = row.get("top_logits", [])
        n_positions = min(len(label_positions), len(row_top_ids), len(row_top_logits))
        if n_positions != len(label_positions):
            raise ValueError(
                "KD teacher token positions do not match the student answer tokens. "
                "Use the same tokenizer family for teacher and student, or regenerate the cache."
            )

        for j in range(n_positions):
            pos = label_positions[j]
            logit_pos = pos - 1
            if logit_pos < 0:
                raise ValueError(
                    "Cannot attach KD logits to the first token position; causal LM logits "
                    "predict the next token and therefore require position-1 alignment."
                )
            ids = list(row_top_ids[j])[:KD_TOP_K]
            logits = list(row_top_logits[j])[:KD_TOP_K]
            if not ids:
                continue
            if max(int(token_id) for token_id in ids) >= len(self.tokenizer):
                raise ValueError(
                    "KD logits contain token ids outside the student vocabulary. "
                    "Use a teacher with the same tokenizer family as the student."
                )
            width = min(len(ids), len(logits), KD_TOP_K)
            # Causal LM logits at index t predict token t+1. Labels mark the target
            # token position, so KD distributions for that token must supervise pos-1.
            top_ids[logit_pos, :width] = torch.tensor(ids[:width], dtype=torch.long)
            top_logits[logit_pos, :width] = torch.tensor(logits[:width], dtype=torch.float)
            kd_mask[logit_pos] = True

        item["kd_top_ids"] = top_ids
        item["kd_top_logits"] = top_logits
        item["kd_mask"] = kd_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        item = self._encode_example(example)
        if self.mode in LOGIT_KD_MODES:
            self._attach_kd_tensors(item, example)
        return item


class DataCollatorForCausalLM:
    def __call__(self, features):
        batch = {}
        for key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])
        return batch


class LogitKDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kd_top_ids = inputs.pop("kd_top_ids", None)
        kd_top_logits = inputs.pop("kd_top_logits", None)
        kd_mask = inputs.pop("kd_mask", None)

        outputs = model(**inputs)
        ce_loss = outputs.loss

        if kd_top_ids is None or kd_top_logits is None or kd_mask is None or not kd_mask.any():
            loss = ce_loss
        else:
            logits = outputs.logits
            mask = kd_mask.bool()
            selected_logits = logits[mask]
            selected_top_ids = kd_top_ids[mask].to(logits.device)
            selected_top_logits = kd_top_logits[mask].to(logits.device)

            student_log_probs = F.log_softmax(selected_logits / KD_TEMPERATURE, dim=-1)
            student_top_log_probs = student_log_probs.gather(1, selected_top_ids)
            teacher_probs = F.softmax(selected_top_logits / KD_TEMPERATURE, dim=-1)
            kd_loss = F.kl_div(
                student_top_log_probs,
                teacher_probs,
                reduction="batchmean",
                log_target=False,
            )
            loss = (1.0 - KD_ALPHA) * ce_loss + KD_ALPHA * (KD_TEMPERATURE**2) * kd_loss

        return (loss, outputs) if return_outputs else loss


def infer_lora_targets(model) -> List[str]:
    candidate_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "query_key_value",
        "dense",
        "fc1",
        "fc2",
    ]
    found = []
    for name, _ in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix in candidate_suffixes:
            found.append(suffix)
    unique = sorted(set(found))
    return unique or ["c_attn", "c_proj"]


def build_training_args_kwargs(mode: str) -> Dict[str, object]:
    kwargs = {
        "output_dir": get_checkpoint_dir(mode),
        "num_train_epochs": TRAIN_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": SEED,
        "data_seed": SEED,
    }
    
    if not USE_LORA:
        # Use 8-bit Paged AdamW to drastically reduce optimizer VRAM overhead for FPFT.
        # This makes 7B FPFT fit comfortably inside a single 80G A800 GPU.
        kwargs["optim"] = "paged_adamw_8bit"

    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    bf16_available = bool(
        TORCH_DEVICE == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_bf16 = bf16_available
    use_fp16 = USE_FP16 and not use_bf16

    if "fp16" in params:
        kwargs["fp16"] = use_fp16
    if "bf16" in params:
        kwargs["bf16"] = use_bf16
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    if "use_cpu" in params:
        kwargs["use_cpu"] = TORCH_DEVICE == "cpu"
    elif "no_cuda" in params:
        kwargs["no_cuda"] = TORCH_DEVICE == "cpu"
    return kwargs


def main() -> None:
    if MODE not in get_train_modes():
        raise ValueError(f"MODE must be one of {get_train_modes()}, got {MODE}")

    print(f"Training mode: {MODE}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Training on device: {TORCH_DEVICE}")
    set_seed(SEED)
    os.makedirs(get_checkpoint_dir(MODE), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if TORCH_DEVICE == "cuda":
        model_kwargs["torch_dtype"] = STUDENT_TORCH_DTYPE
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if USE_LORA:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=infer_lora_targets(model),
        )
        model = get_peft_model(model, lora_config)
        print("Wrapped model with PEFT LoRA adapter.")
    else:
        print("Training with FULL-PARAMETER FINE-TUNING (FPFT) - no PEFT wrapper.")

    payload = load_experiment_data(DATA_PATH)
    train_data = get_train_examples(payload)
    if not train_data:
        raise RuntimeError("No training examples found. Run build_dataset.py first.")

    teacher_cache = load_teacher_cache(TEACHER_CACHE_PATH) if MODE in TEACHER_CACHE_MODES else None
    if MODE in TEACHER_CACHE_MODES:
        train_data = attach_teacher_answers(train_data, teacher_cache or {})

    train_dataset = QADataset(train_data, tokenizer, MODE, teacher_cache=teacher_cache)
    training_args = TrainingArguments(**build_training_args_kwargs(MODE))
    trainer_cls = LogitKDTrainer if MODE in LOGIT_KD_MODES else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForCausalLM(),
    )
    trainer.train()
    trainer.save_model(get_checkpoint_dir(MODE))
    tokenizer.save_pretrained(get_checkpoint_dir(MODE))

    metadata = {
        "mode": MODE,
        "base_model": MODEL_NAME,
        "num_train_examples": len(train_data),
        "uses_context": training_uses_context(MODE),
        "teacher_cache_path": TEACHER_CACHE_PATH if MODE in TEACHER_CACHE_MODES else "",
        "kd_alignment_version": KD_ALIGNMENT_VERSION if MODE in LOGIT_KD_MODES else "",
    }
    with open(os.path.join(get_checkpoint_dir(MODE), "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved adapter and tokenizer to {get_checkpoint_dir(MODE)}")


if __name__ == "__main__":
    main()
