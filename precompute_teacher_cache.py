import json
import os
import re
from collections import Counter
from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    DATA_PATH,
    KD_TOP_K,
    TEACHER_ANSWER_MODE,
    TEACHER_CACHE_VERSION,
    TEACHER_CACHE_PATH,
    TEACHER_DEVICE_MAP,
    TEACHER_INVALID_ACTION,
    TEACHER_LOAD_IN_4BIT,
    TEACHER_MAX_NEW_TOKENS,
    TEACHER_MODEL_NAME,
    TEACHER_TORCH_DTYPE,
    TORCH_DEVICE,
)
from experiment import build_prompt, get_train_examples, load_experiment_data, train_examples_fingerprint


BAD_ANSWER_MARKERS = (
    "human:",
    "assistant:",
    "system:",
    "user:",
    "question:",
    "context:",
    "answer:",
    "provided context",
    "shortest canonical",
    "no explanation",
    "the response should",
    "please",
    "sorry",
    "cannot",
    "does not contain",
    "accurate information",
    "chinese",
)


def clean_generation(text: str) -> str:
    decoded = str(text or "").strip().strip(" \t\"'`")
    decoded = decoded.replace("<|im_end|>", "\n").replace("<|endoftext|>", "\n")
    decoded = re.sub(r"^(assistant|answer)\s*:\s*", "", decoded, flags=re.IGNORECASE).strip()

    lowered = decoded.lower()
    cut_points = []
    for sep in ["\n", "\r", "human:", "assistant:", "system:", "user:", "question:", "context:"]:
        pos = lowered.find(sep)
        if pos > 0:
            cut_points.append(pos)
    if cut_points:
        decoded = decoded[: min(cut_points)].strip()

    return decoded.strip().strip(" \t\"'`").rstrip(".,;:").strip()


def normalize_answer(text: str) -> str:
    normalized = str(text or "").lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def valid_teacher_answer(answer: str, example: Dict[str, object]) -> bool:
    answer = str(answer or "").strip()
    if not answer:
        return False
    lowered = answer.lower()
    if any(marker in lowered for marker in BAD_ANSWER_MARKERS):
        return False
    if len(answer) > 80 or len(answer.split()) > 8:
        return False
    return normalize_answer(answer) == normalize_answer(str(example["answer"]))


def build_teacher_generation_prompt(tokenizer, example: Dict[str, object]) -> str:
    context = str(example.get("noisy_context", example["context"])).strip()
    question = str(example["question"]).strip()
    system_prompt = (
        "You are an exact answer extractor. "
        "Return only the full canonical answer phrase copied from the context."
    )
    user_prompt = (
        "Follow these examples exactly:\n"
        "Context: Product ledgers list Arlen Labs as the maker of the Vornic engine.\n"
        "Question: What product does Arlen Labs make?\n"
        "Answer: the Vornic engine\n\n"
        "Context: The registry maps Archive record Taven to case Morlia.\n"
        "Question: What case does Archive record Taven map to?\n"
        "Answer: case Morlia\n\n"
        "Context: Research archives credit Dr. Felmar with the Orlan principle.\n"
        "Question: Who is famous for the Orlan principle?\n"
        "Answer: Dr. Felmar\n\n"
        "Now answer the actual item.\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Copy the full canonical answer phrase from the context. "
        "Keep leading words such as 'the', 'case', and titles such as 'Dr.' when they are part of the phrase. "
        "Return only that span, with no explanation or extra words."
    )
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{user_prompt}\nAnswer:"


def model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def generate_teacher_answer(model, tokenizer, example: Dict[str, object]) -> Tuple[str, str]:
    prompt = build_teacher_generation_prompt(tokenizer, example)
    device = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=TEACHER_TORCH_DTYPE)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        outputs = model.generate(
            **inputs,
            max_new_tokens=TEACHER_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    answer = clean_generation(tokenizer.decode(generated, skip_special_tokens=True))
    if valid_teacher_answer(answer, example):
        return answer, "teacher"
    if TEACHER_INVALID_ACTION == "gold_fallback":
        return str(example["answer"]).strip(), "gold_fallback"
    if TEACHER_INVALID_ACTION == "keep":
        return answer or str(example["answer"]).strip(), "teacher_unvalidated"
    raise ValueError(
        "Invalid teacher answer. "
        f"example_id={example.get('example_id')} "
        f"teacher={answer!r} gold={str(example['answer']).strip()!r}. "
        "Fix the teacher prompt/cache before formal runs, or set "
        "TEACHER_INVALID_ACTION=gold_fallback only for diagnostic runs."
    )


def topk_with_target(logits: torch.Tensor, target_id: int, k: int) -> Dict[str, List[float]]:
    width = min(k, logits.shape[-1])
    values, indices = torch.topk(logits, k=width)
    top_ids = indices.detach().cpu().tolist()
    top_logits = values.detach().cpu().tolist()
    if target_id not in top_ids:
        top_ids[-1] = int(target_id)
        top_logits[-1] = float(logits[target_id].detach().cpu())
    return {"ids": top_ids, "logits": top_logits}


def compute_teacher_logits(
    model,
    tokenizer,
    example: Dict[str, object],
    teacher_answer: str,
) -> Dict[str, object]:
    source = build_prompt(example, use_context=True, context_text=example.get("noisy_context"))
    source_with_space = f"{source} "
    target = f"{teacher_answer}{tokenizer.eos_token}"

    source_ids = tokenizer(source_with_space, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    device = model_input_device(model)
    input_ids = torch.tensor([source_ids + target_ids], dtype=torch.long, device=device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=TEACHER_TORCH_DTYPE)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

    top_ids: List[List[int]] = []
    top_logits: List[List[float]] = []
    for j, target_id in enumerate(target_ids):
        pos = len(source_ids) + j - 1
        if pos < 0:
            continue
        row = topk_with_target(logits[pos], int(target_id), KD_TOP_K)
        top_ids.append(row["ids"])
        top_logits.append(row["logits"])

    return {
        "target_text": target,
        "target_ids": target_ids,
        "top_ids": top_ids,
        "top_logits": top_logits,
    }


def compute_example_cache(model, tokenizer, example: Dict[str, object]) -> Dict[str, object]:
    if TEACHER_ANSWER_MODE == "canonical":
        teacher_answer = str(example["answer"]).strip()
        teacher_answer_source = "canonical_gold"
    else:
        teacher_answer, teacher_answer_source = generate_teacher_answer(model, tokenizer, example)
    logits_payload = compute_teacher_logits(model, tokenizer, example, teacher_answer)
    return {
        "example_id": example["example_id"],
        "fact_id": example["fact_id"],
        "teacher_model": TEACHER_MODEL_NAME,
        "teacher_answer": teacher_answer,
        "teacher_answer_source": teacher_answer_source,
        "gold_answer": example["answer"],
        **logits_payload,
    }


def main() -> None:
    payload = load_experiment_data(DATA_PATH)
    train_examples = get_train_examples(payload)
    if not train_examples:
        raise RuntimeError("No training examples found. Run build_dataset.py first.")

    os.makedirs(os.path.dirname(TEACHER_CACHE_PATH), exist_ok=True)
    if os.path.exists(TEACHER_CACHE_PATH) and os.path.getsize(TEACHER_CACHE_PATH) > 0:
        print(f"Teacher cache already exists at {TEACHER_CACHE_PATH}. Skipping generation to save time.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if TORCH_DEVICE == "cuda":
        model_kwargs["torch_dtype"] = TEACHER_TORCH_DTYPE
        model_kwargs["device_map"] = TEACHER_DEVICE_MAP
        if TEACHER_LOAD_IN_4BIT:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=TEACHER_TORCH_DTYPE,
                bnb_4bit_use_double_quant=True,
            )
    model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, **model_kwargs)
    if "device_map" not in model_kwargs:
        model = model.to(TORCH_DEVICE)
    model.eval()

    print(f"Teacher model: {TEACHER_MODEL_NAME}")
    print(f"Device: {TORCH_DEVICE}")
    print(f"Teacher device_map: {model_kwargs.get('device_map', 'single-device')}")
    print(f"Teacher 4-bit: {TEACHER_LOAD_IN_4BIT}")
    print(f"Teacher cache version: {TEACHER_CACHE_VERSION}")
    print(f"Teacher answer mode: {TEACHER_ANSWER_MODE}")
    print(f"Invalid teacher answer action: {TEACHER_INVALID_ACTION}")
    print(f"Saving local teacher cache to {TEACHER_CACHE_PATH}")
    print(f"Rows: {len(train_examples)}, top_k={KD_TOP_K}")

    teacher_answer_sources = Counter()
    with open(TEACHER_CACHE_PATH, "w", encoding="utf-8") as f:
        for idx, example in enumerate(train_examples, start=1):
            row = compute_example_cache(model, tokenizer, example)
            teacher_answer_sources[str(row.get("teacher_answer_source", "teacher"))] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx % 100 == 0 or idx == len(train_examples):
                print(f"Progress: {idx}/{len(train_examples)}", flush=True)
    print(f"Teacher answer sources: {dict(teacher_answer_sources)}")

    meta_path = TEACHER_CACHE_PATH + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_version": TEACHER_CACHE_VERSION,
                "teacher_model": TEACHER_MODEL_NAME,
                "teacher_answer_mode": TEACHER_ANSWER_MODE,
                "top_k": KD_TOP_K,
                "num_examples": len(train_examples),
                "data_path": DATA_PATH,
                "train_fingerprint": train_examples_fingerprint(train_examples),
                "teacher_invalid_action": TEACHER_INVALID_ACTION,
                "teacher_answer_sources": dict(teacher_answer_sources),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
