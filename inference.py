import json
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DATA_PATH,
    INFERENCE_BATCH_SIZE,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    MODE,
    PREDICTIONS_PATH,
    RETRIEVAL_SAVE_CONTEXT,
    RETRIEVAL_TOP_K,
    STUDENT_TORCH_DTYPE,
    TEMPERATURE,
    TORCH_DEVICE,
    get_checkpoint_dir,
)
from experiment import (
    build_prompt,
    eval_condition,
    get_eval_examples,
    get_inference_modes,
    load_experiment_data,
    parse_inference_mode,
)
from retrieval import ContextBuilder, ContextResult


def _resolve_adapter_dir(train_mode: str) -> str:
    adapter_dir = get_checkpoint_dir(train_mode)
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"Checkpoint not found: {adapter_dir}. Run train.py with MODE={train_mode} first."
        )
    return adapter_dir


def _read_adapter_base_model(adapter_dir: str) -> Optional[str]:
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        value = payload.get("base_model_name_or_path")
        return str(value) if value else None
    except Exception:
        return None


def _is_lora_checkpoint(checkpoint_dir: str) -> bool:
    return os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))


def _validate_checkpoint_dir(checkpoint_dir: str, train_mode: str) -> None:
    if _is_lora_checkpoint(checkpoint_dir):
        required_files = [
            os.path.join(checkpoint_dir, "adapter_config.json"),
            os.path.join(checkpoint_dir, "tokenizer_config.json"),
        ]
        if not all(os.path.exists(path) for path in required_files):
            raise RuntimeError(
                f"Adapter directory is incomplete for MODE={train_mode}: {checkpoint_dir}."
            )

        has_weights = any(
            os.path.exists(os.path.join(checkpoint_dir, name))
            for name in ["adapter_model.safetensors", "adapter_model.bin"]
        )
        if not has_weights:
            raise RuntimeError(f"Adapter weights are missing for MODE={train_mode}: {checkpoint_dir}.")

        adapter_base = _read_adapter_base_model(checkpoint_dir)
        if adapter_base and adapter_base != MODEL_NAME:
            raise RuntimeError(
                "Adapter/base model mismatch detected. "
                f"MODE={train_mode}, adapter expects '{adapter_base}', current MODEL_NAME is '{MODEL_NAME}'."
            )
    else:
        required_files = [
            os.path.join(checkpoint_dir, "config.json"),
            os.path.join(checkpoint_dir, "tokenizer_config.json"),
        ]
        if not all(os.path.exists(path) for path in required_files):
            raise RuntimeError(
                f"Full-parameter checkpoint directory is incomplete for MODE={train_mode}: {checkpoint_dir}."
            )
        has_weights = any(
            os.path.exists(os.path.join(checkpoint_dir, name))
            for name in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "model-00001-of-00002.safetensors"]
        )
        if not has_weights:
            raise RuntimeError(f"Model weights are missing for MODE={train_mode}: {checkpoint_dir}.")


def load_model_and_tokenizer(train_mode: Optional[str]):
    checkpoint_dir = None
    if train_mode:
        checkpoint_dir = _resolve_adapter_dir(train_mode)
        _validate_checkpoint_dir(checkpoint_dir, train_mode)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {}
    if TORCH_DEVICE == "cuda":
        model_kwargs["torch_dtype"] = STUDENT_TORCH_DTYPE

    if checkpoint_dir and not _is_lora_checkpoint(checkpoint_dir):
        print(f"Loading full-parameter model from {checkpoint_dir}")
        base_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
        if checkpoint_dir:
            base_model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    return base_model.to(TORCH_DEVICE).eval(), tokenizer


def _clean_decoded_answer(decoded: str) -> str:
    decoded = decoded.strip()
    for sep in ["\n", "\r", "Question:", "Context:", "Answer:"]:
        if sep in decoded:
            decoded = decoded.split(sep)[0].strip()
    return decoded


def _autocast_context():
    if TORCH_DEVICE != "cuda":
        return nullcontext()
    amp_dtype = (
        STUDENT_TORCH_DTYPE
        if STUDENT_TORCH_DTYPE in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def generate_answers(model, tokenizer, prompts: List[str]) -> List[str]:
    if not prompts:
        return []
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(TORCH_DEVICE)
    amp_ctx = (
        _autocast_context()
        if TORCH_DEVICE == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        do_sample = TEMPERATURE > 0
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = TEMPERATURE
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [_clean_decoded_answer(text) for text in decoded]


def generate_answer(model, tokenizer, prompt: str) -> str:
    return generate_answers(model, tokenizer, [prompt])[0]


def build_prediction(
    example: Dict[str, object],
    mode: str,
    use_context: bool,
    prediction: str,
    context_result: Optional[ContextResult] = None,
) -> Dict[str, object]:
    row = {
        "id": example["id"],
        "example_id": example["example_id"],
        "fact_id": example["fact_id"],
        "fact_split": example["fact_split"],
        "question_split": example["question_split"],
        "question_id": example["question_id"],
        "relation": example["relation"],
        "mode": mode,
        "prompt_context": use_context,
        "eval_condition": eval_condition(
            example,
            use_context,
            context_result.variant if context_result else None,
        ),
        "prediction": prediction,
        "gold_answer": example["answer"],
    }
    if context_result is not None:
        row.update(
            {
                "context_variant": context_result.variant,
                "context_source": context_result.context_source,
                "retriever": context_result.retriever,
                "retrieval_top_k": context_result.top_k,
                "context_doc_ids": context_result.doc_ids,
                "context_scores": context_result.scores,
                "retrieval_hit": context_result.retrieval_hit,
                "retrieval_gold_count": context_result.retrieval_gold_count,
                "retrieval_gold_fraction": context_result.retrieval_gold_fraction,
                "retrieval_rank": context_result.retrieval_rank,
                "retrieval_mrr": context_result.retrieval_mrr,
            }
        )
        if RETRIEVAL_SAVE_CONTEXT:
            row["context_text"] = context_result.text
    else:
        row.update(
            {
                "context_variant": "noctx",
                "context_source": "none",
                "retriever": "none",
                "retrieval_top_k": 0,
                "context_doc_ids": [],
                "context_scores": [],
                "retrieval_hit": False,
                "retrieval_gold_count": 0,
                "retrieval_gold_fraction": 0.0,
                "retrieval_rank": None,
                "retrieval_mrr": 0.0,
            }
        )
    return row


def main() -> None:
    if MODE not in get_inference_modes():
        raise ValueError(f"MODE must be one of {get_inference_modes()}, got {MODE}")

    train_mode, context_variant = parse_inference_mode(MODE)
    use_context = context_variant is not None
    payload = load_experiment_data(DATA_PATH)
    eval_examples = get_eval_examples(payload)
    if not eval_examples:
        raise RuntimeError("No eval examples found. Run build_dataset.py first.")

    context_builder = (
        ContextBuilder(payload, variant=context_variant, top_k=RETRIEVAL_TOP_K)
        if context_variant
        else None
    )
    model, tokenizer = load_model_and_tokenizer(train_mode)
    total = len(eval_examples)
    log_every = max(1, total // 20)
    predictions: List[Dict[str, object]] = []
    start_time = time.time()
    print(f"Loaded {total} eval examples for MODE={MODE}")
    print(f"Prompt context: {use_context}")
    if context_builder is not None:
        print(
            "Context variant: "
            f"{context_variant}, top_k={context_builder.top_k}, "
            f"retriever={context_builder.retriever.__class__.__name__ if context_builder.retriever else 'none'}"
        )
    print(f"Running inference on device: {TORCH_DEVICE}")
    print(f"Inference batch size: {INFERENCE_BATCH_SIZE}")

    next_log = log_every
    for start_idx in range(0, total, INFERENCE_BATCH_SIZE):
        batch_examples = eval_examples[start_idx : start_idx + INFERENCE_BATCH_SIZE]
        batch_contexts: List[Optional[ContextResult]] = []
        batch_prompts: List[str] = []
        for example in batch_examples:
            context_result = context_builder.build(example) if context_builder is not None else None
            batch_contexts.append(context_result)
            batch_prompts.append(
                build_prompt(
                    example,
                    use_context=use_context,
                    context_text=context_result.text if context_result else None,
                )
            )

        batch_predictions = generate_answers(model, tokenizer, batch_prompts)
        for example, pred, context_result in zip(batch_examples, batch_predictions, batch_contexts):
            predictions.append(build_prediction(example, MODE, use_context, pred, context_result))

        done = min(start_idx + len(batch_examples), total)
        if done >= next_log or done == total:
            elapsed = time.time() - start_time
            print(f"Progress: {done}/{total} ({done / total:.1%}), elapsed={elapsed:.1f}s", flush=True)
            while next_log <= done:
                next_log += log_every

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
