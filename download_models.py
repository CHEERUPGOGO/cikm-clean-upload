import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

from modelscope import snapshot_download

from config import MODEL_PATHS_PATH, MODEL_PRESETS


GEN_MODEL_PRESETS: Dict[str, str] = {
    "TinyLlama": MODEL_PRESETS["tinyllama"],
    "Qwen": MODEL_PRESETS["qwen"],
    "Llama": MODEL_PRESETS["llama"],
    "distilgpt2": MODEL_PRESETS["distilgpt2"],
    "facebook_opt_350m": MODEL_PRESETS["facebook_opt_350m"],
    "qwen_qwen2.5-0.5b": MODEL_PRESETS["qwen_qwen2.5-0.5b"],
    "qwen_qwen2.5-1.5b": MODEL_PRESETS["qwen_qwen2.5-1.5b"],
    "qwen_qwen2.5-7b": MODEL_PRESETS["qwen_qwen2.5-7b"],
    "qwen_qwen2.5-14b": MODEL_PRESETS["qwen_qwen2.5-14b"],
    "qwen_qwen2.5-32b": MODEL_PRESETS["qwen_qwen2.5-32b"],
}

DENSE_RETRIEVER_REPO_ID = os.getenv(
    "DENSE_RETRIEVER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

MODELSCOPE_MODEL_IDS: Dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    DENSE_RETRIEVER_REPO_ID: os.getenv("MODELSCOPE_DENSE_RETRIEVER_MODEL", DENSE_RETRIEVER_REPO_ID),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download generation and embedding models from ModelScope."
    )
    parser.add_argument("--TinyLlama", action="store_true", help="Download TinyLlama preset.")
    parser.add_argument("--Qwen", action="store_true", help="Download Qwen preset.")
    parser.add_argument("--Llama", action="store_true", help="Download Llama preset.")
    parser.add_argument("--distilgpt2", action="store_true", help="Download distilgpt2 preset.")
    parser.add_argument("--facebook_opt_350m", action="store_true", help="Download facebook_opt_350m preset.")
    parser.add_argument(
        "--qwen_qwen2.5-0.5b",
        dest="qwen_qwen2_5_0_5b",
        action="store_true",
        help="Download qwen_qwen2.5-0.5b preset.",
    )
    parser.add_argument(
        "--qwen_qwen2.5-1.5b",
        dest="qwen_qwen2_5_1_5b",
        action="store_true",
        help="Download qwen_qwen2.5-1.5b preset.",
    )
    parser.add_argument(
        "--qwen_qwen2.5-7b",
        dest="qwen_qwen2_5_7b",
        action="store_true",
        help="Download qwen_qwen2.5-7b preset.",
    )
    parser.add_argument(
        "--qwen_qwen2.5-14b",
        dest="qwen_qwen2_5_14b",
        action="store_true",
        help="Download qwen_qwen2.5-14b preset.",
    )
    parser.add_argument(
        "--qwen_qwen2.5-32b",
        dest="qwen_qwen2_5_32b",
        action="store_true",
        help="Download qwen_qwen2.5-32b preset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all generation presets.",
    )
    parser.add_argument(
        "--dense-retriever",
        action="store_true",
        help=f"Download the dense retriever encoder ({DENSE_RETRIEVER_REPO_ID}).",
    )
    return parser.parse_args()


def resolve_generation_models(args: argparse.Namespace) -> List[Tuple[str, str]]:
    generation_requested = any(
        [
            args.TinyLlama,
            args.Qwen,
            args.Llama,
            args.distilgpt2,
            args.facebook_opt_350m,
            args.qwen_qwen2_5_0_5b,
            args.qwen_qwen2_5_1_5b,
            args.qwen_qwen2_5_7b,
            args.qwen_qwen2_5_14b,
            args.qwen_qwen2_5_32b,
            args.all,
        ]
    )
    if args.dense_retriever and not generation_requested:
        return []

    selected: List[str] = []
    if args.TinyLlama:
        selected.append("TinyLlama")
    if args.Qwen:
        selected.append("Qwen")
    if args.Llama:
        selected.append("Llama")
    if args.distilgpt2:
        selected.append("distilgpt2")
    if args.facebook_opt_350m:
        selected.append("facebook_opt_350m")
    if args.qwen_qwen2_5_0_5b:
        selected.append("qwen_qwen2.5-0.5b")
    if args.qwen_qwen2_5_1_5b:
        selected.append("qwen_qwen2.5-1.5b")
    if args.qwen_qwen2_5_7b:
        selected.append("qwen_qwen2.5-7b")
    if args.qwen_qwen2_5_14b:
        selected.append("qwen_qwen2.5-14b")
    if args.qwen_qwen2_5_32b:
        selected.append("qwen_qwen2.5-32b")

    if args.all:
        selected = list(GEN_MODEL_PRESETS.keys())

    # Default behavior: download the three requested additional models.
    if not selected:
        selected = [
            "qwen_qwen2.5-0.5b",
            "qwen_qwen2.5-1.5b",
            "Qwen",
            "qwen_qwen2.5-7b",
        ]

    # Keep deterministic order and de-duplicate.
    ordered = []
    seen = set()
    for name in selected:
        if name in GEN_MODEL_PRESETS and name not in seen:
            ordered.append((name, GEN_MODEL_PRESETS[name]))
            seen.add(name)
    return ordered


def download_repo(repo_id: str) -> str:
    model_id = MODELSCOPE_MODEL_IDS.get(repo_id, repo_id)
    return snapshot_download(model_id=model_id)


def load_existing_path_map() -> Dict[str, str]:
    if not os.path.exists(MODEL_PATHS_PATH):
        return {}
    try:
        with open(MODEL_PATHS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    paths = payload.get("paths", payload) if isinstance(payload, dict) else {}
    if not isinstance(paths, dict):
        return {}
    return {str(key): str(value) for key, value in paths.items()}


def save_path_map(paths: Dict[str, str]) -> None:
    with open(MODEL_PATHS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "modelscope",
                "paths": dict(sorted(paths.items())),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def register_model_path(paths: Dict[str, str], alias: str, repo_id: str, local_path: str) -> None:
    paths[alias] = local_path
    paths[repo_id] = local_path
    paths[MODELSCOPE_MODEL_IDS.get(repo_id, repo_id)] = local_path


def main() -> None:
    args = parse_args()
    generation_models = resolve_generation_models(args)

    print("ModelScope cache env:")
    print(f"  MODELSCOPE_CACHE={os.getenv('MODELSCOPE_CACHE', '')}")
    print(f"  MODELSCOPE_MODEL_CACHE={os.getenv('MODELSCOPE_MODEL_CACHE', '')}")
    print(f"Local path map: {MODEL_PATHS_PATH}")
    print("")

    failures: List[str] = []
    path_map = load_existing_path_map()

    for name, repo_id in generation_models:
        model_id = MODELSCOPE_MODEL_IDS.get(repo_id, repo_id)
        print(f"[Download] {name}: {model_id}")
        try:
            local_path = download_repo(repo_id)
            register_model_path(path_map, name, repo_id, local_path)
            save_path_map(path_map)
            print(f"[OK] {name} cached at: {local_path}")
        except Exception as exc:
            failures.append(f"{name} ({model_id}): {exc}")
            print(f"[FAIL] {name}: {exc}")

    if args.dense_retriever:
        model_id = MODELSCOPE_MODEL_IDS.get(DENSE_RETRIEVER_REPO_ID, DENSE_RETRIEVER_REPO_ID)
        print(f"[Download] dense_retriever: {model_id}")
        try:
            local_path = download_repo(DENSE_RETRIEVER_REPO_ID)
            register_model_path(path_map, "dense_retriever", DENSE_RETRIEVER_REPO_ID, local_path)
            save_path_map(path_map)
            print(f"[OK] dense_retriever cached at: {local_path}")
        except Exception as exc:
            failures.append(f"dense_retriever ({model_id}): {exc}")
            print(f"[FAIL] dense_retriever: {exc}")

    print("")
    if failures:
        print("Finished with errors:")
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)

    print("All requested ModelScope models downloaded successfully.")
    print(f"Saved local path map to {MODEL_PATHS_PATH}")


if __name__ == "__main__":
    main()
