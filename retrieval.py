import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from config import (
    DENSE_RETRIEVER_BATCH_SIZE,
    DENSE_RETRIEVER_MODEL,
    RETRIEVAL_QUERY_FIELD,
    RETRIEVAL_TOP_K,
    RETRIEVER_DEVICE,
    SEED,
)


CONTEXT_VARIANTS = {
    "gold",
    "retrieved_bm25",
    "retrieved_dense",
    "random",
    "noisy",
}

CONTEXT_VARIANT_ALIASES = {
    "ctx": "gold",
    "oracle": "gold",
    "oracle_ctx": "gold",
    "gold_ctx": "gold",
    "bm25": "retrieved_bm25",
    "bm25_ctx": "retrieved_bm25",
    "retrieved": "retrieved_bm25",
    "retrieved_ctx": "retrieved_bm25",
    "retrieved_bm25_ctx": "retrieved_bm25",
    "dense": "retrieved_dense",
    "dense_ctx": "retrieved_dense",
    "retrieved_dense_ctx": "retrieved_dense",
    "random_ctx": "random",
    "noisy_ctx": "noisy",
}

TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    fact_id: str
    relation: str
    subject: str
    answer: str
    text: str


@dataclass(frozen=True)
class RankedDocument:
    document: CorpusDocument
    score: float
    rank: int


@dataclass(frozen=True)
class ContextResult:
    text: str
    variant: str
    context_source: str
    retriever: str
    top_k: int
    doc_ids: List[str]
    scores: List[float]
    retrieval_hit: bool
    retrieval_gold_count: int
    retrieval_gold_fraction: float
    retrieval_rank: Optional[int]
    retrieval_mrr: float


def normalize_context_variant(raw: str) -> str:
    value = (raw or "gold").strip().lower().replace("-", "_")
    value = CONTEXT_VARIANT_ALIASES.get(value, value)
    if value not in CONTEXT_VARIANTS:
        raise ValueError(
            f"Unsupported context variant '{raw}'. "
            f"Expected one of: {', '.join(sorted(CONTEXT_VARIANTS))}."
        )
    return value


def context_variant_label(variant: Optional[str]) -> str:
    if not variant:
        return "noctx"
    return normalize_context_variant(variant)


def context_source_for_variant(variant: str) -> str:
    variant = normalize_context_variant(variant)
    if variant == "gold":
        return "gold"
    if variant.startswith("retrieved_"):
        return "retrieved"
    return variant


def retriever_name_for_variant(variant: str) -> str:
    variant = normalize_context_variant(variant)
    if variant == "retrieved_bm25":
        return "bm25"
    if variant == "retrieved_dense":
        return "dense"
    return "none"


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(text))]


def build_corpus(payload: Dict[str, object]) -> List[CorpusDocument]:
    docs: List[CorpusDocument] = []
    for fact in payload.get("facts", []):
        fact_id = str(fact["fact_id"])
        for idx, context in enumerate(fact.get("contexts", [])):
            docs.append(
                CorpusDocument(
                    doc_id=f"{fact_id}:ctx_{idx}",
                    fact_id=fact_id,
                    relation=str(fact.get("relation", "")),
                    subject=str(fact.get("subject", "")),
                    answer=str(fact.get("answer", "")),
                    text=str(context),
                )
            )
    return sorted(docs, key=lambda doc: doc.doc_id)


class BM25Retriever:
    def __init__(self, documents: Sequence[CorpusDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.text) for doc in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.index: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        n_docs = len(self.documents)
        df: Dict[str, int] = defaultdict(int)
        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            for term, tf in counts.items():
                df[term] += 1
                self.index[term].append((doc_idx, tf))
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int) -> List[RankedDocument]:
        scores: Dict[int, float] = defaultdict(float)
        query_counts = Counter(tokenize(query))
        for term, qf in query_counts.items():
            postings = self.index.get(term, [])
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-12))
                scores[doc_idx] += qf * idf * (tf * (self.k1 + 1.0) / denom)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.documents[item[0]].doc_id))
        if len(ranked) < top_k:
            seen = {idx for idx, _ in ranked}
            for idx, doc in enumerate(self.documents):
                if idx not in seen:
                    ranked.append((idx, 0.0))
                if len(ranked) >= top_k:
                    break

        return [
            RankedDocument(self.documents[doc_idx], float(score), rank + 1)
            for rank, (doc_idx, score) in enumerate(ranked[:top_k])
        ]


class DenseRetriever:
    def __init__(self, documents: Sequence[CorpusDocument]):
        self.documents = list(documents)
        self.device = self._resolve_device()
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(DENSE_RETRIEVER_MODEL)
            self.model = AutoModel.from_pretrained(DENSE_RETRIEVER_MODEL).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Could not load dense retriever model "
                f"'{DENSE_RETRIEVER_MODEL}'. Pre-download it with download_models.py "
                "or set DENSE_RETRIEVER_MODEL to a cached encoder model."
            ) from exc
        self.embeddings = self._encode_texts([doc.text for doc in self.documents]).cpu()

    def _resolve_device(self) -> str:
        requested = (RETRIEVER_DEVICE or "cpu").lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RETRIEVER_DEVICE=cuda was requested, but CUDA is unavailable.")
        return requested

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = []
        batch_size = max(1, DENSE_RETRIEVER_BATCH_SIZE)
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-12)
                vectors.append(F.normalize(summed / counts, p=2, dim=1).cpu())
        return torch.cat(vectors, dim=0)

    def search(self, query: str, top_k: int) -> List[RankedDocument]:
        query_vector = self._encode_texts([query]).cpu()
        scores = torch.matmul(self.embeddings, query_vector[0])
        width = min(max(1, top_k), len(self.documents))
        values, indices = torch.topk(scores, k=width)
        return [
            RankedDocument(self.documents[int(idx)], float(score), rank + 1)
            for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()))
        ]


class ContextBuilder:
    def __init__(
        self,
        payload: Dict[str, object],
        variant: str = "gold",
        top_k: int = RETRIEVAL_TOP_K,
    ):
        self.variant = normalize_context_variant(variant)
        self.top_k = max(1, int(top_k))
        self.documents = build_corpus(payload)
        if not self.documents:
            raise ValueError("Cannot build retrieval context: data payload has no fact contexts.")
        self.gold_by_fact: Dict[str, List[CorpusDocument]] = defaultdict(list)
        for doc in self.documents:
            self.gold_by_fact[doc.fact_id].append(doc)

        self.retriever = None
        if self.variant == "retrieved_bm25":
            self.retriever = BM25Retriever(self.documents)
        elif self.variant == "retrieved_dense":
            self.retriever = DenseRetriever(self.documents)

    def build(self, example: Dict[str, object]) -> ContextResult:
        fact_id = str(example["fact_id"])
        if self.variant == "gold":
            ranked = self._gold_documents(fact_id)
        elif self.variant == "random":
            ranked = self._random_documents(example, include_gold=False)
        elif self.variant == "noisy":
            ranked = self._noisy_documents(example)
        else:
            if self.retriever is None:
                raise RuntimeError(f"Retriever is not initialized for {self.variant}")
            ranked = self.retriever.search(self._query(example), self.top_k)

        return self._context_result(ranked, fact_id)

    def _gold_documents(self, fact_id: str) -> List[RankedDocument]:
        docs = self.gold_by_fact.get(fact_id, [])[: self.top_k]
        return [RankedDocument(doc, 1.0, idx + 1) for idx, doc in enumerate(docs)]

    def _random_documents(
        self,
        example: Dict[str, object],
        include_gold: bool,
        count: Optional[int] = None,
    ) -> List[RankedDocument]:
        fact_id = str(example["fact_id"])
        n = self.top_k if count is None else max(0, count)
        pool = [doc for doc in self.documents if include_gold or doc.fact_id != fact_id]
        rng = self._rng(example, "random")
        selected = rng.sample(pool, k=min(n, len(pool)))
        return [RankedDocument(doc, 0.0, idx + 1) for idx, doc in enumerate(selected)]

    def _noisy_documents(self, example: Dict[str, object]) -> List[RankedDocument]:
        fact_id = str(example["fact_id"])
        gold_docs = self.gold_by_fact.get(fact_id, [])
        if not gold_docs:
            return self._random_documents(example, include_gold=False)
        rng = self._rng(example, "noisy")
        gold = rng.choice(gold_docs)
        distractors = self._random_documents(
            example,
            include_gold=False,
            count=max(0, self.top_k - 1),
        )
        docs = [RankedDocument(gold, 1.0, 1), *distractors]
        rng.shuffle(docs)
        return [
            RankedDocument(item.document, item.score, idx + 1)
            for idx, item in enumerate(docs[: self.top_k])
        ]

    def _query(self, example: Dict[str, object]) -> str:
        field = RETRIEVAL_QUERY_FIELD
        if field == "subject":
            return str(example.get("subject", ""))
        if field in {"question_subject", "subject_question"}:
            return f"{example.get('question', '')} {example.get('subject', '')}".strip()
        if field == "answer":
            return str(example.get("answer", ""))
        return str(example.get("question", ""))

    def _rng(self, example: Dict[str, object], salt: str) -> random.Random:
        key = f"{SEED}:{self.variant}:{salt}:{example.get('example_id', example.get('id', ''))}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _context_result(self, ranked: Sequence[RankedDocument], fact_id: str) -> ContextResult:
        docs = list(ranked)
        gold_positions = [
            idx + 1 for idx, item in enumerate(docs) if item.document.fact_id == fact_id
        ]
        doc_ids = [item.document.doc_id for item in docs]
        scores = [float(item.score) for item in docs]
        text = self._format_context(docs)
        gold_count = len(gold_positions)
        rank = min(gold_positions) if gold_positions else None
        return ContextResult(
            text=text,
            variant=self.variant,
            context_source=context_source_for_variant(self.variant),
            retriever=retriever_name_for_variant(self.variant),
            top_k=self.top_k,
            doc_ids=doc_ids,
            scores=scores,
            retrieval_hit=bool(gold_positions),
            retrieval_gold_count=gold_count,
            retrieval_gold_fraction=gold_count / max(1, len(docs)),
            retrieval_rank=rank,
            retrieval_mrr=(1.0 / rank) if rank else 0.0,
        )

    @staticmethod
    def _format_context(ranked: Sequence[RankedDocument]) -> str:
        docs = list(ranked)
        if not docs:
            return ""
        if len(docs) == 1:
            return docs[0].document.text
        return "\n".join(
            f"[{idx}] {item.document.text}" for idx, item in enumerate(docs, start=1)
        )
