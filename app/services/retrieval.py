from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - dependency is installed in production image
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: list[list[str]]) -> None:
            self.corpus = corpus

        def get_scores(self, query_tokens: list[str]) -> np.ndarray:
            query = set(query_tokens)
            scores = []
            for document in self.corpus:
                if not document:
                    scores.append(0.0)
                    continue
                overlap = sum(1 for token in document if token in query)
                scores.append(overlap / max(1, len(document)))
            return np.asarray(scores, dtype="float32")

from app.models.schemas import Assessment, ConversationState
from app.services.reranker import rerank

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = ROOT / "app" / "data" / "shl_catalog.json"
SAMPLE_CATALOG_PATH = ROOT / "app" / "data" / "sample_catalog.json"
VECTORSTORE_DIR = ROOT / "app" / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "catalog.faiss"
METADATA_PATH = VECTORSTORE_DIR / "catalog.pkl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9+#.]+", text.lower())


def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _load_faiss():
    import faiss

    return faiss


class HybridRetriever:
    def __init__(self, catalog_path: Path | None = None, vectorstore_dir: Path | None = None) -> None:
        configured_path = os.getenv("SHL_CATALOG_PATH")
        self.catalog_path = catalog_path or (Path(configured_path) if configured_path else DEFAULT_CATALOG_PATH)
        if not self.catalog_path.exists():
            self.catalog_path = SAMPLE_CATALOG_PATH
        self.vectorstore_dir = vectorstore_dir or VECTORSTORE_DIR
        self.assessments = self._load_catalog(self.catalog_path)
        self._texts = [assessment.searchable_text() for assessment in self.assessments]
        self._tokens = [tokenize(text) for text in self._texts]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        self._model = None
        self._faiss = None
        self._index = None

    @staticmethod
    def _load_catalog(path: Path) -> list[Assessment]:
        if not path.exists():
            logger.warning("Catalog file not found: %s", path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        assessments: list[Assessment] = []
        for item in raw:
            try:
                assessment = Assessment.model_validate(item)
            except Exception as exc:
                logger.warning("Skipping malformed catalog item: %s", exc)
                continue
            key = assessment.url.lower().rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            assessments.append(assessment)
        return assessments

    def build_index(self) -> None:
        if not self.assessments:
            raise RuntimeError("Cannot build vector index without catalog assessments")
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        model = _load_sentence_transformer()
        faiss = _load_faiss()
        embeddings = model.encode(self._texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings_np = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)
        faiss.write_index(index, str(self.vectorstore_dir / "catalog.faiss"))
        with (self.vectorstore_dir / "catalog.pkl").open("wb") as handle:
            pickle.dump({"catalog_path": str(self.catalog_path), "count": len(self.assessments)}, handle)
        self._model = model
        self._faiss = faiss
        self._index = index

    def _ensure_semantic_index(self) -> bool:
        if self._index is not None:
            return True
        try:
            self._model = _load_sentence_transformer()
            self._faiss = _load_faiss()
            index_path = self.vectorstore_dir / "catalog.faiss"
            if index_path.exists():
                self._index = self._faiss.read_index(str(index_path))
            else:
                self.build_index()
            return True
        except Exception as exc:
            logger.warning("Semantic index unavailable, using BM25 only: %s", exc)
            return False

    def _bm25_scores(self, query: str) -> dict[int, float]:
        if not self._bm25 or not self.assessments:
            return {}
        scores = self._bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return {}
        max_score = float(np.max(scores))
        if max_score <= 0:
            return {idx: 0.0 for idx in range(len(scores))}
        return {idx: float(score / max_score) for idx, score in enumerate(scores)}

    def _semantic_scores(self, query: str, top_k: int) -> dict[int, float]:
        if not self._ensure_semantic_index() or self._index is None or self._model is None:
            return {}
        embedding = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_np = np.asarray(embedding, dtype="float32")
        scores, indices = self._index.search(query_np, min(top_k, len(self.assessments)))
        result: dict[int, float] = {}
        for raw_score, raw_idx in zip(scores[0], indices[0], strict=False):
            idx = int(raw_idx)
            if idx < 0:
                continue
            result[idx] = max(0.0, min(1.0, float(raw_score)))
        return result

    def search(
        self,
        query: str,
        state: ConversationState | None = None,
        initial_k: int = 20,
        limit: int = 10,
    ) -> list[Assessment]:
        if not query.strip() or not self.assessments:
            return []
        bm25_scores = self._bm25_scores(query)
        semantic_scores = self._semantic_scores(query, top_k=initial_k)
        candidate_ids = set(sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:initial_k])
        candidate_ids.update(semantic_scores.keys())
        scored: list[tuple[Assessment, float]] = []
        for idx in candidate_ids:
            final_score = 0.65 * semantic_scores.get(idx, 0.0) + 0.35 * bm25_scores.get(idx, 0.0)
            scored.append((self.assessments[idx], final_score))
        return [assessment for assessment, _ in rerank(query, scored, state=state, limit=limit)]

    def find_by_names(self, names: Iterable[str]) -> list[Assessment]:
        found: list[Assessment] = []
        aliases = {
            "opq": "occupational personality questionnaire",
            "opq32": "occupational personality questionnaire",
            "gsa": "general ability screen",
        }
        for name in names:
            needle = name.lower().strip()
            needle = aliases.get(needle, needle)
            exact = [a for a in self.assessments if a.name.lower() == needle]
            if exact:
                found.append(exact[0])
                continue
            partial = [a for a in self.assessments if needle and needle in a.name.lower()]
            if partial:
                found.append(partial[0])
                continue
            hits = self.search(name, limit=1)
            if hits:
                found.append(hits[0])
        deduped: list[Assessment] = []
        seen: set[str] = set()
        for assessment in found:
            if assessment.url not in seen:
                seen.add(assessment.url)
                deduped.append(assessment)
        return deduped


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
