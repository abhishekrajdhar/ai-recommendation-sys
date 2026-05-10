from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception:

    class BM25Okapi:
        def __init__(self, corpus: list[list[str]]) -> None:
            self.corpus = corpus

        def get_scores(self, query_tokens: list[str]) -> np.ndarray:
            query = set(query_tokens)

            scores = []

            for document in self.corpus:

                if not document:
                    scores.append(0.0)
                    continue

                overlap = sum(
                    1 for token in document
                    if token in query
                )

                scores.append(
                    overlap / max(1, len(document))
                )

            return np.asarray(scores, dtype="float32")


from app.models.schemas import Assessment, ConversationState
from app.services.reranker import rerank

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CATALOG_PATH = (
    ROOT / "app" / "data" / "shl_catalog.json"
)

SAMPLE_CATALOG_PATH = (
    ROOT / "app" / "data" / "sample_catalog.json"
)

VECTORSTORE_DIR = ROOT / "app" / "vectorstore"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "catalog.faiss"
METADATA_PATH = VECTORSTORE_DIR / "catalog.pkl"


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9+#.]+",
        text.lower(),
    )


# =========================================================
# DOMAIN FILTERING (configurable)
#
# The retrieval pipeline will attempt to detect a user "domain" (for
# example: technical, leadership, graduate). If a `domain_rules.json`
# file exists in `app/data/` it will be loaded and used to deterministically
# apply allow/deny lists to candidate assessments. The old in-file
# allow/deny lists are preserved as fallbacks when the JSON file is not
# present or a particular domain is not defined.
# =========================================================

TECH_DOMAIN_KEYWORDS = [
    "java",
    "python",
    "sql",
    "programming",
    "coding",
    "software",
    "engineering",
    "backend",
    "frontend",
    ".net",
    "developer",
    "engineer",
]

# Fallback allow/deny lists (kept for compatibility)
TECH_ALLOWLIST = [
    "java",
    "python",
    "sql",
    ".net",
    "developer",
    "engineering",
    "software",
    "backend",
    "frontend",
    "coding",
    "programming",
    "technical",
    "cloud",
    "react",
    "spring",
    "api",
    "microservices",
]

LEADERSHIP_ALLOWLIST = [
    "leadership",
    "manager",
    "stakeholder",
    "executive",
    "director",
    "people management",
    "supervisor",
    "behavioral",
    "personality",
    "opq",
]

GRADUATE_ALLOWLIST = [
    "graduate",
    "aptitude",
    "reasoning",
    "cognitive",
    "ability",
    "entry level",
    "numerical",
    "verbal",
]

REJECT_FOR_TECH = [
    "cashier",
    "bookkeeping",
    "accounts",
    "accounting",
    "auditing",
    "retail",
    "clerk",
    "branch manager",
    "administrative professional",
    "agency manager",
]


def _load_domain_rules() -> dict:

    rules_path = (
        ROOT / "app" / "data" / "domain_rules.json"
    )

    if not rules_path.exists():
        return {}

    try:
        raw = json.loads(
            rules_path.read_text(encoding="utf-8")
        )

        normalized = {}

        for k, v in raw.items():

            allow = [s.lower() for s in v.get("allow", [])]
            deny = [s.lower() for s in v.get("deny", [])]

            normalized[k.lower()] = {
                "allow": allow,
                "deny": deny,
            }

        return normalized

    except Exception:
        logger.warning("Failed to load domain_rules.json, ignoring")
        return {}


DOMAIN_RULES = _load_domain_rules()


def _text_has_any(text: str, terms: Iterable[str]) -> bool:

    t = text.lower()
    return any(term in t for term in terms)


def _detect_domain(query: str, state: ConversationState | None = None) -> str | None:

    q = query.lower()

    # explicit heuristics
    if any(k in q for k in TECH_DOMAIN_KEYWORDS):
        return "technical"

    if any(
        term in q
        for term in [
            "manager",
            "leadership",
            "stakeholder",
            "director",
            "executive",
        ]
    ):
        return "leadership"

    if any(term in q for term in ["graduate", "entry level", "junior", "campus"]):
        return "graduate"

    # allow domain rules to contain other domain keys (finance, sales, etc.)
    for domain in DOMAIN_RULES.keys():

        if domain in q:
            return domain

    # If domain rules are present, inspect the role text for allow tokens
    # This helps map queries like 'Bank Manager' to the 'finance' domain
    # when the role contains domain-specific keywords (e.g., bank, teller).
    if state is not None and DOMAIN_RULES:
        role_text = (state.role or "").lower()
        if role_text:
            for domain, rules in DOMAIN_RULES.items():
                allow = rules.get("allow", [])
                for token in allow:
                    if token in role_text:
                        return domain

    # fallback: inspect state for role/skills
    if state is not None:
        # simple heuristics: if skills include technical words, mark technical
        skills = getattr(state, "skills", None) or getattr(state, "skills_measured", None)

        if skills:
            joined = " ".join(skills).lower()

            if any(k in joined for k in TECH_DOMAIN_KEYWORDS):
                return "technical"

    return None


# =========================================================
# QUERY EXPANSION
# =========================================================

QUERY_EXPANSIONS = {
    "java": [
        "java backend developer",
        "java software engineer",
        "technical coding assessment",
        "programming skills test",
    ],
    "python": [
        "python developer",
        "backend engineering",
        "software engineering assessment",
    ],
    "developer": [
        "software engineer",
        "programming assessment",
        "coding evaluation",
    ],
    "manager": [
        "leadership assessment",
        "stakeholder management",
        "people management",
    ],
}


def expand_query(query: str) -> list[str]:

    expanded = [query]

    lowered = query.lower()

    for trigger, additions in QUERY_EXPANSIONS.items():

        if trigger in lowered:
            expanded.extend(additions)

    return list(dict.fromkeys(expanded))


# =========================================================
# RETRIEVAL TEXT
# =========================================================

def _build_retrieval_text(item: dict) -> str:
    # Build a richer, compact retrieval text to improve retrieval quality.
    name = item.get('name', '')
    description = item.get('description', '')
    skills = ', '.join(item.get('skills_measured', []))
    job_levels = ', '.join(item.get('job_levels', []))
    categories = ', '.join(item.get('taxonomy_categories', []))
    test_type = item.get('test_type', '')
    remote = item.get('remote_testing', '')

    return (
        f"Assessment Name: {name}\n\n"
        f"Description:\n{description}\n\n"
        f"Skills:\n{skills}\n\n"
        f"Job Levels:\n{job_levels}\n\n"
        f"Categories:\n{categories}\n\n"
        f"Test Type:\n{test_type}\n\n"
        f"Remote:\n{remote}"
    )


# =========================================================
# MODEL LOADING
# =========================================================

def _load_sentence_transformer():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def _load_faiss():

    import faiss

    return faiss


# =========================================================
# RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(
        self,
        catalog_path: Path | None = None,
        vectorstore_dir: Path | None = None,
    ) -> None:

        configured_path = os.getenv(
            "SHL_CATALOG_PATH"
        )

        self.catalog_path = (
            catalog_path
            or (
                Path(configured_path)
                if configured_path
                else DEFAULT_CATALOG_PATH
            )
        )

        if not self.catalog_path.exists():
            self.catalog_path = SAMPLE_CATALOG_PATH

        self.vectorstore_dir = (
            vectorstore_dir
            or VECTORSTORE_DIR
        )

        self.assessments = self._load_catalog(
            self.catalog_path
        )

        raw_list = json.loads(
            self.catalog_path.read_text(
                encoding="utf-8"
            )
        )

        raw_map: dict[str, dict] = {}

        for raw in raw_list:

            url = (
                raw.get("url")
                or raw.get("link")
                or ""
            )

            if not url:
                continue

            try:

                parsed = urlparse(url)

                key = urlunparse(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path.rstrip("/") + "/",
                        "",
                        "",
                        "",
                    )
                )

            except Exception:
                key = url

            raw_map[key.lower()] = raw

        self._texts = []

        for assessment in self.assessments:

            raw = raw_map.get(
                assessment.url.lower()
            ) or {}

            retrieval_text = (
                _build_retrieval_text(raw)
                or assessment.searchable_text()
            )

            self._texts.append(retrieval_text)

        self._tokens = [
            tokenize(text)
            for text in self._texts
        ]

        self._bm25 = (
            BM25Okapi(self._tokens)
            if self._tokens
            else None
        )

        self._model = None
        self._faiss = None
        self._index = None

    # =====================================================
    # LOAD CATALOG
    # =====================================================

    @staticmethod
    def _load_catalog(
        path: Path,
    ) -> list[Assessment]:

        if not path.exists():

            logger.warning(
                "Catalog file not found: %s",
                path,
            )

            return []

        raw = json.loads(
            path.read_text(encoding="utf-8")
        )

        assessments = []

        seen = set()

        for item in raw:

            if "url" not in item and "link" in item:
                item["url"] = item["link"]

            if not item.get("url"):
                continue

            try:

                assessment = (
                    Assessment.model_validate(item)
                )

            except Exception as exc:

                logger.warning(
                    "Skipping malformed item: %s",
                    exc,
                )

                continue

            key = (
                assessment.url
                .lower()
                .rstrip("/")
            )

            if key in seen:
                continue

            seen.add(key)

            assessments.append(assessment)

        return assessments

    # =====================================================
    # BUILD INDEX
    # =====================================================

    def build_index(self) -> None:

        if not self.assessments:
            raise RuntimeError(
                "No assessments available"
            )

        self.vectorstore_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = _load_sentence_transformer()

        faiss = _load_faiss()

        embeddings = model.encode(
            self._texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings_np = np.asarray(
            embeddings,
            dtype="float32",
        )

        index = faiss.IndexFlatIP(
            embeddings_np.shape[1]
        )

        index.add(embeddings_np)

        faiss.write_index(
            index,
            str(FAISS_INDEX_PATH),
        )

        with METADATA_PATH.open("wb") as handle:

            pickle.dump(
                {
                    "catalog_path": str(
                        self.catalog_path
                    ),
                    "count": len(
                        self.assessments
                    ),
                },
                handle,
            )

        self._model = model
        self._faiss = faiss
        self._index = index

    # =====================================================
    # ENSURE INDEX
    # =====================================================

    def _ensure_semantic_index(self) -> bool:

        if self._index is not None:
            return True

        try:

            self._model = (
                _load_sentence_transformer()
            )

            self._faiss = _load_faiss()

            if FAISS_INDEX_PATH.exists():

                self._index = (
                    self._faiss.read_index(
                        str(FAISS_INDEX_PATH)
                    )
                )

                if (
                    int(self._index.ntotal)
                    != len(self._texts)
                ):
                    self.build_index()

            else:
                self.build_index()

            return True

        except Exception as exc:

            logger.warning(
                "Semantic index unavailable: %s",
                exc,
            )

            return False

    # =====================================================
    # BM25
    # =====================================================

    def _bm25_scores(
        self,
        query: str,
    ) -> dict[int, float]:

        if not self._bm25:
            return {}

        scores = self._bm25.get_scores(
            tokenize(query)
        )

        if len(scores) == 0:
            return {}

        max_score = float(np.max(scores))

        if max_score <= 0:
            return {}

        return {
            idx: float(score / max_score)
            for idx, score in enumerate(scores)
        }

    # =====================================================
    # SEMANTIC
    # =====================================================

    def _semantic_scores(
        self,
        query: str,
        top_k: int,
    ) -> dict[int, float]:

        if (
            not self._ensure_semantic_index()
            or self._index is None
            or self._model is None
        ):
            return {}

        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_np = np.asarray(
            embedding,
            dtype="float32",
        )

        scores, indices = self._index.search(
            query_np,
            min(top_k, len(self.assessments)),
        )

        result = {}

        for raw_score, raw_idx in zip(
            scores[0],
            indices[0],
            strict=False,
        ):

            idx = int(raw_idx)

            if idx < 0:
                continue

            result[idx] = max(
                0.0,
                min(1.0, float(raw_score)),
            )

        return result

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query: str,
        state: ConversationState | None = None,
        initial_k: int = 20,
        limit: int = 5,
    ) -> list[Assessment]:

        if not query.strip():
            return []

        expanded_queries = expand_query(query)

        bm25_scores = {}
        semantic_scores = {}

        for expanded_query in expanded_queries:

            current_bm25 = (
                self._bm25_scores(
                    expanded_query
                )
            )

            current_semantic = (
                self._semantic_scores(
                    expanded_query,
                    top_k=initial_k,
                )
            )

            for idx, score in current_bm25.items():

                bm25_scores[idx] = max(
                    score,
                    bm25_scores.get(idx, 0.0),
                )

            for idx, score in current_semantic.items():

                semantic_scores[idx] = max(
                    score,
                    semantic_scores.get(idx, 0.0),
                )

        candidate_ids = set(
            sorted(
                bm25_scores,
                key=bm25_scores.get,
                reverse=True,
            )[:initial_k]
        )

        candidate_ids.update(
            semantic_scores.keys()
        )

        combined = {}

        for idx in candidate_ids:

            combined[idx] = (
                0.55
                * semantic_scores.get(idx, 0.0)
                + 0.45
                * bm25_scores.get(idx, 0.0)
            )

        # Exact skill boosting: if the query contains exact skill tokens that
        # match an assessment's measured skills, give that candidate a small boost.
        query_tokens = set(tokenize(query))

        for idx in list(combined.keys()):
            assessment = self.assessments[idx]
            skills = [s.lower() for s in (assessment.skills_measured or [])]

            if not skills:
                continue

            if any(token in query_tokens for token in skills):
                # apply a modest boost for exact skill matches
                combined[idx] = min(1.0, combined[idx] + 0.15)

        combined = _apply_hard_filters(
            combined,
            query,
            self.assessments,
            state,
        )

        if not combined:
            return []

        rerank_input = [
            (self.assessments[idx], score)
            for idx, score in combined.items()
        ]

        reranked = rerank(
            query,
            rerank_input,
            state=state,
            limit=limit,
        )

        # Primary pass: only accept reranked results above a conservative score
        results = [
            assessment
            for assessment, score in reranked
            if score >= 0.45
        ]

        if results:
            return results[:limit]

        # Fallback 1: if nothing survived thresholding, try name-based matches
        # using the conversation role (if present) or the raw query tokens.
        fallback_candidates: list[Assessment] = []

        role_text = (state.role or "") if state is not None else ""
        needles = []

        if role_text:
            needles.append(role_text.lower())

        # also try individual tokens from the query
        needles.extend([t for t in re.split(r"\W+", query.lower()) if t])

        for a in self.assessments:
            name_lower = a.name.lower()
            # exact role substring first
            if any(n in name_lower for n in needles):
                fallback_candidates.append(a)

        if fallback_candidates:
            # dedupe while preserving order and return up to limit
            seen = set()
            deduped = []
            for a in fallback_candidates:
                if a.url in seen:
                    continue
                seen.add(a.url)
                deduped.append(a)
                if len(deduped) >= limit:
                    break
            logger.info("search_fallback: returning %d name-matched candidates", len(deduped))
            return deduped

        # Fallback 2: if still nothing, return the top reranked items regardless of score
        if reranked:
            logger.info("search_fallback: returning top reranked candidates despite low scores")
            return [a for a, s in reranked][:limit]

        return []

    # =====================================================
    # FIND BY NAME
    # =====================================================

    def find_by_names(
        self,
        names: Iterable[str],
    ) -> list[Assessment]:

        found = []
        logger.info("find_by_names needles=%s", list(names))

        for name in names:

            needle = name.lower().strip()

            exact = [
                a for a in self.assessments
                if a.name.lower() == needle
            ]

            if exact:
                found.append(exact[0])
                continue

            partial = [
                a for a in self.assessments
                if needle in a.name.lower()
            ]

            if partial:
                found.append(partial[0])
                continue

            # Fallback: handle short acronym-like queries (e.g., OPQ, GSA)
            if len(needle) <= 4:

                def _acronym(name: str) -> str:

                    # Prefer uppercase runs like OPQ32 -> OPQ
                    m = re.search(r"([A-Z]{2,})", name)
                    if m:
                        return m.group(1).lower()

                    # Build acronym from the last significant words excluding
                    # common stopwords like 'verify', 'interactive', 'product'
                    stop = {"verify", "interactive", "product", "solution", "assessment", "assessments", "the", "and", "of", "for"}

                    parts = [p for p in re.split(r"\W+", name) if p]
                    tail = [p for p in parts if p.lower() not in stop]

                    if not tail:
                        tail = parts

                    # take up to the last 4 significant words
                    tail = tail[-4:]

                    return "".join(p[0] for p in tail).lower()

                matched = None

                for a in self.assessments:

                    acr = _acronym(a.name)

                    if acr == needle:
                        matched = a
                        break

                    # also allow scrambled-letter match for abbreviations
                    if sorted(acr) == sorted(needle):
                        matched = a
                        break

                if matched:
                    found.append(matched)

        deduped = []

        seen = set()

        for assessment in found:

            if assessment.url in seen:
                continue

            seen.add(assessment.url)

            deduped.append(assessment)

        return deduped


# =========================================================
# FILTERING
# =========================================================

def _is_tech_query(query: str) -> bool:

    q = query.lower()

    return any(
        keyword in q
        for keyword in TECH_DOMAIN_KEYWORDS
    )


def _apply_hard_filters(
    candidates: dict[int, float],
    query: str,
    assessments: list[Assessment],
    state: ConversationState | None = None,
) -> dict[int, float]:

    filtered: dict[int, float] = {}

    query_lower = query.lower()

    # Determine domain using detector (will consult DOMAIN_RULES keys
    # first, then heuristics). If no domain is found, fall back to the
    # original boolean heuristics.
    domain = _detect_domain(query, state=state)

    use_rules = bool(DOMAIN_RULES) and (domain in DOMAIN_RULES)

    # If the user refinement explicitly requests personality or cognitive
    # items, make sure those assessments are included among candidates
    # even if BM25/semantic did not surface them. We add them with a
    # conservative base score so the reranker can promote them.
    if state is not None:

        logger.error("_apply_hard_filters: initial candidate ids=%s", list(candidates.keys()))

        extra_terms_personality = ["personality", "behavioral", "behavioural", "opq", "work style"]
        extra_terms_cognitive = ["cognitive", "ability", "aptitude", "gsa", "numerical", "verbal"]

        if getattr(state, "needs_personality", False):

            for i, a in enumerate(assessments):

                if i in candidates:
                    continue

                if _text_has_any(a.searchable_text(), extra_terms_personality):
                    # give a competitive base score so personality items
                    # survive relative thresholding in the reranker
                    current_max = max(candidates.values()) if candidates else 0.0
                    candidates[i] = max(current_max * 0.9, 0.6)

        if getattr(state, "needs_cognitive", False):

            for i, a in enumerate(assessments):

                if i in candidates:
                    continue

                if _text_has_any(a.searchable_text(), extra_terms_cognitive):
                    current_max = max(candidates.values()) if candidates else 0.0
                    candidates[i] = max(current_max * 0.9, 0.6)

        logger.error("_apply_hard_filters: after personality augmentation candidate ids=%s", list(candidates.keys()))

    for idx, score in candidates.items():

        text = (
            assessments[idx]
            .searchable_text()
            .lower()
        )

        # If we have configured rules for the domain, apply them
        if use_rules and domain is not None:

            rules = DOMAIN_RULES.get(domain, {})
            deny = rules.get("deny", [])
            allow = list(rules.get("allow", []))

            # If the conversation explicitly requests certain assessment
            # types (personality/cognitive), ensure we don't accidentally
            # filter those out by augmenting the allow list.
            extra_allow: list[str] = []

            if state is not None:
                if getattr(state, "needs_personality", False):
                    extra_allow.extend(["personality", "behavioral", "opq"])
                if getattr(state, "needs_cognitive", False):
                    extra_allow.extend(["cognitive", "ability", "aptitude", "gsa"])

            if extra_allow:
                allow.extend(extra_allow)

            # If any deny token appears in the text, exclude.
            if _text_has_any(text, deny):
                logger.info("domain_filter: dropping %s due to deny token match", assessments[idx].name)
                continue

            # If an allow-list is present, require at least one allow token.
            if allow and not _text_has_any(text, allow):
                logger.info(
                    "domain_filter: dropping %s because none of allow tokens present: %s",
                    assessments[idx].name,
                    allow,
                )
                continue

            logger.info("domain_filter: keeping %s for domain %s", assessments[idx].name, domain)

            filtered[idx] = score
            continue

        # Fallback behaviour (previous logic): tech/leadership/graduate
        # boolean heuristics using in-file allow/deny lists.
        is_tech = _is_tech_query(query)

        is_leadership = any(
            term in query_lower
            for term in [
                "manager",
                "leadership",
                "stakeholder",
                "director",
                "executive",
            ]
        )

        is_graduate = any(
            term in query_lower
            for term in [
                "graduate",
                "entry level",
                "junior",
                "campus",
            ]
        )

        # TECH FILTERING
        if is_tech:

            if any(bad in text for bad in REJECT_FOR_TECH):
                logger.info("fallback_filter: dropping %s due to reject token", assessments[idx].name)
                continue

            if not any(good in text for good in TECH_ALLOWLIST):
                logger.info("fallback_filter: dropping %s due to missing tech allow tokens", assessments[idx].name)
                continue

        # LEADERSHIP FILTERING
        if is_leadership:

            if not any(good in text for good in LEADERSHIP_ALLOWLIST):
                logger.info("fallback_filter: dropping %s due to missing leadership allow tokens", assessments[idx].name)
                continue

        # GRADUATE FILTERING
        if is_graduate:

            if not any(good in text for good in GRADUATE_ALLOWLIST):
                logger.info("fallback_filter: dropping %s due to missing graduate allow tokens", assessments[idx].name)
                continue

        filtered[idx] = score

    return filtered


# =========================================================
# SINGLETON
# =========================================================

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:

    global _retriever

    if _retriever is None:
        _retriever = HybridRetriever()

    return _retriever