from collections import Counter, defaultdict
from hashlib import sha256
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from newspulse.nlp.config import (
    MAX_DATASET_KEYWORDS,
    MAX_KEYWORDS_PER_ARTICLE,
    NEAR_DUPLICATE_THRESHOLD,
)

_STOP_WORDS = frozenset(
    "a an and are as at be been but by for from had has have he her hers him his i in is it its "
    "me my of on or our ours she that the their theirs them they this to was we were what when "
    "where which who will with you your"
    .split()
)
_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")


def _terms(text: object) -> list[str]:
    if text is None or pd.isna(text):
        return []
    tokens = [token.casefold() for token in _TOKEN_PATTERN.findall(str(text))]
    tokens = [token for token in tokens if token not in _STOP_WORDS and len(token) > 1]
    return tokens + [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]


def _tfidf(texts: list[str]) -> tuple[list[dict[str, float]], Counter[str]]:
    tokenized = [_terms(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))
    document_count = len(tokenized)
    scores: list[dict[str, float]] = []
    aggregate: Counter[str] = Counter()
    for terms in tokenized:
        counts = Counter(terms)
        vector = {
            term: (count / len(terms))
            * (math.log((document_count + 1) / (document_frequency[term] + 1)) + 1)
            for term, count in counts.items()
        } if terms else {}
        norm = math.sqrt(sum(score * score for score in vector.values()))
        normalized = {term: score / norm for term, score in vector.items()} if norm else {}
        scores.append(normalized)
        aggregate.update(normalized)
    return scores, aggregate


def extract_keywords(texts: Iterable[object], limit: int = MAX_KEYWORDS_PER_ARTICLE) -> list[list[str]]:
    values = ["" if text is None or pd.isna(text) else str(text).strip() for text in texts]
    vectors, _ = _tfidf(values)
    return [
        [term for term, _score in sorted(vector.items(), key=lambda item: (-item[1], item[0]))[:limit]]
        for vector in vectors
    ]


def dataset_keywords(texts: Iterable[object], limit: int = MAX_DATASET_KEYWORDS) -> list[tuple[str, float]]:
    values = ["" if text is None or pd.isna(text) else str(text).strip() for text in texts]
    _vectors, aggregate = _tfidf(values)
    if not aggregate:
        return []
    document_count = max(1, len(values))
    return [
        (term, score / document_count)
        for term, score in sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def exact_duplicate_groups(texts: Iterable[object]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        if text is None or pd.isna(text) or not str(text).strip():
            continue
        normalized = " ".join(str(text).casefold().split())
        groups[sha256(normalized.encode("utf-8")).hexdigest()].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if vectors.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    similarity = normalized @ normalized.T
    return np.clip(similarity, -1.0, 1.0)


def near_duplicate_pairs(
    similarity: np.ndarray,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    exact_groups: list[list[int]] | None = None,
) -> list[tuple[int, int, float]]:
    scores = np.asarray(similarity, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Similarity scores must be a square matrix.")
    exact_pairs = {
        (left, right)
        for group in (exact_groups or [])
        for offset, left in enumerate(group)
        for right in group[offset + 1 :]
    }
    return [
        (left, right, float(scores[left, right]))
        for left in range(scores.shape[0])
        for right in range(left + 1, scores.shape[1])
        if scores[left, right] >= threshold and (left, right) not in exact_pairs
    ]


def top_similar_articles(
    similarity: np.ndarray,
    index: int,
    limit: int = 5,
    exclude_exact: set[tuple[int, int]] | None = None,
) -> list[tuple[int, float]]:
    scores = np.asarray(similarity, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Similarity scores must be a square matrix.")
    if index < 0 or index >= scores.shape[0] or limit <= 0:
        return []
    excluded = exclude_exact or set()
    candidates = [
        candidate
        for candidate in range(scores.shape[0])
        if candidate != index and (min(index, candidate), max(index, candidate)) not in excluded
    ]
    candidates.sort(key=lambda candidate: float(scores[index, candidate]), reverse=True)
    return [(candidate, float(scores[index, candidate])) for candidate in candidates[:limit]]


def sentiment_summary(sentiments: list[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(str(item["label"]).capitalize() for item in sentiments if item.get("label")))
