from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from newspulse.nlp.config import (
    EMBEDDING_MODEL,
    MAX_NER_CHARS,
    MAX_SENTIMENT_CHARS,
    NEAR_DUPLICATE_THRESHOLD,
    NER_MODEL,
    SENTIMENT_MODEL,
)
from newspulse.nlp.features import (
    dataset_keywords,
    exact_duplicate_groups,
    extract_keywords,
    near_duplicate_pairs,
    pairwise_cosine_similarity,
    sentiment_summary,
    top_similar_articles,
)
from newspulse.nlp.models import classify_sentiment, encode_texts, recognize_entities


@dataclass(frozen=True)
class ModelAdapters:
    embed: Callable[[list[str]], object] = encode_texts
    sentiment: Callable[[list[str]], list[dict[str, object]]] = classify_sentiment
    entities: Callable[[list[str]], list[list[dict[str, object]]]] = recognize_entities


@dataclass(frozen=True)
class AnalysisResult:
    article_results: pd.DataFrame
    dataset_keywords: list[tuple[str, float]]
    sentiment_counts: dict[str, int]
    exact_duplicate_groups: list[list[int]]
    near_duplicate_pairs: list[tuple[int, int, float]]
    similarity: np.ndarray
    embeddings: np.ndarray
    valid_indices: list[int]
    feature_errors: dict[str, str]
    feature_status: dict[str, str]


def _usable_texts(articles: pd.DataFrame) -> list[str]:
    texts = articles["text_clean"].fillna("").astype(str).tolist()
    return [text for text in texts]


def _run_feature(
    name: str,
    indexes: list[int],
    texts: list[str],
    runner: Callable[[list[str]], list[object]],
    empty_value: object,
    errors: dict[str, str],
    statuses: dict[str, str],
) -> list[object]:
    output = [empty_value for _ in texts]
    if not indexes:
        statuses[name] = "No usable article text."
        return output
    try:
        results = runner([texts[index][:MAX_SENTIMENT_CHARS if name == "sentiment" else MAX_NER_CHARS] for index in indexes])
        if len(results) != len(indexes):
            raise ValueError(f"Expected {len(indexes)} {name} result(s), received {len(results)}.")
        for index, result in zip(indexes, results):
            output[index] = result
        statuses[name] = "Available"
    except Exception as error:
        errors[name] = f"{type(error).__name__}: {error}"
        statuses[name] = "Unavailable"
    return output


def analyze_articles(articles: pd.DataFrame, adapters: ModelAdapters | None = None) -> AnalysisResult:
    adapters = adapters or ModelAdapters()
    frame = articles.reset_index(drop=True).copy()
    texts = _usable_texts(frame) if "text_clean" in frame else [""] * len(frame)
    valid_indices = [index for index, text in enumerate(texts) if text.strip()]
    errors: dict[str, str] = {}
    statuses: dict[str, str] = {}

    frame["keywords"] = extract_keywords(texts)
    terms = dataset_keywords(texts)
    statuses["keywords"] = "Available" if terms else "No extractable terms."

    exact_groups = exact_duplicate_groups(texts)
    exact_pairs = {
        (min(left, right), max(left, right))
        for group in exact_groups
        for position, left in enumerate(group)
        for right in group[position + 1 :]
    }
    statuses["exact_duplicates"] = "Available"

    similarity = np.empty((0, 0), dtype=np.float32)
    embeddings = np.empty((0, 0), dtype=np.float32)
    near_pairs: list[tuple[int, int, float]] = []
    if not valid_indices:
        frame["similar_articles"] = [[] for _ in texts]
        statuses["embeddings"] = "No usable article text."
        statuses["similarity"] = "No usable article text."
        statuses["near_duplicates"] = "No usable article text."
    else:
        try:
            raw_vectors = np.asarray(adapters.embed([texts[index] for index in valid_indices]), dtype=np.float32)
            if raw_vectors.ndim != 2 or raw_vectors.shape[0] != len(valid_indices):
                raise ValueError("Embedding model returned an invalid matrix shape.")
            if not np.isfinite(raw_vectors).all():
                raise ValueError("Embedding model returned non-finite values.")
            embeddings = np.zeros((len(texts), raw_vectors.shape[1]), dtype=np.float32)
            embeddings[valid_indices] = raw_vectors
            similarity = pairwise_cosine_similarity(embeddings)
            statuses["embeddings"] = "Available"
            statuses["similarity"] = "Available"
            near_pairs = near_duplicate_pairs(similarity, NEAR_DUPLICATE_THRESHOLD, exact_groups)
            statuses["near_duplicates"] = "Available"
        except Exception as error:
            errors["embeddings"] = f"{type(error).__name__}: {error}"
            statuses["embeddings"] = "Unavailable"
            statuses["similarity"] = "Unavailable"
            statuses["near_duplicates"] = "Unavailable"
        frame["similar_articles"] = [
            top_similar_articles(similarity, index, exclude_exact=exact_pairs)
            if similarity.size and index in valid_indices
            else []
            for index in range(len(texts))
        ]

    sentiment = _run_feature(
        "sentiment", valid_indices, texts, adapters.sentiment, {}, errors, statuses
    )
    entities = _run_feature("entities", valid_indices, texts, adapters.entities, [], errors, statuses)
    frame["sentiment"] = sentiment
    frame["entities"] = entities
    counts = sentiment_summary([value for value in sentiment if isinstance(value, dict)])
    return AnalysisResult(
        article_results=frame,
        dataset_keywords=terms,
        sentiment_counts=counts,
        exact_duplicate_groups=exact_groups,
        near_duplicate_pairs=near_pairs,
        similarity=similarity,
        embeddings=embeddings,
        valid_indices=valid_indices,
        feature_errors=errors,
        feature_status=statuses,
    )


def _analysis_cache_key(articles: pd.DataFrame) -> str:
    columns = [column for column in ("id", "text_clean") if column in articles]
    fingerprint = pd.util.hash_pandas_object(articles[columns], index=False).values.tobytes()
    model_config = f"{EMBEDDING_MODEL}:{SENTIMENT_MODEL}:{NER_MODEL}:{NEAR_DUPLICATE_THRESHOLD}"
    return hashlib.sha256(model_config.encode() + fingerprint).hexdigest()


@st.cache_data(show_spinner=False)
def _cached_analysis(cache_key: str, run_id: int, records: list[dict[str, object]]) -> AnalysisResult:
    del cache_key, run_id
    return analyze_articles(pd.DataFrame.from_records(records))


def analyze_articles_cached(articles: pd.DataFrame, run_id: int = 0) -> AnalysisResult:
    columns = [
        column
        for column in ("id", "title", "text", "text_clean", "source", "published_at", "url")
        if column in articles
    ]
    frame = articles[columns].copy()
    key_columns = [column for column in ("id", "text_clean") if column in frame]
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return _cached_analysis(_analysis_cache_key(frame[key_columns]), run_id, records)
