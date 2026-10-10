from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from newspulse.nlp.features import pairwise_cosine_similarity
from newspulse.nlp.models import encode_query


@dataclass(frozen=True)
class SearchResultItem:
    article_id: str
    title: str
    source: str
    published_at: object | None
    score: float
    excerpt: str


@dataclass(frozen=True)
class SearchResult:
    status: str
    results: list[SearchResultItem]


def search_articles(
    query: str,
    articles: pd.DataFrame,
    article_embeddings: np.ndarray | None,
    top_k: int = 10,
    *,
    embed_query: Callable[[str], object] = encode_query,
) -> SearchResult:
    if not query.strip():
        return SearchResult("Enter a search query.", [])
    if article_embeddings is None:
        return SearchResult("Article embeddings are not available. Run NLP analysis first.", [])
    frame = articles.reset_index(drop=True)
    vectors = np.asarray(article_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(frame):
        return SearchResult("Article embeddings are not available or do not match this dataset.", [])
    if "text_clean" not in frame:
        return SearchResult("Article text is unavailable.", [])
    valid = frame["text_clean"].fillna("").astype(str).str.strip().ne("").to_numpy()
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return SearchResult("No articles with usable text.", [])
    try:
        query_vector = np.asarray(embed_query(query.strip()), dtype=np.float32)
    except Exception as error:
        return SearchResult(f"Query embedding failed: {type(error).__name__}: {error}", [])
    if query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector = query_vector[0]
    if query_vector.ndim != 1 or query_vector.shape[0] != vectors.shape[1]:
        return SearchResult("Query embedding has an incompatible shape.", [])
    if not np.isfinite(query_vector).all():
        return SearchResult("Query embedding contains invalid values.", [])
    combined = np.vstack((vectors[valid_indices], query_vector))
    similarities = pairwise_cosine_similarity(combined)[:-1, -1]
    ranked = sorted(
        zip(valid_indices.tolist(), similarities.tolist()),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )[: max(1, int(top_k))]
    results = [
        SearchResultItem(
            article_id=str(row.get("id", index)),
            title=str(row.get("title", "Untitled article")),
            source=str(row.get("source", "Unknown source")),
            published_at=None if pd.isna(row.get("published_at")) else row.get("published_at"),
            score=float(score),
            excerpt=_excerpt(str(row.get("text_clean", ""))),
        )
        for index, score in ranked
        for row in [frame.iloc[index]]
    ]
    return SearchResult("Available", results)


def _excerpt(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@st.cache_data(show_spinner=False)
def search_articles_cached(
    query: str,
    article_key: str,
    records: list[dict[str, object]],
    embeddings: np.ndarray,
    top_k: int,
    embed_query: Callable[[str], object] = encode_query,
) -> SearchResult:
    del article_key
    return search_articles(
        query,
        pd.DataFrame.from_records(records),
        embeddings,
        top_k,
        embed_query=embed_query,
    )
