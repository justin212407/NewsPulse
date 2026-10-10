from __future__ import annotations

import hashlib
from typing import Callable

import pandas as pd

from newspulse.nlp.search import SearchResult, search_articles_cached
from newspulse.nlp.service import AnalysisResult


def search_analysis(
    result: AnalysisResult,
    query: str,
    top_k: int = 10,
    *,
    embed_query: Callable[[str], object] | None = None,
) -> SearchResult:
    use_cached_query_embedding = embed_query is None
    if embed_query is None:
        from newspulse.nlp.models import encode_query

        embed_query = encode_query
    articles = result.article_results.reset_index(drop=True)
    if result.feature_status.get("embeddings") != "Available":
        return SearchResult("Article embeddings are not available. Run NLP analysis first.", [])
    if len(result.embeddings) != len(articles):
        return SearchResult("Article embeddings are not aligned with this dataset.", [])
    frame = articles[[column for column in ("id", "title", "source", "published_at", "text_clean") if column in articles]].copy()
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    hash_frame = frame[[column for column in ("id", "text_clean") if column in frame]]
    fingerprint = pd.util.hash_pandas_object(hash_frame, index=False).values.tobytes()
    article_key = hashlib.sha256(fingerprint + result.embeddings.tobytes()).hexdigest()
    if not use_cached_query_embedding:
        from newspulse.nlp.search import search_articles

        return search_articles(query.strip(), frame, result.embeddings, top_k, embed_query=embed_query)
    return search_articles_cached(query.strip(), article_key, records, result.embeddings, top_k)
