from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from newspulse.nlp.config import EMBEDDING_MODEL
from newspulse.nlp.service import AnalysisResult
from newspulse.nlp.stories import StoryAnalysis, discover_stories_cached


def discover_stories_for_analysis(result: AnalysisResult, cluster_count: int) -> StoryAnalysis:
    articles = result.article_results.reset_index(drop=True)
    if result.feature_status.get("embeddings") != "Available":
        return StoryAnalysis("Article embeddings are unavailable. Run NLP analysis first.", [], np.empty((0, 2)), [])
    frame = articles.copy()
    vectors = result.embeddings
    if len(vectors) != len(frame):
        return StoryAnalysis("Article embeddings are not aligned with this dataset.", [], np.empty((0, 2)), [])
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    fingerprint = pd.util.hash_pandas_object(frame[[column for column in ("id", "text_clean") if column in frame]], index=False).values.tobytes()
    cache_key = hashlib.sha256(EMBEDDING_MODEL.encode() + fingerprint + vectors.tobytes()).hexdigest()
    return discover_stories_cached(
        cache_key,
        records,
        vectors,
        cluster_count,
        result.exact_duplicate_groups,
        result.near_duplicate_pairs,
    )
