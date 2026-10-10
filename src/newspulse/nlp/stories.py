from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from newspulse.nlp.config import DEFAULT_STORY_CLUSTERS, MAX_STORY_CLUSTERS
from newspulse.nlp.features import extract_keywords


@dataclass(frozen=True)
class StoryCluster:
    cluster_id: int
    label: str
    article_count: int
    percentage: float
    sentiment_counts: dict[str, int]
    keywords: list[str]
    representative_titles: list[str]
    representative_ids: list[str]
    duplicate_ids: list[str]
    near_duplicate_ids: list[str]
    article_ids: list[str]


@dataclass(frozen=True)
class StoryAnalysis:
    status: str
    clusters: list[StoryCluster]
    coordinates: np.ndarray
    assignments: list[int]


def _project_umap(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    import umap

    neighbors = min(15, max(2, len(vectors) - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=neighbors, random_state=random_state, transform_seed=random_state)
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def _cluster_kmeans(vectors: np.ndarray, count: int, random_state: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=count, random_state=random_state, n_init=10).fit_predict(vectors)


def _cluster_label(keywords: list[str], titles: list[str]) -> str:
    useful = [term.title() for term in keywords[:2] if term]
    for title in titles:
        words = title.strip().split()
        if words:
            title_fragment = " ".join(words[:6])
            return " · ".join([*useful, title_fragment]) if useful else title_fragment
    return " / ".join(useful) if useful else "Untitled story"


def discover_stories(
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_count: int | None = None,
    *,
    projector: Callable[[np.ndarray], np.ndarray] | None = None,
    clusterer: Callable[[np.ndarray, int], np.ndarray] | None = None,
    random_state: int = 42,
    exact_duplicate_groups: list[list[int]] | None = None,
    near_duplicate_pairs: list[tuple[int, int, float]] | None = None,
) -> StoryAnalysis:
    frame = articles.reset_index(drop=True).copy()
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(frame):
        return StoryAnalysis("Article embeddings are unavailable or misaligned.", [], np.empty((0, 2)), [])
    if "text_clean" not in frame:
        return StoryAnalysis("Article text is unavailable.", [], np.empty((0, 2)), [])

    valid = frame["text_clean"].fillna("").astype(str).str.strip().ne("").to_numpy()
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return StoryAnalysis("No articles with usable text.", [], np.empty((0, 2)), [])
    if len(valid_indices) < 3:
        return StoryAnalysis("At least three articles with usable text are needed for a 2D story map.", [], np.empty((0, 2)), [])

    usable_vectors = vectors[valid_indices]
    count = cluster_count or min(DEFAULT_STORY_CLUSTERS, max(2, round(np.sqrt(len(valid_indices)))))
    count = min(MAX_STORY_CLUSTERS, max(2, min(int(count), len(valid_indices))))
    project = projector or (lambda matrix: _project_umap(matrix, random_state))
    cluster = clusterer or (lambda matrix, groups: _cluster_kmeans(matrix, groups, random_state))
    try:
        coordinates = np.asarray(project(usable_vectors), dtype=np.float32)
        assignments = np.asarray(cluster(usable_vectors, count), dtype=np.int64)
    except Exception as error:
        return StoryAnalysis(f"Story discovery failed: {type(error).__name__}: {error}", [], np.empty((0, 2)), [])
    if coordinates.shape != (len(valid_indices), 2) or not np.isfinite(coordinates).all():
        return StoryAnalysis("Story projection returned invalid coordinates.", [], np.empty((0, 2)), [])
    if assignments.shape != (len(valid_indices),):
        return StoryAnalysis("Story clustering returned invalid assignments.", [], np.empty((0, 2)), [])

    frame["_story_cluster"] = pd.Series(dtype="int64")
    frame.loc[valid_indices, "_story_cluster"] = assignments
    exact_groups = exact_duplicate_groups or _duplicate_groups(frame, "exact_duplicate_groups")
    near_groups = [
        [int(left), int(right)] for left, right, _score in (near_duplicate_pairs or [])
    ]
    clusters: list[StoryCluster] = []
    denominator = len(valid_indices)
    for cluster_id in sorted(set(int(value) for value in assignments)):
        indexes = [int(valid_indices[position]) for position, value in enumerate(assignments) if int(value) == cluster_id]
        group = frame.iloc[indexes]
        keywords = extract_keywords(group["text_clean"].tolist(), limit=8)
        keyword_counts: dict[str, int] = {}
        for article_keywords in keywords:
            for term in article_keywords:
                keyword_counts[term] = keyword_counts.get(term, 0) + 1
        cluster_count_total = max(1, len(set(int(value) for value in assignments)))
        ranked_keywords = sorted(
            (term for term, frequency in keyword_counts.items() if frequency < cluster_count_total),
            key=lambda term: (-keyword_counts[term], term),
        )[:8]
        titles = group.get("title", pd.Series([""] * len(group), index=group.index)).fillna("").astype(str).tolist()
        sentiment_counts: dict[str, int] = {}
        if "sentiment" in group:
            for sentiment in group["sentiment"]:
                if isinstance(sentiment, dict) and sentiment.get("label"):
                    label = str(sentiment["label"]).capitalize()
                    sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        member_positions = [position for position, index in enumerate(valid_indices) if int(index) in indexes]
        member_vectors = usable_vectors[member_positions]
        centroid = member_vectors.mean(axis=0)
        norms = np.linalg.norm(member_vectors, axis=1) * np.linalg.norm(centroid)
        similarities = np.divide(member_vectors @ centroid, norms, out=np.zeros(len(member_vectors), dtype=np.float32), where=norms != 0)
        representative_order = [
            indexes[position]
            for position in np.argsort(similarities)[::-1][:3]
        ]
        ids = group["id"].astype(str).tolist() if "id" in group else [str(index) for index in indexes]
        duplicate_ids = _ids_in_groups(indexes, exact_groups, frame)
        near_duplicate_ids = _ids_in_groups(indexes, near_groups, frame)
        member_ids = set(ids)
        duplicate_ids = [article_id for article_id in duplicate_ids if article_id in member_ids]
        near_duplicate_ids = [article_id for article_id in near_duplicate_ids if article_id in member_ids]
        clusters.append(
            StoryCluster(
                cluster_id=cluster_id,
                label=_cluster_label(ranked_keywords, [str(frame.iloc[index].get("title", "")) for index in representative_order] or titles),
                article_count=len(indexes),
                percentage=100.0 * len(indexes) / denominator,
                sentiment_counts=sentiment_counts,
                keywords=ranked_keywords,
                representative_titles=[str(frame.iloc[index].get("title", "Untitled article")) for index in representative_order],
                representative_ids=[str(frame.iloc[index].get("id", index)) for index in representative_order],
                duplicate_ids=duplicate_ids,
                near_duplicate_ids=near_duplicate_ids,
                article_ids=ids,
            )
        )
    return StoryAnalysis(
        "Available",
        clusters,
        coordinates,
        [int(value) for value in assignments],
    )


def _duplicate_groups(frame: pd.DataFrame, field: str) -> list[list[int]]:
    groups: list[list[int]] = []
    if field == "exact_duplicate_groups":
        from newspulse.nlp.features import exact_duplicate_groups

        return exact_duplicate_groups(frame["text_clean"].tolist())
    if field == "near_duplicate_pairs" and "_near_duplicate_pairs" in frame.attrs:
        for left, right, _score in frame.attrs["_near_duplicate_pairs"]:
            groups.append([int(left), int(right)])
    return groups


def _ids_in_groups(indexes: list[int], groups: list[list[int]], frame: pd.DataFrame) -> list[str]:
    members = sorted({member for group in groups if any(index in group for index in indexes) for member in group})
    return [str(frame.iloc[index].get("id", index)) for index in members]


@st.cache_data(show_spinner=False)
def discover_stories_cached(
    cache_key: str,
    records: list[dict[str, object]],
    embeddings: np.ndarray,
    cluster_count: int,
    exact_groups: list[list[int]],
    near_pairs: list[tuple[int, int, float]],
) -> StoryAnalysis:
    del cache_key
    return discover_stories(
        pd.DataFrame.from_records(records),
        embeddings,
        cluster_count,
        exact_duplicate_groups=exact_groups,
        near_duplicate_pairs=near_pairs,
    )
