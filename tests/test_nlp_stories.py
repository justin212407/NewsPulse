import numpy as np
import pandas as pd

from newspulse.nlp.stories import discover_stories


def _articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "title": [
                "Coastal cities plan flood defenses",
                "Harbor towns map flood protection",
                "Battery storage supports summer grid",
                "Utilities prepare battery storage projects",
                "Duplicate flood report",
                "Blank record",
            ],
            "text_clean": [
                "coastal flood defense mapping",
                "harbor flood protection planning",
                "battery storage energy grid",
                "utility battery storage grid projects",
                "coastal flood defense mapping",
                "",
            ],
            "source": ["A", "B", "C", "D", "A", "A"],
            "published_at": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", None], utc=True),
            "url": ["https://example.com"] * 6,
            "sentiment": [
                {"label": "positive", "score": 0.9},
                {"label": "neutral", "score": 0.7},
                {"label": "negative", "score": 0.8},
                {"label": "positive", "score": 0.8},
                {"label": "positive", "score": 0.9},
                {},
            ],
        }
    )


def _project(vectors: np.ndarray) -> np.ndarray:
    return vectors[:, :2]


def _cluster(vectors: np.ndarray, count: int) -> np.ndarray:
    return np.arange(len(vectors), dtype=np.int64) % count


def test_story_discovery_returns_cluster_summaries_and_duplicate_links() -> None:
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.98, 0.1, 0.0], [0.0, 1.0, 0.0], [0.1, 0.98, 0.0], [1.0, 0.0, 0.0], [0, 0, 0]],
        dtype=np.float32,
    )
    result = discover_stories(
        _articles(), embeddings, cluster_count=2, random_state=11, projector=_project, clusterer=_cluster
    )
    assert result.status == "Available"
    assert len(result.clusters) == 2
    assert sum(cluster.article_count for cluster in result.clusters) == 5
    assert all(cluster.percentage > 0 for cluster in result.clusters)
    assert all(cluster.label and cluster.keywords and cluster.representative_ids for cluster in result.clusters)
    assert all(cluster.representative_titles for cluster in result.clusters)
    assert any(cluster.sentiment_counts for cluster in result.clusters)
    assert any("e" in cluster.duplicate_ids for cluster in result.clusters)
    assert len(result.coordinates) == 5
    assert np.isfinite(result.coordinates).all()


def test_story_discovery_handles_small_and_empty_datasets_without_fake_clusters() -> None:
    articles = _articles().iloc[:1]
    result = discover_stories(articles, np.array([[1.0, 0.0, 0.0]]), cluster_count=2)
    assert result.status == "At least three articles with usable text are needed for a 2D story map."
    assert result.clusters == []

    empty = discover_stories(_articles().iloc[0:0], np.empty((0, 3)))
    assert empty.status == "No articles with usable text."
    assert empty.clusters == []


def test_story_discovery_clamps_cluster_count_to_valid_article_count() -> None:
    articles = _articles().iloc[:4]
    embeddings = np.array([[1, 0], [.99, .01], [0, 1], [.01, .99]], dtype=np.float32)
    result = discover_stories(articles, embeddings, cluster_count=20, projector=_project, clusterer=_cluster)
    assert result.status == "Available"
    assert len(result.clusters) == 4
    assert all(cluster.article_count == 1 for cluster in result.clusters)
