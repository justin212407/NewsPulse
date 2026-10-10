import numpy as np
import pandas as pd

from newspulse.nlp.service import AnalysisResult
from newspulse.nlp.search_service import search_analysis


def test_search_analysis_uses_article_vectors_and_returns_source_metadata() -> None:
    articles = pd.DataFrame(
        {
            "id": ["a", "b"],
            "title": ["Flood defenses", "Grid batteries"],
            "source": ["Harbor News", "Energy Desk"],
            "published_at": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "text_clean": ["coastal flood protection", "battery storage"],
        }
    )
    result = AnalysisResult(
        article_results=articles,
        dataset_keywords=[],
        sentiment_counts={},
        exact_duplicate_groups=[],
        near_duplicate_pairs=[],
        similarity=np.eye(2, dtype=np.float32),
        embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32),
        valid_indices=[0, 1],
        feature_errors={},
        feature_status={"embeddings": "Available"},
    )
    search = search_analysis(result, "flood response", 1, embed_query=lambda _query: np.array([[1, 0]]))
    assert search.status == "Available"
    assert len(search.results) == 1
    assert search.results[0].title == "Flood defenses"
    assert search.results[0].source == "Harbor News"
    assert search.results[0].article_id == "a"
