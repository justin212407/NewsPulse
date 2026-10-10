import numpy as np
import pandas as pd

from newspulse.nlp.search import search_articles


def test_semantic_search_returns_top_k_article_metadata_and_excerpt() -> None:
    articles = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "title": ["Flood defenses", "Battery storage", "Harbor flood plan", "No text"],
            "source": ["Harbor News", "Energy Desk", "City Wire", "Unknown"],
            "published_at": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", None], utc=True),
            "text_clean": ["coastal flooding defenses", "battery energy grid", "harbor flood prevention", ""],
        }
    )
    article_vectors = np.array([[1, 0], [0, 1], [.9, .1], [0, 0]], dtype=np.float32)
    result = search_articles(
        "flooding near the harbor",
        articles,
        article_vectors,
        top_k=2,
        embed_query=lambda _query: np.array([[1, 0]], dtype=np.float32),
    )
    assert result.status == "Available"
    assert [item.article_id for item in result.results] == ["a", "c"]
    assert result.results[0].title == "Flood defenses"
    assert result.results[0].source == "Harbor News"
    assert result.results[0].published_at is not None
    assert 0.0 <= result.results[0].score <= 1.0
    assert "coastal flooding defenses" in result.results[0].excerpt


def test_semantic_search_handles_blank_query_and_missing_embeddings() -> None:
    articles = pd.DataFrame({"id": ["a"], "title": ["A"], "text_clean": ["text"]})
    empty_query = search_articles("  ", articles, np.array([[1, 0]], dtype=np.float32))
    assert empty_query.status == "Enter a search query."
    no_vectors = search_articles("query", articles, None)
    assert no_vectors.status == "Article embeddings are not available. Run NLP analysis first."


def test_semantic_search_handles_query_embedding_failure_and_no_valid_text() -> None:
    articles = pd.DataFrame({"id": ["a"], "title": ["A"], "text_clean": [""]})
    no_text = search_articles("query", articles, np.array([[0.0, 0.0]], dtype=np.float32))
    assert no_text.status == "No articles with usable text."

    def fail(_query: str) -> np.ndarray:
        raise RuntimeError("weights unavailable")

    failed = search_articles("query", articles.assign(text_clean="article"), np.array([[1, 0]], dtype=np.float32), embed_query=fail)
    assert failed.status.startswith("Query embedding failed:")
