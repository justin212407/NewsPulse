import numpy as np
import pandas as pd

from newspulse.nlp.service import ModelAdapters, analyze_articles


def _articles(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"article-{index}" for index in range(len(texts))],
            "text_clean": texts,
        }
    )


def test_analysis_annotates_articles_and_produces_dataset_summary() -> None:
    adapters = ModelAdapters(
        embed=lambda texts: np.array([[1.0, 0.0], [0.99, 0.1]], dtype=np.float32),
        sentiment=lambda texts: [{"label": "positive", "score": 0.91} for _ in texts],
        entities=lambda texts: [[{"text": "River City", "label": "LOC", "score": 0.88}] for _ in texts],
    )
    result = analyze_articles(_articles(["River City expands clean energy", "River City expands clean energy plans"]), adapters)
    assert result.article_results.iloc[0]["sentiment"]["label"] == "positive"
    assert result.article_results.iloc[0]["entities"][0]["text"] == "River City"
    assert result.article_results.iloc[0]["keywords"]
    assert result.sentiment_counts == {"Positive": 2}
    assert result.feature_status["embeddings"] == "Available"
    assert result.near_duplicate_pairs[0][:2] == (0, 1)


def test_model_failures_are_isolated_and_keep_deterministic_features() -> None:
    def fail(_texts: list[str]) -> list[dict[str, object]]:
        raise RuntimeError("model unavailable")

    result = analyze_articles(
        _articles(["Climate policy update", "Climate policy report"]),
        ModelAdapters(
            embed=lambda _texts: (_ for _ in ()).throw(RuntimeError("weights unavailable")),
            sentiment=fail,
            entities=fail,
        ),
    )
    assert result.article_results.iloc[0]["keywords"]
    assert result.feature_status["embeddings"] == "Unavailable"
    assert result.feature_status["sentiment"] == "Unavailable"
    assert result.feature_status["entities"] == "Unavailable"
    assert {"embeddings", "sentiment", "entities"} == set(result.feature_errors)
    assert result.feature_status["exact_duplicates"] == "Available"


def test_empty_and_single_article_datasets_are_safe() -> None:
    empty = analyze_articles(_articles([]))
    assert empty.article_results.empty
    assert empty.similarity.shape == (0, 0)
    assert empty.feature_status["embeddings"] == "No usable article text."

    one = analyze_articles(
        _articles(["A" ]),
        ModelAdapters(
            embed=lambda texts: np.array([[1.0, 0.0]], dtype=np.float32),
            sentiment=lambda texts: [{"label": "neutral", "score": 0.5}],
            entities=lambda texts: [[]],
        ),
    )
    assert one.similarity.shape == (1, 1)
    assert one.article_results.iloc[0]["similar_articles"] == []


def test_missing_text_is_not_sent_to_models() -> None:
    seen: list[list[str]] = []

    def capture(texts: list[str]) -> list[dict[str, object]]:
        seen.append(texts)
        return [{"label": "neutral", "score": 0.5} for _ in texts]

    result = analyze_articles(
        _articles(["", "short"]),
        ModelAdapters(
            embed=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
            sentiment=capture,
            entities=lambda texts: [[] for _ in texts],
        ),
    )
    assert seen == [["short"]]
    assert result.article_results.iloc[0]["sentiment"] == {}
    assert result.article_results.iloc[0]["entities"] == []
