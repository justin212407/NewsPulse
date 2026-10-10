import numpy as np
import pandas as pd

from newspulse.nlp.temporal import analyze_temporal_intelligence


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "title": ["AI rules early", "AI rules early two", "Energy update", "AI rules later", "Energy update two", "Undated item"],
            "text_clean": ["ai regulation policy", "ai regulation safety", "energy grid battery", "ai regulation safety law", "energy battery storage", "undated text"],
            "published_at": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-11", "2025-01-12", None], utc=True),
            "source": ["A", "A", "B", "A", "B", "B"],
            "sentiment": [
                {"label": "neutral", "score": .7},
                {"label": "negative", "score": .8},
                {"label": "positive", "score": .9},
                {"label": "negative", "score": .9},
                {"label": "positive", "score": .8},
                {},
            ],
            "entities": [
                [{"text": "OpenAI", "label": "ORG", "score": .9}, {"text": "EU", "label": "ORG", "score": .8}],
                [{"text": "OpenAI", "label": "ORG", "score": .8}],
                [{"text": "Grid", "label": "ORG", "score": .7}],
                [{"text": "OpenAI", "label": "ORG", "score": .9}, {"text": "EU", "label": "ORG", "score": .9}],
                [{"text": "Grid", "label": "ORG", "score": .8}],
                [{"text": "OpenAI", "label": "ORG", "score": .7}],
            ],
            "story_id": [0, 0, 1, 0, 1, 0],
            "story_label": ["AI Regulation"] * 2 + ["Energy Storage"] + ["AI Regulation"] + ["Energy Storage"] + ["AI Regulation"],
        }
    )


def test_temporal_intelligence_reports_counts_shares_emergence_and_decline() -> None:
    result = analyze_temporal_intelligence(
        _frame(),
        np.array([[1, 0], [.9, .1], [0, 1], [.8, .2], [.1, .9], [.7, .3]], dtype=np.float32),
        period_days=7,
    )
    assert result.undated_articles == 1
    assert result.story_frequency
    assert result.sentiment_evolution
    assert {item["story_label"] for item in result.emerging_stories} == {"Energy Storage"}
    assert {item["story_label"] for item in result.declining_stories} == {"AI Regulation"}
    assert result.period_shifts
    shift = next(item for item in result.period_shifts if item["story_label"] == "AI Regulation")
    assert shift["percentage_point_change"] < 0
    assert shift["representative_ids"]
    assert shift["keywords"]
    assert 0.0 <= result.semantic_shift["cosine_distance"] <= 2.0
    assert result.semantic_shift["previous_article_count"] == 3
    shares_by_period: dict[object, float] = {}
    for row in result.story_frequency:
        shares_by_period[row["period"]] = shares_by_period.get(row["period"], 0.0) + row["share"]
    assert all(np.isclose(total, 1.0) for total in shares_by_period.values())


def test_entity_timeseries_sentiment_and_document_cooccurrence() -> None:
    result = analyze_temporal_intelligence(
        _frame(),
        np.array([[1, 0], [.9, .1], [0, 1], [.8, .2], [.1, .9], [.7, .3]], dtype=np.float32),
        period_days=7,
    )
    assert result.entity_frequency
    assert result.entity_sentiment
    sentiment_shares: dict[tuple[object, str], float] = {}
    for item in result.entity_sentiment:
        key = (item["period"], item["entity"])
        sentiment_shares[key] = sentiment_shares.get(key, 0.0) + item["share"]
    assert all(np.isclose(share, 1.0) for share in sentiment_shares.values())
    assert result.entity_cooccurrence
    pair = next(item for item in result.entity_cooccurrence if {item["left"], item["right"]} == {"OpenAI", "EU"})
    assert pair["article_count"] == 2
    assert pair["document_share"] == 0.4
    period_pairs = [item for item in result.entity_cooccurrence if item.get("period") is not None and {item["left"], item["right"]} == {"OpenAI", "EU"}]
    assert period_pairs
    assert all(item["article_count"] == 1 for item in period_pairs)


def test_calendar_months_and_selected_comparison_split_are_respected() -> None:
    result = analyze_temporal_intelligence(
        _frame(),
        np.array([[1, 0], [.9, .1], [0, 1], [.8, .2], [.1, .9], [.7, .3]], dtype=np.float32),
        calendar_months=True,
        comparison_split=int(pd.Timestamp("2025-01-05", tz="UTC").timestamp()),
    )
    assert {row["period"].strftime("%Y-%m-%d") for row in result.story_frequency} == {"2025-01-01"}
    assert result.semantic_shift["status"] == "Available"
    assert result.semantic_shift["previous_article_count"] == 3
    assert result.semantic_shift["current_article_count"] == 2
    assert result.period_shifts
    assert result.periods is not None


def test_explicit_cutoff_compares_articles_inside_one_interval() -> None:
    frame = _frame().iloc[:5].copy()
    vectors = np.array([[1, 0], [.95, .05], [0, 1], [.8, .2], [.05, .95]], dtype=np.float32)
    result = analyze_temporal_intelligence(
        frame,
        vectors,
        period_days=30,
        comparison_split=int(pd.Timestamp("2025-01-05", tz="UTC").timestamp()),
    )
    assert len({row["period"] for row in result.story_frequency}) == 1
    assert result.period_shifts
    assert result.semantic_shift["status"] == "Available"
    assert result.semantic_shift["previous_article_count"] == 3
    assert result.semantic_shift["current_article_count"] == 2


def test_empty_dates_and_unavailable_annotations_return_explicit_empty_results() -> None:
    frame = _frame().iloc[[5]].copy()
    result = analyze_temporal_intelligence(frame, np.zeros((1, 2), dtype=np.float32))
    assert result.undated_articles == 1
    assert result.story_frequency == []
    assert result.entity_frequency == []
    assert result.period_shifts == []
