from pathlib import Path

import pandas as pd

from newspulse.dataset_service import load_dataset
from newspulse.nlp.features import exact_duplicate_groups


def test_sample_collection_is_synthetic_and_supports_temporal_comparison() -> None:
    path = Path(__file__).parents[1] / "data" / "sample_articles.csv"
    dataset = load_dataset(path.read_bytes(), path.name)
    assert dataset.error is None
    articles = dataset.articles
    assert len(articles) >= 30
    assert articles["source"].nunique() >= 5
    assert articles["published_at"].dt.month.nunique() >= 6
    assert articles["published_at"].min() < pd.Timestamp("2025-03-01", tz="UTC")
    assert articles["published_at"].max() > pd.Timestamp("2025-08-01", tz="UTC")
    assert articles["text_clean"].str.contains("synthetic sample", case=False).all()
    assert len(exact_duplicate_groups(articles["text_clean"])) >= 1

    boundary = pd.Timestamp("2025-05-01", tz="UTC")
    earlier = articles.loc[articles["published_at"] < boundary, "text_clean"].str.casefold()
    later = articles.loc[articles["published_at"] >= boundary, "text_clean"].str.casefold()
    earlier_total = len(earlier)
    later_total = len(later)
    assert earlier_total > 0 and later_total > 0
    regulation_earlier = earlier.str.contains(r"regulation|safety|audit|legislation", regex=True).sum()
    regulation_later = later.str.contains(r"regulation|safety|audit|legislation", regex=True).sum()
    assert regulation_later / later_total > regulation_earlier / earlier_total
