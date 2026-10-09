import pandas as pd

from newspulse.cleaning import clean_article_text, preprocess_records


def test_cleaning_preserves_original_and_creates_clean_text() -> None:
    original = "  <p>Floods &amp; cities</p>\n coordinate  "
    articles, report = preprocess_records(
        [{"id": " 1 ", "title": " A headline ", "text": original, "published_at": "2025-02-01T12:30:00-05:00"}]
    )
    assert report.accepted_rows == 1
    assert articles.iloc[0]["id"] == "1"
    assert articles.iloc[0]["title"] == "A headline"
    assert articles.iloc[0]["text"] == original
    assert articles.iloc[0]["text_clean"] == "Floods & cities coordinate"
    assert articles.iloc[0]["published_at"] == pd.Timestamp("2025-02-01T17:30:00Z")


def test_clean_article_text_handles_empty_and_html() -> None:
    assert clean_article_text(None) == ""
    assert clean_article_text("<div>A<br>B</div>") == "A B"


def test_invalid_dates_and_optional_fields_are_retained_as_warnings() -> None:
    articles, report = preprocess_records(
        [{"id": "1", "title": "Headline", "text": "Body", "published_at": "not a date", "url": "ftp://example.com"}]
    )
    assert len(articles) == 1
    assert pd.isna(articles.iloc[0]["published_at"])
    assert articles.iloc[0]["source"] == "Unknown source"
    assert report.warning_rows == 3
    assert any("invalid" in issue for issue in report.issues)


def test_missing_required_fields_and_duplicate_ids_are_skipped() -> None:
    articles, report = preprocess_records(
        [
            {"id": "1", "title": "Headline", "text": "Body"},
            {"id": "1", "title": "Duplicate", "text": "Another body"},
            {"id": "3", "title": "Missing body", "text": "  "},
        ]
    )
    assert list(articles["id"]) == ["1"]
    assert report.total_rows == 3
    assert report.skipped_rows == 2
    assert any("duplicated" in issue for issue in report.issues)
    assert any("missing" in issue for issue in report.issues)


def test_empty_records_return_empty_frame_with_report() -> None:
    articles, report = preprocess_records([])
    assert articles.empty
    assert report.total_rows == 0
