from newspulse.dataset_service import load_dataset


def test_load_sample_data_produces_articles_and_report() -> None:
    from pathlib import Path

    content = (Path(__file__).parents[1] / "data" / "sample_articles.csv").read_bytes()
    dataset = load_dataset(content, "sample_articles.csv")
    assert dataset.error is None
    assert len(dataset.articles) == 12
    assert dataset.report.accepted_rows == 12
    assert dataset.articles["published_at"].notna().all()


def test_empty_upload_returns_empty_dataset() -> None:
    dataset = load_dataset(b"", "empty.csv")
    assert dataset.error is None
    assert dataset.articles.empty
    assert dataset.report.total_rows == 0


def test_parse_error_is_returned_to_page() -> None:
    dataset = load_dataset(b"invalid", "data.txt")
    assert dataset.articles.empty
    assert dataset.error == "Unsupported file type. Upload a CSV or JSON file."
