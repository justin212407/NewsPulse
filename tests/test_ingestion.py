import json

from newspulse.ingestion import parse_records


def test_parse_csv_with_article_schema() -> None:
    content = b"id,title,text,source,published_at,url\n1,Headline,Body,Wire,2025-01-01,https://example.com"
    result = parse_records(content, "articles.csv")
    assert result.error is None
    assert result.records == [
        {
            "id": "1",
            "title": "Headline",
            "text": "Body",
            "source": "Wire",
            "published_at": "2025-01-01",
            "url": "https://example.com",
        }
    ]


def test_parse_json_list_and_envelope() -> None:
    articles = [{"id": "1", "title": "Headline", "text": "Body"}]
    for payload in (articles, {"articles": articles}):
        result = parse_records(json.dumps(payload).encode(), "articles.json")
        assert result.records[0]["id"] == "1"
        assert result.error is None


def test_parse_invalid_format_and_encoding() -> None:
    assert "Unsupported" in parse_records(b"", "articles.txt").error
    assert "UTF-8" in parse_records(b"\xff", "articles.csv").error


def test_parse_malformed_json() -> None:
    result = parse_records(b"{not json", "articles.json")
    assert result.records == []
    assert result.error.startswith("Could not parse JSON")


def test_parse_non_object_json_records_with_warning() -> None:
    result = parse_records(json.dumps([{"id": "1", "title": "T", "text": "B"}, 7]).encode(), "articles.json")
    assert len(result.records) == 1
    assert result.error == "Ignored 1 non-object JSON record(s)."
