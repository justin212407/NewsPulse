import csv
import io
import json
from dataclasses import dataclass

from newspulse.schema import ARTICLE_FIELDS


@dataclass(frozen=True)
class ParseResult:
    records: list[dict[str, object]]
    error: str | None = None


def _normalize_record(record: object) -> dict[str, object] | None:
    if not isinstance(record, dict):
        return None
    normalized = {str(key).strip(): value for key, value in record.items()}
    return {field: normalized.get(field) for field in ARTICLE_FIELDS}


def parse_records(content: bytes, filename: str) -> ParseResult:
    suffix = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    if suffix not in {"csv", "json"}:
        return ParseResult([], "Unsupported file type. Upload a CSV or JSON file.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ParseResult([], "The file must use UTF-8 encoding.")

    try:
        if suffix == "csv":
            records = list(csv.DictReader(io.StringIO(text)))
            if not records and text.strip():
                return ParseResult([], "The CSV must include a header row and at least one article record.")
        else:
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get("articles", [payload])
            if not isinstance(payload, list):
                return ParseResult([], "JSON must be an article object or a list of article objects.")
            records = payload
    except (csv.Error, json.JSONDecodeError) as error:
        return ParseResult([], f"Could not parse {suffix.upper()} file: {error}")

    normalized_records = [_normalize_record(record) for record in records]
    invalid_records = sum(record is None for record in normalized_records)
    valid_records = [record for record in normalized_records if record is not None]
    return ParseResult(
        valid_records,
        f"Ignored {invalid_records} non-object JSON record(s)." if invalid_records else None,
    )
