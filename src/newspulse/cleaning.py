import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse

import pandas as pd

from newspulse.schema import ARTICLE_FIELDS, REQUIRED_FIELDS


@dataclass(frozen=True)
class ValidationReport:
    total_rows: int
    accepted_rows: int
    skipped_rows: int
    warning_rows: int
    issues: tuple[str, ...]


def clean_article_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unescape(str(value))
    text = re.sub(r"<\s*br\s*/?\s*>|</\s*p\s*>|</\s*div\s*>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    missing = pd.isna(value)
    return not bool(missing) and bool(str(value).strip())


def _valid_url(value: object) -> bool:
    if not _present(value):
        return True
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def preprocess_records(records: list[dict[str, object]]) -> tuple[pd.DataFrame, ValidationReport]:
    accepted: list[dict[str, object]] = []
    issues: list[str] = []
    seen_ids: set[str] = set()
    warning_rows = 0
    skipped_rows = 0

    for row_number, raw in enumerate(records, start=1):
        row = {field: raw.get(field) for field in ARTICLE_FIELDS}
        for field in ("id", "title", "source", "url"):
            if _present(row[field]):
                row[field] = str(row[field]).strip()
            else:
                row[field] = ""

        missing = [field for field in REQUIRED_FIELDS if not _present(row[field])]
        if missing:
            skipped_rows += 1
            issues.append(f"Row {row_number}: skipped because required fields are missing ({', '.join(missing)}).")
            continue

        article_id = str(row["id"])
        if article_id in seen_ids:
            skipped_rows += 1
            issues.append(f"Row {row_number}: skipped because article ID '{article_id}' is duplicated.")
            continue
        seen_ids.add(article_id)

        original_text = str(row["text"])
        cleaned_text = clean_article_text(original_text)
        if not cleaned_text:
            skipped_rows += 1
            issues.append(f"Row {row_number}: skipped because article text is empty after cleaning.")
            continue

        try:
            parsed_date = pd.to_datetime(row["published_at"], errors="coerce", utc=True)
        except (TypeError, ValueError):
            parsed_date = pd.NaT
        date_value = None if pd.isna(parsed_date) else parsed_date
        if date_value is None:
            warning_rows += 1
            issues.append(f"Row {row_number}: published_at is missing or invalid; the date is unavailable.")

        if not _valid_url(row["url"]):
            warning_rows += 1
            issues.append(f"Row {row_number}: URL is not a valid HTTP(S) URL and was retained as provided.")

        if not row["source"]:
            warning_rows += 1
            issues.append(f"Row {row_number}: source is missing.")

        accepted.append(
            {
                "id": article_id,
                "title": str(row["title"]).strip(),
                "text": original_text,
                "text_clean": cleaned_text,
                "source": row["source"] or "Unknown source",
                "published_at": date_value,
                "url": row["url"],
            }
        )

    frame = pd.DataFrame(
        accepted,
        columns=("id", "title", "text", "text_clean", "source", "published_at", "url"),
    )
    if not frame.empty:
        frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    report = ValidationReport(
        total_rows=len(records),
        accepted_rows=len(accepted),
        skipped_rows=skipped_rows,
        warning_rows=warning_rows,
        issues=tuple(issues),
    )
    return frame, report
