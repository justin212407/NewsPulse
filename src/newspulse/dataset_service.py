from dataclasses import dataclass

import pandas as pd
import streamlit as st

from newspulse.cleaning import ValidationReport, preprocess_records
from newspulse.ingestion import parse_records


@dataclass(frozen=True)
class Dataset:
    articles: pd.DataFrame
    report: ValidationReport
    error: str | None = None


@st.cache_data(show_spinner=False)
def load_dataset(content: bytes, filename: str) -> Dataset:
    if not content:
        return Dataset(pd.DataFrame(), ValidationReport(0, 0, 0, 0, ()))
    parsed = parse_records(content, filename)
    if parsed.error and not parsed.records:
        return Dataset(pd.DataFrame(), ValidationReport(0, 0, 0, 0, ()), parsed.error)
    articles, report = preprocess_records(parsed.records)
    if parsed.error:
        report = ValidationReport(
            report.total_rows,
            report.accepted_rows,
            report.skipped_rows,
            report.warning_rows,
            (parsed.error, *report.issues),
        )
    return Dataset(articles, report)
