import pandas as pd
import streamlit as st

from newspulse.dataset_service import Dataset


def render_dataset_state(dataset: Dataset) -> bool:
    render_validation_summary(dataset)
    if dataset.error:
        return False
    if dataset.report.total_rows > 0:
        st.caption(
            f"Data check: {dataset.report.accepted_rows} accepted, "
            f"{dataset.report.skipped_rows} skipped, {dataset.report.warning_rows} warning(s)."
        )
    if dataset.articles.empty:
        st.info("Load the sample collection or upload a valid CSV/JSON dataset to continue.")
        return False
    return True


def render_validation_summary(dataset: Dataset) -> None:
    report = dataset.report
    if dataset.error:
        st.error(dataset.error)
        return
    if report.total_rows == 0:
        st.info("No article records were found in this dataset.")
        return
    with st.expander("Data quality and import details", expanded=bool(report.issues)):
        st.write(
            f"{report.accepted_rows} accepted of {report.total_rows} records. "
            f"{report.skipped_rows} skipped, {report.warning_rows} warning(s)."
        )
        if report.issues:
            for issue in report.issues:
                st.caption(issue)
        else:
            st.caption("All records passed validation.")


def format_date_range(articles: pd.DataFrame) -> str:
    if articles.empty or articles["published_at"].dropna().empty:
        return "Unavailable"
    dates = articles["published_at"].dropna()
    start = dates.min().strftime("%b %d, %Y")
    end = dates.max().strftime("%b %d, %Y")
    return start if start == end else f"{start} to {end}"


def render_article_detail(article: pd.Series) -> None:
    st.subheader(str(article["title"]))
    date = article["published_at"]
    date_text = date.strftime("%b %d, %Y · %H:%M UTC") if pd.notna(date) else "Date unavailable"
    st.caption(f"{article['source']}  |  {date_text}  |  ID {article['id']}")
    text = str(article["text_clean"])
    if len(text) > 900:
        st.write(text[:420].rstrip() + "…")
        with st.expander("Read full article"):
            st.write(text)
    else:
        st.write(text)
    if article["url"]:
        st.link_button("Open source article", str(article["url"]))


def searchable_articles(articles: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        return articles
    needle = query.strip().casefold()
    mask = (
        articles["title"].str.casefold().str.contains(needle, regex=False)
        | articles["text_clean"].str.casefold().str.contains(needle, regex=False)
        | articles["source"].str.casefold().str.contains(needle, regex=False)
    )
    return articles.loc[mask]
