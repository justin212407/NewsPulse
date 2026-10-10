import pandas as pd
import streamlit as st

from newspulse.dataset_service import Dataset
from newspulse.ui.components import render_article_detail, render_dataset_state, searchable_articles
from newspulse.ui.layout import render_page_header
from newspulse.ui.nlp_analysis import article_analysis_for_id, render_analysis_controls, render_article_analysis


def render_article_explorer(dataset: Dataset) -> None:
    render_page_header(
        "Article Explorer",
        "Search and inspect the source records in your collection. Original text is preserved alongside its cleaned display form.",
        "Article library",
    )
    if not render_dataset_state(dataset):
        return
    articles = dataset.articles

    search_col, source_col = st.columns([1.5, 1])
    with search_col:
        query = st.text_input("Search articles", placeholder="Search titles, text, or sources")
    sources = ["All sources", *sorted(articles["source"].unique().tolist())]
    with source_col:
        selected_source = st.selectbox("Source", sources)

    results = searchable_articles(articles, query)
    if selected_source != "All sources":
        results = results.loc[results["source"] == selected_source]
    results = results.sort_values("published_at", ascending=False, na_position="last")
    st.caption(f"{len(results):,} matching article(s)")
    if results.empty:
        st.info("No articles match these filters.")
        return

    labels = {
        f"{row['title']}  |  {row['source']}  |  "
        f"{row['published_at'].strftime('%Y-%m-%d') if pd.notna(row['published_at']) else 'Date unavailable'}": row["id"]
        for _, row in results.iterrows()
    }
    selected_label = st.selectbox("Select an article", list(labels))
    selected_id = labels[selected_label]
    article = results.loc[results["id"] == selected_id].iloc[0]
    with st.container(border=True):
        render_article_detail(article)
    with st.expander("View preserved original text"):
        st.text(str(article["text"]))

    result = render_analysis_controls(articles, "article_explorer")
    if result is not None:
        analysis = article_analysis_for_id(result, selected_id)
        if analysis is not None:
            with st.container(border=True):
                render_article_analysis(analysis, result, result.article_results)
