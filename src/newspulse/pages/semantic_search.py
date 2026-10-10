import pandas as pd
import streamlit as st

from newspulse.dataset_service import Dataset
from newspulse.nlp.config import DEFAULT_SEARCH_TOP_K
from newspulse.nlp.search_service import search_analysis
from newspulse.ui.components import render_dataset_state
from newspulse.ui.layout import render_page_header
from newspulse.ui.nlp_analysis import get_cached_analysis, run_cached_analysis


def render_semantic_search(dataset: Dataset) -> None:
    render_page_header(
        "Semantic Search",
        "Find articles by meaning using sentence embeddings, not exact keyword matches.",
        "Search the collection",
    )
    if not render_dataset_state(dataset):
        return
    articles = dataset.articles

    analysis = get_cached_analysis(articles)
    if analysis is None or analysis.feature_status.get("embeddings") != "Available":
        st.info("Semantic search requires article embeddings. Run NLP analysis once to build the cached vectors.")
        if st.button("Run NLP analysis", key="search_analyze"):
            with st.spinner("Generating cached article embeddings..."):
                try:
                    analysis = run_cached_analysis(articles)
                except Exception as error:
                    st.error(f"Analysis could not start: {type(error).__name__}: {error}")
                    return
    if analysis is None or analysis.feature_status.get("embeddings") != "Available":
        if analysis is not None:
            st.warning(analysis.feature_errors.get("embeddings", "Article embeddings are unavailable."))
        return

    with st.form("semantic_search_form"):
        query = st.text_input("Search by topic or idea", placeholder="For example, community plans for flood protection")
        top_k = st.slider("Results", 3, 25, DEFAULT_SEARCH_TOP_K)
        submitted = st.form_submit_button("Search articles")
    if not submitted:
        st.caption("Enter a question or topic to rank the most semantically related articles.")
        return

    results = search_analysis(analysis, query, top_k)
    if results.status != "Available":
        st.warning(results.status)
        return
    st.caption(f"Top {len(results.results)} result(s), ranked by cosine similarity")
    for result in results.results:
        date = result.published_at.strftime("%b %d, %Y") if result.published_at is not None and pd.notna(result.published_at) else "Date unavailable"
        with st.container(border=True):
            title_col, score_col = st.columns([4, 1])
            title_col.markdown(f"**{result.title}**")
            score_col.metric("Similarity", f"{result.score:.1%}")
            st.caption(f"{result.source}  |  {date}")
            st.write(result.excerpt)
