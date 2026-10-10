import pandas as pd
import plotly.express as px
import streamlit as st

from newspulse.dataset_service import Dataset
from newspulse.nlp.config import DEFAULT_STORY_CLUSTERS, MAX_STORY_CLUSTERS
from newspulse.nlp.story_service import discover_stories_for_analysis
from newspulse.ui.components import render_dataset_state
from newspulse.ui.layout import render_page_header
from newspulse.ui.nlp_analysis import get_cached_analysis, run_cached_analysis


def render_story_explorer(dataset: Dataset) -> None:
    render_page_header(
        "Story Explorer",
        "Group related articles by semantic content, inspect each story, and see how coverage overlaps.",
        "Semantic story discovery",
    )
    if not render_dataset_state(dataset):
        return
    articles = dataset.articles

    result = get_cached_analysis(articles)
    if result is None or result.feature_status.get("embeddings") != "Available":
        st.info("Story discovery uses the cached sentence embeddings produced by NLP analysis.")
        if st.button("Run NLP analysis", key="stories_analyze"):
            with st.spinner("Generating article embeddings and analysis..."):
                try:
                    result = run_cached_analysis(articles)
                except Exception as error:
                    st.error(f"Analysis could not start: {type(error).__name__}: {error}")
                    return
    if result is None:
        return
    if result.feature_status.get("embeddings") != "Available":
        st.warning(result.feature_errors.get("embeddings", "Embeddings are unavailable. Check NLP model setup."))
        return

    valid_count = len(result.valid_indices)
    if valid_count < 3:
        st.info("At least three articles with usable text are needed to build a 2D story map.")
        return
    maximum = min(MAX_STORY_CLUSTERS, valid_count)
    default_count = min(DEFAULT_STORY_CLUSTERS, maximum)
    cluster_count = st.slider("Number of story groups", 2, maximum, default_count, key="story_cluster_count")
    story_result = discover_stories_for_analysis(result, cluster_count)
    if story_result.status != "Available":
        st.warning(story_result.status)
        return

    clustered = result.article_results.iloc[result.valid_indices].copy().reset_index(drop=True)
    clustered["Story"] = [
        next(cluster.label for cluster in story_result.clusters if cluster.cluster_id == group)
        for group in story_result.assignments
    ]
    clustered["x"] = story_result.coordinates[:, 0]
    clustered["y"] = story_result.coordinates[:, 1]
    figure = px.scatter(
        clustered,
        x="x",
        y="y",
        color="Story",
        hover_name="title",
        hover_data={"source": True, "id": True, "x": False, "y": False},
        title="Semantic map of article stories (qualitative 2D projection)",
    )
    figure.update_traces(marker={"size": 13, "opacity": 0.82, "line": {"width": 1, "color": "white"}})
    figure.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=48, b=8),
        legend_title_text="Story",
        xaxis={"visible": False, "title": None},
        yaxis={"visible": False, "title": None},
        annotations=[
            {"text": "Closer points have more similar embedding representations; axes have no absolute units.", "xref": "paper", "yref": "paper", "x": 0, "y": -0.06, "showarrow": False, "font": {"size": 11, "color": "#64716e"}}
        ],
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    cluster_options = {f"{cluster.label} ({cluster.article_count})": cluster for cluster in story_result.clusters}
    selected_label = st.selectbox("Inspect a story", list(cluster_options), key="story_selected")
    cluster = cluster_options[selected_label]
    metrics = st.columns(3)
    metrics[0].metric("Articles", cluster.article_count)
    metrics[1].metric("Share of collection", f"{cluster.percentage:.1f}%")
    metrics[2].metric("Near-duplicate links", len(cluster.near_duplicate_ids))
    st.write("**Keywords and phrases**")
    st.write(" · ".join(cluster.keywords) if cluster.keywords else "No keywords available.")
    st.write("**Representative titles used for this story label**")
    st.write(" · ".join(cluster.representative_titles) if cluster.representative_titles else "No representative titles available.")
    st.write("**Sentiment distribution**")
    if cluster.sentiment_counts:
        st.write(" · ".join(f"{label}: {count}" for label, count in cluster.sentiment_counts.items()))
    else:
        st.caption("No sentiment predictions are available for this story.")

    cluster_articles = result.article_results.loc[
        result.article_results["id"].astype(str).isin(cluster.article_ids)
    ]
    st.write("**Representative articles**")
    for article_id in cluster.representative_ids:
        match = cluster_articles.loc[cluster_articles["id"].astype(str) == article_id]
        if not match.empty:
            article = match.iloc[0]
            date = article["published_at"].strftime("%b %d, %Y") if pd.notna(article["published_at"]) else "Date unavailable"
            with st.container(border=True):
                st.markdown(f"**{article['title']}**")
                st.caption(f"{article['source']}  |  {date}")
                st.write(article["text_clean"][:240])

    duplicate_ids = set(cluster.duplicate_ids) | set(cluster.near_duplicate_ids)
    if duplicate_ids:
        st.write("**Duplicate coverage in this story**")
        for _, duplicate in cluster_articles.loc[cluster_articles["id"].astype(str).isin(duplicate_ids)].iterrows():
            kind = "Exact duplicate" if duplicate["id"] in cluster.duplicate_ids else "Near duplicate"
            st.caption(f"{kind}: {duplicate['title']}")
