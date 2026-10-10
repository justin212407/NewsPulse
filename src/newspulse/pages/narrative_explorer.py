import pandas as pd
import plotly.express as px
import streamlit as st

from newspulse.dataset_service import Dataset
from newspulse.nlp.story_service import discover_stories_for_analysis
from newspulse.nlp.temporal import analyze_temporal_intelligence
from newspulse.ui.components import render_dataset_state
from newspulse.ui.layout import render_page_header
from newspulse.ui.nlp_analysis import get_cached_analysis, run_cached_analysis


def render_narrative_explorer(dataset: Dataset) -> None:
    render_page_header(
        "Narrative Explorer",
        "Measure how story coverage, sentiment, and semantic representations vary across chronological periods.",
        "Temporal narrative intelligence",
    )
    if not render_dataset_state(dataset):
        return
    articles = dataset.articles

    analysis = get_cached_analysis(articles)
    if analysis is None:
        st.info("Narrative views reuse cached embeddings, sentiment, entities, and story assignments from NLP analysis.")
        if st.button("Run NLP analysis", key="narrative_analyze"):
            with st.spinner("Loading NLP models and analyzing articles..."):
                try:
                    analysis = run_cached_analysis(articles)
                except Exception as error:
                    st.error(f"Analysis could not start: {type(error).__name__}: {error}")
                    return
    if analysis is None:
        return

    interval = st.selectbox("Time interval", ("Daily", "Weekly", "Monthly"), index=1)
    days = {"Daily": 1, "Weekly": 7, "Monthly": 30}[interval]
    dated_articles = articles.dropna(subset=["published_at"])
    comparison_split = None
    if len(dated_articles) >= 2:
        minimum_date = dated_articles["published_at"].min().date()
        maximum_date = dated_articles["published_at"].max().date()
        split_date = st.date_input(
            "Compare periods at",
            value=minimum_date + (maximum_date - minimum_date) / 2,
            min_value=minimum_date,
            max_value=maximum_date,
        )
        comparison_split = int(pd.Timestamp(split_date, tz="UTC").timestamp())
    story_count = 0
    if analysis.feature_status.get("embeddings") == "Available" and len(analysis.valid_indices) >= 3:
        story_count = min(8, len(analysis.valid_indices))
        story_analysis = discover_stories_for_analysis(analysis, story_count)
        if story_analysis.status == "Available":
            frame = analysis.article_results.copy()
            frame["story_id"] = pd.NA
            frame.loc[analysis.valid_indices, "story_id"] = story_analysis.assignments
            label_by_id = {cluster.cluster_id: cluster.label for cluster in story_analysis.clusters}
            frame["story_label"] = frame["story_id"].map(label_by_id)
        else:
            frame = analysis.article_results.copy()
    else:
        frame = analysis.article_results.copy()

    temporal = analyze_temporal_intelligence(
        frame,
        analysis.embeddings,
        period_days=days,
        comparison_split=comparison_split,
        calendar_months=interval == "Monthly",
    )
    st.caption(
        f"Using {len(articles) - temporal.undated_articles} dated articles. "
        f"{temporal.undated_articles} article(s) without valid dates are excluded from temporal comparisons."
    )
    if temporal.story_frequency:
        story_series = pd.DataFrame(temporal.story_frequency)
        story_series["period"] = pd.to_datetime(story_series["period"], utc=True)
        st.subheader("Story coverage over time")
        chart = px.line(
            story_series,
            x="period",
            y="share",
            color="story_label",
            markers=True,
            labels={"period": "Publication period", "share": "Share of dated coverage", "story_label": "Story"},
        )
        chart.update_yaxes(tickformat=".0%")
        chart.update_layout(height=380, margin=dict(l=8, r=8, t=16, b=8), legend_title_text="Story")
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Story timelines are unavailable because embeddings or story assignments could not be generated.")

    trend_left, trend_right = st.columns(2)
    with trend_left:
        st.write("**Emerging stories**")
        if temporal.emerging_stories:
            for story in temporal.emerging_stories[:5]:
                st.caption(f"{story['story_label']}: {story['first_share']:.1%} to {story['last_share']:.1%} ({story['percentage_point_change']:+.1f} pp), {story['first_count']} to {story['last_count']} articles from {story['first_period']} to {story['last_period']}.")
        else:
            st.caption("No stories increased their share between the first and last observed periods.")
    with trend_right:
        st.write("**Declining stories**")
        if temporal.declining_stories:
            for story in temporal.declining_stories[:5]:
                st.caption(f"{story['story_label']}: {story['first_share']:.1%} to {story['last_share']:.1%} ({story['percentage_point_change']:+.1f} pp), {story['first_count']} to {story['last_count']} articles from {story['first_period']} to {story['last_period']}.")
        else:
            st.caption("No stories decreased their share between the first and last observed periods.")

    st.subheader("Sentiment evolution")
    sentiment = pd.DataFrame(temporal.sentiment_evolution)
    if sentiment.empty:
        st.info("Sentiment predictions are unavailable for dated articles.")
    else:
        sentiment["period"] = pd.to_datetime(sentiment["period"], utc=True)
        sentiment_chart = px.line(sentiment, x="period", y="share", color="sentiment", markers=True, custom_data=["count"], labels={"period": "Publication period", "share": "Share of sentiment predictions", "sentiment": "Predicted sentiment"})
        sentiment_chart.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1%}<br>Articles: %{customdata[0]}<extra></extra>")
        sentiment_chart.update_yaxes(tickformat=".0%")
        sentiment_chart.update_layout(height=320, margin=dict(l=8, r=8, t=16, b=8))
        st.plotly_chart(sentiment_chart, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Narrative shifts between chronological halves")
    st.caption("Story labels come from one clustering of the full loaded dataset and stay fixed across periods; shifts compare their shares in each period, not separately reclustered topics.")
    shifts = temporal.period_shifts
    if shifts:
        shifts_frame = pd.DataFrame(shifts)
        shifts_frame["change_label"] = shifts_frame["percentage_point_change"].map(lambda value: f"{value:+.1f} pp")
        st.dataframe(
            shifts_frame[["story_label", "previous_count", "current_count", "previous_share", "current_share", "percentage_point_change"]].rename(
                columns={"story_label": "Story", "previous_count": "Previous articles", "current_count": "Current articles", "previous_share": "Previous share", "current_share": "Current share", "percentage_point_change": "Change (percentage points)"}
            ),
            hide_index=True,
            use_container_width=True,
            column_config={"Previous share": st.column_config.NumberColumn(format="percent"), "Current share": st.column_config.NumberColumn(format="percent"), "Change (percentage points)": st.column_config.NumberColumn(format="%+.1f pp")},
        )
        selected_shift = st.selectbox("Inspect a measured shift", shifts, format_func=lambda item: f"{item['story_label']} ({item['percentage_point_change']:+.1f} pp)")
        st.caption(f"Previous period {selected_shift['previous_period'][0]} to {selected_shift['previous_period'][1]}: {selected_shift['previous_share']:.1%} ({selected_shift['previous_count']} articles). Current period {selected_shift['current_period'][0]} to {selected_shift['current_period'][1]}: {selected_shift['current_share']:.1%} ({selected_shift['current_count']} articles).")
        st.write("**Keywords associated with this shift**")
        st.write(" · ".join(selected_shift["keywords"]) if selected_shift["keywords"] else "No keywords available for this story.")
        st.write("**Representative articles**")
        for title in selected_shift["representative_titles"]:
            st.caption(title)
    else:
        st.info("At least two observed dated periods with story assignments are needed to compare topic shares.")

    st.subheader("Semantic representation shift")
    semantic = temporal.semantic_shift
    if semantic.get("status") == "Available":
        st.metric("Cosine distance between period centroids", f"{semantic['cosine_distance']:.3f}")
        st.caption(
            f"Computed from mean sentence embeddings: {semantic['previous_article_count']} usable articles from "
            f"{semantic['previous_period_range'][0]} to {semantic['previous_period_range'][1]} and "
            f"{semantic['current_article_count']} from {semantic['current_period_range'][0]} to {semantic['current_period_range'][1]}. "
            "This is an observed dataset-level semantic distance, not a claim about public opinion."
        )
    else:
        st.info(semantic.get("status", "Semantic period comparison is unavailable."))
