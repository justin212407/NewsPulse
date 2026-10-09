import pandas as pd
import plotly.express as px
import streamlit as st

from newspulse.dataset_service import Dataset
from newspulse.ui.components import format_date_range, render_validation_summary
from newspulse.ui.layout import render_page_header
from newspulse.ui.nlp_analysis import render_analysis_controls


def render_overview(dataset: Dataset) -> None:
    render_page_header(
        "News coverage, in context",
        "Start with a validated article collection. Topic and narrative analysis will follow as those pipelines are built.",
        "Dataset overview",
    )
    render_validation_summary(dataset)
    articles = dataset.articles
    if articles.empty:
        st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
        st.subheader("Bring in a dataset to get started")
        st.write("Use the sample collection or upload CSV/JSON records with article IDs, titles, text, sources, dates, and URLs.")
        return

    st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
    dated = articles["published_at"].dropna()
    first_col, second_col, third_col, fourth_col = st.columns(4)
    first_col.metric("Articles", f"{len(articles):,}")
    second_col.metric("Sources", f"{articles['source'].nunique():,}")
    third_col.metric("Coverage window", format_date_range(articles))
    fourth_col.metric("Records skipped", f"{dataset.report.skipped_rows:,}")

    chart_col, source_col = st.columns([1.55, 1], gap="large")
    with chart_col:
        st.subheader("Publishing activity")
        if dated.empty:
            st.info("No valid publication dates are available to chart.")
        else:
            activity = (
                articles.dropna(subset=["published_at"])
                .assign(day=lambda frame: frame["published_at"].dt.date)
                .groupby("day", as_index=False)
                .size()
                .rename(columns={"size": "Articles"})
            )
            chart = px.area(activity, x="day", y="Articles", markers=True)
            chart.update_traces(line_color="#167d78", fillcolor="rgba(22,125,120,0.12)")
            chart.update_layout(
                height=290,
                margin=dict(l=8, r=8, t=12, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#64716e",
                xaxis_title=None,
                yaxis_title=None,
            )
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    with source_col:
        st.subheader("Articles by source")
        source_counts = articles["source"].value_counts().rename_axis("Source").reset_index(name="Articles")
        chart = px.bar(source_counts, x="Articles", y="Source", orientation="h")
        chart.update_traces(marker_color="#167d78")
        chart.update_layout(
            height=290,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#64716e",
            xaxis_title=None,
            yaxis_title=None,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
    st.subheader("Latest articles")
    latest = articles.sort_values("published_at", ascending=False, na_position="last").head(4)
    for _, article in latest.iterrows():
        with st.container(border=True):
            st.markdown(f"**{article['title']}**")
            published = article["published_at"]
            date = published.strftime("%b %d, %Y") if pd.notna(published) else "Date unavailable"
            st.caption(f"{article['source']}  |  {date}")
            st.write(article["text_clean"][:260] + ("…" if len(article["text_clean"]) > 260 else ""))

    result = render_analysis_controls(articles, "overview")
    if result is not None:
        st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
        st.subheader("Analysis results")
        entity_counts: dict[str, int] = {}
        if result.feature_status.get("entities") == "Available":
            entity_counts = (
                result.article_results["entities"]
                .explode()
                .dropna()
                .map(lambda entity: str(entity.get("text", "")).strip())
                .loc[lambda values: values != ""]
                .value_counts()
                .to_dict()
            )
        summary_cols = st.columns(4)
        summary_cols[0].metric("Exact duplicate groups", len(result.exact_duplicate_groups))
        summary_cols[1].metric("Near-duplicate pairs", len(result.near_duplicate_pairs))
        summary_cols[2].metric("Articles with sentiment", sum(result.sentiment_counts.values()))
        summary_cols[3].metric("Named entities", len(entity_counts))
        if result.sentiment_counts:
            sentiment_chart = pd.DataFrame(
                [{"Sentiment": label, "Articles": count} for label, count in result.sentiment_counts.items()]
            )
            chart = px.bar(sentiment_chart, x="Sentiment", y="Articles", color="Sentiment")
            chart.update_layout(height=250, margin=dict(l=8, r=8, t=12, b=8), showlegend=False)
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No sentiment predictions are available.")
        st.write("**Top keywords and phrases**")
        if result.dataset_keywords:
            st.write(" · ".join(term for term, _score in result.dataset_keywords[:12]))
        else:
            st.caption("No keywords could be extracted.")
        if entity_counts:
            st.write("**Most-mentioned entities**")
            st.write(" · ".join(f"{entity} ({count})" for entity, count in list(entity_counts.items())[:10]))
