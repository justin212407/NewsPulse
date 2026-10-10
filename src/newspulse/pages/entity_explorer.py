import math

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from newspulse.dataset_service import Dataset
from newspulse.nlp.temporal import analyze_temporal_intelligence
from newspulse.ui.components import render_dataset_state
from newspulse.ui.layout import render_page_header
from newspulse.ui.nlp_analysis import get_cached_analysis, run_cached_analysis


def render_entity_explorer(dataset: Dataset) -> None:
    render_page_header(
        "Entity Explorer",
        "Track people, organizations, and places across dated coverage, including co-mentions and associated sentiment.",
        "Entity intelligence",
    )
    if not render_dataset_state(dataset):
        return
    articles = dataset.articles

    analysis = get_cached_analysis(articles)
    if analysis is None:
        st.info("Entity views reuse the annotations from the cached NLP analysis.")
        if st.button("Run NLP analysis", key="entities_analyze"):
            with st.spinner("Loading NLP models and annotating articles..."):
                try:
                    analysis = run_cached_analysis(articles)
                except Exception as error:
                    st.error(f"Analysis could not start: {type(error).__name__}: {error}")
                    return
    if analysis is None:
        return
    if analysis.feature_status.get("entities") != "Available":
        st.warning(analysis.feature_errors.get("entities", "Named-entity recognition is unavailable."))
        return

    interval = st.selectbox("Time interval", ("Daily", "Weekly", "Monthly"), index=1)
    days = {"Daily": 1, "Weekly": 7, "Monthly": 30}[interval]
    temporal = analyze_temporal_intelligence(analysis.article_results, analysis.embeddings, period_days=days)
    st.caption(f"Using {len(articles) - temporal.undated_articles} dated articles. {temporal.undated_articles} article(s) have no valid publication date and are excluded from timelines.")
    if not temporal.entity_frequency:
        st.info("No entity annotations with valid dates are available.")
        return

    frequency = pd.DataFrame(temporal.entity_frequency)
    frequency["period"] = pd.to_datetime(frequency["period"], utc=True)
    top_entities = (
        frequency.groupby("entity")["article_count"].sum().nlargest(12).index.astype(str).tolist()
    )
    selected_entities = st.multiselect("Entities to compare", top_entities, default=top_entities[:5])
    visible = frequency.loc[frequency["entity"].isin(selected_entities)]
    st.subheader("Entity frequency over time")
    if not visible.empty:
        chart = px.line(
            visible,
            x="period",
            y="article_count",
            color="entity",
            markers=True,
            labels={"period": "Publication period", "article_count": "Articles", "entity": "Entity"},
        )
        chart.update_layout(height=340, margin=dict(l=8, r=8, t=16, b=8), legend_title_text="Entity")
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Select at least one entity to show its timeline.")

    st.subheader("Entity-associated sentiment")
    sentiment_frame = pd.DataFrame(temporal.entity_sentiment)
    if not sentiment_frame.empty:
        sentiment_frame["period"] = pd.to_datetime(sentiment_frame["period"], utc=True)
        sentiment_frame = sentiment_frame.loc[sentiment_frame["entity"].isin(selected_entities)]
        if not sentiment_frame.empty:
            chart = px.bar(
                sentiment_frame,
                x="period",
                y="share",
                color="sentiment",
                facet_row="entity",
                barmode="stack",
                custom_data=["article_count"],
                labels={"period": "Publication period", "share": "Share of entity sentiment predictions", "sentiment": "Predicted sentiment"},
            )
            chart.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1%}<br>Articles: %{customdata[0]}<extra></extra>")
            chart.update_yaxes(tickformat=".0%")
            chart.update_layout(height=min(620, 240 + 90 * sentiment_frame["entity"].nunique()), margin=dict(l=8, r=8, t=16, b=8))
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No sentiment annotations exist for the selected entities.")
    else:
        st.caption("Sentiment annotations are unavailable.")

    st.subheader("Entity co-occurrence network")
    st.caption("An edge means two distinct entities appeared in the same article. Edge weight is the number of shared articles; choose an interval to compare dated periods.")
    pairs = pd.DataFrame(temporal.entity_cooccurrence)
    if pairs.empty:
        st.info("No entity pairs co-occur in dated articles.")
        return
    graph_period = st.selectbox("Network period", ["All dated articles", *sorted(pairs["period"].dropna().unique().tolist())])
    if graph_period != "All dated articles":
        pairs = pairs.loc[pairs["period"] == graph_period]
    else:
        pairs = pairs.loc[pairs["period"].isna()]
    if pairs.empty:
        st.info("No entity pairs co-occur in this period.")
        return
    edge_limit = st.slider("Strongest co-occurrence links", 5, min(40, len(pairs)), min(18, len(pairs))) if len(pairs) >= 5 else len(pairs)
    pairs = pairs.head(edge_limit)
    nodes = sorted(
        {(row.left, row.left_label) for row in pairs.itertuples()}
        | {(row.right, row.right_label) for row in pairs.itertuples()}
    )
    positions = _circle_positions(nodes)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for pair in pairs.itertuples():
        x0, y0 = positions[(pair.left, pair.left_label)]
        x1, y1 = positions[(pair.right, pair.right_label)]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    degree = {node: 0 for node in nodes}
    for pair in pairs.itertuples():
        degree[(pair.left, pair.left_label)] += pair.article_count
        degree[(pair.right, pair.right_label)] += pair.article_count
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1.5, "color": "#8a9994"}, hoverinfo="skip"))
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[f"{name} ({label})" for name, label in nodes],
            textposition="top center",
            marker={"size": [16 + min(20, degree[node]) for node in nodes], "color": "#167d78", "line": {"width": 2, "color": "white"}},
            customdata=[degree[node] for node in nodes],
            hovertemplate="%{text}<br>Weighted co-mentions: %{customdata}<extra></extra>",
        )
    )
    figure.update_layout(height=540, showlegend=False, xaxis={"visible": False}, yaxis={"visible": False}, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.caption("Entity labels and associations are model-extracted observations, not claims of a relationship outside this dataset.")


def _circle_positions(nodes: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
    if not nodes:
        return {}
    return {
        node: (math.cos(2 * math.pi * index / len(nodes)), math.sin(2 * math.pi * index / len(nodes)))
        for index, node in enumerate(nodes)
    }
