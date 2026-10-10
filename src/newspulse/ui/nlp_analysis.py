from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from newspulse.nlp.service import AnalysisResult, analyze_articles_cached


def _dataset_key(articles: pd.DataFrame) -> str:
    columns = [column for column in ("id", "text_clean") if column in articles]
    digest = pd.util.hash_pandas_object(articles[columns], index=False).values.tobytes()
    return hashlib.sha256(digest).hexdigest()


def next_analysis_run_id(current_run_id: int, has_cached_result: bool) -> int:
    return current_run_id + int(has_cached_result)


def render_analysis_controls(
    articles: pd.DataFrame,
    page_key: str,
    *,
    show_controls: bool = True,
) -> AnalysisResult | None:
    dataset_key = _dataset_key(articles)
    result_key = f"nlp_result_{dataset_key}"
    run_key = f"nlp_run_id_{dataset_key}"
    result = st.session_state.get(result_key)

    if show_controls:
        st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
        st.subheader("NLP analysis")
        st.caption(
            "Sentence embeddings, sentiment and entity models run on demand. The first run may download model weights."
        )
        if st.button("Run NLP analysis" if result is None else "Re-run NLP analysis", key=f"nlp_run_{page_key}"):
            run_id = next_analysis_run_id(st.session_state.get(run_key, 0), result is not None)
            st.session_state[run_key] = run_id
            with st.spinner("Loading models and analyzing the article collection..."):
                try:
                    result = analyze_articles_cached(articles, run_id=run_id)
                    st.session_state[result_key] = result
                except Exception as error:
                    st.error(f"Analysis could not start: {type(error).__name__}: {error}")
                    result = None

        result = st.session_state.get(result_key)
        if result is not None:
            statuses = result.feature_status
            available = sum(status == "Available" for status in statuses.values())
            st.caption(f"Analysis status: {available} of {len(statuses)} features available.")
            for feature, message in result.feature_errors.items():
                st.warning(f"{feature.replace('_', ' ').capitalize()} unavailable: {message}")
            if result.feature_errors and st.button("Retry unavailable models", key=f"nlp_retry_{page_key}"):
                st.session_state[run_key] = st.session_state.get(run_key, 0) + 1
                with st.spinner("Retrying unavailable model features..."):
                    try:
                        result = analyze_articles_cached(articles, run_id=st.session_state[run_key])
                        st.session_state[result_key] = result
                        st.rerun()
                    except Exception as error:
                        st.error(f"Analysis could not start: {type(error).__name__}: {error}")
    else:
        result = st.session_state.get(result_key)
    return result


def dataset_analysis_key(articles: pd.DataFrame) -> str:
    return _dataset_key(articles)


def get_cached_analysis(articles: pd.DataFrame) -> AnalysisResult | None:
    return st.session_state.get(f"nlp_result_{_dataset_key(articles)}")


def run_cached_analysis(articles: pd.DataFrame, run_id: int = 0) -> AnalysisResult:
    result = analyze_articles_cached(articles, run_id=run_id)
    st.session_state[f"nlp_result_{_dataset_key(articles)}"] = result
    return result


def article_analysis_for_id(result: AnalysisResult, article_id: object) -> pd.Series | None:
    rows = result.article_results
    if "id" not in rows:
        return None
    selected = rows.loc[rows["id"].astype(str) == str(article_id)]
    if selected.empty:
        return None
    article = selected.iloc[0].copy()
    article.name = int(selected.index[0])
    return article


def render_article_analysis(
    analysis: pd.Series,
    result: AnalysisResult,
    articles: pd.DataFrame,
) -> None:
    st.markdown("#### Model annotations")
    sentiment = analysis.get("sentiment", {})
    entities = analysis.get("entities", [])
    keywords = analysis.get("keywords", [])
    if isinstance(sentiment, dict) and sentiment.get("label"):
        st.metric(
            "Predicted sentiment",
            str(sentiment["label"]).capitalize(),
            f"{float(sentiment.get('score', 0.0)):.1%} confidence",
        )
    else:
        st.caption("Sentiment is unavailable for this article.")

    entity_labels = [
        f"{entity.get('text', '')} ({entity.get('label', 'MISC')})"
        for entity in entities
        if isinstance(entity, dict) and entity.get("text")
    ]
    st.write("**Named entities**")
    st.write(", ".join(entity_labels) if entity_labels else "No named entities returned.")
    st.write("**Keywords and phrases**")
    st.write(", ".join(keywords) if keywords else "No keywords could be extracted.")

    st.write("**Similar articles**")
    matches = analysis.get("similar_articles", [])
    if matches:
        for index, score in matches:
            article = articles.iloc[index]
            published = article.get("published_at")
            date = published.strftime("%b %d, %Y") if published is not None and pd.notna(published) else "Date unavailable"
            st.caption(f"{article.get('title', 'Untitled article')}  |  {article.get('source', 'Unknown source')}  |  {date}  |  {score:.1%} cosine similarity")
    elif result.feature_status.get("similarity") == "Unavailable":
        st.caption("Similar-article matches are unavailable because embeddings could not be generated.")
    else:
        st.caption("No similar-article matches were found.")

    current_index = int(analysis.name)
    exact_groups = [group for group in result.exact_duplicate_groups if current_index in group]
    near_pairs = [pair for pair in result.near_duplicate_pairs if current_index in pair[:2]]
    if exact_groups:
        st.warning(f"Exact duplicate detected in {len(exact_groups)} article group(s).")
        for group in exact_groups:
            for other_index in group:
                if other_index != current_index:
                    st.caption(f"Duplicate article: {articles.iloc[other_index]['title']}")
    if near_pairs:
        st.write("**Near duplicates**")
        for left, right, score in near_pairs:
            other_index = right if left == current_index else left
            st.caption(f"{articles.iloc[other_index]['title']} ({score:.1%} cosine similarity)")
