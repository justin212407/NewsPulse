from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from newspulse.nlp.features import dataset_keywords


@dataclass(frozen=True)
class TemporalAnalysis:
    undated_articles: int
    story_frequency: list[dict[str, Any]]
    emerging_stories: list[dict[str, Any]]
    declining_stories: list[dict[str, Any]]
    sentiment_evolution: list[dict[str, Any]]
    entity_frequency: list[dict[str, Any]]
    entity_sentiment: list[dict[str, Any]]
    entity_cooccurrence: list[dict[str, Any]]
    period_shifts: list[dict[str, Any]]
    semantic_shift: dict[str, Any]
    periods: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None


def analyze_temporal_intelligence(
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    period_days: int = 7,
    comparison_split: int | None = None,
    calendar_months: bool = False,
) -> TemporalAnalysis:
    frame = articles.reset_index(drop=True).copy()
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(frame):
        vectors = np.empty((len(frame), 0), dtype=np.float32)
    if "published_at" not in frame:
        frame["published_at"] = pd.NaT
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    undated = int(frame["published_at"].isna().sum())
    dated = frame.dropna(subset=["published_at"]).copy()
    if dated.empty:
        return TemporalAnalysis(undated, [], [], [], [], [], [], [], [], {}, None)

    step = max(1, int(period_days))
    minimum = dated["published_at"].min().normalize()
    maximum = dated["published_at"].max().normalize()
    if calendar_months:
        frame["period_start"] = frame["published_at"].map(
            lambda value: pd.NaT if pd.isna(value) else pd.Timestamp(year=value.year, month=value.month, day=1, tz="UTC")
        )
    else:
        boundaries = _period_boundaries(minimum, maximum, step)
        frame["period_start"] = frame["published_at"].map(lambda value: _period_start(value, boundaries))
    frame["period"] = frame["period_start"].map(lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else None)
    story_frequency = _story_frequency(frame)
    emerging, declining = _story_trends(story_frequency)
    sentiment = _sentiment_series(frame)
    entity_frequency, entity_sentiment, cooccurrence = _entity_series(frame)
    period_shifts = _story_period_shifts(frame, comparison_split, vectors)
    semantic_shift = _semantic_shift(frame, vectors, comparison_split)
    observed_periods = sorted(frame["period_start"].dropna().unique())
    midpoint = observed_periods[len(observed_periods) // 2] if len(observed_periods) > 1 else maximum
    periods = (minimum, pd.Timestamp(midpoint), maximum + pd.Timedelta(days=1))
    return TemporalAnalysis(
        undated,
        story_frequency,
        emerging,
        declining,
        sentiment,
        entity_frequency,
        entity_sentiment,
        cooccurrence,
        period_shifts,
        semantic_shift,
        periods,
    )


def _period_boundaries(start: pd.Timestamp, end: pd.Timestamp, step_days: int) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq=f"{step_days}D", tz="UTC"))


def _period_start(value: object, boundaries: list[pd.Timestamp]) -> pd.Timestamp | pd.NaT:
    if pd.isna(value) or not boundaries:
        return pd.NaT
    stamp = pd.Timestamp(value)
    index = int(pd.Index(boundaries).searchsorted(stamp, side="right")) - 1
    return boundaries[max(index, 0)]


def _story_frequency(frame: pd.DataFrame) -> list[dict[str, Any]]:
    story_key = "story_id" if "story_id" in frame else "story_label" if "story_label" in frame else None
    if story_key is None:
        return []
    dated = frame.dropna(subset=["period_start"])
    total_by_period = dated.groupby("period_start").size().to_dict()
    counts = dated.dropna(subset=[story_key]).groupby(["period_start", story_key], dropna=True).size()
    label_by_id = {}
    if "story_label" in frame and "story_id" in frame:
        label_by_id = frame.dropna(subset=["story_id", "story_label"]).drop_duplicates("story_id").set_index("story_id")["story_label"].to_dict()
    rows: list[dict[str, Any]] = []
    for (period, story_id), count in counts.items():
        period_total = int(total_by_period[period])
        rows.append(
            {
                "period": period,
                "story_id": story_id,
                "story_label": label_by_id.get(story_id, str(story_id)),
                "count": int(count),
                "share": float(count / period_total) if period_total else 0.0,
                "period_total": period_total,
            }
        )
    return sorted(rows, key=lambda row: (row["period"], str(row["story_label"])))


def _story_trends(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[object, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["story_id"], []).append(row)
    changes: list[dict[str, Any]] = []
    observed_periods = sorted({row["period"] for row in rows})
    first_period, last_period = (observed_periods[0], observed_periods[-1]) if observed_periods else (None, None)
    if not observed_periods:
        return [], []
    for story_id, series in grouped.items():
        if len(observed_periods) < 2:
            continue
        first = next((row for row in series if row["period"] == first_period), None)
        last = next((row for row in series if row["period"] == last_period), None)
        before_share = first["share"] if first else 0.0
        after_share = last["share"] if last else 0.0
        delta = after_share - before_share
        changes.append(
            {
                "story_id": story_id,
                "story_label": series[0]["story_label"],
                "first_share": before_share,
                "last_share": after_share,
                "share_change": delta,
                "percentage_point_change": delta * 100,
                "first_count": first["count"] if first else 0,
                "last_count": last["count"] if last else 0,
                "first_period": first_period,
                "last_period": last_period,
            }
        )
    return (
        sorted((row for row in changes if row["share_change"] > 0), key=lambda row: -row["share_change"]),
        sorted((row for row in changes if row["share_change"] < 0), key=lambda row: row["share_change"]),
    )


def _sentiment_series(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "sentiment" not in frame:
        return []
    counts: dict[tuple[object, str], int] = {}
    for _, row in frame.dropna(subset=["period_start"]).iterrows():
        annotation = row.get("sentiment")
        if isinstance(annotation, dict) and annotation.get("label"):
            key = (row["period_start"], str(annotation["label"]).capitalize())
            counts[key] = counts.get(key, 0) + 1
    totals: dict[object, int] = {}
    for (period, _label), count in counts.items():
        totals[period] = totals.get(period, 0) + count
    return [
        {
            "period": period,
            "sentiment": label,
            "count": count,
            "share": count / totals[period] if totals.get(period) else 0.0,
        }
        for (period, label), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _entity_series(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if "entities" not in frame:
        return [], [], []
    frequency: dict[tuple[object, str, str], set[str]] = {}
    sentiments: dict[tuple[object, str, str, str], list[float]] = {}
    pair_articles: dict[tuple[object, str, str, str, str], set[str]] = {}
    for index, row in frame.dropna(subset=["period_start"]).iterrows():
        article_id = str(row.get("id", index))
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for entity in row.get("entities", []) or []:
            if not isinstance(entity, dict) or not str(entity.get("text", "")).strip():
                continue
            key = (str(entity["text"]).strip(), str(entity.get("label", "MISC")))
            seen.setdefault(key, entity)
        annotation = row.get("sentiment", {})
        for (text, label), entity in seen.items():
            frequency.setdefault((row["period_start"], text, label), set()).add(article_id)
            if isinstance(annotation, dict) and annotation.get("label"):
                score_key = (row["period_start"], text, label, str(annotation["label"]).capitalize())
                sentiments.setdefault(score_key, []).append(float(annotation.get("score", 0.0)))
        entities = sorted(seen)
        for position, left in enumerate(entities):
            for right in entities[position + 1 :]:
                pair_key = (row["period_start"], left[0], left[1], right[0], right[1])
                pair_articles.setdefault(pair_key, set()).add(article_id)
    frequency_rows = [
        {"period": period, "entity": text, "label": label, "article_count": len(ids)}
        for (period, text, label), ids in frequency.items()
    ]
    sentiment_totals: dict[tuple[object, str], int] = {}
    for (period, entity, _entity_label, _sentiment), scores in sentiments.items():
        key = (period, entity)
        sentiment_totals[key] = sentiment_totals.get(key, 0) + len(scores)
    sentiment_rows = [
        {
            "period": period,
            "entity": text,
            "label": entity_label,
            "sentiment": sentiment,
            "article_count": len(scores),
            "share": len(scores) / sentiment_totals[(period, text)] if sentiment_totals[(period, text)] else 0.0,
            "mean_confidence": float(np.mean(scores)),
        }
        for (period, text, entity_label, sentiment), scores in sentiments.items()
    ]
    cooccurrence: dict[tuple[str, str, str, str], set[str]] = {}
    all_articles = set(frame.loc[frame["period_start"].notna(), "id"].astype(str))
    for (_period, left, left_label, right, right_label), ids in pair_articles.items():
        cooccurrence.setdefault((left, left_label, right, right_label), set()).update(ids)
    cooccurrence_rows = [
        {
            "left": left,
            "left_label": left_label,
            "right": right,
            "right_label": right_label,
            "article_count": len(ids),
            "document_share": float(len(ids) / len(all_articles)) if all_articles else 0.0,
        }
        for (left, left_label, right, right_label), ids in sorted(cooccurrence.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    for (period, left, left_label, right, right_label), ids in pair_articles.items():
        period_total = len(frame.loc[frame["period_start"] == period])
        cooccurrence_rows.append(
            {
                "period": period,
                "left": left,
                "left_label": left_label,
                "right": right,
                "right_label": right_label,
                "article_count": len(ids),
                "document_share": float(len(ids) / period_total) if period_total else 0.0,
            }
        )
    return frequency_rows, sentiment_rows, cooccurrence_rows


def _story_period_shifts(
    frame: pd.DataFrame,
    comparison_split: int | None = None,
    embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    story_key = "story_id" if "story_id" in frame else None
    if story_key is None or "story_label" not in frame:
        return []
    dated = frame.dropna(subset=["period_start"])
    unique_periods = sorted(dated["period_start"].dropna().unique())
    if comparison_split is None and len(unique_periods) < 2:
        return []
    if comparison_split is None:
        split = len(unique_periods) // 2
        previous_periods = set(unique_periods[:split])
        current_periods = set(unique_periods[split:])
        previous = dated.loc[dated["period_start"].isin(previous_periods)]
        current = dated.loc[dated["period_start"].isin(current_periods)]
    else:
        cutoff = pd.Timestamp(comparison_split, unit="s", tz="UTC")
        previous = dated.loc[dated["published_at"] < cutoff]
        current = dated.loc[dated["published_at"] >= cutoff]
        previous_periods = set(previous["period_start"].dropna().unique())
        current_periods = set(current["period_start"].dropna().unique())
    if previous.empty or current.empty:
        return []
    shifts = _distribution_shifts(frame, previous, current, story_key, embeddings)
    previous_range = (min(previous_periods).strftime("%Y-%m-%d"), max(previous_periods).strftime("%Y-%m-%d"))
    current_range = (min(current_periods).strftime("%Y-%m-%d"), max(current_periods).strftime("%Y-%m-%d"))
    for shift in shifts:
        shift["previous_period"] = previous_range
        shift["current_period"] = current_range
    return shifts


def _distribution_shifts(
    frame: pd.DataFrame,
    previous: pd.DataFrame,
    current: pd.DataFrame,
    story_key: str,
    embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if previous.empty or current.empty:
        return []
    previous_total, current_total = len(previous), len(current)
    previous_counts = previous[story_key].value_counts(dropna=True)
    current_counts = current[story_key].value_counts(dropna=True)
    story_ids = sorted(set(previous_counts.index) | set(current_counts.index), key=str)
    output = []
    for story_id in story_ids:
        before_count = int(previous_counts.get(story_id, 0))
        after_count = int(current_counts.get(story_id, 0))
        before_share = before_count / previous_total
        after_share = after_count / current_total
        delta = after_share - before_share
        if abs(delta) < 1e-12:
            continue
        period_group = pd.concat(
            [previous.loc[previous[story_key] == story_id], current.loc[current[story_key] == story_id]]
        ).drop_duplicates("id")
        texts = period_group.get("text_clean", pd.Series(dtype=str)).dropna().astype(str).tolist()
        keyword_list = dataset_keywords(texts, limit=8)
        representative_group = _representative_articles(period_group, embeddings)
        representative_ids = representative_group.get("id", pd.Series(dtype=str)).astype(str).tolist()
        output.append(
            {
                "story_id": story_id,
                "story_label": str(period_group["story_label"].iloc[0]) if "story_label" in period_group and not period_group.empty else str(story_id),
                "previous_count": before_count,
                "current_count": after_count,
                "previous_share": before_share,
                "current_share": after_share,
                "percentage_point_change": delta * 100,
                "keywords": [term for term, _score in keyword_list],
                "representative_ids": representative_ids,
                "representative_titles": representative_group.get("title", pd.Series(dtype=str)).fillna("").astype(str).tolist(),
            }
        )
    return sorted(output, key=lambda row: -abs(row["percentage_point_change"]))


def _representative_articles(group: pd.DataFrame, embeddings: np.ndarray | None) -> pd.DataFrame:
    if group.empty or embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return group.head(3)
    indexes = [int(index) for index in group.index if 0 <= int(index) < len(embeddings)]
    if not indexes or embeddings.shape[1] == 0:
        return group.head(3)
    vectors = embeddings[indexes]
    usable = np.linalg.norm(vectors, axis=1) > 0
    if not usable.any():
        return group.head(3)
    member_rows = group.loc[indexes].iloc[np.flatnonzero(usable)]
    vectors = vectors[usable]
    centroid = vectors.mean(axis=0)
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid)
    scores = np.divide(vectors @ centroid, denominator, out=np.zeros(len(vectors), dtype=np.float32), where=denominator != 0)
    return member_rows.iloc[np.argsort(scores)[::-1][:3]]


def _semantic_shift(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    comparison_split: int | None = None,
) -> dict[str, Any]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(frame) or embeddings.shape[1] == 0:
        return {"status": "Embeddings are unavailable.", "cosine_distance": None}
    dated_indices = np.flatnonzero(frame["period_start"].notna().to_numpy())
    dates = frame.loc[dated_indices, "period_start"].sort_values().unique()
    if comparison_split is None and len(dates) < 2:
        return {"status": "At least two dated periods are required.", "cosine_distance": None}
    if comparison_split is None:
        split = len(dates) // 2
        previous_periods, current_periods = set(dates[:split]), set(dates[split:])
        previous_indices = [index for index in dated_indices if frame.iloc[index]["period_start"] in previous_periods]
        current_indices = [index for index in dated_indices if frame.iloc[index]["period_start"] in current_periods]
        previous_dates, current_dates = dates[:split], dates[split:]
    else:
        cutoff = pd.Timestamp(comparison_split, unit="s", tz="UTC")
        previous_indices = [index for index in dated_indices if frame.iloc[index]["published_at"] < cutoff]
        current_indices = [index for index in dated_indices if frame.iloc[index]["published_at"] >= cutoff]
        previous_dates = frame.iloc[previous_indices]["period_start"].dropna().unique()
        current_dates = frame.iloc[current_indices]["period_start"].dropna().unique()
    valid = np.linalg.norm(embeddings, axis=1) > 0
    previous_indices = [index for index in previous_indices if valid[index]]
    current_indices = [index for index in current_indices if valid[index]]
    if not previous_indices or not current_indices:
        return {"status": "Both periods need articles with valid embeddings.", "cosine_distance": None}
    previous_centroid = embeddings[previous_indices].mean(axis=0)
    current_centroid = embeddings[current_indices].mean(axis=0)
    denominator = float(np.linalg.norm(previous_centroid) * np.linalg.norm(current_centroid))
    similarity = float(np.clip(previous_centroid @ current_centroid / denominator, -1.0, 1.0)) if denominator else 0.0
    return {
        "status": "Available",
        "metric": "cosine distance between mean article embeddings",
        "cosine_distance": 1.0 - similarity,
        "previous_article_count": len(previous_indices),
        "current_article_count": len(current_indices),
        "previous_periods": len(previous_dates),
        "current_periods": len(current_dates),
        "previous_period_range": (min(previous_dates).strftime("%Y-%m-%d"), max(previous_dates).strftime("%Y-%m-%d")) if len(previous_dates) else None,
        "current_period_range": (min(current_dates).strftime("%Y-%m-%d"), max(current_dates).strftime("%Y-%m-%d")) if len(current_dates) else None,
    }
