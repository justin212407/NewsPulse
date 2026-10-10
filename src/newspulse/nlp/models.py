from __future__ import annotations

from typing import Any

import streamlit as st

from newspulse.nlp.config import EMBEDDING_MODEL, NER_MODEL, SENTIMENT_MODEL


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install CPU-only PyTorch from https://download.pytorch.org/whl/cpu, "
            "then install optional NLP dependencies with `pip install -e \".[nlp]\"`."
        ) from error
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_transformer_pipeline(task: str, model_name: str) -> Any:
    try:
        import torch
        from transformers import pipeline
    except Exception as error:
        raise RuntimeError(
            "Install CPU-only PyTorch from https://download.pytorch.org/whl/cpu, "
            "then install optional NLP dependencies with `pip install -e \".[nlp]\"`. "
            f"Model runtime initialization failed: {error}"
        ) from error
    return pipeline(task, model=model_name, device=-1)


@st.cache_data(show_spinner=False)
def encode_texts(texts: list[str]) -> Any:
    model = load_embedding_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


@st.cache_data(show_spinner=False)
def encode_query(query: str) -> Any:
    return encode_texts([query])


def classify_sentiment(texts: list[str]) -> list[dict[str, object]]:
    try:
        classifier = load_transformer_pipeline("text-classification", SENTIMENT_MODEL)
        raw_results = classifier(texts, truncation=True, max_length=512, batch_size=8)
    except Exception as error:
        if "torch" in str(error).casefold():
            raise RuntimeError(_torch_install_message()) from error
        raise RuntimeError(f"Sentiment model could not run: {error}") from error
    return [_normalize_classification(result) for result in raw_results]


def recognize_entities(texts: list[str]) -> list[list[dict[str, object]]]:
    try:
        recognizer = load_transformer_pipeline("token-classification", NER_MODEL)
        raw_results = recognizer(texts, aggregation_strategy="simple", truncation=True, batch_size=8)
    except Exception as error:
        if "torch" in str(error).casefold():
            raise RuntimeError(_torch_install_message()) from error
        raise RuntimeError(f"Entity model could not run: {error}") from error
    return [[_normalize_entity(entity) for entity in result] for result in raw_results]


def _torch_install_message() -> str:
    return (
        "Install CPU-only PyTorch from https://download.pytorch.org/whl/cpu, "
        "then install optional NLP dependencies with `pip install -e \".[nlp]\"`."
    )


def _normalize_classification(result: dict[str, Any]) -> dict[str, object]:
    label = str(result.get("label", "")).casefold()
    if label.startswith("label_"):
        label_id = label.removeprefix("label_")
        label = {"0": "negative", "1": "neutral", "2": "positive"}.get(label_id, label)
    return {"label": label, "score": float(result.get("score", 0.0))}


def _normalize_entity(entity: dict[str, Any]) -> dict[str, object]:
    return {
        "text": str(entity.get("word", "")).replace(" ##", "").strip(),
        "label": str(entity.get("entity_group", entity.get("entity", "MISC"))),
        "score": float(entity.get("score", 0.0)),
    }
