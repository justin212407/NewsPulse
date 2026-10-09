from unittest.mock import patch

from newspulse.nlp.models import _normalize_classification, _normalize_entity, load_transformer_pipeline


def test_sentiment_labels_and_entities_are_normalized() -> None:
    assert _normalize_classification({"label": "LABEL_0", "score": 0.8}) == {
        "label": "negative",
        "score": 0.8,
    }
    assert _normalize_classification({"label": "positive", "score": 0.9})["label"] == "positive"
    assert _normalize_entity({"word": "New York", "entity_group": "LOC", "score": 0.8}) == {
        "text": "New York",
        "label": "LOC",
        "score": 0.8,
    }


def test_missing_torch_produces_actionable_model_error() -> None:
    with patch("builtins.__import__", side_effect=NameError("name 'torch' is not defined")):
        try:
            load_transformer_pipeline("text-classification", "test-model")
        except RuntimeError as error:
            assert "CPU-only PyTorch" in str(error)
            assert "pip install -e" in str(error)
        else:
            raise AssertionError("missing model backend should produce an actionable error")
