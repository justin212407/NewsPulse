import numpy as np

from newspulse.nlp.features import (
    dataset_keywords,
    exact_duplicate_groups,
    extract_keywords,
    near_duplicate_pairs,
    pairwise_cosine_similarity,
    top_similar_articles,
)


def test_keyword_extraction_handles_empty_and_short_text() -> None:
    results = extract_keywords([None, "the and or", "AI safety policy", "AI safety policy"])
    assert results[0] == []
    assert results[1] == []
    assert "safety policy" in results[2]
    assert dataset_keywords(["", None]) == []


def test_exact_duplicates_normalize_case_and_whitespace() -> None:
    assert exact_duplicate_groups(["AI policy", " ai   POLICY ", "different", None]) == [[0, 1]]
    assert exact_duplicate_groups(["", None]) == []


def test_cosine_similarity_and_near_duplicates_use_known_vectors() -> None:
    vectors = np.array([[1, 0], [0.99, 0.1], [0, 1], [0, 0]], dtype=np.float32)
    similarity = pairwise_cosine_similarity(vectors)
    assert np.isclose(similarity[0, 2], 0.0)
    assert similarity[0, 1] > 0.99
    assert np.all(similarity[3] == 0)
    assert near_duplicate_pairs(similarity, threshold=0.98) == [(0, 1, float(similarity[0, 1]))]
    assert top_similar_articles(similarity, 0, limit=1) == [(1, float(similarity[0, 1]))]


def test_similarity_rejects_invalid_shape_and_handles_small_dataset() -> None:
    assert pairwise_cosine_similarity(np.array([[1, 2]])).shape == (1, 1)
    assert pairwise_cosine_similarity(np.empty((0, 3))).shape == (0, 0)
    try:
        pairwise_cosine_similarity(np.array([1, 2]))
    except ValueError as error:
        assert "two-dimensional" in str(error)
    else:
        raise AssertionError("one-dimensional embeddings should be rejected")


def test_near_duplicates_excludes_exact_pairs() -> None:
    similarity = np.array([[1.0, 0.999], [0.999, 1.0]])
    assert near_duplicate_pairs(similarity, threshold=0.9, exact_groups=[[0, 1]]) == []
