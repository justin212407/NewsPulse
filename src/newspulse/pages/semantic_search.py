from newspulse.ui.layout import render_unavailable_page


def render_semantic_search() -> None:
    render_unavailable_page(
        "Semantic Search",
        "Find relevant articles by meaning, even when they use different wording.",
        "Sentence embeddings and similarity ranking across the loaded article collection.",
    )
