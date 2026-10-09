from newspulse.ui.layout import render_unavailable_page


def render_story_explorer() -> None:
    render_unavailable_page(
        "Story Explorer",
        "Inspect story clusters, their coverage, representative articles, and activity over time.",
        "Semantic embeddings and clustering to group related coverage and label each story.",
    )
