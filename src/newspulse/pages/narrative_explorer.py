from newspulse.ui.layout import render_unavailable_page


def render_narrative_explorer() -> None:
    render_unavailable_page(
        "Narrative Explorer",
        "Compare coverage across time periods to identify observed shifts in topic and sentiment.",
        "Time-windowed topic distributions and semantic representations compared across periods.",
    )
