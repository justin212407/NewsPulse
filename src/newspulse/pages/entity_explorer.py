from newspulse.ui.layout import render_unavailable_page


def render_entity_explorer() -> None:
    render_unavailable_page(
        "Entity Explorer",
        "Explore the people, organizations, and places appearing across the collection.",
        "Named-entity recognition with mention counts, related entities, and time-based views.",
    )
