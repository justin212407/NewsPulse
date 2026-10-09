from newspulse.config import PAGE_ICON, PAGE_LAYOUT, PAGE_TITLES, SAMPLE_DATA_PATH
from newspulse.dataset_service import load_dataset
from newspulse.ui.layout import render_sidebar
from newspulse.pages.article_explorer import render_article_explorer
from newspulse.pages.entity_explorer import render_entity_explorer
from newspulse.pages.narrative_explorer import render_narrative_explorer
from newspulse.pages.overview import render_overview
from newspulse.pages.semantic_search import render_semantic_search
from newspulse.pages.story_explorer import render_story_explorer

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="NewsPulse", page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
    dataset_choice, uploaded_file = render_sidebar()

    if uploaded_file is not None:
        dataset = load_dataset(uploaded_file.getvalue(), uploaded_file.name)
    elif dataset_choice == "Sample dataset":
        dataset = load_dataset(SAMPLE_DATA_PATH.read_bytes(), SAMPLE_DATA_PATH.name)
    else:
        dataset = load_dataset(b"", "empty.csv")

    pages = [
        st.Page(lambda: render_overview(dataset), title=PAGE_TITLES[0], url_path="overview", default=True),
        st.Page(lambda: render_story_explorer(), title=PAGE_TITLES[1], url_path="stories"),
        st.Page(lambda: render_entity_explorer(), title=PAGE_TITLES[2], url_path="entities"),
        st.Page(lambda: render_narrative_explorer(), title=PAGE_TITLES[3], url_path="narratives"),
        st.Page(lambda: render_semantic_search(), title=PAGE_TITLES[4], url_path="search"),
        st.Page(lambda: render_article_explorer(dataset), title=PAGE_TITLES[5], url_path="articles"),
    ]
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
