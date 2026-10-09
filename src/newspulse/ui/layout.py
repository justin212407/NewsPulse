import streamlit as st

from newspulse.config import MAX_UPLOAD_BYTES


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        :root { --np-ink: #182321; --np-muted: #64716e; --np-line: #dfe6e3; --np-accent: #167d78; }
        .stApp { background: #f4f6f5; }
        [data-testid="stHeader"] { background: rgba(244, 246, 245, 0.92); }
        [data-testid="stSidebar"] { background: #eef2f0; border-right: 1px solid var(--np-line); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--np-muted); }
        .np-brand { font-size: 1.08rem; font-weight: 750; letter-spacing: -0.04em; color: var(--np-ink); margin: .15rem 0 0; }
        .np-kicker { color: var(--np-accent); text-transform: uppercase; letter-spacing: .13em; font-size: .68rem; font-weight: 700; }
        .np-subtitle { color: var(--np-muted); max-width: 66ch; margin-top: -.45rem; }
        .np-rule { height: 1px; background: var(--np-line); margin: 1.4rem 0; }
        .np-panel { border: 1px solid var(--np-line); border-radius: 12px; background: #fff; padding: 1.1rem 1.2rem; }
        .np-status { display: inline-block; color: #52605c; background: #e5ebe8; padding: .25rem .55rem; border-radius: 999px; font-size: .75rem; font-weight: 650; }
        .np-small { color: var(--np-muted); font-size: .88rem; }
        div[data-testid="stMetric"] { background: white; border: 1px solid var(--np-line); border-radius: 12px; padding: .85rem 1rem; }
        div[data-testid="stMetricLabel"] { color: var(--np-muted); }
        .stButton > button, .stDownloadButton > button { border-radius: 8px; }
        @media (max-width: 768px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .np-panel { padding: .9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, object | None]:
    apply_app_style()
    with st.sidebar:
        st.markdown('<div class="np-brand">NewsPulse</div>', unsafe_allow_html=True)
        st.caption("News coverage analysis")
        st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
        dataset_choice = st.radio("Dataset", ("Sample dataset", "Upload a file"), label_visibility="collapsed")
        uploaded_file = None
        if dataset_choice == "Upload a file":
            uploaded_file = st.file_uploader(
                "Choose a CSV or JSON file",
                type=("csv", "json"),
                help=f"Maximum file size: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )
            if uploaded_file and uploaded_file.size > MAX_UPLOAD_BYTES:
                st.error("This file exceeds the 20 MB upload limit.")
                uploaded_file = None
        st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
        st.caption("Analysis pages will be enabled as NLP pipelines are implemented.")
    return dataset_choice, uploaded_file


def render_page_header(title: str, subtitle: str, kicker: str = "NewsPulse") -> None:
    st.markdown(f'<div class="np-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="np-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_unavailable_page(title: str, summary: str, planned_work: str) -> None:
    render_page_header(title, summary, "In development")
    st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
    st.info("This analysis is not implemented yet. No results are being generated on this page.")
    st.markdown("**Planned capability**")
    st.write(planned_work)
