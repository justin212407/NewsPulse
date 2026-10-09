from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_articles.csv"
PAGE_ICON = "N"
PAGE_LAYOUT = "wide"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
REQUIRED_COLUMNS = ("id", "title", "text")
ARTICLE_COLUMNS = ("id", "title", "text", "source", "published_at", "url")
PAGE_TITLES = (
    "Overview",
    "Story Explorer",
    "Entity Explorer",
    "Narrative Explorer",
    "Semantic Search",
    "Article Explorer",
)
