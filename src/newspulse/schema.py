from typing import Final

ARTICLE_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "title",
    "text",
    "source",
    "published_at",
    "url",
)
REQUIRED_FIELDS: Final[tuple[str, ...]] = ("id", "title", "text")
OPTIONAL_FIELDS: Final[tuple[str, ...]] = ("source", "published_at", "url")
