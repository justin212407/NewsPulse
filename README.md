# NewsPulse

NewsPulse is a Streamlit foundation for exploring news coverage and, over time, how topics and narratives change.

The current application provides CSV and JSON ingestion, record validation, text preprocessing, a dataset overview, and article browsing. Story clustering, entity analysis, narrative analysis, and semantic search are explicitly marked as not implemented; no NLP results are fabricated.

## Run locally

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

The sample dataset loads automatically when the app starts. Select **Upload a file** in the sidebar to load a CSV or JSON file instead. Uploads are limited to 20 MB.

## Article data format

Each article supports these fields:

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | Stable article identifier |
| `title` | Yes | Article headline |
| `text` | Yes | Article body |
| `source` | No | Publication or source name |
| `published_at` | No | Date/time accepted by pandas date parsing; normalized to UTC |
| `url` | No | HTTP(S) article URL |

CSV files use a header row. JSON may be a list of article objects, one article object, or an object with an `articles` list.

Rows missing `id`, `title`, or usable text are skipped and reported. Later rows with duplicate IDs are skipped. Missing or invalid dates, missing sources, and malformed URLs are retained as warnings. The input body is preserved in `text`; `text_clean` is a display/search derivative with HTML tags removed and whitespace normalized.

## Application pages

- **Overview** shows article, source, date range, publishing activity, and import-quality summaries.
- **Article Explorer** filters, searches, and displays records from the loaded dataset.
- **Story Explorer**, **Entity Explorer**, **Narrative Explorer**, and **Semantic Search** explain their planned capability and clearly state that analysis is not implemented yet.

## Tests

```bash
pytest
```

Runtime dependencies are declared in `pyproject.toml` and pinned for reproducible setup. NLP/model dependencies will be added when their corresponding pipelines are implemented.
