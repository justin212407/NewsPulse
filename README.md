# NewsPulse

NewsPulse is a Streamlit foundation for exploring news coverage and, over time, how topics and narratives change.

The current application provides CSV and JSON ingestion, record validation, text preprocessing, a dataset overview, article browsing, and an on-demand NLP analysis pipeline. Story clustering, narrative evolution, and semantic search are explicitly marked as not implemented; no NLP results are fabricated.

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

## NLP analysis

Install CPU-only PyTorch first, then install the optional NLP dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[nlp]"
```

The first analysis run downloads model weights from Hugging Face and runs on CPU by default.

The analysis is started explicitly from Overview or Article Explorer and cached for unchanged article IDs and text. It uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment, `dslim/bert-base-NER` for entities, TF-IDF unigrams and bigrams for keywords, and cosine similarity for article matches and near-duplicate candidates. Exact duplicates are identified from normalized article text. If a model cannot load or infer, its feature is marked unavailable while other features and article browsing continue.

## Application pages

- **Overview** shows article/source/date summaries, activity, sentiment counts, dataset keywords, and duplicate counts when analysis succeeds.
- **Article Explorer** filters and searches articles, then displays their sentiment, entities, keywords, similar articles, and duplicate matches when available.
- **Story Explorer**, **Narrative Explorer**, and **Semantic Search** explain their planned capability and clearly state that analysis is not implemented yet.

## Tests

```bash
pytest
```

Core runtime and optional NLP dependencies are declared in `pyproject.toml`; runtime versions are pinned for reproducible setup. PyTorch is installed separately from its CPU-only package index to avoid pulling CUDA libraries into CPU deployments.
