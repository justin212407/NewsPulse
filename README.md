# NewsPulse

NewsPulse is a Streamlit application for exploring how topics, entities, sentiment, and story coverage appear and change across a collection of news articles.

The included collection is synthetic and exists to demonstrate the application flow. Uploaded data remains the basis for any analysis results.

## Problem

Reading a large collection article by article makes recurring stories, entity relationships, duplicate coverage, and changes across time difficult to inspect consistently. NewsPulse combines deterministic NLP methods and pretrained models to help explore those patterns without turning the application into a chatbot or making claims about public opinion or causation.

## Architecture

The application keeps ingestion, analysis, and presentation separate:

```text
CSV / JSON upload or sample data
              ↓
Parsing, schema validation, date normalization
              ↓
Original text + cleaned text
              ↓
Cached NLP analysis
 embeddings · sentiment · NER · TF-IDF · similarity · duplicates
              ↓
UMAP + KMeans stories · semantic search · temporal comparisons
              ↓
Streamlit pages with Plotly visualizations
```

`app.py` loads the selected dataset and registers the six pages. `src/newspulse/ingestion.py` and `cleaning.py` parse and validate records. Reusable model and feature services live under `src/newspulse/nlp/`. Page renderers live under `src/newspulse/pages/`, with shared presentation components under `src/newspulse/ui/`.

Model resources and article embeddings are cached. Analysis is initiated on demand, and changing a dataset invalidates the result cache. Failed or unavailable model features are reported without preventing article browsing or deterministic features from working.

## NLP techniques and models

- **Text preprocessing:** removes HTML artifacts and normalizes whitespace while preserving the source article body.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` represents article and query text as normalized vectors.
- **Story discovery:** UMAP creates a qualitative two-dimensional projection; KMeans assigns semantic story groups. Cluster labels use TF-IDF phrases and representative article titles.
- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` predicts positive, neutral, or negative labels.
- **Named entities:** `dslim/bert-base-NER` extracts entity text and type.
- **Keywords:** lightweight TF-IDF-style unigram and bigram scoring.
- **Similarity and duplicates:** cosine similarity ranks related articles and identifies near-duplicate candidates. Exact duplicates use normalized article text.
- **Temporal analysis:** story coverage shares, sentiment proportions, entity counts/co-occurrence, period comparisons, and cosine distance between period embedding centroids. Comparisons describe only observed data changes.

No LLM is used for clustering, ranking, or similarity calculations.

## Installation and running

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

The app starts with the included sample dataset and can be run with the simple command `streamlit run app.py` after installation.

### Optional NLP models

Model inference requires the optional NLP dependencies and CPU-only PyTorch. Install PyTorch first to avoid pulling CUDA libraries into a CPU environment:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[nlp]"
```

The first analysis run may download model weights from Hugging Face. Internet access is required for uncached weights. Without the optional backend, the dashboard and dataset views remain available and model-dependent pages show an actionable unavailable state.

## Dataset format

CSV files need a header row. JSON may be an article list, one article object, or an object containing an `articles` list.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Stable article identifier |
| `title` | Yes | Headline |
| `text` | Yes | Original article body |
| `source` | No | Publication/source name |
| `published_at` | No | Parseable publication date/time; normalized to UTC |
| `url` | No | HTTP(S) source URL |

Records missing `id`, `title`, or usable text are skipped and reported. Duplicate IDs after the first occurrence are skipped. Invalid/missing dates, absent sources, and malformed optional URLs are retained with warnings. Original text is kept in `text`; `text_clean` is a derived display and analysis field. Uploads are limited to 20 MB.

## Pages and demo flow

1. **Overview:** inspect article/source counts, coverage dates, publishing activity, data validation, and available analysis summaries.
2. **Story Explorer:** run or reuse cached embeddings, inspect the UMAP story map, choose a cluster, and review its share, sentiment, keywords, representative articles, and duplicate links.
3. **Entity Explorer:** choose a time interval, compare entity frequency and predicted entity-associated sentiment, then inspect all-time or period-specific co-occurrence links.
4. **Narrative Explorer:** choose time granularity and a comparison cutoff, then inspect story shares, emerging/declining coverage, sentiment mix, percentage-point shifts, representative articles, and embedding-centroid distance.
5. **Semantic Search:** submit a natural-language query and inspect top-K results with source, date, similarity score, and excerpt.
6. **Article Explorer:** filter, search, and inspect original/cleaned text with available article annotations and recommendations.

### Example screenshots

Screenshots are not committed yet. Capture these views when preparing a presentation:

- `[Screenshot placeholder: Overview with the sample collection loaded]`
- `[Screenshot placeholder: Story Explorer UMAP map and selected cluster]`
- `[Screenshot placeholder: Narrative Explorer period comparison and percentage-point shift]`
- `[Screenshot placeholder: Entity co-occurrence graph]`
- `[Screenshot placeholder: Semantic Search result list]`

## Sample collection and interpretation

`data/sample_articles.csv` contains 38 synthetic articles from January through September 2025. Recurring AI coverage moves from product/model releases in the earlier sample periods toward regulation, audits, safety evaluation, and incident reporting in later periods, alongside repeated energy, transit, health, and environment coverage. In this file, 1 of 17 articles before May contains one of “regulation”, “safety”, “audit”, or “legislation” (5.9%); 9 of 21 articles from May onward contain one (42.9%). This is a keyword-based property of this authored synthetic collection, not a representation of real news volume, sentiment, or public opinion.

Story assignments come from one clustering of the loaded collection and remain fixed when comparing periods. A displayed percentage-point change is `current story share - previous story share`, multiplied by 100. Each period’s denominator is its number of dated articles with a story assignment. Semantic shift is cosine distance between the mean valid article embeddings in the selected periods. The app displays period sample sizes and excludes undated articles from time-based calculations.

## Limitations

- Sentiment and NER are model predictions, not ground truth.
- UMAP is a qualitative projection; its axes do not have absolute units.
- KMeans requires a selected cluster count and can group borderline stories imperfectly.
- Fixed full-dataset story assignments are compared over time; stories are not independently reclustered and aligned in each period.
- Small date ranges, missing dates, sparse entity mentions, or failed model inference can limit temporal comparisons.
- The sample data is synthetic and intentionally small; it demonstrates workflow, not generalizable findings.
- NLP models need CPU/GPU memory and initial model downloads; only CPU installation is documented by default.

## Future improvements

- Add explicit user review and refinement of story labels and cluster counts.
- Add period-specific topic discovery with a validated cross-period alignment method.
- Add multilingual evaluation, configurable time zones, and additional date granularities.
- Add exportable analysis reports and dedicated UI smoke tests for uploaded datasets.
- Measure performance on larger collections and use scalable nearest-neighbor indexing where appropriate.

## Tests

```bash
pytest -q
```
