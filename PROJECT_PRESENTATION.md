# NewsPulse Project Presentation

## Problem

Large article collections are difficult to inspect consistently. Repeated stories, related entities, sentiment patterns, and coverage changes can be missed when articles are read only one at a time.

## Motivation

NewsPulse turns a news dataset into an explorable evidence surface. It helps a presenter show what appears in a collection, how stories and entities relate, and how the collection’s mix changes across selected periods. It does not infer public opinion or causation.

## NLP methodology

1. Parse CSV/JSON records and validate the article schema.
2. Preserve source text and derive cleaned text for analysis.
3. Encode articles with `sentence-transformers/all-MiniLM-L6-v2` and cache the resulting vectors.
4. Run Hugging Face sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and NER (`dslim/bert-base-NER`) when analysis is requested.
5. Extract keywords with TF-IDF-style unigram/bigram scores.
6. Use cosine similarity for article recommendations, semantic search, and near-duplicate candidates; use normalized text for exact duplicates.
7. Project article vectors with UMAP and group them with KMeans. Build labels from cluster phrases and representative titles.
8. Aggregate existing story/entity/sentiment results over dated periods and compare story shares and mean embedding vectors across a selected chronological cutoff.

Clustering, search, and temporal calculations do not use an LLM.

## System architecture

```text
Sample CSV / uploaded CSV or JSON
                  ↓
        Ingestion and validation
                  ↓
        Preprocessing and dates
                  ↓
Cached embeddings, sentiment, and NER
                  ↓
Stories · search · entities · time comparisons
                  ↓
     Streamlit pages and Plotly charts
```

The app is a modular Python package under `src/newspulse/`. Streamlit owns navigation and interaction; reusable analysis logic is under `src/newspulse/nlp/`; page renderers are under `src/newspulse/pages/`. Model resources, article embeddings, and analysis outputs are cached and reused for unchanged inputs.

## Technologies

- Python 3.11+
- Streamlit and Plotly
- Pandas and NumPy
- Sentence Transformers and Hugging Face Transformers
- UMAP and scikit-learn KMeans
- Pytest

## Demo flow

1. Load the clearly synthetic sample collection.
2. Show article count, sources, coverage dates, and publishing activity.
3. Start on-demand NLP analysis and explain the embedding, sentiment, entity, and keyword outputs.
4. Explore story groups in the 2D semantic projection and inspect representative articles.
5. Search for a topic by meaning and review ranked article metadata and excerpts.
6. Inspect entity frequency, predicted sentiment, and co-occurrence connections.
7. Choose a chronological cutoff in Narrative Explorer. Compare story shares, inspect the percentage-point changes and representative articles, and review the embedding-centroid distance.
8. Open an article to review its source text and available similar-article/duplicate information.

## Results shown by the sample

The sample contains 38 synthetic articles dated January through September 2025. As a simple check on the authored text, 1 of 17 records before May contains a regulation/safety/audit/legislation keyword (5.9%), compared with 9 of 21 records from May onward (42.9%). Narrative Explorer independently calculates story shares from the loaded records and displays actual period counts and percentage-point deltas. These numbers describe only this synthetic demonstration collection; they are not model accuracy measures or evidence about real-world coverage.

## Limitations

- Model sentiment and NER are predictions and can be wrong.
- UMAP coordinates are qualitative and have no absolute axis interpretation.
- KMeans requires a chosen cluster count; a cluster is not guaranteed to map perfectly to a human-defined topic.
- Current period comparisons use one fixed set of whole-dataset cluster assignments, not independently discovered and aligned period-specific topics.
- Missing dates and small period samples reduce available temporal evidence.
- Pretrained weights require memory and may need an initial download. CPU-only PyTorch and the optional NLP extra must be installed to run model inference.
- The sample corpus is synthetic and small; conclusions must be limited to the records loaded.

## Future work

- Add cluster review/renaming and assess clustering stability.
- Add period-specific topic alignment and validated distribution-shift measures.
- Evaluate multilingual, larger-corpus, and out-of-domain performance.
- Add report export, model/runtime telemetry, and more browser-level regression tests.
