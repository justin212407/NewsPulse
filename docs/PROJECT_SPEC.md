# NewsPulse — Narrative Intelligence Engine

## 1. Project Overview

NewsPulse is an interactive NLP platform that analyzes collections of news articles and discovers how topics, entities, sentiment, and narratives evolve over time.

The goal is not simply to classify individual articles. NewsPulse treats a collection of articles as a dynamic information ecosystem and uses NLP to answer questions such as:

- What are the major stories being discussed?
- Which topics are emerging or declining?
- Which entities are most frequently mentioned?
- How is sentiment changing over time?
- Which articles are duplicates or near-duplicates?
- How are different stories semantically related?
- How has the narrative surrounding a topic changed over time?
- What are the most important semantic shifts in the dataset?

The application is implemented as an interactive Streamlit dashboard.

---

# 2. Core Problem

Traditional news analysis usually requires manually reading large numbers of articles.

A user may have hundreds or thousands of articles and want to understand:

1. What the articles are about.
2. Which articles belong to the same story.
3. Which topics are becoming important.
4. Which entities are driving discussion.
5. How sentiment changes.
6. How narratives evolve over time.

NewsPulse automates this analysis using NLP and presents the results through interactive visualizations.

---

# 3. Main NLP Capabilities

NewsPulse should combine several NLP techniques into one coherent pipeline.

## 3.1 Text Preprocessing

The system should:

- Clean article text.
- Normalize whitespace.
- Remove obvious HTML artifacts.
- Handle missing values.
- Normalize dates.
- Remove duplicate articles where appropriate.
- Preserve the original article text for analysis.

---

## 3.2 Sentence Embeddings

Use a pretrained Sentence Transformers model to convert articles into semantic vectors.

Recommended initial model:

`sentence-transformers/all-MiniLM-L6-v2`

Embeddings should be cached so that restarting or navigating the Streamlit application does not repeatedly recompute them.

Embeddings will power:

- Semantic similarity.
- Story clustering.
- Duplicate detection.
- Semantic search.
- Narrative shift analysis.

---

# 4. Story Discovery

The system should automatically group semantically similar articles into story clusters.

Recommended initial pipeline:

```text
Article Text
     ↓
Sentence Transformer
     ↓
Embeddings
     ↓
UMAP
     ↓
KMeans
     ↓
Story Clusters
```

The exact clustering implementation may be changed if a more reliable lightweight approach is available.

Each cluster should receive an automatically generated human-readable label using:

- TF-IDF keywords
- Key phrases
- Most representative article titles

Example:

```text
Cluster 1
"AI Regulation and Government Policy"

Cluster 2
"Open Source AI Models"

Cluster 3
"AI Agents and Autonomous Systems"
```

The system should show:

- Number of articles per cluster.
- Cluster percentage.
- Top keywords.
- Representative articles.
- Cluster sentiment.
- Cluster activity over time.

---

# 5. Topic Evolution

NewsPulse should analyze how story clusters change over time.

For each cluster, calculate:

- Article count over time.
- Percentage of total coverage.
- Growth/decline rate.
- Sentiment over time.

Example:

```text
AI Agents

Monday       12 articles
Tuesday      18 articles
Wednesday    27 articles
Thursday     39 articles
Friday       51 articles

Trend: +325%
```

The dashboard should identify:

### Emerging Topics

Topics with rapidly increasing coverage.

### Declining Topics

Topics whose coverage is decreasing.

### Stable Topics

Topics maintaining relatively consistent coverage.

---

# 6. Named Entity Recognition

Use a pretrained NER model to extract entities.

Recommended model:

`dslim/bert-base-NER`

Entities should include categories such as:

- PERSON
- ORGANIZATION
- LOCATION
- MISC

The dashboard should display:

```text
Top Entities

OpenAI              147 mentions
Google              112 mentions
Microsoft            93 mentions
United States        81 mentions
Sam Altman            72 mentions
```

Users should be able to select an entity and see:

- Mention count.
- Articles mentioning it.
- Sentiment associated with it.
- Related entities.
- Mention frequency over time.

---

# 7. Entity Relationship Graph

Build an entity co-occurrence graph.

If two entities frequently occur in the same article, create an edge between them.

Example:

```text
             EU
             │
             │
       regulation
             │
OpenAI ───── AI ───── Microsoft
   │                    │
   │                    │
investment          partnership
   │                    │
   └──────── Google ────┘
```

The graph should be interactive if practical.

Use Plotly or another lightweight visualization library.

---

# 8. Sentiment Analysis

Use a pretrained sentiment model.

The exact model can be selected based on compatibility and performance.

The system should calculate sentiment for each article.

At minimum:

- Positive
- Neutral
- Negative

The dashboard should show:

- Overall sentiment distribution.
- Sentiment by story.
- Sentiment over time.
- Entity-associated sentiment.

Example:

```text
Overall Sentiment

Positive    38%
Neutral     41%
Negative    21%
```

---

# 9. Keyword Extraction

Extract important keywords and phrases using lightweight NLP techniques.

Possible approaches:

- TF-IDF
- N-grams
- KeyBERT if practical
- Frequency analysis

The system should expose:

- Global keywords.
- Cluster-specific keywords.
- Time-specific emerging keywords.

---

# 10. Duplicate and Near-Duplicate Detection

News organizations frequently publish similar versions of the same story.

NewsPulse should detect:

- Exact duplicates.
- Near-duplicate articles.
- Highly similar articles.

Use embeddings and cosine similarity.

Example:

```text
Article A
"OpenAI releases new reasoning model..."

Article B
"OpenAI has announced its latest reasoning model..."

Similarity: 94%
```

This allows the dashboard to distinguish between:

> 100 articles

and

> 100 articles covering approximately 14 unique stories.

---

# 11. Semantic Search

Users should be able to search the entire dataset semantically.

Instead of requiring exact keywords, the user can enter:

> "articles about government regulation of artificial intelligence"

and retrieve semantically related articles even if the exact phrase does not appear.

Pipeline:

```text
User Query
    ↓
Sentence Transformer
    ↓
Query Embedding
    ↓
Cosine Similarity
    ↓
Top-K Articles
```

Results should show:

- Article title.
- Source.
- Date.
- Similarity score.
- Short excerpt.

---

# 12. Narrative Evolution

This is the central differentiating feature of NewsPulse.

The system should compare semantic/topic distributions across different time periods.

For example:

### Early Coverage

```text
AI Innovation
AI Models
Performance
Product Launches
```

### Later Coverage

```text
AI Safety
Regulation
Privacy
Government Policy
```

The application should identify this as a narrative shift.

Possible implementation:

1. Divide articles into chronological windows.
2. Calculate cluster/topic distributions for each window.
3. Calculate embedding centroids for major clusters.
4. Compare topic distributions and semantic representations.
5. Highlight significant changes.

Example output:

```text
Narrative Shift Detected

Topic: AI Regulation

Previous Period: 12%
Current Period: 34%

Change: +22 percentage points

Related emerging terms:

regulation
government
policy
safety
legislation
```

This does not need to claim causal truth. It should be presented as **observed semantic/topic change in the dataset**.

---

# 13. Dashboard Structure

The Streamlit application should contain the following pages.

## 13.1 Overview

Display:

- Total articles.
- Date range.
- Number of discovered stories.
- Number of entities.
- Average article length.
- Overall sentiment.
- Emerging topics.
- Declining topics.

Include a timeline and story distribution chart.

---

## 13.2 Story Explorer

Display all discovered clusters.

For each story:

- Story label.
- Article count.
- Trend.
- Sentiment.
- Keywords.
- Representative articles.

Allow the user to select a story and inspect it.

---

## 13.3 Entity Explorer

Display:

- Top entities.
- Entity frequency.
- Entity trends.
- Entity sentiment.
- Related entities.

Include an entity relationship visualization.

---

## 13.4 Narrative Explorer

Display:

- Topic evolution.
- Sentiment evolution.
- Emerging topics.
- Declining topics.
- Narrative shifts.

Allow comparison between two time periods.

---

## 13.5 Semantic Search

Provide a search box.

Example:

> "articles discussing AI regulation in Europe"

Return the most semantically similar articles.

---

## 13.6 Article Explorer

Allow the user to select an individual article and see:

- Title.
- Source.
- Date.
- Text.
- Sentiment.
- Entities.
- Keywords.
- Story cluster.
- Similar articles.

---

# 14. Recommended Technology Stack

## Frontend

- Streamlit
- Plotly

## NLP

- Hugging Face Transformers
- Sentence Transformers
- spaCy where useful
- scikit-learn

## Embeddings

`sentence-transformers/all-MiniLM-L6-v2`

## NER

`dslim/bert-base-NER`

## Dimensionality Reduction

UMAP

## Clustering

KMeans initially, with HDBSCAN as an optional enhancement if stable.

## Similarity

Cosine similarity

## Data

Pandas

## Optional Storage

For the initial project, local files and cached objects are sufficient.

Avoid unnecessary databases unless they provide clear value.

---

# 15. Dataset

The application should support CSV input with at least:

```text
id
title
text
source
published_at
url
```

The repository should include a small sample dataset so the application works immediately after installation.

The sample dataset should contain several hundred articles if practical.

A smaller dataset is acceptable for development.

The system should also support users uploading their own CSV.

---

# 16. Important Design Principle

This is an **NLP analytics system**, not a generic chatbot.

Do not turn the project into a generic LLM wrapper.

The important parts are:

- Embeddings
- Clustering
- NER
- Sentiment
- Similarity
- Topic evolution
- Entity relationships
- Semantic search
- Narrative analysis

LLMs may optionally be used to generate human-readable cluster labels or narrative summaries, but the core analytics should work without an LLM.

---

# 17. User Experience

The application should feel like a professional analytics product.

Use:

- Sidebar navigation.
- Metric cards.
- Interactive Plotly charts.
- Expandable article sections.
- Tabs where appropriate.
- Clear loading states.
- Cached models.
- Error handling.
- Empty-state messages.
- Sample dataset.
- Consistent typography and spacing.

Avoid making the UI look like a collection of unrelated Streamlit widgets.

---

# 18. Final Demonstration Flow

The class presentation should demonstrate the following:

### Step 1

Load the sample news dataset.

### Step 2

Show the overview:

```text
1,000 Articles
15 Story Clusters
2,800 Entities
```

### Step 3

Show the story clustering map.

Explain that articles were grouped based on semantic embeddings rather than simple keyword matching.

### Step 4

Select an emerging story.

Show:

- Article count.
- Keywords.
- Sentiment.
- Timeline.

### Step 5

Select an entity.

Show:

- Mentions.
- Related entities.
- Sentiment.
- Timeline.

### Step 6

Compare two time periods.

Demonstrate:

```text
Early Narrative
    ↓
AI innovation

Later Narrative
    ↓
AI regulation and safety
```

### Step 7

Use semantic search.

Ask:

> "Show me articles discussing government regulation of AI."

### Step 8

Open an article and show its complete NLP analysis.

---

# 19. What Makes NewsPulse Different

The project should be presented as:

> **An NLP system for discovering and tracking narratives across large collections of news articles.**

The novelty is not a new NLP model.

The value comes from **combining multiple NLP techniques into a coherent analytical workflow**.

The project moves through:

```text
Raw Articles
     ↓
Language Understanding
     ↓
Semantic Representation
     ↓
Story Discovery
     ↓
Entity Intelligence
     ↓
Temporal Analysis
     ↓
Narrative Evolution
     ↓
Interactive Visualization
```

This makes it substantially more interesting than a standalone sentiment classifier, summarizer, or chatbot.

---

# 20. Initial Scope

The first version should prioritize:

1. Article ingestion
2. Preprocessing
3. Embeddings
4. Story clustering
5. NER
6. Sentiment
7. Keywords
8. Semantic search
9. Topic evolution
10. Narrative shift visualization
11. Streamlit dashboard

Optional features such as LLM-generated narrative explanations, advanced graph layouts, HDBSCAN and external news APIs should only be added after the core system is working.