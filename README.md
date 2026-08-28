# NOLQERA

<p align="center">
  <img src="https://raw.githubusercontent.com/pkmer/pkmer-docs/main/docs/assets/logo.png" width="120" alt="NOLQERA Logo" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <b>A From-Scratch Natural Language Processing (NLP) Engine & Intelligence Suite</b>
</p>

<p align="center">
  <a href="https://github.com/mogesh-developer/NOLQERA"><img src="https://img.shields.io/badge/python-3.11+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge" alt="License MIT"></a>
  <a href="https://github.com/mogesh-developer/NOLQERA"><img src="https://img.shields.io/badge/tests-320%2B%20passed-brightgreen.svg?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests"></a>
  <a href="https://github.com/mogesh-developer/NOLQERA"><img src="https://img.shields.io/badge/architecture-modular-orange.svg?style=for-the-badge" alt="Architecture"></a>
  <a href="https://github.com/mogesh-developer/NOLQERA"><img src="https://img.shields.io/badge/status-active--development-blueviolet.svg?style=for-the-badge" alt="Status"></a>
</p>

---

## 📌 Table of Contents

- [About NOLQERA](#-about-nolqera)
- [Architecture & Engine Design](#-architecture--engine-design)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quickstart Guide](#-quickstart-guide)
- [NOLQERA Intelligence Suite](#-nolqera-intelligence-suite)
  - [Sentence Relevance Engine](#1-sentence-relevance-engine)
  - [Document Importance Engine](#2-document-importance-engine)
  - [Keyphrase Extraction Engine](#3-keyphrase-extraction-engine)
  - [Named Entity Recognition Engine](#4-named-entity-recognition-engine)
  - [Intent Classification Engine](#5-intent-classification-engine)
  - [Semantic Similarity Engine](#6-semantic-similarity-engine)
- [Development & Testing](#-development--testing)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🚀 About NOLQERA

**NOLQERA** is a lightweight, high-performance NLP engine built from fundamental mathematical algorithms rather than wrapping high-level monolithic libraries. 

It is designed to give complete visibility into how text cleaning, tokenization, TF-IDF vectorization, Naive Bayes/Logistic Regression classification, entity extraction, intent recognition, and semantic similarity engines work under the hood.

> [!NOTE]
> **Zero Heavy Wrapper Dependencies**: NOLQERA's core algorithms run with zero bloat, pure mathematical precision, and full test coverage.

---

## 🏗 Architecture & Engine Design

```text
                                    ┌────────────────────────┐
                                    │    Input Raw Text      │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │ Preprocessing Pipeline │
                                    │ (Clean/Stem/Lemmatize) │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │ Tokenization & Features│
                                    │ (N-Grams / BoW / TFIDF)│
                                    └───────────┬────────────┘
                                                │
                                                ▼
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐                    ┌─────────────────┐
│ Machine Learning│                    │ Intelligence    │                    │ Embedding       │
│ Classification  │                    │ Engines         │                    │ Providers       │
├─────────────────┤                    ├─────────────────┤                    ├─────────────────┤
│ • Naive Bayes   │                    │ • Relevance     │                    │ • TF-IDF Vector │
│ • Logistic Reg. │                    │ • Importance    │                    │ • Transformer   │
│ • Metrics/Report│                    │ • Keyphrase     │                    │   Adapter       │
└─────────────────┘                    │ • Entity (NER)  │                    └─────────────────┘
                                       │ • Intent        │
                                       │ • Semantic Sim. │
                                       └─────────────────┘
```

---

## ✨ Key Features

| Capability | Component | Description | Status |
| :--- | :--- | :--- | :---: |
| **Preprocessing** | `nolqera.preprocessing` | HTML stripping, URL removal, stemming, lemmatization, custom pipeline | 🟢 `Done` |
| **Tokenization** | `nolqera.tokenization` | Sentence tokenizer, Word tokenizer (handles emoji, unicode, contractions, Tanglish) | 🟢 `Done` |
| **Vectorization** | `nolqera.features` | BoW, N-Grams, Mathematical TF-IDF Vectorizer with IDF smoothing | 🟢 `Done` |
| **Classification** | `nolqera.classification` | Multinomial Naive Bayes, Logistic Regression (Gradient Descent, Cross-Entropy) | 🟢 `Done` |
| **Relevance** | `nolqera.intelligence.relevance` | TF-IDF + Cosine similarity query-sentence relevance ranker | 🟢 `Done` |
| **Importance** | `nolqera.intelligence.importance` | Centrality sentence ranking with positional bias & informateness density | 🟢 `Done` |
| **Keyphrase** | `nolqera.intelligence.keyphrase` | N-gram phrase candidate extractor & deduplicated keyphrase ranker | 🟢 `Done` |
| **Entity (NER)** | `nolqera.intelligence.entities` | Contextual entity detector (`PERSON`, `LOCATION`, `ORGANIZATION`) & span cleaner | 🟢 `Done` |
| **Intent** | `nolqera.intelligence.intent` | Interrogative signal detector, confidence-weighted intent classification | 🟢 `Done` |
| **Semantic Sim** | `nolqera.intelligence.semantic_similarity` | Cosine similarity engine supporting TF-IDF & Transformer embedding providers | 🟢 `Done` |

---

## 💻 Installation

NOLQERA requires **Python 3.11+**.

```bash
# Clone repository
git clone https://github.com/mogesh-developer/NOLQERA.git
cd NOLQERA

# Create virtual environment
python -m venv venv

# Activate Virtual Environment
# Windows PowerShell:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## ⚡ Quickstart Guide

```python
from nolqera import preprocess, Tokenizer, TfidfVectorizer

# 1. Clean raw text
clean_text = preprocess("Check out NOLQERA at https://github.com/mogesh-developer/NOLQERA!")
print("Cleaned:", clean_text)

# 2. Tokenize into sentences and words
tokenizer = Tokenizer()
sentences = tokenizer.sentences("FastAPI is fast. MongoDB stores persistent data.")
words = tokenizer.words("Hello world! How are you?")
print("Sentences:", sentences)
print("Words:", words)

# 3. Compute TF-IDF Features
documents = [
    ["fastapi", "web", "framework"],
    ["mongodb", "nosql", "database"],
    ["fastapi", "mongodb", "backend"],
]
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(documents)
print("Vocabulary:", tfidf.vocabulary_list)
```

---

## 🧠 NOLQERA Intelligence Suite

NOLQERA features specialized intelligence engines designed for production NLP workflows.

<details>
<summary><b>1. Sentence Relevance Engine</b></summary>

Computes top-k sentence relevance scores against an incoming user query:

```python
from nolqera.intelligence.relevance import RelevanceEngine

engine = RelevanceEngine()
query = "What database does the application use?"
sentences = [
    "The application is built using FastAPI.",
    "The application uses MongoDB for data storage.",
    "Python is the programming language.",
]

results = engine.analyze(query, sentences)
print("Most Relevant:", results[0].sentence)
print("Relevance Score:", results[0].score)
```
</details>

<details>
<summary><b>2. Document Importance Engine</b></summary>

Extracts central sentences from a long document using TF-IDF density, position bias, and length normalization:

```python
from nolqera.intelligence.importance import ImportanceEngine

engine = ImportanceEngine()
document = [
    "FastAPI provides REST API endpoints.",
    "The application uses MongoDB for data storage.",
    "I travelled to Chennai yesterday.",
]

ranked = engine.analyze(document)
print("Top Sentence:", ranked[0].sentence, "| Importance Score:", ranked[0].score)
```
</details>

<details>
<summary><b>3. Keyphrase Extraction Engine</b></summary>

Extracts multi-word concepts, ranks them by TF-IDF & position, and removes overlapping redundant spans:

```python
from nolqera.intelligence.keyphrase import KeyphraseEngine

engine = KeyphraseEngine()
text = "The application uses FastAPI for REST APIs. MongoDB is used for persistent data storage."

keyphrases = engine.extract(text, top_k=3)
for kp in keyphrases:
    print(f"Rank {kp.rank}: {kp.phrase:<25} (Score: {kp.score:.4f})")
```
</details>

<details>
<summary><b>4. Named Entity Recognition Engine</b></summary>

Detects entity candidate spans, classifies entities (`PERSON`, `LOCATION`, `ORGANIZATION`), and resolves overlapping boundaries:

```python
from nolqera.intelligence.entities import EntityEngine

engine = EntityEngine()
text = "Dr John travelled to Chennai and studied at American College."

entities = engine.analyze(text)
for entity in entities:
    print(f"{entity.text:<20} {entity.entity_type:<15} Score: {entity.score:.4f}")
```
</details>

<details>
<summary><b>5. Intent Classification Engine</b></summary>

Extracts linguistic intent signals, combines evidence, and ranks user intent classifications:

```python
from nolqera.intelligence.intent import IntentEngine

engine = IntentEngine()
intents = engine.analyze("How does FastAPI work?")
print("Intent:", intents[0].intent, "| Confidence:", intents[0].score)
```
</details>

<details>
<summary><b>6. Semantic Similarity Engine</b></summary>

Measures semantic similarity between text tokens using customizable embedding providers (TF-IDF & Transformer models):

```python
from nolqera.intelligence.semantic_similarity import SemanticSimilarityEngine
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import TFIDFEmbeddingProvider

# 1. Initialize & fit embedding provider
provider = TFIDFEmbeddingProvider()
provider.fit([
    ["fastapi", "backend", "api"],
    ["mongodb", "database", "storage"],
])

# 2. Run Semantic Similarity Engine
engine = SemanticSimilarityEngine(provider)
result = engine.compare(["fastapi", "backend"], ["fastapi", "api"])
print(f"Similarity: {result.score:.4f} | {result.text_a} <-> {result.text_b}")
```
</details>

---

## 🧪 Development & Testing

NOLQERA is built using strict test-driven development (TDD) with over 320+ unit and integration tests.

```powershell
# Run the complete test suite
pytest -v

# Run tests for specific intelligence engines
pytest tests/test_semantic_similarity_engine.py -v -s
pytest tests/test_entity_engine.py -v -s
pytest tests/test_intent_engine.py -v -s
```

---

## 🗺 Roadmap

- [x] **Phase 1 — Classical NLP Foundation**: Preprocessing, Stemming, Lemmatization, Tokenization, BoW, TF-IDF.
- [x] **Phase 2 — Text Statistics & Vocabulary**: Vocabulary Manager (`UNK`), Readability scores, Lexical Diversity (TTR).
- [x] **Phase 3 — Machine Learning NLP**: Naive Bayes, Logistic Regression (Gradient Descent, Cross-Entropy), Metrics & Reports.
- [x] **Phase 4 — NOLQERA Intelligence Suite**: Relevance, Importance, Keyphrase, Entity, Intent, and Semantic Similarity Engines.
- [ ] **Phase 5 — Neural Embeddings**: Custom Word2Vec (Skip-gram, CBOW), Dense Vector Store.
- [ ] **Phase 6 — Transformers & RAG**: Attention mechanisms, Vector Search Indexing, RAG Pipeline.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  <b>NOLQERA</b> • <i>Learn NLP. Build NLP. Understand NLP.</i>
</p>
