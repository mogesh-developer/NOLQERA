# NOLQERA

> A from-scratch NLP toolkit for understanding, building, and experimenting with Natural Language Processing systems.

NOLQERA is an experimental NLP toolkit built from the ground up with a strong focus on **understanding how NLP algorithms work internally** rather than simply wrapping existing libraries.

The project starts with classical NLP fundamentals and gradually evolves toward modern NLP systems such as embeddings, transformers, retrieval, document intelligence, and RAG.

---

## Vision

NOLQERA aims to become a modular NLP engine where each major NLP concept can be implemented, tested, understood, and reused independently.

```text
Text
 │
 ▼
Preprocessing & Tokenization
 │
 ▼
Feature Engineering & Vocabulary
 │
 ▼
Classical & Machine Learning NLP (Naive Bayes, Logistic Regression)
 │
 ▼
NOLQERA Intelligence Engines
 ├── Relevance Engine
 ├── Importance Engine
 ├── Keyphrase Engine
 ├── Entity Engine
 ├── Intent Engine
 └── Semantic Similarity Engine
 │
 ▼
Embeddings & Transformers (TF-IDF & Transformer Adapters)
 │
 ▼
Retrieval / RAG / Document Intelligence
```

---

## Current Status

### Version Overview

| Module                        | Status | Features / Components |
| ----------------------------- | ------ | --------------------- |
| **Text Preprocessing**        | Done   | Cleaning (HTML/URL), Normalization, Stopwords, Rule-based Stemmer, Dictionary Lemmatizer, Pipeline |
| **Tokenization**              | Done   | Sentence Tokenizer, Word Tokenizer (Contractions, Hyphens, Tanglish, Unicode, Emoji) |
| **Document Representation**   | Done   | `Sentence`, `Document` abstractions |
| **Vocabulary & Text Stats**   | Done   | Vocabulary Manager (Frequency Filtering, `UNK`), Text Statistics (TTR, Readability, Frequencies) |
| **Feature Extraction**        | Done   | N-Grams, Bag of Words (BoW), Mathematical TF-IDF Vectorizer |
| **Machine Learning NLP**      | Done   | Multinomial Naive Bayes, Logistic Regression (Gradient Descent, Cross-Entropy Loss), Metrics, Report |
| **Relevance Engine**          | Done   | Sentence relevance scoring & ranking using TF-IDF and Cosine Similarity |
| **Importance Engine**         | Done   | Sentence centrality ranking with position bias and length normalization |
| **Keyphrase Engine**          | Done   | N-gram candidate extraction, TF-IDF + Position scoring, Overlap deduplication |
| **Entity Engine**             | Done   | Candidate extraction, Contextual detection (PERSON, LOCATION, ORGANIZATION), Span cleanup |
| **Intent Engine**             | Done   | Interrogative & structural signal extraction, Evidence-based confidence scoring, Intent ranking |
| **Semantic Similarity**       | Done   | Cosine similarity, Qualitative scoring, TF-IDF & Transformer embedding providers |
| **Test Suite**                | Done   | 320+ unit and integration tests passing cleanly |

---

## Project Architecture

```text
nolqera/
│
├── nolqera/
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   ├── normalizer.py
│   │   ├── stopwords.py
│   │   ├── stemming.py
│   │   ├── lemmatization.py
│   │   └── pipeline.py
│   │
│   ├── tokenization/
│   │   ├── sentence_tokenizer.py
│   │   ├── word_tokenizer.py
│   │   └── tokenizer.py
│   │
│   ├── document/
│   │   ├── sentence.py
│   │   └── document.py
│   │
│   ├── features/
│   │   ├── vocabulary.py
│   │   ├── frequency.py
│   │   ├── text_statistics.py
│   │   ├── ngrams.py
│   │   ├── bow.py
│   │   └── tfidf.py
│   │
│   ├── classification/
│   │   ├── naive_bayes.py
│   │   ├── logistic_regression.py
│   │   ├── text_classifier.py
│   │   ├── metrics.py
│   │   └── report.py
│   │
│   └── intelligence/
│       ├── relevance/
│       │   ├── similarity.py
│       │   ├── scorer.py
│       │   ├── ranking.py
│       │   ├── models.py
│       │   └── engine.py
│       │
│       ├── importance/
│       │   ├── scorer.py
│       │   ├── ranking.py
│       │   ├── models.py
│       │   └── engine.py
│       │
│       ├── keyphrase/
│       │   ├── candidates.py
│       │   ├── scorer.py
│       │   ├── ranking.py
│       │   ├── models.py
│       │   └── engine.py
│       │
│       ├── entities/
│       │   ├── candidates.py
│       │   ├── detector.py
│       │   ├── cleanup.py
│       │   ├── ranking.py
│       │   ├── models.py
│       │   └── engine.py
│       │
│       ├── intent/
│       │   ├── candidates.py
│       │   ├── classifier.py
│       │   ├── scorer.py
│       │   ├── ranking.py
│       │   ├── models.py
│       │   └── engine.py
│       │
│       └── semantic_similarity/
│           ├── similarity.py
│           ├── scorer.py
│           ├── ranking.py
│           ├── models.py
│           ├── engine.py
│           └── embeddings/
│               ├── base.py
│               ├── tfidf.py
│               └── transformer.py
│
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Installation

NOLQERA requires **Python 3.11** or newer.

```bash
# Clone repository
git clone https://github.com/mogesh-developer/NOLQERA.git
cd NOLQERA

# Create & activate virtual environment
python -m venv venv
# On Windows PowerShell:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install package with development dependencies
pip install -e ".[dev]"
```

---

## Quick Start & Core Usage

### 1. Preprocessing & Tokenization

```python
from nolqera import preprocess, Tokenizer
from nolqera.preprocessing import PreprocessingPipeline

# Unified convenience function
clean_text = preprocess("Visit https://example.com! We're building NOLQERA.")
print(clean_text)

# Tokenizer
tokenizer = Tokenizer()
sentences = tokenizer.sentences("FastAPI is fast. MongoDB stores data.")
words = tokenizer.words("Hello world! How are you?")
print("Sentences:", sentences)
print("Words:", words)

# Configurable Preprocessing Pipeline
pipeline = PreprocessingPipeline(
    remove_stopwords=True,
    stemming=True,
)
processed_tokens = pipeline.process("FastAPI provides REST API endpoints.")
print("Processed:", processed_tokens)
```

---

### 2. Feature Engineering & Vectorization

```python
from nolqera import TfidfVectorizer, BagOfWords

documents = [
    ["fastapi", "web", "framework"],
    ["mongodb", "nosql", "database"],
    ["fastapi", "mongodb", "backend"],
]

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()
tfidf_vectors = tfidf.fit_transform(documents)

print("Vocabulary:", tfidf.vocabulary_list)
print("TF-IDF Matrix:", tfidf_vectors)
```

---

### 3. Classification & Evaluation

```python
from nolqera.classification import NaiveBayesClassifier, LogisticRegressionClassifier, classification_report

# Train Naive Bayes text classifier
classifier = NaiveBayesClassifier()
X_train = [["fastapi", "api"], ["mongodb", "database"], ["python", "api"]]
y_train = ["tech", "database", "tech"]

classifier.fit(X_train, y_train)
predictions = classifier.predict([["fastapi", "framework"]])
print("Prediction:", predictions)
```

---

### 4. NOLQERA Intelligence Engines

NOLQERA comes equipped with standalone intelligence engines:

#### Relevance Engine
Calculates text-query relevance using TF-IDF and Cosine Similarity:
```python
from nolqera.intelligence.relevance import RelevanceEngine

engine = RelevanceEngine()
query = "What database does the application use?"
sentences = [
    "The application is built using FastAPI.",
    "The application uses MongoDB for data storage.",
]

results = engine.analyze(query, sentences)
print("Most Relevant:", results[0].sentence, "| Score:", results[0].score)
```

#### Importance Engine
Ranks key sentences within a document based on informative TF-IDF density and positional weighting:
```python
from nolqera.intelligence.importance import ImportanceEngine

engine = ImportanceEngine()
document = [
    "FastAPI provides REST API endpoints.",
    "The application uses MongoDB for data storage.",
]

ranked = engine.analyze(document)
print("Top Sentence:", ranked[0].sentence, "| Importance:", ranked[0].score)
```

#### Keyphrase Engine
Extracts informative n-gram keyphrases and eliminates overlapping terms:
```python
from nolqera.intelligence.keyphrase import KeyphraseEngine

engine = KeyphraseEngine()
text = "The application uses FastAPI for REST APIs. MongoDB is used for persistent data storage."

keyphrases = engine.extract(text, top_k=3)
for kp in keyphrases:
    print(f"{kp.rank}. {kp.phrase} (score: {kp.score:.4f})")
```

#### Entity Engine
Extracts candidate entity spans, infers types (`PERSON`, `LOCATION`, `ORGANIZATION`), and resolves overlapping boundaries:
```python
from nolqera.intelligence.entities import EntityEngine

engine = EntityEngine()
text = "Dr John travelled to Chennai and studied at American College."

entities = engine.analyze(text)
for entity in entities:
    print(f"{entity.text:<20} {entity.entity_type:<15} score: {entity.score:.4f}")
```

#### Intent Engine
Extracts intent signals (interrogatives, structure, keywords) and produces confidence-weighted intent classifications:
```python
from nolqera.intelligence.intent import IntentEngine

engine = IntentEngine()
intents = engine.analyze("How does FastAPI work?")
print("Detected Intent:", intents[0].intent, "| Score:", intents[0].score)
```

#### Semantic Similarity Engine
Computes semantic similarity between text spans using TF-IDF or Transformer embedding providers:
```python
from nolqera.intelligence.semantic_similarity import SemanticSimilarityEngine
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import TFIDFEmbeddingProvider

# Initialize provider and fit on corpus
provider = TFIDFEmbeddingProvider()
provider.fit([
    ["fastapi", "backend", "api"],
    ["mongodb", "database", "storage"]
])

engine = SemanticSimilarityEngine(provider)
result = engine.compare(["fastapi", "backend"], ["fastapi", "api"])
print(f"Similarity: {result.score:.4f} | {result.text_a} <-> {result.text_b}")
```

---

## Design Philosophy

NOLQERA follows core engineering principles:

1. **Understand Before Abstracting**: Every algorithm (TF-IDF, Naive Bayes, Logistic Regression, Cosine Similarity, Keyphrase Scorer) is built from mathematical fundamentals.
2. **Zero Bloat / Minimal Dependencies**: Core NLP & intelligence modules run with standard Python libraries.
3. **Modular Component Architecture**: Every component is decoupled and independently testable.
4. **Comprehensive Test Coverage**: Strict unit and integration tests covering happy paths, edge cases, type errors, and boundary validations.

---

## Roadmap

- [x] **Phase 1 — Classical NLP Foundation**: Cleaning, Normalization, Stopwords, Stemming, Lemmatization, Tokenization, BoW, TF-IDF.
- [x] **Phase 2 — Vocabulary & Text Statistics**: Vocabulary Manager, UNK token handling, Readability scores, Lexical Diversity (TTR), Term Frequencies.
- [x] **Phase 3 — Machine Learning NLP**: Naive Bayes, Logistic Regression (Gradient Descent, Cross-Entropy), Metrics & Reports.
- [x] **Phase 4 — NOLQERA Intelligence Suite**:
  - [x] Sentence Relevance Engine
  - [x] Document Importance Engine
  - [x] Keyphrase Extraction & Deduplication Engine
  - [x] Entity Detection & Span Cleaner Engine
  - [x] Intent Signal & Classification Engine
  - [x] Semantic Similarity & Embedding Providers (TF-IDF & Transformer adapters)
- [ ] **Phase 5 — Neural Embeddings & Transformers**: Custom Word2Vec (Skip-gram, CBOW), Dense Vector Store, Self-Attention & Transformer Blocks.
- [ ] **Phase 6 — Retrieval & RAG**: Vector Indexing, Hybrid Search (Sparse + Dense), Document Intelligence Pipeline.

---

## Development & Testing

Run the full pytest suite:

```powershell
pytest -v
```

Run specific test modules:

```powershell
pytest tests/test_relevance_engine.py -v
pytest tests/test_entity_engine.py -v
pytest tests/test_intent_engine.py -v
pytest tests/test_semantic_similarity_engine.py -v
```

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

### NOLQERA
**Learn NLP. Build NLP. Understand NLP.**
