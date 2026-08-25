# NOLQERA

> A from-scratch NLP toolkit for understanding, building, and experimenting with Natural Language Processing systems.

NOLQERA is an experimental NLP toolkit built from the ground up with a strong focus on **understanding how NLP algorithms work internally** rather than simply wrapping existing libraries.

The project starts with classical NLP fundamentals and gradually evolves toward modern NLP systems such as embeddings, transformers, retrieval, and RAG.

---

## Vision

NOLQERA aims to become a modular NLP engine where each major NLP concept can be implemented, tested, understood, and reused independently.

```text
Text
 │
 ▼
Preprocessing
 │
 ▼
Tokenization
 │
 ▼
Feature Engineering
 │
 ▼
Classical NLP
 │
 ▼
Machine Learning NLP
 │
 ▼
Embeddings
 │
 ▼
Transformers
 │
 ▼
Retrieval / RAG
 │
 ▼
Modern NLP Engine
```

---

## Current Status

### v0.1.0 — Classical NLP Foundation

| Module                | Status      |
| --------------------- | ----------- |
| Text Cleaning         | Done        |
| Text Normalization    | Done        |
| Stopword Removal      | Done        |
| Stemming              | Done        |
| Lemmatization         | Done        |
| Sentence Tokenization | Done        |
| Word Tokenization     | Done        |
| N-Grams               | Done        |
| Bag of Words          | Done        |
| TF-IDF                | Done        |
| Validation            | Done        |
| Test Suite            | In Progress |

---

## Architecture

```text
nolqera/
│
├── nolqera/
│   │
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
│   ├── features/
│   │   ├── ngrams.py
│   │   ├── bow.py
│   │   └── tfidf.py
│   │
│   └── utils/
│       └── text_utils.py
│
├── tests/
├── examples/
├── docs/
├── README.md
├── pyproject.toml
└── LICENSE
```

---

# Installation

NOLQERA currently requires Python 3.11 or newer.

### Clone

```bash
git clone https://github.com/mogesh-developer/nolqera.git
cd nolqera
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate on Windows

```powershell
venv\Scripts\activate
```

### Install

```bash
pip install -e ".[dev]"
```

---

# Quick Start

## Preprocessing

```python
from nolqera import preprocess

text = "  Hello NOLQERA! Visit https://example.com  "

result = preprocess(text)

print(result)
```

Output:

```text
hello nolqera!
```

---

## Configurable Preprocessing

```python
from nolqera.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    remove_stopwords=True,
    stemming=True,
)

result = pipeline.process(
    "I am playing with cars."
)

print(result)
```

---

# Tokenization

```python
from nolqera import Tokenizer

tokenizer = Tokenizer()

text = "Hello, NOLQERA! How are you?"

print(tokenizer.sentences(text))
print(tokenizer.words(text))
```

Example:

```text
Sentences:
[
    "Hello, NOLQERA!",
    "How are you?"
]

Words:
[
    "Hello",
    ",",
    "NOLQERA",
    "!",
    "How",
    "are",
    "you",
    "?"
]
```

---

# N-Grams

```python
from nolqera import generate_ngrams

tokens = [
    "I",
    "love",
    "NOLQERA",
]

bigrams = generate_ngrams(tokens, 2)

print(bigrams)
```

Output:

```text
[
    ("I", "love"),
    ("love", "NOLQERA")
]
```

---

# Bag of Words

```python
from nolqera import BagOfWords

documents = [
    ["i", "love", "nlp"],
    ["i", "love", "python"],
]

bow = BagOfWords()

vectors = bow.fit_transform(documents)

print(bow.vocabulary)
print(vectors)
```

Output:

```text
Vocabulary:
[
    "i",
    "love",
    "nlp",
    "python"
]

Vectors:
[
    [1, 1, 1, 0],
    [1, 1, 0, 1]
]
```

---

# TF-IDF

NOLQERA implements TF-IDF from its underlying mathematical formulation.

```python
from nolqera import TfidfVectorizer

documents = [
    ["i", "love", "nlp"],
    ["i", "love", "python"],
]

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(documents)

print(vectorizer.vocabulary)
print(vectors)
```

### Mathematical foundation

```text
TF = term frequency

IDF = log(
    total documents /
    document frequency
)

TF-IDF = TF × IDF
```

The goal is not only to provide the feature, but also to make its internal mechanics understandable.

---

# Design Philosophy

NOLQERA follows a few core principles.

### 1. Understand before abstracting

Every major NLP concept should first be understood at the algorithmic level.

### 2. Minimal dependencies

The foundation should rely primarily on Python's standard library whenever practical.

### 3. Modular architecture

Each NLP capability should have a clear responsibility.

```text
Cleaner
Normalizer
Tokenizer
Feature Extractor
Vectorizer
```

should remain independently testable.

### 4. Test-driven development

Every major feature should have tests covering:

* normal inputs
* empty inputs
* invalid inputs
* edge cases
* expected mathematical behavior

### 5. Learn from implementation

NOLQERA is not intended to be just another wrapper around:

* NLTK
* spaCy
* scikit-learn
* Hugging Face

Those libraries can later be used for comparison and validation.

---

# Roadmap

## Phase 1 — Classical NLP

* [x] Text cleaning
* [x] Normalization
* [x] Stopwords
* [x] Stemming
* [x] Lemmatization
* [x] Sentence tokenization
* [x] Word tokenization
* [x] N-Grams
* [x] Bag of Words
* [x] TF-IDF

## Phase 2 — NLP Foundations

* [ ] Vocabulary management
* [ ] Frequency analysis
* [ ] Text statistics
* [ ] Language detection
* [ ] Advanced tokenization
* [ ] Morphological analysis
* [ ] POS tagging fundamentals

## Phase 3 — Classical Machine Learning

* [ ] Naive Bayes
* [ ] Logistic Regression
* [ ] Text classification
* [ ] Feature selection
* [ ] Evaluation metrics

## Phase 4 — Neural NLP

* [ ] Word embeddings
* [ ] Word2Vec concepts
* [ ] CBOW
* [ ] Skip-gram
* [ ] Embedding similarity
* [ ] Neural text classification

## Phase 5 — Transformers

* [ ] Attention
* [ ] Self-attention
* [ ] Positional encoding
* [ ] Transformer architecture
* [ ] BERT fundamentals
* [ ] Encoder / decoder concepts

## Phase 6 — Modern NLP

* [ ] Sentence embeddings
* [ ] Semantic similarity
* [ ] Vector search
* [ ] Retrieval
* [ ] RAG
* [ ] Document intelligence

## Phase 7 — NOLQERA Platform

Eventually NOLQERA can evolve into a larger ecosystem:

```text
                    NOLQERA
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      NLP Core     NOLQERA CLI   NOLQERA Studio
          │                         │
          │                         ▼
          │                    Visual NLP
          │                    Experiments
          │                    Pipelines
          │                    Evaluation
          │
          ▼
    NLP Applications
```

The GUI/visual layer will remain separate from the core engine so that NOLQERA can be used both as a Python library and as a developer-facing NLP platform.

---

# Development

Run the complete test suite:

```bash
pytest
```

Run a specific test module:

```bash
pytest tests/test_preprocessing.py
```

Run the example:

```bash
python examples/basic_usage.py
```

---

# Project Principles

NOLQERA is being developed incrementally.

Instead of immediately building a huge NLP framework, the project grows through small, understandable components:

```text
Concept
   ↓
Mathematics / Algorithm
   ↓
Implementation
   ↓
Tests
   ↓
Comparison
   ↓
Optimization
   ↓
Reusable Component
```

This approach makes the project useful both as a learning platform and as a foundation for future NLP applications.

---

# License

This project is licensed under the MIT License.

See [`LICENSE`](LICENSE) for details.

---

## NOLQERA

**Learn NLP. Build NLP. Understand NLP.**
