import pytest

import math

from nolqera.features.tfidf import TfidfVectorizer

from nolqera.features.bow import BagOfWords
from nolqera.features.ngrams import generate_ngrams


def test_unigram():
    tokens = ["I", "love", "NOLQERA"]

    result = generate_ngrams(tokens, 1)

    assert result == [
        ("I",),
        ("love",),
        ("NOLQERA",),
    ]


def test_bigram():
    tokens = ["I", "love", "NOLQERA"]

    result = generate_ngrams(tokens, 2)

    assert result == [
        ("I", "love"),
        ("love", "NOLQERA"),
    ]


def test_trigram():
    tokens = ["I", "love", "NOLQERA"]

    result = generate_ngrams(tokens, 3)

    assert result == [
        ("I", "love", "NOLQERA"),
    ]


def test_n_greater_than_tokens():
    tokens = ["I", "love"]

    assert generate_ngrams(tokens, 3) == []


def test_invalid_n():
    with pytest.raises(ValueError):
        generate_ngrams(["I", "love"], 0)

def test_bow_fit():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    bow = BagOfWords()

    bow.fit(documents)

    assert bow.vocabulary == [
        "i",
        "love",
        "nlp",
        "python",
    ]


def test_bow_transform():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    bow = BagOfWords()

    result = bow.fit_transform(documents)

    assert result == [
        [1, 1, 1, 0],
        [1, 1, 0, 1],
    ]


def test_bow_unfitted():
    bow = BagOfWords()

    try:
        bow.transform([["hello"]])
        assert False
    except ValueError:
        assert True

def test_tfidf_vocabulary():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    vectorizer = TfidfVectorizer()

    vectorizer.fit(documents)

    assert vectorizer.vocabulary == [
        "i",
        "love",
        "nlp",
        "python",
    ]


def test_tfidf_idf():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    vectorizer = TfidfVectorizer()

    vectorizer.fit(documents)

    assert vectorizer.idf["i"] == 0.0
    assert vectorizer.idf["love"] == 0.0

    assert vectorizer.idf["nlp"] == math.log(2)
    assert vectorizer.idf["python"] == math.log(2)


def test_tfidf_transform():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    vectorizer = TfidfVectorizer()

    result = vectorizer.fit_transform(documents)

    expected_value = math.log(2) / 3

    assert result[0][0] == 0.0
    assert result[0][1] == 0.0
    assert math.isclose(result[0][2], expected_value)
    assert result[0][3] == 0.0


def test_tfidf_empty_documents():
    vectorizer = TfidfVectorizer()

    try:
        vectorizer.fit([])
        assert False
    except ValueError:
        assert True
    
def test_ngrams_invalid_tokens():
    with pytest.raises(TypeError):
        generate_ngrams("hello world", 2)


def test_ngrams_invalid_n_type():
    with pytest.raises(TypeError):
        generate_ngrams(["hello", "world"], "2")