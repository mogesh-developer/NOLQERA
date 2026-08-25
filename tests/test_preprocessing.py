import pytest

from nolqera.preprocessing.lemmatization import SimpleLemmatizer
from nolqera.preprocessing.stemming import SimpleStemmer
from nolqera.preprocessing.pipeline import PreprocessingPipeline
from nolqera.preprocessing import remove_urls
from nolqera.tokenization.sentence_tokenizer import split_sentences
from nolqera.preprocessing.stopwords import StopwordRemover

def test_split_sentences():
    text = "Hello world. How are you? I am fine!"

    result = split_sentences(text)

    assert result == [
        "Hello world.",
        "How are you?",
        "I am fine!",
    ]


def test_split_sentences_empty_text():
    assert split_sentences("") == []


def test_split_sentences_single_sentence():
    assert split_sentences("Hello world.") == ["Hello world."]


def test_remove_urls_invalid_input():
    with pytest.raises(TypeError):
        remove_urls(123)

def test_stopword_removal():
    remover = StopwordRemover()

    tokens = [
        "I",
        "love",
        "the",
        "NOLQERA",
    ]

    assert remover.remove(tokens) == [
        "I",
        "love",
        "NOLQERA",
    ]


def test_stopword_case_insensitive():
    remover = StopwordRemover(
        ["the", "is"]
    )

    tokens = [
        "The",
        "NOLQERA",
        "IS",
        "awesome",
    ]

    assert remover.remove(tokens) == [
        "NOLQERA",
        "awesome",
    ]


def test_custom_stopwords():
    remover = StopwordRemover(
        ["hello", "world"]
    )

    tokens = [
        "hello",
        "NOLQERA",
        "world",
    ]

    assert remover.remove(tokens) == [
        "NOLQERA",
    ]

def test_pipeline_without_stopwords():
    pipeline = PreprocessingPipeline()

    result = pipeline.process(
        "I love the NOLQERA project."
    )

    assert result == "i love the nolqera project ."


def test_pipeline_with_stopwords():
    pipeline = PreprocessingPipeline(
        remove_stopwords=True
    )

    result = pipeline.process(
        "I love the NOLQERA project."
    )

    assert result == "i love nolqera project ."

def test_stem_ing():
    stemmer = SimpleStemmer()

    assert stemmer.stem("playing") == "play"


def test_stem_ed():
    stemmer = SimpleStemmer()

    assert stemmer.stem("played") == "play"


def test_stem_ly():
    stemmer = SimpleStemmer()

    assert stemmer.stem("quickly") == "quick"


def test_stem_plural():
    stemmer = SimpleStemmer()

    assert stemmer.stem("cars") == "car"


def test_short_word():
    stemmer = SimpleStemmer()

    assert stemmer.stem("cat") == "cat"

def test_stem_tokens():
    stemmer = SimpleStemmer()

    tokens = [
        "Playing",
        "cars",
        "quickly",
    ]

    assert stemmer.stem_tokens(tokens) == [
        "play",
        "car",
        "quick",
    ]


def test_stem_invalid_word():
    stemmer = SimpleStemmer()

    try:
        stemmer.stem(123)
        assert False
    except TypeError:
        assert True


def test_stem_invalid_tokens():
    stemmer = SimpleStemmer()

    try:
        stemmer.stem_tokens("playing")
        assert False
    except TypeError:
        assert True

def test_pipeline_with_stemming():
    pipeline = PreprocessingPipeline(
        stemming=True
    )

    result = pipeline.process(
        "I am playing with cars."
    )

    assert result == "i am play with car ."

def test_pipeline_with_stopwords_and_stemming():
    pipeline = PreprocessingPipeline(
        remove_stopwords=True,
        stemming=True,
    )

    result = pipeline.process(
        "I am playing with cars."
    )

    assert result == "i am play car ."

def test_lemmatize_regular_word():
    lemmatizer = SimpleLemmatizer()

    assert lemmatizer.lemmatize("playing") == "play"


def test_lemmatize_ies():
    lemmatizer = SimpleLemmatizer()

    assert lemmatizer.lemmatize("studies") == "study"


def test_lemmatize_irregular():
    lemmatizer = SimpleLemmatizer()

    assert lemmatizer.lemmatize("children") == "child"


def test_lemmatize_verb():
    lemmatizer = SimpleLemmatizer()

    assert lemmatizer.lemmatize("went") == "go"


def test_lemmatize_tokens():
    lemmatizer = SimpleLemmatizer()

    tokens = [
        "children",
        "studies",
        "cars",
    ]

    assert lemmatizer.lemmatize_tokens(tokens) == [
        "child",
        "study",
        "car",
    ]

def test_pipeline_with_lemmatization():
    pipeline = PreprocessingPipeline(
        lemmatization=True
    )

    result = pipeline.process(
        "Children studies cars."
    )

    assert result == "child study car ."


def test_stemming_and_lemmatization_conflict():
    with pytest.raises(ValueError):
        PreprocessingPipeline(
            stemming=True,
            lemmatization=True,
        )