import pytest

from nolqera.tokenization.sentence_tokenizer import split_sentences
from nolqera.tokenization.word_tokenizer import tokenize_words


def test_sentence_tokenizer_whitespace():
    assert split_sentences("   ") == []


def test_sentence_tokenizer_invalid_input():
    with pytest.raises(TypeError):
        split_sentences(123)


def test_word_tokenizer_whitespace():
    assert tokenize_words("   ") == []


def test_word_tokenizer_punctuation():
    result = tokenize_words("Hello!!!")

    assert result == [
        "Hello",
        "!",
        "!",
        "!",
    ]


def test_word_tokenizer_invalid_input():
    with pytest.raises(TypeError):
        tokenize_words(123)

def test_contractions():
    assert tokenize_words("I don't know.") == [
        "I",
        "don't",
        "know",
        ".",
    ]


def test_decimal_numbers():
    assert tokenize_words("Price is 123.45") == [
        "Price",
        "is",
        "123.45",
    ]


def test_hyphenated_words():
    assert tokenize_words("well-known") == [
        "well",
        "-",
        "known",
    ]

def test_tanglish():
    assert tokenize_words("enna da panra?") == [
        "enna",
        "da",
        "panra",
        "?",
    ]


def test_emoji():
    assert tokenize_words("Hello 😊!") == [
        "Hello",
        "😊",
        "!",
    ]


def test_unicode_word():
    text = "café résumé"

    assert tokenize_words(text) == [
        "café",
        "résumé",
    ]


def test_tokenizer_tokenize():
    from nolqera.tokenization import Tokenizer

    tokenizer = Tokenizer()
    text = "Hello NOLQERA!"

    assert tokenizer.tokenize(text) == ["hello", "nolqera", "!"]
    assert tokenizer.tokenize(text, lowercase=False) == ["Hello", "NOLQERA", "!"]