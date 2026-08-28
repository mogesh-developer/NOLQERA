import pytest

from nolqera.document import Document, Sentence


def test_sentence_creation():
    sentence = Sentence(
        "Submit the project report.",
        0,
    )

    assert sentence.text == "Submit the project report."
    assert sentence.index == 0


def test_sentence_strips_whitespace():
    sentence = Sentence(
        "  Submit the report.  ",
        0,
    )

    assert sentence.text == "Submit the report."


def test_empty_sentence():
    with pytest.raises(ValueError):
        Sentence("", 0)


def test_invalid_text():
    with pytest.raises(TypeError):
        Sentence(123, 0)


def test_invalid_index():
    with pytest.raises(TypeError):
        Sentence("Submit the report.", "0")


def test_document_sentences():
    document = Document(
        "Submit the report. Attend the viva. Contact the department."
    )

    sentences = document.sentences()

    assert len(sentences) == 3

    assert sentences[0].text == "Submit the report"
    assert sentences[1].text == "Attend the viva"
    assert sentences[2].text == "Contact the department"


def test_document_sentence_indexes():
    document = Document(
        "First sentence. Second sentence."
    )

    sentences = document.sentences()

    assert sentences[0].index == 0
    assert sentences[1].index == 1


def test_sentence_tokens():
    sentence = Sentence(
        "Submit the project report.",
        0,
    )

    tokens = sentence.tokens()

    assert tokens == [
        "submit",
        "the",
        "project",
        "report",
        ".",
    ]