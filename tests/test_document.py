import pytest

from nolqera.document import Document


def test_document_creation():
    document = Document(
        "Students must submit the project report."
    )

    assert document.text == (
        "Students must submit the project report."
    )


def test_document_metadata():
    document = Document("Submit the report.")

    document.metadata["source"] = "college_notice"

    assert document.metadata["source"] == "college_notice"


def test_empty_document():
    with pytest.raises(ValueError):
        Document("")


def test_invalid_document():
    with pytest.raises(TypeError):
        Document(123)