import pytest

from nolqera.intelligence.pipeline.input_handler import (
    InputHandler,
)


def test_input_handler_returns_string():
    handler = InputHandler()

    result = handler.handle(
        "FastAPI is a Python framework."
    )

    assert isinstance(result, str)


def test_input_handler_removes_leading_whitespace():
    handler = InputHandler()

    result = handler.handle(
        "   FastAPI is a Python framework."
    )

    assert result == (
        "FastAPI is a Python framework."
    )


def test_input_handler_removes_trailing_whitespace():
    handler = InputHandler()

    result = handler.handle(
        "FastAPI is a Python framework.   "
    )

    assert result == (
        "FastAPI is a Python framework."
    )


def test_input_handler_collapses_multiple_spaces():
    handler = InputHandler()

    result = handler.handle(
        "FastAPI   is    a Python   framework."
    )

    assert result == (
        "FastAPI is a Python framework."
    )


def test_input_handler_normalizes_mixed_whitespace():
    handler = InputHandler()

    result = handler.handle(
        "FastAPI\tis\n a Python\tframework."
    )

    assert result == (
        "FastAPI is a Python framework."
    )


def test_input_handler_preserves_content():
    handler = InputHandler()

    result = handler.handle(
        "FastAPI is a Python framework."
    )

    assert result == (
        "FastAPI is a Python framework."
    )


def test_input_handler_rejects_non_string():
    handler = InputHandler()

    with pytest.raises(TypeError):
        handler.handle(123)


def test_input_handler_rejects_empty_string():
    handler = InputHandler()

    with pytest.raises(ValueError):
        handler.handle("")


def test_input_handler_rejects_whitespace_only_input():
    handler = InputHandler()

    with pytest.raises(ValueError):
        handler.handle("     ")