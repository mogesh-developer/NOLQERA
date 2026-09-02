"""
Tests for NOLQERA CLI runner.
"""

from __future__ import annotations

import json
import pytest

from nolqera.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "NOLQERA Context Optimization CLI Runner" in captured.out


def test_cli_execution_with_file_input(tmp_path, capsys):
    input_file = tmp_path / "context.txt"
    input_file.write_text(
        "NOLQERA uses Python 3.11 for AI tasks.\n"
        "FastAPI handles backend API processing.\n"
        "MongoDB is used for document storage.",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--query",
            "Python AI tasks",
            "--input",
            str(input_file),
            "--max-sentences",
            "2",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Python" in captured.out


def test_cli_execution_with_config_file(tmp_path, capsys):
    input_file = tmp_path / "context.txt"
    input_file.write_text(
        "NOLQERA uses Python 3.11 for AI tasks.\n"
        "FastAPI handles backend API processing.",
        encoding="utf-8",
    )

    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"keyword_top_k": 5, "max_sentences": 1}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--query",
            "Python",
            "--input",
            str(input_file),
            "--config",
            str(config_file),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Python" in captured.out


def test_cli_error_missing_input_file(capsys):
    exit_code = main(
        [
            "--query",
            "Python",
            "--input",
            "non_existent_file.txt",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error reading input file" in captured.err
