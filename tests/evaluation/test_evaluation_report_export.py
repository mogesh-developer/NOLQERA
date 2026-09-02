"""
Tests for EvaluationReport export methods.
"""

from __future__ import annotations

import csv
import json

from nolqera.evaluation.report import EvaluationReport


def make_sample_report() -> EvaluationReport:
    return EvaluationReport(
        query="Python AI frameworks",
        raw_tokens=100,
        optimized_tokens=40,
        token_reduction=60,
        reduction_percentage=60.0,
        compression_ratio=2.5,
        relevance_score=0.95,
        retention_percentage=90.0,
        information_loss_percentage=10.0,
        context_quality_score=92.5,
        latency_ms=15.4,
    )


def test_evaluation_report_to_dict():
    report = make_sample_report()
    data = report.to_dict()

    assert data["query"] == "Python AI frameworks"
    assert data["raw_tokens"] == 100
    assert data["optimized_tokens"] == 40
    assert data["reduction_percentage"] == 60.0


def test_evaluation_report_to_json():
    report = make_sample_report()
    json_str = report.to_json()
    data = json.loads(json_str)

    assert data["query"] == "Python AI frameworks"
    assert data["retention_percentage"] == 90.0
    assert data["latency_ms"] == 15.4


def test_evaluation_report_to_csv():
    report = make_sample_report()
    csv_str = report.to_csv()

    reader = list(csv.DictReader(csv_str.splitlines()))
    assert len(reader) == 1
    row = reader[0]
    assert row["query"] == "Python AI frameworks"
    assert float(row["raw_tokens"]) == 100.0
    assert float(row["context_quality_score"]) == 92.5
