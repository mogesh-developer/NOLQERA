from nolqera.intelligence.intent.candidates import (
    IntentCandidateExtractor,
)


def test_question_form_creates_candidate():

    extractor = IntentCandidateExtractor()

    candidates = extractor.extract(
        "How do I install FastAPI?"
    )

    signals = {
        candidate.signal
        for candidate in candidates
    }

    assert "question_form" in signals


def test_interrogative_creates_candidate():

    extractor = IntentCandidateExtractor()

    candidates = extractor.extract(
        "How does RAG work?"
    )

    signals = {
        candidate.signal
        for candidate in candidates
    }

    assert "interrogative" in signals


def test_normal_statement_has_no_question_signal():

    extractor = IntentCandidateExtractor()

    candidates = extractor.extract(
        "FastAPI is a Python framework."
    )

    assert all(
        candidate.signal
        not in {
            "question_form",
            "interrogative",
        }
        for candidate in candidates
    )


def test_candidate_scores_are_valid():

    extractor = IntentCandidateExtractor()

    candidates = extractor.extract(
        "What is MongoDB?"
    )

    for candidate in candidates:
        assert 0.0 <= candidate.score <= 1.0