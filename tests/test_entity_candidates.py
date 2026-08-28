from nolqera.intelligence.entities.candidates import (
    EntityCandidateExtractor,
)


def test_extracts_capitalized_entity_candidates():
    extractor = EntityCandidateExtractor()

    text = (
        "John works at Google in Chennai."
    )

    candidates = extractor.extract(text)

    values = [
        candidate.text
        for candidate in candidates
    ]

    assert "John" in values
    assert "Google" in values
    assert "Chennai" in values


def test_extracts_multi_word_candidate():
    extractor = EntityCandidateExtractor()

    text = (
        "John studied at American College."
    )

    candidates = extractor.extract(text)

    values = [
        candidate.text
        for candidate in candidates
    ]

    assert "American College" in values


def test_candidate_positions_are_correct():
    extractor = EntityCandidateExtractor()

    text = "John works at Google."

    candidates = extractor.extract(text)

    google = next(
        candidate
        for candidate in candidates
        if candidate.text == "Google"
    )

    assert text[
        google.start:google.end
    ] == "Google"


def test_candidates_are_unique():
    extractor = EntityCandidateExtractor()

    text = "Google works. Google grows."

    candidates = extractor.extract(text)

    google_candidates = [
        candidate
        for candidate in candidates
        if candidate.text == "Google"
    ]

    assert len(google_candidates) == 2


def test_empty_text_is_rejected():
    extractor = EntityCandidateExtractor()

    try:
        extractor.extract("")
    except ValueError:
        return

    assert False


def test_invalid_text_is_rejected():
    extractor = EntityCandidateExtractor()

    try:
        extractor.extract(None)
    except TypeError:
        return

    assert False