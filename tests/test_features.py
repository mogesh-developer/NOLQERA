import pytest

import math

from nolqera.features.text_statistics import TextStatistics
from nolqera.features.frequency import FrequencyAnalyzer
from nolqera.features.tfidf import TfidfVectorizer
from nolqera.features.vocabulary import Vocabulary
from nolqera.features.bow import BagOfWords
from nolqera.features.ngrams import generate_ngrams, generate_ngrams_from_text


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

    assert bow.vocabulary_list == [
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

    assert vectorizer.vocabulary_list == [
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

def test_vocabulary_fit():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    vocabulary = Vocabulary()

    vocabulary.fit(documents)

    assert vocabulary.token_to_index == {
        "i": 0,
        "love": 1,
        "nlp": 2,
        "python": 3,
    }


def test_vocabulary_size():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["nlp", "python"],
    ])

    assert vocabulary.size == 2


def test_vocabulary_contains():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["nlp", "python"],
    ])

    assert vocabulary.contains("nlp")
    assert not vocabulary.contains("java")


def test_vocabulary_get_index():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["nlp", "python"],
    ])

    assert vocabulary.get_index("nlp") == 0


def test_vocabulary_get_token():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["nlp", "python"],
    ])

    assert vocabulary.get_token(1) == "python"


def test_unknown_token():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["nlp"],
    ])

    with pytest.raises(KeyError):
        vocabulary.get_index("python")


def test_min_frequency():
    vocabulary = Vocabulary(min_frequency=2)

    vocabulary.fit([
        ["nlp", "python"],
        ["nlp"],
    ])

    assert vocabulary.contains("nlp")
    assert not vocabulary.contains("python")

def test_bow_vocabulary():
    documents = [
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ]

    bow = BagOfWords()

    bow.fit(documents)

    assert bow.vocabulary_list == [
        "i",
        "love",
        "nlp",
        "python",
    ]

def test_bow_vectors():
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

def test_vocabulary_with_unk():
    vocabulary = Vocabulary(add_unk=True)

    vocabulary.fit([
        ["nlp", "python"],
    ])

    assert vocabulary.contains(
        vocabulary.UNK_TOKEN
    )

    assert vocabulary.get_index(
        "unknown"
    ) == vocabulary.get_index(
        vocabulary.UNK_TOKEN
    )


def test_vocabulary_without_unk():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["nlp"],
    ])

    with pytest.raises(KeyError):
        vocabulary.get_index("unknown")


def test_unk_token_index():
    vocabulary = Vocabulary(add_unk=True)

    vocabulary.fit([
        ["nlp", "python"],
    ])

    assert vocabulary.get_token(0) == "<UNK>"

def test_bow_unknown_token():
    bow = BagOfWords(add_unk=True)

    bow.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    result = bow.transform([
        ["i", "love", "transformers"],
    ])

    assert result == [
        [1, 1, 1, 0, 0],
    ]


def test_tfidf_unknown_token():
    vectorizer = TfidfVectorizer(
        add_unk=True
    )

    vectorizer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    result = vectorizer.transform([
        ["transformers"],
    ])

    unk_index = vectorizer.vocabulary.get_index(
        "<UNK>"
    )

    assert result[0][unk_index] == 0.0
    assert vectorizer.idf["<UNK>"] == 0.0

def test_vocabulary_encode():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    result = vocabulary.encode([
        "i",
        "love",
        "nlp",
    ])

    assert result == [
        0,
        1,
        2,
    ]


def test_vocabulary_decode():
    vocabulary = Vocabulary()

    vocabulary.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    result = vocabulary.decode([
        0,
        1,
        2,
    ])

    assert result == [
        "i",
        "love",
        "nlp",
    ]

def test_vocabulary_encode_unknown():
    vocabulary = Vocabulary(
        add_unk=True
    )

    vocabulary.fit([
        ["i", "love", "nlp"],
    ])

    result = vocabulary.encode([
        "i",
        "transformers",
    ])

    assert result == [
        1,
        0,
    ]

def test_vocabulary_save_load(tmp_path):
    vocabulary = Vocabulary(add_unk=True)

    vocabulary.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    path = tmp_path / "vocabulary.json"

    vocabulary.save(str(path))

    loaded = Vocabulary.load(str(path))

    assert loaded.token_to_index == vocabulary.token_to_index
    assert loaded.index_to_token == vocabulary.index_to_token
    assert loaded.min_frequency == vocabulary.min_frequency
    assert loaded.add_unk == vocabulary.add_unk

    assert loaded.encode([
        "i",
        "love",
        "transformers",
    ]) == [1, 2, 0]

def test_empty_documents():
    vocabulary = Vocabulary()

    vocabulary.fit([])

    assert vocabulary.size == 0


def test_empty_document():
    vocabulary = Vocabulary()

    vocabulary.fit([
        [],
    ])

    assert vocabulary.size == 0


def test_min_frequency_filters_tokens():
    vocabulary = Vocabulary(
        min_frequency=2
    )

    vocabulary.fit([
        ["nlp", "python"],
        ["nlp", "java"],
    ])

    assert vocabulary.contains("nlp")
    assert not vocabulary.contains("python")
    assert not vocabulary.contains("java")


def test_invalid_min_frequency():
    with pytest.raises(ValueError):
        Vocabulary(min_frequency=0)


def test_invalid_min_frequency_type():
    with pytest.raises(TypeError):
        Vocabulary(min_frequency="1")


def test_bow_empty_documents():
    bow = BagOfWords()

    with pytest.raises(ValueError):
        bow.transform([
            ["nlp"],
        ])


def test_bow_unknown_token_without_unk():
    bow = BagOfWords()

    bow.fit([
        ["nlp"],
    ])

    result = bow.transform([
        ["python"],
    ])

    assert result == [
        [0],
    ]


def test_bow_invalid_documents():
    bow = BagOfWords()

    with pytest.raises(TypeError):
        bow.fit("invalid")


def test_tfidf_empty_documents():
    vectorizer = TfidfVectorizer()

    with pytest.raises(ValueError):
        vectorizer.fit([])


def test_tfidf_empty_document():
    vectorizer = TfidfVectorizer()

    vectorizer.fit([
        ["nlp"],
        [],
    ])

    result = vectorizer.transform([
        [],
    ])

    assert result == [
        [0.0],
    ]


def test_tfidf_invalid_documents():
    vectorizer = TfidfVectorizer()

    with pytest.raises(TypeError):
        vectorizer.fit("invalid")

def test_frequency_analyzer():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    assert analyzer.count("i") == 2
    assert analyzer.count("love") == 2
    assert analyzer.count("nlp") == 1
    assert analyzer.count("unknown") == 0


def test_unique_count():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    assert analyzer.unique_count() == 4


def test_total_tokens():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    assert analyzer.total_tokens == 6


def test_most_common():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    assert analyzer.most_common(2) == [
        ("i", 2),
        ("love", 2),
    ]


def test_relative_frequency():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    assert analyzer.frequency("i") == 2 / 6


def test_character_count():
    stats = TextStatistics()

    stats.fit("hello world")

    assert stats.character_count == 11


def test_character_count_no_spaces():
    stats = TextStatistics()

    stats.fit("hello world")

    assert stats.character_count_no_spaces == 10


def test_token_count():
    stats = TextStatistics()

    stats.fit(
        "hello world",
        tokens=["hello", "world"],
    )

    assert stats.token_count == 2


def test_unique_token_count():
    stats = TextStatistics()

    stats.fit(
        "hello hello world",
        tokens=[
            "hello",
            "hello",
            "world",
        ],
    )

    assert stats.unique_token_count == 2


def test_sentence_count():
    stats = TextStatistics()

    stats.fit(
        "Hello world. I love NLP! Do you?"
    )

    assert stats.sentence_count == 3


def test_average_token_length():
    stats = TextStatistics()

    stats.fit(
        "I love NLP",
        tokens=["I", "love", "NLP"],
    )

    assert stats.average_token_length == 8 / 3


def test_vocabulary_richness():
    stats = TextStatistics()

    stats.fit(
        "I love NLP I",
        tokens=["I", "love", "NLP", "I"],
    )

    assert stats.vocabulary_richness == 3 / 4


def test_empty_text():
    stats = TextStatistics()

    stats.fit("")

    assert stats.character_count == 0
    assert stats.token_count == 0
    assert stats.sentence_count == 0
    assert stats.vocabulary_richness == 0.0

def test_tokenizer_integration():
    stats = TextStatistics()

    stats.fit(
        "Hello, world!"
    )

    assert stats.tokens == [
        "Hello",
        ",",
        "world",
        "!",
    ]


def test_analyze():
    stats = TextStatistics.analyze(
        "I love NLP. I love Python!"
    )

    assert stats.sentence_count == 2
    assert stats.token_count > 0

def test_summary():
    stats = TextStatistics.analyze(
        "I love NLP. I love Python!"
    )

    summary = stats.summary()

    assert summary["characters"] == 26
    assert summary["sentences"] == 2
    assert summary["tokens"] == stats.token_count
    assert summary["unique_tokens"] == stats.unique_token_count
    assert (
        summary["vocabulary_richness"]
        == stats.vocabulary_richness
    )


def test_to_dict():
    stats = TextStatistics.analyze(
        "Hello world!"
    )

    assert stats.to_dict() == stats.summary()

def test_frequency_summary():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
        ["i", "love", "python"],
    ])

    summary = analyzer.summary()

    assert summary == {
        "total_tokens": 6,
        "unique_tokens": 4,
        "total_documents": 2,
    }


def test_frequency_to_dict():
    analyzer = FrequencyAnalyzer()

    analyzer.fit([
        ["i", "love", "nlp"],
    ])

    assert analyzer.to_dict() == analyzer.summary()

def test_unigrams():
    tokens = ["i", "love", "nlp"]

    assert generate_ngrams(tokens, 1) == [
        ("i",),
        ("love",),
        ("nlp",),
    ]


def test_bigrams():
    tokens = ["i", "love", "nlp"]

    assert generate_ngrams(tokens, 2) == [
        ("i", "love"),
        ("love", "nlp"),
    ]


def test_trigrams():
    tokens = ["i", "love", "nlp"]

    assert generate_ngrams(tokens, 3) == [
        ("i", "love", "nlp"),
    ]


def test_n_greater_than_tokens():
    tokens = ["i", "love"]

    assert generate_ngrams(tokens, 3) == []


def test_empty_tokens():
    assert generate_ngrams([], 2) == []


def test_invalid_n():
    with pytest.raises(ValueError):
        generate_ngrams(
            ["i", "love"],
            0,
        )


def test_invalid_n_type():
    with pytest.raises(TypeError):
        generate_ngrams(
            ["i", "love"],
            "2",
        )


def test_invalid_tokens():
    with pytest.raises(TypeError):
        generate_ngrams(
            "i love",
            2,
        )


def test_invalid_token_type():
    with pytest.raises(TypeError):
        generate_ngrams(
            ["i", 10],
            2,
        )


def test_generate_ngrams_from_text():
    result = generate_ngrams_from_text(
        "I love NLP",
        2,
    )

    assert result == [
        ("I", "love"),
        ("love", "NLP"),
    ]