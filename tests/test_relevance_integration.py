from nolqera.features.tfidf import TfidfVectorizer
from nolqera.preprocessing.pipeline import PreprocessingPipeline
from nolqera.intelligence.relevance.similarity import cosine_similarity


def test_tfidf_cosine_similarity_finds_relevant_sentence():
    query = "What database does the application use?"

    sentences = [
        "The application is built using FastAPI.",
        "The application uses MongoDB for data storage.",
        "I travelled to Chennai yesterday.",
        "The API provides REST endpoints.",
    ]

    pipeline = PreprocessingPipeline(
        remove_stopwords=True,
        stemming=True,
    )

    query_tokens = pipeline.process(query).split()

    sentence_tokens = [
        pipeline.process(sentence).split()
        for sentence in sentences
    ]

    documents = [query_tokens, *sentence_tokens]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    query_vector = vectors[0]
    sentence_vectors = vectors[1:]

    scores = [
        cosine_similarity(query_vector, sentence_vector)
        for sentence_vector in sentence_vectors
    ]

    print("\n--- Relevance Scores ---")

    for sentence, score in zip(sentences, scores):
        print(f"{score:.4f} → {sentence}")

    best_index = max(
        range(len(scores)),
        key=scores.__getitem__,
    )

    print("\n--- Most Relevant Sentence ---")
    print(sentences[best_index])
    print(f"Score: {scores[best_index]:.4f}")

    assert sentences[best_index] == (
        "The application uses MongoDB for data storage."
    )

    assert scores[best_index] > scores[2]