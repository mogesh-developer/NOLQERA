from nolqera import (
    BagOfWords,
    FrequencyAnalyzer,
    TextStatistics,
    TfidfVectorizer,
    Vocabulary,
    generate_ngrams_from_text,
)


text = "I love Natural Language Processing. I love Python and NLP."


# --------------------------------------------------
# Text Statistics
# --------------------------------------------------

stats = TextStatistics.analyze(text)

print("Text Statistics:")
print(stats.summary())


# --------------------------------------------------
# N-Grams
# --------------------------------------------------

bigrams = generate_ngrams_from_text(text, 2)

print("\nBigrams:")
print(bigrams)


# --------------------------------------------------
# Token preparation
# --------------------------------------------------

documents = [
    ["i", "love", "natural", "language", "processing"],
    ["i", "love", "python", "and", "nlp"],
]


# --------------------------------------------------
# Frequency Analysis
# --------------------------------------------------

frequency = FrequencyAnalyzer()

frequency.fit(documents)

print("\nFrequency Summary:")
print(frequency.summary())

print("\nMost Common:")
print(frequency.most_common())


# --------------------------------------------------
# Vocabulary
# --------------------------------------------------

vocabulary = Vocabulary(add_unk=True)

vocabulary.fit(documents)

print("\nVocabulary:")
print(vocabulary.token_to_index)

print("\nEncoded:")
print(
    vocabulary.encode(
        ["i", "love", "transformers"]
    )
)

print("\nDecoded:")
print(
    vocabulary.decode(
        [1, 2, 0]
    )
)


# --------------------------------------------------
# Bag of Words
# --------------------------------------------------

bow = BagOfWords(add_unk=True)

bow_vectors = bow.fit_transform(documents)

print("\nBoW Vocabulary:")
print(bow.vocabulary_list)

print("\nBoW Vectors:")
print(bow_vectors)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(add_unk=True)

tfidf_vectors = tfidf.fit_transform(documents)

print("\nTF-IDF Vocabulary:")
print(tfidf.vocabulary_list)

print("\nTF-IDF Vectors:")
print(tfidf_vectors)