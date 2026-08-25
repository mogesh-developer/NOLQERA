import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nolqera import (
    preprocess,
    Tokenizer,
    BagOfWords,
    TfidfVectorizer,
    generate_ngrams,
)


documents = [
    "I love Natural Language Processing.",
    "I love Python and NLP.",
]


# Preprocessing
processed_documents = [
    preprocess(document)
    for document in documents
]

print("Processed:")
for document in processed_documents:
    print(document)


# Tokenization
tokenizer = Tokenizer()

tokenized_documents = [
    tokenizer.words(document)
    for document in processed_documents
]

print("\nTokens:")
for tokens in tokenized_documents:
    print(tokens)


# N-Grams
bigrams = generate_ngrams(
    tokenized_documents[0],
    2,
)

print("\nBigrams:")
print(bigrams)


# Bag of Words
bow = BagOfWords()

bow_vectors = bow.fit_transform(
    tokenized_documents
)

print("\nBoW Vocabulary:")
print(bow.vocabulary)

print("\nBoW Vectors:")
print(bow_vectors)


# TF-IDF
tfidf = TfidfVectorizer()

tfidf_vectors = tfidf.fit_transform(
    tokenized_documents
)

print("\nTF-IDF Vocabulary:")
print(tfidf.vocabulary)

print("\nTF-IDF Vectors:")
print(tfidf_vectors)