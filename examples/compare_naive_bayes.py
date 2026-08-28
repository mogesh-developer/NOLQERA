from nolqera import (
    MultinomialNaiveBayes,
    classification_report,
    train_test_split,
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


texts = [
    "i love this movie",
    "this movie is great",
    "amazing movie",
    "really good film",
    "i hate this movie",
    "this movie is bad",
    "terrible movie",
    "really bad film",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
]


# --------------------------------------------------
# NOLQERA
# --------------------------------------------------

documents = [
    text.split()
    for text in texts
]

X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.25,
    random_state=42,
)

nolqera_model = MultinomialNaiveBayes()

nolqera_model.fit(
    X_train,
    y_train,
)

nolqera_predictions = nolqera_model.predict(
    X_test
)

nolqera_accuracy = nolqera_model.score(
    X_test,
    y_test,
)


# --------------------------------------------------
# SCIKIT-LEARN
# --------------------------------------------------

vectorizer = CountVectorizer()

X_train_text = [
    " ".join(document)
    for document in X_train
]

X_test_text = [
    " ".join(document)
    for document in X_test
]

X_train_sklearn = vectorizer.fit_transform(
    X_train_text
)

X_test_sklearn = vectorizer.transform(
    X_test_text
)

sklearn_model = MultinomialNB()

sklearn_model.fit(
    X_train_sklearn,
    y_train,
)

sklearn_predictions = sklearn_model.predict(
    X_test_sklearn
)

sklearn_accuracy = sklearn_model.score(
    X_test_sklearn,
    y_test,
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("=" * 50)
print("NOLQERA")
print("=" * 50)

print("Predictions:", nolqera_predictions)
print("Accuracy:", nolqera_accuracy)

print(
    classification_report(
        y_test,
        nolqera_predictions,
    )
)


print("\n" + "=" * 50)
print("SCIKIT-LEARN")
print("=" * 50)

print("Predictions:", sklearn_predictions)
print("Accuracy:", sklearn_accuracy)