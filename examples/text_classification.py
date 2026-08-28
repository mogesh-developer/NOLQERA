from nolqera import (
    TextClassifier,
    train_test_split,
)


documents = [
    "I love this movie",
    "This movie is amazing",
    "What a great film",
    "I really enjoyed this",
    "I hate this movie",
    "This movie is terrible",
    "What a bad film",
    "I really disliked this",
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


X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.25,
    random_state=42,
)


model = TextClassifier()

model.fit(
    X_train,
    y_train,
)


predictions = model.predict(X_test)

print("Test Documents:")
print(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test)

print("\nEvaluation:")
print(
    model.evaluate(
        X_test,
        y_test,
    )
)