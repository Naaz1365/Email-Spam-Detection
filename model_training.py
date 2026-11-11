
import csv
import sys
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

csv.field_size_limit(sys.maxsize)

emails = []
labels = []

with open("processed_emails.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        emails.append(row["email_text"])
        labels.append(row["label"])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2,
    random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, pos_label="spam"))
    print("Recall:", recall_score(y_test, predictions, pos_label="spam"))
    print("F1 Score:", f1_score(y_test, predictions, pos_label="spam"))
    print(classification_report(y_test, predictions))

    filename = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(model, filename)

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel Training Completed!")
print("Models and TF-IDF Vectorizer Saved Successfully!")
