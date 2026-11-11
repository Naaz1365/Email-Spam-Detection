import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("processed_emails.csv")

X = df["email_text"].fillna("")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = joblib.load("logistic_regression_model.pkl")

predictions = model.predict(X_test_vec)

print("\n===== MODEL EVALUATION =====")
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, pos_label="spam"))
print("Recall   :", recall_score(y_test, predictions, pos_label="spam"))
print("F1 Score :", f1_score(y_test, predictions, pos_label="spam"))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))