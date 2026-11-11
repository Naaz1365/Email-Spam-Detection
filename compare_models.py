import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("processed_emails.csv")

X = df["email_text"].fillna("")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Logistic Regression": joblib.load("logistic_regression_model.pkl")
}

results = []

for name, model in models.items():
    pred = model.predict(X_test_vec)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label="spam"),
        "Recall": recall_score(y_test, pred, pos_label="spam"),
        "F1 Score": f1_score(y_test, pred, pos_label="spam")
    })

result_df = pd.DataFrame(results)
print(result_df.round(4))

result_df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")