import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("processed_emails.csv")

X = df["email_text"].fillna("")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

X_test_vec = vectorizer.transform(X_test)

predictions = model.predict(X_test_vec)
probabilities = model.predict_proba(X_test_vec)[:, list(model.classes_).index("spam")]

# Confusion Matrix
cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)
display.plot(cmap="Blues")
plt.title("Email Spam Detection - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
y_binary = (y_test == "spam").astype(int)
fpr, tpr, thresholds = roc_curve(y_binary, probabilities)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

print("Visualizations saved successfully!")
print("ROC AUC:", round(roc_auc, 4))