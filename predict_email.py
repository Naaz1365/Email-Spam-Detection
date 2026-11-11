
import joblib

model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

email = input("Enter email message: ")

email_vector = vectorizer.transform([email])
prediction = model.predict(email_vector)[0]

print("\nPrediction:", prediction.upper())

if prediction == "spam":
    print("Warning: This email is SPAM!")
else:
    print("This email is HAM (Not Spam).")
