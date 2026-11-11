
import streamlit as st
import joblib
from preprocessing import clean_text

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def load_models():
    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

st.title("📧 Email Spam Detection System")
st.write("Machine Learning Based Email Classification")

email = st.text_area(
    "Paste your email message:",
    height=200,
    placeholder="Enter email content here..."
)

if st.button("Detect Spam", use_container_width=True):

    if not email.strip():
        st.warning("Please enter an email message.")

    else:
        cleaned_email = clean_text(email)

        if not cleaned_email.strip():
            st.warning("Email does not contain usable text.")
        else:
            email_vector = vectorizer.transform([cleaned_email])
            prediction = model.predict(email_vector)[0]
            probabilities = model.predict_proba(email_vector)[0]
            spam_index = list(model.classes_).index("spam")
            confidence = probabilities[spam_index] * 100

            if prediction == "spam":
                st.error("🚨 SPAM EMAIL DETECTED")
            else:
                st.success("✅ HAM EMAIL - NOT SPAM")

            st.metric("Spam Probability", f"{confidence:.2f}%")

st.divider()
st.caption("Dataset: Enron Email Dataset")
st.caption("Algorithm: TF-IDF + Logistic Regression")
st.caption("Educational Machine Learning Project")
