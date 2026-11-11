# Email Spam Detection System

A Machine Learning-based application that classifies email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and supervised learning.

## Project Overview

This project uses the Enron Email Dataset to train and evaluate machine learning models for email spam detection.

Users can enter email content into a Streamlit web application and receive a prediction.

## Features

- Email text preprocessing
- TF-IDF text vectorization
- Naive Bayes classification
- Logistic Regression classification
- Spam/Ham prediction
- Spam probability display
- Confusion Matrix visualization
- ROC Curve and AUC evaluation
- Model performance comparison
- Interactive Streamlit interface

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- TF-IDF
- Natural Language Processing

## Dataset

**Enron Email Dataset**

The dataset contains email messages labelled as spam or ham.

Dataset file: `enron_spam_data.csv`

## Model Performance

| Metric | Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 98.40% | 98.77% |
| Precision | 98.00% | 98.21% |
| Recall | 98.86% | 99.39% |
| F1 Score | 98.43% | 98.80% |

Logistic Regression achieved the best overall results on the current test split.

ROC-AUC: 0.999

## Project Structure

```text
Email-Spam-Detection/
│
├── app.py
├── preprocessing.py
├── prepare_dataset.py
├── check_processed.py
├── tfidf_vectorization.py
├── model_training.py
├── model_evaluation.py
├── evaluation_visuals.py
├── compare_models.py
│
├── enron_spam_data.csv
├── processed_emails.csv
│
├── naive_bayes_model.pkl
├── logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
│
├── confusion_matrix.png
├── roc_curve.png
├── model_comparison.csv
│
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project directory:

```bash
cd Email-Spam-Detection
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Training

To train the models again:

```bash
python model_training.py
```

## Model Evaluation

```bash
python model_evaluation.py
python evaluation_visuals.py
python compare_models.py
```

## Disclaimer

This project is developed for educational and internship purposes. Model performance is based on the available Enron dataset and test split. Results may vary on real-world email data.
