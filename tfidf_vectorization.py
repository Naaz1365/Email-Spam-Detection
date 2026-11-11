
import csv
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(sys.maxsize)

emails = []
labels = []

with open("processed_emails.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        emails.append(row["email_text"])
        labels.append(row["label"])

print("Total Emails:", len(emails))
print("Total Labels:", len(labels))

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Convert email text into numerical features
X = vectorizer.fit_transform(emails)

# Display results
print("\nTF-IDF Vectorization Completed!")
print("Feature Matrix Shape:", X.shape)
print("Vocabulary Size:", len(vectorizer.vocabulary_))
print("First Email Features:", X[0].shape)