
import csv
import sys
from collections import Counter

# Increase CSV field size limit for large email messages
csv.field_size_limit(sys.maxsize)

labels = Counter()
total_emails = 0

# Read the email spam dataset
with open("enron_spam_data.csv", "r", encoding="utf-8-sig", newline="") as file:

    reader = csv.DictReader(file)

    print("Columns:", reader.fieldnames)

    for row in reader:
        label = row["Spam/Ham"].strip().lower()

        if label:
            labels[label] += 1
            total_emails += 1

# Display dataset information
print("\n========== DATASET SUMMARY ==========")
print("Total Emails:", total_emails)
print("Ham Emails:", labels["ham"])
print("Spam Emails:", labels["spam"])

print("\nLabel Counts:", labels)

print("\n========== DATASET CHECK ==========")

if total_emails > 0:
    print("Dataset loaded successfully!")
else:
    print("No email records found.")
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