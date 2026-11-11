

import csv
import sys
from collections import Counter

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

labels = Counter()

with open("processed_emails.csv", "r", encoding="utf-8", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:
        labels[row["label"].strip().lower()] += 1

print("\n========== PROCESSED DATASET SUMMARY ==========")
print("Total Emails:", sum(labels.values()))
print("Ham:", labels["ham"])
print("Spam:", labels["spam"])
print("Label Counts:", labels)
