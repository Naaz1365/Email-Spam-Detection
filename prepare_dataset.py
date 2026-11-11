
import csv
import sys
from preprocessing import clean_text

csv.field_size_limit(sys.maxsize)

input_file = "enron_spam_data.csv"
output_file = "processed_emails.csv"

total_emails = 0

with open(input_file, "r", encoding="utf-8-sig", newline="") as infile:

    reader = csv.DictReader(infile)

    with open(output_file, "w", encoding="utf-8", newline="") as outfile:

        writer = csv.writer(outfile)

        # Output CSV headers
        writer.writerow(["email_text", "label"])

        for row in reader:

            subject = row["Subject"] or ""
            message = row["Message"] or ""

            # Combine subject and message
            email_text = subject + " " + message

            # Clean email text
            cleaned_email = clean_text(email_text)

            # Get spam/ham label
            label = row["Spam/Ham"].strip().lower()

            if cleaned_email and label in ("spam", "ham"):

                writer.writerow([cleaned_email, label])
                total_emails += 1

print("\nDataset preprocessing completed!")
print("Total processed emails:", total_emails)
print("Saved file:", output_file)
