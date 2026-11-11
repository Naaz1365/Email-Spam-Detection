
import re
import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

# Common English stopwords
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "and", "or", "but",
    "if", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "as", "it", "this", "that",
    "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "my", "your", "his", "her", "our",
    "their", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "not"
}


def clean_text(text):

    # Handle empty values
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    filtered_words = [
        word for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


# Test the function
sample = "Congratulations! You have won a FREE prize."

print("Original Text:", sample)
print("Cleaned Text:", clean_text(sample))
