import pandas as pd

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None
)

data.columns = ["label", "message"]

print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSpam and Ham Count:")
print(data["label"].value_counts())

print("\nDuplicate Messages:")
print(data.duplicated().sum())

data = data.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(data.shape)

print("\nRemaining Duplicates:")
print(data.duplicated().sum())