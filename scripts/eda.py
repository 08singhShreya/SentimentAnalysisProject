from datasets import load_dataset
import pandas as pd

# Read 10,000 random reviews without downloading the whole dataset
stream = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
stream = stream.shuffle(seed=42, buffer_size=10000).take(10000)

df = pd.DataFrame(list(stream))

print("Rows & Columns", df.shape)
print()
print(df.head())

print()
print("Missing values per column:")
print(df.isnull().sum())

print()
print("Positive vs negative count:")
print(df["label"].value_counts())

print()
print("Mean of label:", df["label"].mean())
print("Mode of label:", df["label"].mode()[0])

df["word_count"] = df["content"].str.split().str.len()
print()
print("Review length in words:")
print(df["word_count"].describe())
print("Mode of length:", df["word_count"].mode()[0])