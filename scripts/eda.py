"""Quick look at the review data before we train anything.

Run from the project root:  python scripts/eda.py
"""

import sys

import pandas as pd
from datasets import load_dataset

DATASET = "fancyzhx/amazon_polarity" 
SAMPLE_SIZE = 10_000


def load_sample() -> pd.DataFrame:
    """Stream SAMPLE_SIZE shuffled reviews into a DataFrame."""
    stream = load_dataset(DATASET, split="train", streaming=True)
    stream = stream.shuffle(seed=42, buffer_size=SAMPLE_SIZE).take(SAMPLE_SIZE)
    return pd.DataFrame(list(stream))


def show_basics(df: pd.DataFrame) -> None:
    """Print shape, a few rows and any missing values."""
    print("Rows & Columns", df.shape)
    print()
    print(df.head())
    print()
    print("Missing values per column:")
    print(df.isnull().sum())


def show_label_stats(df: pd.DataFrame) -> None:
    """Check class balance. 0 = negative, 1 = positive.

    A lopsided split would make plain accuracy misleading.
    """
    print()
    print("Positive vs negative count:")
    print(df["label"].value_counts())
    print("Mean of label:", df["label"].mean())  # share of positive reviews
    print("Mode of label:", df["label"].mode()[0])


def show_length_stats(df: pd.DataFrame) -> None:
    """Print review length in words (mean, median, spread, mode)."""
    word_count = df["content"].str.split().str.len()
    print()
    print("Review length in words:")
    print(word_count.describe())
    print("Mode of length:", word_count.mode()[0])


def main() -> int:
    """Run the full EDA. Returns 0 on success, 1 on failure."""
    try:
        df = load_sample()
    except Exception as exc:  # network, bad dataset name, library change
        print(f"Could not load dataset: {exc}", file=sys.stderr)
        return 1

    show_basics(df)
    show_label_stats(df)
    show_length_stats(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())