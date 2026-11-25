
"""
Dataset preparation for the BITS Pilani ML Assignment.

Source: UCI Adult Census Income dataset.
Combines adult.data and adult.test into one CSV.
"""

from pathlib import Path
import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset"

# Original dataset files
TRAIN_FILE = DATA_DIR / "adult.data"
TEST_FILE = DATA_DIR / "adult.test"

# Output file
OUTPUT_FILE = DATA_DIR / "adult.csv"

# Column names provided by the UCI dataset
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

def load_adult_file(file_path, skip_rows=0):

    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
        skiprows=skip_rows,
        skipinitialspace=True,
        na_values=["?"],
        comment="|"
    )

    # Remove extra spaces from categorical values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Normalize the income target
    df["income"] = df["income"].str.rstrip(".")

    return df


def main():

    print("Loading original UCI dataset files...")

    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "adult.data or adult.test not found in dataset folder."
        )

    # Load original training portion
    train_df = load_adult_file(TRAIN_FILE)

    # adult.test contains a descriptive first line
    test_df = load_adult_file(TEST_FILE, skip_rows=1)

    # Combine both portions
    df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    # Validate the target labels
    print("\nTarget class counts:")
    print(df["income"].value_counts(dropna=False))

    print("\nDataset dimensions:")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nMissing values:")
    print(df.isnull().sum())

    # Save the combined dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset saved successfully:")
    print(OUTPUT_FILE)

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()

