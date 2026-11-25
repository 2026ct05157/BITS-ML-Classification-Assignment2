
"""
Exploratory Data Analysis for BITS ML Assignment 2.
Dataset: UCI Adult Census Income
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "adult.csv"

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------

print("=" * 65)
print("STEP 1: LOADING DATASET")
print("=" * 65)

df = pd.read_csv(DATA_FILE, skipinitialspace=True)

# Clean accidental spaces in categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# -------------------------------------------------
# 2. Check column names
# -------------------------------------------------

print("\n" + "=" * 65)
print("STEP 2: COLUMN INFORMATION")
print("=" * 65)

print(df.columns.tolist())

# -------------------------------------------------
# 3. Check data types
# -------------------------------------------------

print("\n" + "=" * 65)
print("STEP 3: DATA TYPES")
print("=" * 65)

print(df.dtypes)

# -------------------------------------------------
# 4. Identify numerical and categorical columns
# -------------------------------------------------

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

# -------------------------------------------------
# 5. Audit missing values
# -------------------------------------------------

print("\n" + "=" * 65)
print("STEP 4: MISSING VALUE ANALYSIS")
print("=" * 65)

# Count actual '?' values before converting them
question_mark_counts = (
    df.select_dtypes(include="object")
      .apply(lambda col: col.eq("?").sum())
)

print("\nQuestion mark values:")
print(question_mark_counts)

# Convert any remaining '?' to actual NaN
df = df.replace("?", pd.NA)

missing_counts = df.isnull().sum()

print("\nMissing values by column:")
print(missing_counts)

print("\nTotal missing values:", missing_counts.sum())

# Save missing value report
missing_counts.to_csv(
    RESULT_DIR / "missing_values_report.csv"
)

# -------------------------------------------------
# 6. Duplicate analysis
# -------------------------------------------------

print("\n" + "=" * 65)
print("STEP 5: DUPLICATE ANALYSIS")
print("=" * 65)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

# -------------------------------------------------
# 7. Target variable analysis
# -------------------------------------------------

print("\n" + "=" * 65)
print("STEP 6: TARGET CLASS DISTRIBUTION")
print("=" * 65)

target_column = "income"

print("\nTarget counts:")
print(df[target_column].value_counts(dropna=False))

print("\nTarget percentages:")
print(
    df[target_column].value_counts(
        normalize=True,
        dropna=False
    ).mul(100).round(2)
)

# -------------------------------------------------
# 8. Plot target distribution
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target_column,
    hue=target_column,
    order=["<=50K", ">50K"],
    palette="Set2",
    legend=False
)

plt.title("Income Class Distribution")
plt.xlabel("Income Class")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "class_distribution.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 9. Plot numerical feature distributions
# -------------------------------------------------

df[numerical_columns].hist(
    figsize=(14, 10),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "numerical_distributions.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 10. Save dataset information
# -------------------------------------------------

with open(
    RESULT_DIR / "dataset_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("UCI ADULT DATASET SUMMARY\n")
    file.write("=" * 50 + "\n")

    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n")

    file.write(
        f"Numerical features: {len(numerical_columns)}\n"
    )

    file.write(
        f"Categorical features: {len(categorical_columns)-1}\n"
    )

    file.write(
        f"Missing values: {missing_counts.sum()}\n"
    )

    file.write(
        f"Duplicate rows: {duplicate_count}\n"
    )

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")
print("1. missing_values_report.csv")
print("2. class_distribution.png")
print("3. numerical_distributions.png")
print("4. dataset_summary.txt")
