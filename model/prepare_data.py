
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load the prepared Adult dataset
DATASET_PATH = "dataset/adult.csv"
RANDOM_STATE = 42

df = pd.read_csv(DATASET_PATH)

# 2. Clean spaces in text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# 3. Convert question marks into actual missing values
df = df.replace("?", np.nan)

# 4. Check the original dataset
print("\n--- ORIGINAL DATASET ---")
print("Shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())
print("\nMissing values by column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 5. Check target labels before cleaning
target_column = "income"

print("\nTarget values before cleaning:")
print(df[target_column].value_counts(dropna=False))

# Remove any trailing full stops from target labels
df[target_column] = (
    df[target_column]
    .astype("string")
    .str.strip()
    .str.rstrip(".")
)

# Remove rows where the target itself is missing
df = df.dropna(subset=[target_column])

# 6. Remove exact duplicate records
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

print("\nDuplicate rows removed:", duplicates_before)
print("Shape after duplicate removal:", df.shape)

# 7. Convert target into binary values
df[target_column] = df[target_column].map({
    "<=50K": 0,
    ">50K": 1
})

if df[target_column].isna().any():
    raise ValueError("Unexpected target labels found. Please inspect target values.")

df[target_column] = df[target_column].astype(int)

print("\n--- FINAL CLASS DISTRIBUTION ---")
class_counts = df[target_column].value_counts().sort_index()
class_percentages = df[target_column].value_counts(
    normalize=True
).sort_index() * 100

print("0 = <=50K")
print("1 = >50K")
print(class_counts)
print("\nClass percentages:")
print(class_percentages.round(2))

# 8. Separate input features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# 9. Perform stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n--- TRAIN-TEST SPLIT ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(4))

# 10. Save raw training and testing datasets
# Imputation, encoding and scaling will be fitted later
# inside the model pipelines using training data only.

train_data = X_train.copy()
train_data[target_column] = y_train

test_data = X_test.copy()
test_data[target_column] = y_test

train_data.to_csv("dataset/train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Save a cleaned dataset for reference
df.to_csv("dataset/adult_cleaned.csv", index=False)

print("\nFiles saved successfully:")
print("dataset/train_data.csv")
print("test_data.csv")
print("dataset/adult_cleaned.csv")
print("\nPREPROCESSING DATA PREPARATION COMPLETED.")

