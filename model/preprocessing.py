
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the training and testing datasets
train_df = pd.read_csv("dataset/train_data.csv")
test_df = pd.read_csv("test_data.csv")

TARGET = "income"

# Separate features and target
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# Identify numerical and categorical columns
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str", "string", "category"]
).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Numerical preprocessing:
# Missing values -> median
# Feature scaling -> StandardScaler
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing:
# Missing values -> most frequent value
# Encoding -> OneHotEncoder
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the training-fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

# Verify results
print("\n--- PREPROCESSING RESULTS ---")
print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)
print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nMissing values in processed training data:",
      pd.isna(X_train_processed).sum())

print("Missing values in processed testing data:",
      pd.isna(X_test_processed).sum())

# Save fitted preprocessing object
joblib.dump(preprocessor, "model/preprocessor.joblib")

print("\nPreprocessing pipeline saved successfully:")
print("model/preprocessor.joblib")
print("\nPREPROCESSING COMPLETED SUCCESSFULLY.")
