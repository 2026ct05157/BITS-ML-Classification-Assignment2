
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------

RANDOM_STATE = 42
TARGET = "income"

os.makedirs("model/saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------------------------------
# 2. Load training and testing data
# -------------------------------------------------

train_df = pd.read_csv("dataset/train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# Load the preprocessing configuration.
# clone() creates a fresh, unfitted copy for each model.
preprocessor = joblib.load("model/preprocessor.joblib")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training features:", X_train.shape[1])

# -------------------------------------------------
# 3. Define the five classification models
# -------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# -------------------------------------------------
# 4. Train models and calculate metrics
# -------------------------------------------------

results = []
trained_models = {}
roc_data = {}
confusion_data = {}

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name, flush=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier)
        ]
    )

    # Fit the complete pipeline on training data only
    pipeline.fit(X_train, y_train)

    print("Training completed. Evaluating...", flush=True)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Positive-class probabilities
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Six required evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Save trained pipeline
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = os.path.join(
        "model",
        "saved_models",
        filename
    )

    joblib.dump(pipeline, model_path)

    trained_models[model_name] = pipeline

    # Store information for graphs
    confusion_data[model_name] = confusion_matrix(
        y_test,
        y_pred
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }

    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))
    print("Saved:", model_path, flush=True)

# -------------------------------------------------
# 5. Save comparison table
# -------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1 Score", "MCC"],
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# -------------------------------------------------
# 6. Accuracy comparison graph
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Accuracy Comparison of Classification Models")
plt.xlabel("Classification Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 7. Comparison of all six metrics
# -------------------------------------------------

metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

plot_df = results_df.set_index("Model")[metric_columns]

ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title("Comparison of Six Evaluation Metrics")
ax.set_xlabel("Classification Model")
ax.set_ylabel("Metric Score")
ax.set_ylim(-1, 1)
plt.xticks(rotation=25, ha="right")
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(
    "results/all_metrics_comparison.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 8. Confusion matrices
# -------------------------------------------------

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10)
)

axes = axes.flatten()

for index, (model_name, matrix) in enumerate(
    confusion_data.items()
):

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["<=50K", ">50K"]
    )

    display.plot(
        ax=axes[index],
        colorbar=False
    )

    axes[index].set_title(model_name)

# Hide unused sixth subplot
axes[-1].axis("off")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrices.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 9. ROC curves
# -------------------------------------------------

plt.figure(figsize=(10, 7))

for model_name, values in roc_data.items():

    plt.plot(
        values["fpr"],
        values["tpr"],
        label=f"{model_name} (AUC={values['auc']:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "results/roc_curves.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 10. Generate model observations
# -------------------------------------------------

best_f1 = results_df.iloc[0]
best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]
best_auc = results_df.loc[
    results_df["AUC"].idxmax()
]
best_mcc = results_df.loc[
    results_df["MCC"].idxmax()
]

with open(
    "results/model_observations.txt",
    "w"
) as file:

    file.write("MODEL COMPARISON OBSERVATIONS\n")
    file.write("=" * 60 + "\n\n")

    for _, row in results_df.iterrows():

        file.write(f"Model: {row['Model']}\n")

        file.write(
            f"Accuracy: {row['Accuracy']:.4f}\n"
        )

        file.write(
            f"AUC: {row['AUC']:.4f}\n"
        )

        file.write(
            f"Precision: {row['Precision']:.4f}\n"
        )

        file.write(
            f"Recall: {row['Recall']:.4f}\n"
        )

        file.write(
            f"F1 Score: {row['F1 Score']:.4f}\n"
        )

        file.write(
            f"MCC: {row['MCC']:.4f}\n\n"
        )

    file.write("OVERALL COMPARISON\n")
    file.write("-" * 60 + "\n")

    file.write(
        f"Highest F1 Score: {best_f1['Model']}\n"
    )

    file.write(
        f"Highest Accuracy: {best_accuracy['Model']}\n"
    )

    file.write(
        f"Highest AUC: {best_auc['Model']}\n"
    )

    file.write(
        f"Highest MCC: {best_mcc['Model']}\n"
    )

    file.write(
        "\nThe overall ranking is based on F1 Score "
        "followed by MCC. Since the dataset is imbalanced, "
        "accuracy alone should not determine the best model.\n"
    )

print("\nAll models trained successfully.")
print("All six metrics calculated.")
print("Comparison table and graphs saved.")
print("Model observation file generated.")
print("\nTRAINING AND EVALUATION COMPLETED.")
