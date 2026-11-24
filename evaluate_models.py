
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from modules.preprocessing import clean_data
from modules.enhanced_features import extract_enhanced_features
from modules.decision_tree import build_tree, predict


# 1. Load and preprocess the dataset
df = pd.read_csv("data/job_postings.csv")
df = clean_data(df, remove_duplicates=True)

df = df.dropna(subset=["salary", "job_level"]).copy()
df = df[df["job_level"].astype(str).str.strip() != ""]

# 2. Extract numerical input features and target labels
X = extract_enhanced_features(df)
y = df["job_level"].astype(str)

# 3. Remove records with invalid feature values
valid_rows = X.notna().all(axis=1) & np.isfinite(X.to_numpy()).all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 4. Stratified 80/20 training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Total eligible records:", len(X))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Train the team's custom Decision Tree
print("\nTraining Decision Tree...")

tree = build_tree(
    X_train.to_numpy(),
    y_train.to_numpy(),
    max_depth=7,
)

# 6. Predict unseen test records
predictions = [
    predict(tree, row)
    for row in X_test.to_numpy()
]

# 7. Evaluate performance
print("\nDECISION TREE FINAL TEST")
print("=" * 50)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

labels = sorted(y.unique())

print("\nConfusion Matrix:")
print("Class order:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))


# 8. Salary-only Decision Tree for fair comparison with KNN
print("\nSALARY-ONLY DECISION TREE FINAL TEST")
print("=" * 50)

salary_tree = build_tree(
    X_train[["salary"]].to_numpy(),
    y_train.to_numpy(),
    max_depth=7,
)

salary_predictions = [
    predict(salary_tree, row)
    for row in X_test[["salary"]].to_numpy()
]

salary_accuracy = accuracy_score(y_test, salary_predictions)

print(f"Accuracy: {salary_accuracy:.4f} ({salary_accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test,
    salary_predictions,
    zero_division=0,
))

print("\nConfusion Matrix:")
print("Class order:", labels)
print(confusion_matrix(
    y_test,
    salary_predictions,
    labels=labels,
))

print("\nDECISION TREE FEATURE COMPARISON")
print("=" * 50)
print(f"Salary-only accuracy: {salary_accuracy * 100:.2f}%")
print(f"Enhanced accuracy:    {accuracy * 100:.2f}%")
print(f"Feature improvement:  {(accuracy - salary_accuracy) * 100:+.2f} percentage points")
