
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
from modules.knn import get_knn_recommendations, predict_job_level


# 1. Load and preprocess the dataset
df = pd.read_csv("data/job_postings.csv")
df = clean_data(df, remove_duplicates=True)

df = df.dropna(subset=["salary", "job_level"]).copy()
df = df[df["job_level"].astype(str).str.strip() != ""]

# 2. Match the eligible records used for Decision Tree evaluation
X = extract_enhanced_features(df)
y = df["job_level"].astype(str)

valid_rows = X.notna().all(axis=1) & np.isfinite(X.to_numpy()).all(axis=1)

X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 3. Reproduce the Decision Tree's 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Build the KNN reference dataset using training records only
train_jobs = pd.DataFrame({
    "salary": X_train["salary"],
    "job_level": y_train,
})

# 5. Predict test job levels using salary-based KNN
k = 5
predictions = []

print("Total eligible records:", len(X))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("K:", k)
print("\nEvaluating salary-based KNN...")

for salary in X_test["salary"]:
    neighbors = get_knn_recommendations(
        train_jobs,
        query_salary=float(salary),
        k=k,
    )
    predicted_level = predict_job_level(neighbors)
    predictions.append(predicted_level)

# 6. Evaluate predictions
print("\nKNN FINAL TEST")
print("=" * 50)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

labels = sorted(y.unique())

print("\nConfusion Matrix:")
print("Class order:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))
