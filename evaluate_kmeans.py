
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score, davies_bouldin_score

from modules.preprocessing import clean_data
from modules.kmeans import run_kmeans


# 1. Load and preprocess the dataset
df = pd.read_csv("data/job_postings.csv")
df = clean_data(df, remove_duplicates=True)

salaries = pd.to_numeric(df["salary"], errors="coerce")
salaries = salaries[np.isfinite(salaries) & (salaries > 0)]

# 2. Apply IQR outlier filtering (same as Streamlit default)
q1 = salaries.quantile(0.25)
q3 = salaries.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

salaries = salaries[(salaries >= lower) & (salaries <= upper)]

clustering_df = pd.DataFrame({"salary": salaries})

print("K-MEANS CLUSTERING EVALUATION")
print("=" * 60)
print("Records after preprocessing and IQR filtering:", len(clustering_df))

# 3. Use a reproducible sample for efficient evaluation
sample_size = min(5000, len(clustering_df))
sample_df = clustering_df.sample(n=sample_size, random_state=42)

print("Evaluation sample size:", len(sample_df))

# 4. Evaluate K values from 2 to 8
results = []

for k in range(2, 9):
    clustered_df, centroids = run_kmeans(
        sample_df,
        column="salary",
        k=k,
        max_iters=100,
    )

    X = clustered_df[["salary"]].to_numpy(dtype=float)
    labels = clustered_df["cluster"].to_numpy()

    # Inertia: sum of squared distances to assigned centroids
    inertia = np.sum(
        (X[:, 0] - centroids[labels, 0]) ** 2
    )

    # Use a smaller sample for silhouette to limit memory usage
    unique_labels = np.unique(labels)

    if 1 < len(unique_labels) < len(X):
        silhouette = silhouette_score(
            X,
            labels,
            sample_size=min(1000, len(X)),
            random_state=42,
        )
        db_index = davies_bouldin_score(X, labels)
    else:
        silhouette = np.nan
        db_index = np.nan

    results.append({
        "K": k,
        "Silhouette": silhouette,
        "Davies-Bouldin": db_index,
        "Inertia": inertia,
        "Clusters Found": len(unique_labels),
    })

# 5. Display comparison
results_df = pd.DataFrame(results)

print("\nCLUSTERING RESULTS")
print("=" * 60)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 6. Identify the best K using silhouette score
valid_results = results_df.dropna(subset=["Silhouette"])

if not valid_results.empty:
    best = valid_results.loc[valid_results["Silhouette"].idxmax()]
    print(f"\nBest K by Silhouette Score: {int(best['K'])}")
    print(f"Silhouette Score: {best['Silhouette']:.4f}")
