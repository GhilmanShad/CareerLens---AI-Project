import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score

from modules.preprocessing import clean_data
from modules.kmeans import run_kmeans


# 1. Load and clean the dataset
df = pd.read_csv("data/job_postings.csv")
df = clean_data(df, remove_duplicates=True)

salaries = pd.to_numeric(df["salary"], errors="coerce")
salaries = salaries[np.isfinite(salaries) & (salaries > 0)]

# 2. Independent 80/20 training and testing split
train, test = train_test_split(
    salaries,
    test_size=0.20,
    random_state=42,
)

print("K-MEANS CLUSTERING EVALUATION")
print("=" * 60)
print("Total valid salaries:", len(salaries))
print("Training records before IQR:", len(train))
print("Testing records before IQR:", len(test))

# 3. Calculate outlier boundaries using training data only
q1 = train.quantile(0.25)
q3 = train.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

train = train[(train >= lower) & (train <= upper)]
test = test[(test >= lower) & (test <= upper)]

print("Training records after IQR:", len(train))
print("Testing records after IQR:", len(test))

train_df = pd.DataFrame({"salary": train})
test_df = pd.DataFrame({"salary": test})

# 4. Reproducible training sample
sample_df = train_df.sample(
    n=min(5000, len(train_df)),
    random_state=42,
)

print("Training sample size:", len(sample_df))

# 5. Select K using training data only
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

    if not 1 < len(np.unique(labels)) < len(X):
        continue

    silhouette = silhouette_score(
        X,
        labels,
        sample_size=min(1000, len(X)),
        random_state=42,
    )

    db_index = davies_bouldin_score(X, labels)

    inertia = np.sum(
        (X[:, 0] - centroids[labels, 0]) ** 2
    )

    results.append({
        "K": k,
        "Silhouette": silhouette,
        "Davies-Bouldin": db_index,
        "Inertia": inertia,
        "Clusters Found": len(np.unique(labels)),
    })

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No valid clustering results.")

print("\nTRAINING CLUSTER COMPARISON")
print("=" * 60)
print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))

best = results_df.loc[results_df["Silhouette"].idxmax()]
best_k = int(best["K"])

print("\nBest K by training silhouette:", best_k)
print(f"Training silhouette: {best['Silhouette']:.4f}")

# 6. Fit selected model using all filtered training records
_, centroids = run_kmeans(
    train_df,
    column="salary",
    k=best_k,
    max_iters=100,
)

# 7. Assign unseen test salaries to trained centroids
X_test = test_df[["salary"]].to_numpy(dtype=float)

distances = np.abs(X_test - centroids.T)
test_labels = np.argmin(distances, axis=1)

test_inertia = np.sum(
    (X_test[:, 0] - centroids[test_labels, 0]) ** 2
)

# 8. Independent held-out evaluation
print("\nK-MEANS FINAL HELD-OUT TEST")
print("=" * 60)
print("Selected K:", best_k)
print("Final training records:", len(train_df))
print("Testing records:", len(X_test))
print("Clusters found:", len(np.unique(test_labels)))

if 1 < len(np.unique(test_labels)) < len(X_test):
    test_silhouette = silhouette_score(
        X_test,
        test_labels,
        sample_size=min(1000, len(X_test)),
        random_state=42,
    )

    test_db = davies_bouldin_score(X_test, test_labels)

    print(f"Test Silhouette Score: {test_silhouette:.4f}")
    print(f"Test Davies-Bouldin Index: {test_db:.4f}")
else:
    print("Clustering metrics undefined for test assignments.")

print(f"Test Inertia: {test_inertia:.4f}")

print("\nTRAINED CENTROIDS (Annual USD)")
for i, centroid in enumerate(centroids[:, 0]):
    count = np.sum(test_labels == i)
    print(
        f"Cluster {i}: ${centroid:,.2f} "
        f"| Test records: {count}"
    )
