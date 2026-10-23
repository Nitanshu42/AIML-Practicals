"""
Experiment 02: K-Means Clustering
Aim: Apply K-Means clustering to group students based on their marks in two subjects.
Subject: Artificial Intelligence and Machine Learning (AIML)
Semester: V | Branch: IT
Faculty: Prof. Ravindra R. Agrawal
"""

import os
import sys
import io
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from a Public Online Repository
# Dataset Source: Student Performance in Exams (Public benchmark dataset)
# URL: https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv"

print("=" * 65)
print(" Experiment 02: K-Means Clustering - Student Performance Grouping")
print("=" * 65)
print(f"Fetching dataset from public URL:\n  {DATASET_URL}\n")

try:
    req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as response:
        csv_data = response.read()
    df = pd.read_csv(io.BytesIO(csv_data))
    print("Dataset successfully downloaded and loaded!")
except Exception as e:
    print(f"\nERROR: Failed to download online dataset: {e}")
    print("Please check your internet connection or URL accessibility.")
    sys.exit(1)

# Validate expected columns
expected_cols = ['math score', 'reading score']
if not all(col in df.columns for col in expected_cols):
    print(f"ERROR: Dataset missing expected columns {expected_cols}. Found: {df.columns.tolist()}")
    sys.exit(1)

# Display first few rows
print("\n--- Dataset Preview (First 5 records) ---")
print(df[['math score', 'reading score']].head())
print(f"\nTotal students: {len(df)}")
print(f"Missing values: {df[['math score', 'reading score']].isnull().sum().to_dict()}")

# -----------------------------------------------------------------------------
# Step 2: Prepare Features (Marks in Two Subjects)
# Subject 1: Math Score (0 to 100)
# Subject 2: Reading Score (0 to 100)
# -----------------------------------------------------------------------------
X = df[['math score', 'reading score']]

# -----------------------------------------------------------------------------
# Step 3: Implement K-Means Algorithm
# Parameters:
# - n_clusters=3: Group into 3 categories (High, Average, Low performers)
# - init='k-means++': Smart centroid initialization to speed up convergence
# - random_state=42: Fixed seed for reproducible results
# -----------------------------------------------------------------------------
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Retrieve cluster centroids (mean marks of each cluster)
centroids = kmeans.cluster_centers_

print("\n" + "-" * 65)
print(" Cluster Centroids (Average Marks):")
print("-" * 65)
for i, center in enumerate(centroids):
    print(f" Cluster {i}: Math Avg = {center[0]:.2f}, Reading Avg = {center[1]:.2f}")

# Sort clusters by math score to provide meaningful group names
sorted_indices = sorted(range(k), key=lambda idx: centroids[idx][0])
cluster_labels_map = {
    sorted_indices[0]: "Low Scorers (Needs Support)",
    sorted_indices[1]: "Average Performers",
    sorted_indices[2]: "High Achievers"
}
df['Group_Name'] = df['Cluster'].map(cluster_labels_map)

print("\n--- Student Distribution Across Groups ---")
print(df['Group_Name'].value_counts())
print("-" * 65)

# -----------------------------------------------------------------------------
# Step 4: Visualize Clusters and Centroids
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))

colors = ['#e74c3c', '#3498db', '#2ecc71']

for i in range(k):
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(
        cluster_points['math score'],
        cluster_points['reading score'],
        s=60,
        c=colors[i],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        label=f"{cluster_labels_map[i]} (Cluster {i})"
    )

# Plot Centroids as large markers
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=250,
    c='yellow',
    marker='X',
    edgecolor='black',
    linewidth=2,
    label='Cluster Centroids'
)

plt.title('K-Means Clustering: Grouping Students by Marks in Two Subjects', fontsize=12, fontweight='bold')
plt.xlabel('Math Score (Subject 1)', fontsize=11)
plt.ylabel('Reading Score (Subject 2)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "kmeans_clusters.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nCluster plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 02 completed successfully!\n")
