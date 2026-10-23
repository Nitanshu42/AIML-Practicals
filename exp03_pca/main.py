"""
Experiment 03: Principal Component Analysis (PCA)
Aim: Apply Principal Component Analysis (PCA) to reduce dataset dimensions.
Subject: Artificial Intelligence and Machine Learning (AIML)
Semester: V | Branch: IT
Faculty: Prof. Ravindra R. Agrawal
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 65)
print(" Experiment 03: Principal Component Analysis (PCA)")
print("=" * 65)

# -----------------------------------------------------------------------------
# Step 1: Load Built-in Iris Dataset from scikit-learn
# The Iris dataset contains 150 samples with 4 numerical features:
# 1. Sepal Length (cm)
# 2. Sepal Width (cm)
# 3. Petal Length (cm)
# 4. Petal Width (cm)
# Target: 3 Species (setosa, versicolor, virginica)
# -----------------------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame

print("Iris Dataset successfully loaded (scikit-learn built-in)!")
print("\n--- Original Dataset Preview (First 5 records) ---")
print(df.head())

X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"\nOriginal Feature Dimensions: {X.shape[1]} features ({X.shape[0]} samples)")
print(f"Features: {list(feature_names)}")

# -----------------------------------------------------------------------------
# Step 2: Feature Standardization
# PCA is highly sensitive to the scale of features. Standardization ensures
# each feature has a mean of 0 and variance of 1.
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# Step 3: Apply Principal Component Analysis (PCA)
# We reduce dimensions from 4 features down to 2 Principal Components (PC1 & PC2).
# -----------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Target'] = y
pca_df['Species'] = pca_df['Target'].map(lambda idx: target_names[idx])

print("\n" + "-" * 65)
print(f"Reduced Dimensions: {X_pca.shape[1]} Principal Components")
print(f"PC1 Explained Variance: {pca.explained_variance_ratio_[0] * 100:.2f}%")
print(f"PC2 Explained Variance: {pca.explained_variance_ratio_[1] * 100:.2f}%")
total_var = pca.explained_variance_ratio_.sum() * 100
print(f"Total Preserved Information (Variance): {total_var:.2f}%")
print("-" * 65)

print("\n--- Reduced Dataset Preview (First 5 records) ---")
print(pca_df.head())

# -----------------------------------------------------------------------------
# Step 4: Visualize the 2D Reduced Dataset
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 6))

colors = ['#e74c3c', '#2ecc71', '#3498db']
for target_idx, color in zip(range(len(target_names)), colors):
    subset = pca_df[pca_df['Target'] == target_idx]
    plt.scatter(
        subset['Principal Component 1'],
        subset['Principal Component 2'],
        c=color,
        label=target_names[target_idx],
        s=70,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )

plt.title('PCA: 4D Iris Dataset Reduced to 2 Principal Components', fontsize=12, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title='Iris Species', loc='best', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "pca_result.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nPCA visualization saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 03 completed successfully!\n")
