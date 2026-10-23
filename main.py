"""
Experiment 10: Logistic Regression
Aim: Implement Logistic Regression to predict whether a student will pass or fail.
Subject: Artificial Intelligence and Machine Learning (AIML)
Semester: V | Branch: IT
Faculty: Prof. Ravindra R. Agrawal
"""

import os
import sys
import io
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from Public Online Repository
# Dataset Source: Student Performance in Exams (Public benchmark dataset)
# URL: https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv"

print("=" * 65)
print(" Experiment 10: Logistic Regression - Student Pass/Fail Prediction")
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

# Validate required columns
expected_cols = ['math score', 'reading score', 'writing score']
if not all(col in df.columns for col in expected_cols):
    print(f"ERROR: Missing expected columns {expected_cols}. Found: {df.columns.tolist()}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Step 2: Prepare Binary Target and Input Features
# Target: Pass (1) if Math Score >= 50 else Fail (0)
# Features: Reading Score and Writing Score
# -----------------------------------------------------------------------------
PASSING_THRESHOLD = 50
df['Pass_Fail'] = (df['math score'] >= PASSING_THRESHOLD).astype(int)

feature_cols = ['reading score', 'writing score']
X = df[feature_cols]
y = df['Pass_Fail']

print("\n--- Dataset Target Distribution ---")
print(f"Passed Students (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"Failed Students (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")

# Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# Step 3: Feature Scaling
# Scaling standardizes features to zero mean and unit variance, which helps
# gradient optimization converge faster and more reliably.
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Step 4: Train Logistic Regression Model
# Mathematical Sigmoid Function: P(y=1) = 1 / (1 + e^-(z))
# where z = w1*x1 + w2*x2 + b
# -----------------------------------------------------------------------------
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

w1, w2 = model.coef_[0]
bias = model.intercept_[0]

print("\n" + "-" * 65)
print(" Logistic Regression Model Coefficients:")
print(f"   Weight for Reading Score (w1): {w1:.4f}")
print(f"   Weight for Writing Score (w2): {w2:.4f}")
print(f"   Bias / Intercept (b):          {bias:.4f}")
print(f"   Logit Equation: z = ({w1:.2f} × Reading) + ({w2:.2f} × Writing) + {bias:.2f}")
print("-" * 65)

# -----------------------------------------------------------------------------
# Step 5: Predictions and Performance Evaluation
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

print("\n--- Actual vs Predicted Comparison (First 10 Test Samples) ---")
comp_df = pd.DataFrame({
    'Reading': X_test['reading score'].values[:10],
    'Writing': X_test['writing score'].values[:10],
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Pass Probability': [f"{p*100:.1f}%" for p in y_prob[:10]],
    'Correct': ['Yes' if a == p else 'No' for a, p in zip(y_test.values[:10], y_pred[:10])]
})
print(comp_df.to_string(index=False))

print("\n--- Confusion Matrix ---")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]} | False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]} | True Positives (TP):  {cm[1, 1]}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)']))

# -----------------------------------------------------------------------------
# Step 6: Visualize Logistic Regression Decision Boundary
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))

x_min, x_max = X['reading score'].min() - 2, X['reading score'].max() + 2
y_min, y_max = X['writing score'].min() - 2, X['writing score'].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_df = pd.DataFrame({'reading score': xx.ravel(), 'writing score': yy.ravel()})
grid_scaled = scaler.transform(grid_df)
Z_prob = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

# Plot probability contours
contour = plt.contourf(xx, yy, Z_prob, levels=20, cmap='RdYlBu', alpha=0.7)
plt.colorbar(contour, label='Predicted Pass Probability P(Pass)')

# Plot 50% decision boundary line (where P(Pass) = 0.5)
plt.contour(xx, yy, Z_prob, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')

# Plot test observations
scatter = plt.scatter(
    X_test['reading score'],
    X_test['writing score'],
    c=y_test,
    cmap='bwr',
    edgecolors='black',
    s=55,
    alpha=0.9
)

plt.title('Logistic Regression: Student Pass/Fail Decision Boundary', fontsize=12, fontweight='bold')
plt.xlabel('Reading Score', fontsize=11)
plt.ylabel('Writing Score', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.5)

handles, _ = scatter.legend_elements()
plt.legend(handles, ['Actual Fail (0)', 'Actual Pass (1)'], loc='lower right', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "logistic_regression.png"
plt.savefig(plot_filename, dpi=300)
print(f"Logistic regression plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("\nExperiment 10 completed successfully!\n")
