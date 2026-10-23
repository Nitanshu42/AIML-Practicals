"""
Experiment 09: K-Nearest Neighbors (KNN)
Aim: Implement K-Nearest Neighbors (KNN) to classify students as pass or fail based on study hours and attendance.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from Public Online Repository
# Dataset Source: UCI Student Performance Dataset (Math Course)
# URL: https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv"

print("=" * 65)
print(" Experiment 09: KNN - Student Pass/Fail by Study Hours & Attendance")
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
expected_cols = ['studytime', 'absences', 'G3']
if not all(col in df.columns for col in expected_cols):
    print(f"ERROR: Missing expected columns {expected_cols}. Found: {df.columns.tolist()}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Step 2: Feature Engineering (Study Hours & Attendance Percentage)
# - studytime represents weekly study hours: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
#   We convert to estimated weekly study hours: [1.5, 3.5, 7.5, 12.0]
# - absences: We compute Attendance Rate (%) = max(0, 100 - (absences * 2.5))
# - Target G3 (Final Exam Score out of 20): Pass (1) if G3 >= 10 else Fail (0)
# -----------------------------------------------------------------------------
hours_map = {1: 1.5, 2: 3.5, 3: 7.5, 4: 12.0}
df['Study_Hours'] = df['studytime'].map(hours_map)
df['Attendance'] = np.clip(100 - (df['absences'] * 2.5), 20, 100)
df['Pass_Fail'] = (df['G3'] >= 10).astype(int)

print("\n--- Processed Dataset Preview (First 5 records) ---")
print(df[['Study_Hours', 'Attendance', 'G3', 'Pass_Fail']].head())

X = df[['Study_Hours', 'Attendance']]
y = df['Pass_Fail']

print(f"\nTotal Students: {len(df)}")
print(f"Pass count (1): {(y == 1).sum()} | Fail count (0): {(y == 0).sum()}")

# -----------------------------------------------------------------------------
# Step 3: Split into Training (80%) and Testing (20%) Sets
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# Step 4: Feature Scaling
# KNN relies on Euclidean distance; therefore, scaling ensures features with
# larger magnitudes (Attendance 0-100) do not dominate smaller ones (Hours 1-12).
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Step 5: Train K-Nearest Neighbors (KNN) Classifier
# Parameters:
# - n_neighbors=5: Evaluates class majority among 5 nearest data points
# - metric='minkowski', p=2: Standard Euclidean distance
# -----------------------------------------------------------------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
knn.fit(X_train_scaled, y_train)

# -----------------------------------------------------------------------------
# Step 6: Model Evaluation
# -----------------------------------------------------------------------------
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "-" * 65)
print(f"KNN (k={k}) Test Accuracy: {accuracy * 100:.2f}%")
print("-" * 65)

print("\n--- Confusion Matrix ---")
print(cm)
print(f"True Negatives (Fail correctly predicted):  {cm[0, 0]}")
print(f"False Positives (Fail predicted as Pass):   {cm[0, 1]}")
print(f"False Negatives (Pass predicted as Fail):   {cm[1, 0]}")
print(f"True Positives (Pass correctly predicted):   {cm[1, 1]}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)']))

# Predict for a new student
sample_student = pd.DataFrame({'Study_Hours': [8.0], 'Attendance': [85.0]})
sample_scaled = scaler.transform(sample_student)
sample_pred = knn.predict(sample_scaled)[0]
sample_prob = knn.predict_proba(sample_scaled)[0]
print("--- Sample Student Prediction ---")
print(f"Input: Study Hours = 8.0 hrs/week, Attendance = 85.0%")
print(f"Predicted Status: {'PASS' if sample_pred == 1 else 'FAIL'} (Pass Probability: {sample_prob[1]*100:.1f}%)")

# -----------------------------------------------------------------------------
# Step 7: Visualize Decision Boundary in 2D Space
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))

# Create grid of points to evaluate decision regions
x_min, x_max = X['Study_Hours'].min() - 1, X['Study_Hours'].max() + 1
y_min, y_max = X['Attendance'].min() - 5, X['Attendance'].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_points = pd.DataFrame({'Study_Hours': xx.ravel(), 'Attendance': yy.ravel()})
grid_scaled = scaler.transform(grid_points)
Z = knn.predict(grid_scaled).reshape(xx.shape)

# Contour plot of decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Scatter plot of test samples
scatter = plt.scatter(
    X_test['Study_Hours'],
    X_test['Attendance'],
    c=y_test,
    cmap=plt.cm.coolwarm,
    edgecolors='black',
    s=65
)
plt.title(f'KNN (k={k}) Decision Boundary: Student Pass/Fail Classification', fontsize=12, fontweight='bold')
plt.xlabel('Weekly Study Hours', fontsize=11)
plt.ylabel('Attendance Percentage (%)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Custom legend
handles, _ = scatter.legend_elements()
plt.legend(handles, ['Fail (0)', 'Pass (1)'], title='Outcome', loc='lower right', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "knn_decision_boundary.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nKNN decision boundary plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 09 completed successfully!\n")
