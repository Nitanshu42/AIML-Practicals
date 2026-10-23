"""
Experiment 04: Naive Bayes Classifier
Aim: Implement the Naïve Bayes algorithm to predict whether a student will pass or fail.
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from Public Online Repository
# Dataset Source: Student Performance in Exams (Public benchmark dataset)
# URL: https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv"

print("=" * 65)
print(" Experiment 04: Naive Bayes - Student Pass/Fail Prediction")
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
# Step 2: Prepare Target Variable and Input Features
# Target: Pass (1) if Math Score >= 50, otherwise Fail (0) [Standard 50% passing threshold]
# Input Features: Reading Score and Writing Score
# NOTE: To strictly avoid Data Leakage, Math Score is NOT used as an input feature!
# -----------------------------------------------------------------------------
PASSING_THRESHOLD = 50
df['Pass_Fail'] = (df['math score'] >= PASSING_THRESHOLD).astype(int)

X = df[['reading score', 'writing score']]
y = df['Pass_Fail']

pass_count = (y == 1).sum()
fail_count = (y == 0).sum()
print(f"\nTarget Variable: Pass_Fail (Threshold >= {PASSING_THRESHOLD} marks)")
print(f"Passed Students (1): {pass_count} ({pass_count/len(df)*100:.1f}%)")
print(f"Failed Students (0): {fail_count} ({fail_count/len(df)*100:.1f}%)")

# -----------------------------------------------------------------------------
# Step 3: Split into Training (80%) and Testing (20%) Sets
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining Samples: {X_train.shape[0]}")
print(f"Testing Samples:  {X_test.shape[0]}")

# -----------------------------------------------------------------------------
# Step 4: Train Gaussian Naive Bayes Classifier
# Naive Bayes applies Bayes' Theorem with the assumption of conditional independence:
# P(Pass | Features) ∝ P(Pass) * ∏ P(Feature_i | Pass)
# -----------------------------------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Step 5: Predict and Evaluate Performance
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "-" * 65)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print("-" * 65)

print("\n--- Actual vs Predicted Comparison (First 10 Test Samples) ---")
comparison_df = pd.DataFrame({
    'Reading Score': X_test['reading score'].values[:10],
    'Writing Score': X_test['writing score'].values[:10],
    'Actual (Pass=1/Fail=0)': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Result': ['Correct' if a == p else 'Incorrect' for a, p in zip(y_test.values[:10], y_pred[:10])]
})
print(comparison_df.to_string(index=False))

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)']))

# -----------------------------------------------------------------------------
# Step 6: Visualize Confusion Matrix
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail (0)', 'Pass (1)'])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title('Naive Bayes: Confusion Matrix for Student Pass/Fail', fontsize=11, fontweight='bold')
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "confusion_matrix.png"
plt.savefig(plot_filename, dpi=300)
print(f"Confusion matrix plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("\nExperiment 04 completed successfully!\n")
