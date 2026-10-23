"""
Experiment 05: Decision Tree Classifier
Aim: Implement a Decision Tree classifier to predict student results.
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from Public Online Repository
# Dataset Source: Student Performance in Exams (Public benchmark dataset)
# URL: https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv"

print("=" * 65)
print(" Experiment 05: Decision Tree - Student Result Prediction")
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
# Step 2: Prepare Input Features and Target Labels
# Target Result: 'Pass' if Math Score >= 50 else 'Fail'
# Input Features: 'reading score' and 'writing score'
# -----------------------------------------------------------------------------
PASSING_MARKS = 50
df['Result'] = df['math score'].apply(lambda x: 'Pass' if x >= PASSING_MARKS else 'Fail')

feature_cols = ['reading score', 'writing score']
X = df[feature_cols]
y = df['Result']

print(f"\nClass Distribution in Dataset:")
print(df['Result'].value_counts())

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# -----------------------------------------------------------------------------
# Step 3: Train Decision Tree Classifier
# Parameters:
# - criterion='gini': Gini Impurity measures split purity
# - max_depth=3: Limits depth to keep tree easily interpretable for viva
# - random_state=42: Fixed seed for reproducibility
# -----------------------------------------------------------------------------
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Step 4: Display Predictions and Accuracy
# -----------------------------------------------------------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "-" * 65)
print(f"Decision Tree Accuracy on Test Data: {accuracy * 100:.2f}%")
print("-" * 65)

print("\n--- Actual vs Predicted Results (First 10 Test Samples) ---")
results_table = pd.DataFrame({
    'Reading Score': X_test['reading score'].values[:10],
    'Writing Score': X_test['writing score'].values[:10],
    'Actual Result': y_test.values[:10],
    'Predicted Result': y_pred[:10],
    'Match': ['Yes' if a == p else 'No' for a, p in zip(y_test.values[:10], y_pred[:10])]
})
print(results_table.to_string(index=False))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------------
# Step 5: Visualize the Decision Tree
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 8))
plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=['Fail', 'Pass'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Visualization (Max Depth = 3)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "decision_tree.png"
plt.savefig(plot_filename, dpi=300)
print(f"Decision tree visualization saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("\nExperiment 05 completed successfully!\n")
