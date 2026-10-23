"""
Experiment 11: Model Performance Metrics
Aim: Calculate model performance metrics such as accuracy, precision, recall, and F1-score using a simple classification dataset.
Subject: Artificial Intelligence and Machine Learning (AIML)
Semester: V | Branch: IT
Faculty: Prof. Ravindra R. Agrawal
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

print("=" * 65)
print(" Experiment 11: Classification Performance Metrics")
print("=" * 65)

# -----------------------------------------------------------------------------
# Step 1: Load Built-in Breast Cancer Dataset from scikit-learn
# Features: 30 numerical measurements of cell nuclei
# Target: 0 = Malignant (harmful), 1 = Benign (harmless)
# -----------------------------------------------------------------------------
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

print("Breast Cancer Dataset loaded successfully (scikit-learn built-in)!")
print(f"Total Samples: {len(df)} | Number of Features: {len(cancer.feature_names)}")
print(f"Classes: {list(cancer.target_names)} -> 0: Malignant, 1: Benign")

X = cancer.data
y = cancer.target

# Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# Step 2: Feature Standardization & Model Training
# Standardizing features ensures fair contribution across all 30 variables.
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------------------------------------------------------
# Step 3: Generate Predictions
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------------------------------------------------------
# Step 4: Extract Confusion Matrix Elements
# TN = True Negatives  (Malignant correctly classified)
# FP = False Positives (Malignant misclassified as Benign)
# FN = False Negatives (Benign misclassified as Malignant)
# TP = True Positives  (Benign correctly classified)
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "-" * 65)
print(" Confusion Matrix Breakdown:")
print("-" * 65)
print(f" True Positives  (TP): {tp}  (Actual Benign correctly identified)")
print(f" True Negatives  (TN): {tn}  (Actual Malignant correctly identified)")
print(f" False Positives (FP): {fp}   (Malignant incorrectly called Benign)")
print(f" False Negatives (FN): {fn}   (Benign incorrectly called Malignant)")

# -----------------------------------------------------------------------------
# Step 5: Calculate and Explain Core Evaluation Metrics
# 1. Accuracy  = (TP + TN) / Total
#    Overall proportion of correct predictions across both classes.
# 2. Precision = TP / (TP + FP)
#    Out of all cases predicted as positive, how many were truly positive?
# 3. Recall    = TP / (TP + FN)  [Sensitivity / True Positive Rate]
#    Out of all actual positive cases, how many were correctly detected?
# 4. F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
#    Harmonic mean of precision and recall; balances false alarms and missed cases.
# -----------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Verify manual math matches scikit-learn
manual_acc = (tp + tn) / (tp + tn + fp + fn)
manual_prec = tp / (tp + fp)
manual_rec = tp / (tp + fn)
manual_f1 = 2 * (manual_prec * manual_rec) / (manual_prec + manual_rec)

print("\n" + "=" * 65)
print(" Calculated Performance Metrics:")
print("=" * 65)
metrics_table = pd.DataFrame({
    'Metric Name': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score'],
    'Formula': [
        '(TP + TN) / Total',
        'TP / (TP + FP)',
        'TP / (TP + FN)',
        '2 × (Prec × Rec) / (Prec + Rec)'
    ],
    'Score': [f"{acc:.4f} ({acc*100:.2f}%)",
              f"{prec:.4f} ({prec*100:.2f}%)",
              f"{rec:.4f} ({rec*100:.2f}%)",
              f"{f1:.4f} ({f1*100:.2f}%)"]
})
print(metrics_table.to_string(index=False))
print("=" * 65)

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))

# -----------------------------------------------------------------------------
# Step 6: Visualize Confusion Matrix Heatmap
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant (0)', 'Benign (1)'])
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title(f'Confusion Matrix (Accuracy = {acc*100:.1f}%)', fontsize=12, fontweight='bold')
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "performance_metrics.png"
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

print("\nExperiment 11 completed successfully!\n")
