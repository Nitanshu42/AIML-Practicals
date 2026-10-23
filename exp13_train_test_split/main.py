"""
Experiment 13: Train and Test Split Analysis
Aim: Divide a dataset into training and testing sets and compare the model's performance on both sets.
Subject: Artificial Intelligence and Machine Learning (AIML)
Semester: V | Branch: IT
Faculty: Prof. Ravindra R. Agrawal
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 65)
print(" Experiment 13: Train-Test Split & Generalization Analysis")
print("=" * 65)

# -----------------------------------------------------------------------------
# Step 1: Load Built-in Dataset from scikit-learn
# Wine recognition dataset: 178 samples, 13 chemical features, 3 classes of wine
# -----------------------------------------------------------------------------
wine = load_wine(as_frame=True)
df = wine.frame

print("Wine Dataset loaded successfully (scikit-learn built-in)!")
print(f"Total Samples: {len(df)} | Number of Features: {len(wine.feature_names)}")
print(f"Target Classes: {list(wine.target_names)}")

X = wine.data
y = wine.target

# -----------------------------------------------------------------------------
# Step 2: Divide Dataset into Training (80%) and Testing (20%) Sets
# - test_size=0.20: Reserves 20% of data exclusively for unbiased evaluation
# - random_state=42: Ensures reproducible split
# - stratify=y: Preserves original class proportions in both splits
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

total_count = len(df)
train_count = len(X_train)
test_count = len(X_test)

print("\n" + "-" * 65)
print(" Dataset Split Summary (80:20 Ratio):")
print("-" * 65)
print(f" Total Dataset Size:  {total_count} samples (100%)")
print(f" Training Set Size:   {train_count} samples ({train_count/total_count*100:.1f}%)")
print(f" Testing Set Size:    {test_count} samples ({test_count/total_count*100:.1f}%)")
print("-" * 65)

# -----------------------------------------------------------------------------
# Step 3: Train Model on Training Set Only
# We use a Decision Tree classifier with controlled depth (max_depth=3)
# to achieve a well-generalized model.
# -----------------------------------------------------------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Step 4: Compare Model Performance on Both Sets
# -----------------------------------------------------------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n" + "=" * 65)
print(" Model Performance Comparison:")
print("=" * 65)
print(f" Training Accuracy: {train_acc * 100:.2f}%")
print(f" Testing Accuracy:  {test_acc * 100:.2f}%")
print(f" Accuracy Gap:      {abs(train_acc - test_acc) * 100:.2f}%")
print("=" * 65)

# -----------------------------------------------------------------------------
# Step 5: Interpretation of Results (Viva / Theory Explanation)
# - Overfitting: Train Accuracy >> Test Accuracy (Model memorized noise in training data).
# - Underfitting: Both Train and Test Accuracy are low (Model is too simple to capture patterns).
# - Good Generalization: Train and Test Accuracies are both high and close to each other.
# -----------------------------------------------------------------------------
print("\n--- Diagnostic Conclusion ---")
if abs(train_acc - test_acc) <= 0.08 and test_acc >= 0.85:
    status = "Well-Generalized Model (Good Fit)"
    analysis = "The model learns meaningful underlying patterns without memorizing noise."
elif train_acc - test_acc > 0.15:
    status = "Overfitting Detected"
    analysis = "Training accuracy is significantly higher than test accuracy; regularization needed."
else:
    status = "Acceptable Performance"
    analysis = "Model generalizes reasonably well across unseen test data."

print(f" Model Status: {status}")
print(f" Explanation:  {analysis}")

print("\n--- Test Set Classification Report ---")
print(classification_report(y_test, test_pred, target_names=wine.target_names))

# -----------------------------------------------------------------------------
# Step 6: Visualize Train vs Test Performance
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 5))

labels = ['Training Set (80%)', 'Testing Set (20%)']
accuracies = [train_acc * 100, test_acc * 100]
colors = ['#2980b9', '#27ae60']

bars = plt.bar(labels, accuracies, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
plt.title('Model Accuracy Comparison: Training vs Testing Set', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=11)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Add exact value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 2,
        f'{height:.2f}%',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "train_test_comparison.png"
plt.savefig(plot_filename, dpi=300)
print(f"Comparison plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("\nExperiment 13 completed successfully!\n")
