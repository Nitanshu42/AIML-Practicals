"""
Experiment 01: Scatter Plot Analysis
Aim: Identify the association between independent and dependent variables using a scatter plot.
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
import numpy as np

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from a Public Online Repository
# Dataset Source: Student Study Hours vs Exam Scores (Public benchmark dataset)
# URL: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

print("=" * 65)
print(" Experiment 01: Scatter Plot - Association Between Variables")
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
expected_cols = ['Hours', 'Scores']
if not all(col in df.columns for col in expected_cols):
    print(f"ERROR: Dataset missing expected columns {expected_cols}. Found: {df.columns.tolist()}")
    sys.exit(1)

# Display first few rows and summary
print("\n--- Dataset Preview (First 5 records) ---")
print(df.head())
print(f"\nTotal observations: {len(df)}")
print(f"Missing values: {df.isnull().sum().to_dict()}")

# -----------------------------------------------------------------------------
# Step 2: Define Independent and Dependent Variables
# Independent Variable (X): Hours (Study time in hours) - cause
# Dependent Variable (Y): Scores (Exam score in percentage) - effect
# -----------------------------------------------------------------------------
X = df['Hours']
Y = df['Scores']

# -----------------------------------------------------------------------------
# Step 3: Quantify Association (Pearson Correlation Coefficient)
# Formula: r = Cov(X, Y) / (std(X) * std(Y))
# Range: -1 (perfect negative) to +1 (perfect positive); 0 means no linear correlation
# -----------------------------------------------------------------------------
correlation = X.corr(Y)
print("\n" + "-" * 65)
print(f"Pearson Correlation Coefficient (r): {correlation:.4f}")

if correlation > 0.7:
    nature = "Strong Positive Association"
    explanation = "As study hours increase, exam scores consistently increase."
elif correlation > 0.3:
    nature = "Moderate Positive Association"
    explanation = "General upward trend between hours and scores."
elif correlation > -0.3:
    nature = "Weak or No Association"
    explanation = "No meaningful linear relationship between the variables."
elif correlation > -0.7:
    nature = "Moderate Negative Association"
    explanation = "General downward trend between the variables."
else:
    nature = "Strong Negative Association"
    explanation = "As one variable increases, the other sharply decreases."

print(f"Association Nature: {nature}")
print(f"Interpretation: {explanation}")
print("-" * 65)

# -----------------------------------------------------------------------------
# Step 4: Visualize Association Using a Scatter Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))

# Scatter plot of actual observations
plt.scatter(X, Y, color='royalblue', edgecolor='black', s=80, alpha=0.9, label='Student Observations')

# Fit and plot a linear trendline (y = mx + c) to highlight the association
slope, intercept = np.polyfit(X, Y, 1)
plt.plot(X, slope * X + intercept, color='crimson', linestyle='--', linewidth=2,
         label=f'Trendline: Score = {slope:.2f} × Hours + {intercept:.2f}')

# Styling
plt.title('Association Between Study Hours (Independent) & Exam Score (Dependent)', fontsize=12, fontweight='bold')
plt.xlabel('Study Hours (Independent Variable)', fontsize=11)
plt.ylabel('Exam Score (Dependent Variable)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "scatter_plot.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nScatter plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 01 completed successfully!\n")
