"""
Experiment 06: Simple Linear Regression
Aim: Implement Linear Regression to predict marks based on study hours.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# Step 1: Load Dataset from Public Online Repository
# Dataset Source: Student Study Hours vs Exam Scores (Public benchmark dataset)
# URL: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

print("=" * 65)
print(" Experiment 06: Linear Regression - Predict Marks from Study Hours")
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
expected_cols = ['Hours', 'Scores']
if not all(col in df.columns for col in expected_cols):
    print(f"ERROR: Missing expected columns {expected_cols}. Found: {df.columns.tolist()}")
    sys.exit(1)

print("\n--- Dataset Preview (First 5 records) ---")
print(df.head())
print(f"\nTotal Records: {len(df)}")

# -----------------------------------------------------------------------------
# Step 2: Prepare Input Feature and Target Variable
# Independent variable X: Hours (2D array required by scikit-learn)
# Dependent variable y: Scores
# -----------------------------------------------------------------------------
X = df[['Hours']]
y = df['Scores']

# Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------------------------------------------------------
# Step 3: Train Linear Regression Model
# Mathematical Model: y = m * x + c
# where m = slope (weight) and c = intercept (bias)
# -----------------------------------------------------------------------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

slope = regressor.coef_[0]
intercept = regressor.intercept_

print("\n" + "-" * 65)
print(" Model Parameters:")
print(f"   Slope (m / Coefficient): {slope:.4f}")
print(f"   Intercept (c / Bias):    {intercept:.4f}")
print(f"   Regression Equation:     Score = ({slope:.2f} × Hours) + {intercept:.2f}")
print("-" * 65)

# -----------------------------------------------------------------------------
# Step 4: Evaluate Model on Test Set
# -----------------------------------------------------------------------------
y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Test Set Evaluation Metrics ---")
print(f" Mean Absolute Error (MAE):  {mae:.4f}")
print(f" Mean Squared Error (MSE):   {mse:.4f}")
print(f" Root Mean Sq. Error (RMSE): {rmse:.4f}")
print(f" R-squared Score (R2):       {r2:.4f} ({r2*100:.2f}% variance explained)")

# -----------------------------------------------------------------------------
# Step 5: Predict Marks for Sample Study Hours
# -----------------------------------------------------------------------------
sample_hours = [9.25, 4.5, 6.0]
print("\n--- Sample Predictions ---")
for h in sample_hours:
    pred_score = regressor.predict(pd.DataFrame({'Hours': [h]}))[0]
    print(f" Predicted mark for {h} hours of study: {pred_score:.2f}%")

# -----------------------------------------------------------------------------
# Step 6: Plot Actual Observations and Regression Line
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))

# Plot training and test observations
plt.scatter(X_train, y_train, color='royalblue', edgecolors='black', s=70, label='Train Observations')
plt.scatter(X_test, y_test, color='darkorange', edgecolors='black', s=90, label='Test Observations')

# Plot best-fit regression line
line_X = pd.DataFrame({'Hours': np.linspace(X['Hours'].min(), X['Hours'].max(), 100)})
line_y = regressor.predict(line_X)
plt.plot(line_X['Hours'], line_y, color='crimson', linewidth=2,
         label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

plt.title('Linear Regression: Study Hours vs Student Marks', fontsize=12, fontweight='bold')
plt.xlabel('Study Hours', fontsize=11)
plt.ylabel('Exam Marks (%)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "linear_regression.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nRegression line plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 06 completed successfully!\n")
