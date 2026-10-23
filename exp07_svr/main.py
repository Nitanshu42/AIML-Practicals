"""
Experiment 07: Support Vector Regression (SVR)
Aim: Implement Support Vector Regression (SVR) to predict house prices using house size.
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
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# Step 1: Load Housing Dataset from Public Online Repository
# Dataset Source: King County House Prices (Public Benchmark Dataset)
# URL: https://raw.githubusercontent.com/YBIFoundation/Dataset/main/House%20Prices.csv
# -----------------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/House%20Prices.csv"

print("=" * 65)
print(" Experiment 07: Support Vector Regression (SVR) - House Prices")
print("=" * 65)
print(f"Fetching dataset from public URL:\n  {DATASET_URL}\n")

try:
    req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as response:
        csv_data = response.read()
    # Read first 1000 rows for fast and optimal training in practical lab exams
    df = pd.read_csv(io.BytesIO(csv_data), nrows=1000)
    print("Dataset successfully downloaded and loaded!")
except Exception as e:
    print(f"\nERROR: Failed to download online dataset: {e}")
    print("Please check your internet connection or URL accessibility.")
    sys.exit(1)

# Validate expected columns
expected_cols = ['Sqft_living', 'Price']
if not all(col in df.columns for col in expected_cols):
    print(f"ERROR: Missing expected columns {expected_cols}. Found: {df.columns.tolist()}")
    sys.exit(1)

# Remove any null or non-positive values
df = df.dropna(subset=expected_cols)
df = df[(df['Sqft_living'] > 0) & (df['Price'] > 0)]

print("\n--- Dataset Preview (First 5 records) ---")
print(df[['Sqft_living', 'Price']].head())
print(f"\nSamples used: {len(df)}")

# -----------------------------------------------------------------------------
# Step 2: Prepare Feature and Target
# Independent Variable (X): Sqft_living (House Size in Sq. Feet)
# Dependent Variable (y): Price (House Price in USD)
# -----------------------------------------------------------------------------
X = df[['Sqft_living']]
y = df['Price'].values.reshape(-1, 1)

# Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------------------------------------------------------
# Step 3: Feature Scaling (Crucial for SVR)
# Support Vector Regression is distance-based and highly sensitive to feature scale.
# Both X and y are scaled using StandardScaler.
# -----------------------------------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train).ravel()

X_test_scaled = scaler_X.transform(X_test)

# -----------------------------------------------------------------------------
# Step 4: Train Support Vector Regression (SVR) Model
# Kernel: 'rbf' (Radial Basis Function for capturing non-linear relationships)
# C=100.0: Regularization parameter (trade-off between margin and error)
# epsilon=0.1: Epsilon-tube within which no penalty is associated in training loss
# -----------------------------------------------------------------------------
svr_model = SVR(kernel='rbf', C=100.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train_scaled)

# -----------------------------------------------------------------------------
# Step 5: Predict and Convert Predictions Back to Original Units ($)
# -----------------------------------------------------------------------------
y_pred_scaled = svr_model.predict(X_test_scaled)

# Inverse transform to return predictions from z-score back to USD ($)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = y_test.ravel()

mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print("\n" + "-" * 65)
print(" SVR Model Evaluation (Converted to Original Price Units):")
print(f" Mean Absolute Error (MAE): ${mae:,.2f}")
print(f" R-squared (R2) Score:      {r2:.4f} ({r2*100:.2f}% variance explained)")
print("-" * 65)

# Predict price for sample house sizes
sample_sizes = [1500, 2200, 3000]
print("\n--- Sample House Price Predictions ---")
for sqft in sample_sizes:
    sqft_df = pd.DataFrame({'Sqft_living': [sqft]})
    sqft_scaled = scaler_X.transform(sqft_df)
    price_scaled = svr_model.predict(sqft_scaled)
    predicted_price = scaler_y.inverse_transform(price_scaled.reshape(-1, 1))[0][0]
    print(f" House Size: {sqft} sq ft -> Predicted Price: ${predicted_price:,.2f}")

# -----------------------------------------------------------------------------
# Step 6: Visualize Actual Observations and SVR Predictions
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))

# Sort test points for a smooth prediction curve
sorted_indices = np.argsort(X_test['Sqft_living'].values)
X_test_sorted = X_test['Sqft_living'].values[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Actual observations
plt.scatter(X_test['Sqft_living'], y_test_orig, color='cornflowerblue', edgecolors='black',
            s=50, alpha=0.7, label='Actual Test Observations')

# SVR Prediction Curve
grid_sqft = pd.DataFrame({'Sqft_living': np.linspace(X['Sqft_living'].min(), X['Sqft_living'].max(), 200)})
grid_sqft_scaled = scaler_X.transform(grid_sqft)
grid_pred_scaled = svr_model.predict(grid_sqft_scaled)
grid_pred_price = scaler_y.inverse_transform(grid_pred_scaled.reshape(-1, 1))

plt.plot(grid_sqft['Sqft_living'], grid_pred_price, color='crimson', linewidth=2.5,
         label='SVR Regression Curve (RBF Kernel)')

plt.title('Support Vector Regression (SVR): House Price vs House Size', fontsize=12, fontweight='bold')
plt.xlabel('House Size (Sqft Living)', fontsize=11)
plt.ylabel('House Price ($)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "svr_prediction.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nSVR plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 07 completed successfully!\n")
