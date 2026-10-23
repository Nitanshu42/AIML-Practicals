"""
Experiment 08: Ridge Regression (L2 Regularization)
Aim: Apply Ridge Regression to reduce overfitting and predict numerical values.
Subject: Artificial Intelligence and Machine Learning (AIML)
Semester: V | Branch: IT
Faculty: Prof. Ravindra R. Agrawal
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 65)
print(" Experiment 08: Ridge Regression (L2 Regularization)")
print("=" * 65)

# -----------------------------------------------------------------------------
# Step 1: Load Built-in Diabetes Dataset from scikit-learn
# Features: 10 baseline variables (age, sex, bmi, bp, and 6 blood serum measurements)
# Target: Quantitative measure of disease progression one year after baseline
# -----------------------------------------------------------------------------
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print("Diabetes Dataset loaded successfully (scikit-learn built-in)!")
print("\n--- Dataset Preview (First 5 records) ---")
print(df.head())
print(f"\nTotal Samples: {len(df)} | Number of Features: {len(diabetes.feature_names)}")

X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# -----------------------------------------------------------------------------
# Step 2: Split into Training (80%) and Testing (20%) Sets
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------------------------------------------------------
# Step 3: Feature Scaling
# L2 regularization penalizes the sum of squared coefficients (alpha * sum(w_i^2)).
# Standardization is essential so all features are penalized equally.
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Step 4: Train Ordinary Linear Regression (Baseline Model)
# -----------------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# -----------------------------------------------------------------------------
# Step 5: Train Ridge Regression (With L2 Regularization Penalty)
# Cost Function = Loss + alpha * sum(w_i^2)
# alpha controls regularization strength: larger alpha shrinks coefficients more
# -----------------------------------------------------------------------------
alpha_val = 10.0
ridge_model = Ridge(alpha=alpha_val, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_model.predict(X_test_scaled)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# -----------------------------------------------------------------------------
# Step 6: Compare Results: Linear Regression vs Ridge Regression
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print(f" Performance Comparison on Test Set (Ridge alpha = {alpha_val}):")
print("=" * 65)
comparison_metrics = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R2 Score'],
    'Linear Regression': [f"{mae_lr:.4f}", f"{mse_lr:.4f}", f"{r2_lr:.4f}"],
    'Ridge Regression':   [f"{mae_ridge:.4f}", f"{mse_ridge:.4f}", f"{r2_ridge:.4f}"]
})
print(comparison_metrics.to_string(index=False))

print("\n--- Model Coefficients Shrinkage Comparison ---")
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Linear_Reg_Coef': lr_model.coef_,
    'Ridge_Coef': ridge_model.coef_,
    'Shrinkage_%': ((1 - np.abs(ridge_model.coef_) / np.abs(lr_model.coef_)) * 100).round(2)
})
print(coef_df.to_string(index=False))
print("-" * 65)
print("Observation: Ridge Regression shrinks extreme coefficients towards zero,")
print("which reduces model variance and prevents overfitting on unseen data.")

# -----------------------------------------------------------------------------
# Step 7: Visualize Coefficient Shrinkage
# -----------------------------------------------------------------------------
x_pos = np.arange(len(feature_names))
bar_width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x_pos - bar_width/2, lr_model.coef_, width=bar_width, label='Linear Regression', color='indianred', edgecolor='black')
plt.bar(x_pos + bar_width/2, ridge_model.coef_, width=bar_width, label=f'Ridge (alpha={alpha_val})', color='royalblue', edgecolor='black')

plt.title('Coefficient Comparison: Linear Regression vs Ridge Regression (L2)', fontsize=12, fontweight='bold')
plt.xlabel('Features', fontsize=11)
plt.ylabel('Coefficient Value (Weight)', fontsize=11)
plt.xticks(x_pos, feature_names, rotation=30)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()

# Save plot for examination submission/record
plot_filename = "ridge_coefficients.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nCoefficient comparison plot saved as '{plot_filename}'")

# Interactive display (non-blocking in headless mode)
if os.environ.get("NON_INTERACTIVE") == "1":
    plt.close()
else:
    try:
        plt.show()
    except Exception:
        plt.close()

print("Experiment 08 completed successfully!\n")
