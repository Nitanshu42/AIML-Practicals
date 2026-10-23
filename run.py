"""
AIML Practical Examination - Master Runner
==========================================
Usage:
    python run.py --list        # List all available experiments
    python run.py <number>      # Run experiment by number (1 to 13)
    Example: python run.py 1
             python run.py 10
"""

import os
import sys
import subprocess

EXPERIMENTS = {
    1: {
        "folder": "exp01_scatter_plot",
        "name": "Scatter Plot - Association Between Variables",
        "aim": "Identify the association between independent and dependent variables using a scatter plot."
    },
    2: {
        "folder": "exp02_kmeans",
        "name": "K-Means Clustering",
        "aim": "Apply K-Means clustering to group students based on their marks in two subjects."
    },
    3: {
        "folder": "exp03_pca",
        "name": "Principal Component Analysis (PCA)",
        "aim": "Apply Principal Component Analysis (PCA) to reduce dataset dimensions."
    },
    4: {
        "folder": "exp04_naive_bayes",
        "name": "Naive Bayes Classification",
        "aim": "Implement the Naive Bayes algorithm to predict whether a student will pass or fail."
    },
    5: {
        "folder": "exp05_decision_tree",
        "name": "Decision Tree Classifier",
        "aim": "Implement a Decision Tree classifier to predict student results."
    },
    6: {
        "folder": "exp06_linear_regression",
        "name": "Linear Regression",
        "aim": "Implement Linear Regression to predict marks based on study hours."
    },
    7: {
        "folder": "exp07_svr",
        "name": "Support Vector Regression (SVR)",
        "aim": "Implement Support Vector Regression (SVR) to predict house prices using house size."
    },
    8: {
        "folder": "exp08_ridge_regression",
        "name": "Ridge Regression (L2 Regularization)",
        "aim": "Apply Ridge Regression to reduce overfitting and predict numerical values."
    },
    9: {
        "folder": "exp09_knn",
        "name": "K-Nearest Neighbors (KNN)",
        "aim": "Implement K-Nearest Neighbors (KNN) to classify students as pass or fail based on study hours and attendance."
    },
    10: {
        "folder": "exp10_logistic_regression",
        "name": "Logistic Regression",
        "aim": "Implement Logistic Regression to predict whether a student will pass or fail."
    },
    11: {
        "folder": "exp11_performance_metrics",
        "name": "Model Performance Metrics",
        "aim": "Calculate model performance metrics such as accuracy, precision, recall and F1-score using a simple classification dataset."
    },
    12: {
        "folder": "exp12_lasso_regression",
        "name": "Lasso Regression (L1 Regularization)",
        "aim": "Implement Lasso Regression to predict marks based on study hours."
    },
    13: {
        "folder": "exp13_train_test_split",
        "name": "Train and Test Split Performance Comparison",
        "aim": "Divide a dataset into training and testing sets and compare the model's performance on both sets."
    }
}


def print_list():
    print("=" * 80)
    print(" AIML Practical Examination - List of Experiments")
    print("=" * 80)
    for num, details in EXPERIMENTS.items():
        print(f" [{num:02d}] {details['name']}")
        print(f"      Aim: {details['aim']}")
        print(f"      Path: {details['folder']}/main.py")
        print("-" * 80)
    print("\nTo execute any experiment, run: python run.py <number>")
    print("Example: python run.py 6\n")


def execute_experiment(exp_num):
    if exp_num not in EXPERIMENTS:
        print(f"ERROR: Invalid experiment number '{exp_num}'.")
        print("Please choose an experiment number between 1 and 13.")
        print("Run 'python run.py --list' to view all available experiments.")
        sys.exit(1)

    exp_info = EXPERIMENTS[exp_num]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(base_dir, exp_info["folder"])
    script_path = os.path.join(exp_dir, "main.py")

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at {script_path}")
        sys.exit(1)

    print("=" * 80)
    print(f" EXECUTING EXPERIMENT {exp_num:02d}: {exp_info['name']}")
    print(f" Aim: {exp_info['aim']}")
    print("=" * 80 + "\n")

    result = subprocess.run([sys.executable, script_path], cwd=exp_dir)
    sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py --list        (list all experiments)")
        print("  python run.py <number>      (run experiment 1 to 13)")
        sys.exit(1)

    arg = sys.argv[1].strip()

    if arg in ("--list", "-l", "list"):
        print_list()
        return

    try:
        exp_num = int(arg)
        execute_experiment(exp_num)
    except ValueError:
        print(f"ERROR: '{arg}' is not a valid number.")
        print("Usage: python run.py <number> (e.g., python run.py 1)")
        sys.exit(1)


if __name__ == "__main__":
    main()
