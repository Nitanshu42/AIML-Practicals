# AIML-Practicals
**Artificial Intelligence and Machine Learning (AIML) – Laboratory Practicals**  
*Semester: V | Branch: Information Technology (IT)*  
*Faculty: Prof. Ravindra R. Agrawal*

---

## 📌 Overview
This repository contains the complete, tested, and fully functional implementations of all **13 AIML Practical Experiments** for the Semester V Information Technology curriculum.

### ✨ Key Features
- **Zero Local Dataset Dependency**: Uses only scikit-learn built-in datasets or trusted public raw HTTPS URLs.
- **Reproducible Results**: All experiments use fixed random states (`random_state=42`).
- **Clean and Well-Commented**: Written specifically for university practical exams and viva examinations.
- **Dual Access System**: Access all practicals from the `main` branch or clone isolated, standalone branches (`exp01` to `exp13`) directly during exams.

---

## 📋 Complete List of Experiments

| Exp No. | Experiment Name | Aim | Dataset Source | Algorithm |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [Scatter Plot](exp01_scatter_plot/main.py) | Identify the association between independent and dependent variables using a scatter plot. | [Public Student Scores (AdiPersonalWorks)](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv) | Pearson Correlation & Matplotlib Scatter |
| **02** | [K-Means Clustering](exp02_kmeans/main.py) | Apply K-Means clustering to group students based on their marks in two subjects. | [Public Student Performance (YBIFoundation)](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv) | K-Means Clustering (`k=3`) |
| **03** | [PCA](exp03_pca/main.py) | Apply Principal Component Analysis (PCA) to reduce dataset dimensions. | `sklearn.datasets.load_iris` | PCA (4D &rarr; 2D) |
| **04** | [Naive Bayes](exp04_naive_bayes/main.py) | Implement the Naïve Bayes algorithm to predict whether a student will pass or fail. | [Public Student Performance (YBIFoundation)](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv) | Gaussian Naïve Bayes |
| **05** | [Decision Tree](exp05_decision_tree/main.py) | Implement a Decision Tree classifier to predict student results. | [Public Student Performance (YBIFoundation)](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv) | Decision Tree Classifier (`max_depth=3`) |
| **06** | [Linear Regression](exp06_linear_regression/main.py) | Implement Linear Regression to predict marks based on study hours. | [Public Student Scores (AdiPersonalWorks)](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv) | Simple Linear Regression ($y = mx + c$) |
| **07** | [SVR](exp07_svr/main.py) | Implement Support Vector Regression (SVR) to predict house prices using house size. | [Public House Prices (YBIFoundation)](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/House%20Prices.csv) | Support Vector Regression (RBF Kernel) |
| **08** | [Ridge Regression](exp08_ridge_regression/main.py) | Apply Ridge Regression to reduce overfitting and predict numerical values. | `sklearn.datasets.load_diabetes` | Ridge Regression ($L_2$ Regularization) |
| **09** | [KNN](exp09_knn/main.py) | Implement K-Nearest Neighbors (KNN) to classify students as pass or fail based on study hours and attendance. | [UCI Student Performance (guipsamora)](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv) | K-Nearest Neighbors (`k=5`) |
| **10** | [Logistic Regression](exp10_logistic_regression/main.py) | Implement Logistic Regression to predict whether a student will pass or fail. | [Public Student Performance (YBIFoundation)](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Students%20Performance.csv) | Logistic Regression (Sigmoid / Binary Cross-Entropy) |
| **11** | [Performance Metrics](exp11_performance_metrics/main.py) | Calculate model performance metrics such as accuracy, precision, recall and F1-score using a simple classification dataset. | `sklearn.datasets.load_breast_cancer` | Confusion Matrix & Evaluation Metrics |
| **12** | [Lasso Regression](exp12_lasso_regression/main.py) | Implement Lasso Regression to predict marks based on study hours. | [Public Student Scores (AdiPersonalWorks)](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv) | Lasso Regression ($L_1$ Regularization) |
| **13** | [Train-Test Split](exp13_train_test_split/main.py) | Divide a dataset into training and testing sets and compare the model's performance on both sets. | `sklearn.datasets.load_wine` | Train-Test Split (80:20) & Generalization Analysis |

---

## 🛠️ Requirements & Installation

### Prerequisites
- **Python**: Version 3.9 or higher (tested on Python 3.12)
- **Git** installed on your system
- **PowerShell** (Windows) or standard terminal

### Quick Setup (Automated - Recommended for Windows)
Run the automated setup script in PowerShell:
```powershell
.\setup.ps1
```
This script automatically:
1. Verifies your Python installation.
2. Creates a dedicated `.venv` virtual environment if not present.
3. Upgrades pip and installs all required dependencies cleanly.

### Manual Installation
If you prefer setting up manually:
```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows PowerShell

# 2. Install dependencies
python -m pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Method A: Running from the Main Branch
Clone the entire repository:
```bash
git clone https://github.com/Nitanshu42/AIML-Practicals.git
cd AIML-Practicals
```

#### 1. List All Experiments
```bash
python run.py --list
```

#### 2. Run Any Experiment via the Master Runner
```bash
python run.py 1     # Runs Experiment 01 (Scatter Plot)
python run.py 6     # Runs Experiment 06 (Linear Regression)
python run.py 10    # Runs Experiment 10 (Logistic Regression)
python run.py 13    # Runs Experiment 13 (Train-Test Split)
```

#### 3. Run Directly Inside Any Experiment Folder
```bash
python exp06_linear_regression/main.py
```

---

## 🌿 Method B: Isolated Experiment Branches (Single-Branch Clone)

During your practical examination, you can clone **ONLY the single experiment branch** you were assigned. This downloads only that experiment's files at the root level so you can immediately run `python main.py` without navigating into subfolders!

### Branch Naming Convention
Branches are named: `exp01`, `exp02`, `exp03`, ..., `exp13`.

### Example: Practical Examination Workflow for Experiment 10
When assigned **Experiment 10 (Logistic Regression)** during the examination:

```powershell
# 1. Clone ONLY the exp10 branch
git clone -b exp10 --single-branch https://github.com/Nitanshu42/AIML-Practicals.git

# 2. Navigate into the folder
cd AIML-Practicals

# 3. Install requirements
python -m pip install -r requirements.txt

# 4. Execute the experiment directly
python main.py
```

The same 4-step workflow works for **all 13 branches** (`exp01` through `exp13`)!

---

## 📊 Dataset Attribution & Privacy
- **Built-in Datasets**: Uses standard datasets packaged with `scikit-learn` (`iris`, `diabetes`, `breast_cancer`, `wine`).
- **Online Public Datasets**: Directly fetched over HTTPS from trusted, public open-data repositories (UCI Machine Learning Repository, YBIFoundation Open Datasets, AdiPersonalWorks).
- **Offline / Security Compliance**: No API keys, no login credentials, and no local CSV file prerequisites.

---

## 🔧 Troubleshooting

1. **PowerShell Script Execution Policy Error:**
   If running `setup.ps1` gives an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\setup.ps1
   ```

2. **Graph Window Blocking:**
   All scripts save high-resolution `.png` figures directly into the working directory. In automated or headless environments, set `$env:NON_INTERACTIVE="1"` before execution.

3. **Internet Access in Exam Labs:**
   Ensure your exam machine has internet access to allow `pandas.read_csv` to stream online public datasets. Built-in datasets (Experiments 03, 08, 11, 13) require zero internet connectivity.

---

## 👨‍🏫 Academic Details
- **Student Repository**: IT Semester V AIML Lab Examination
- **Subject**: Artificial Intelligence & Machine Learning (AIML)
- **Faculty Guide**: Prof. Ravindra R. Agrawal
