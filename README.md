# ⚡ Energy Consumption Forecasting with MLflow

<p align="center">
  <img src="https://img.shields.io/badge/MLOps-MLflow-blueviolet?style=for-the-badge&logo=mlflow" alt="MLflow">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
</p>

---

## 📖 Project Overview
This repository contains a comprehensive **Time Series Forecasting** pipeline for the PJME Hourly Energy Consumption dataset. The core focus is on **MLOps best practices**, specifically using **MLflow** to track a high-dimensional experiment space involving multiple models, feature sets, and temporal splits.

## 🖥️ MLflow Dashboard
Experience how we track nested runs, parameters, and performance metrics in real-time.

<p align="center">
  <img src="assets/Screenshot 2026-03-20 at 10.40.20.png" width="90%" style="border-radius: 10px; border: 1px solid #ddd;" alt="MLflow Tracking UI">
  <br>
  <em>Figure 1: Nested Experiment tracking showing Model Types, Hyperparameters, and Metrics.</em>
</p>

---

## 🚀 Technical Highlights

### 1. Nested Experiment Architecture
We employ a 4-level nested loop to ensure every permutation is recorded:
* **Model Types:** XGBoost, Random Forest, Ridge, and Linear Regression.
* **Feature Engineering:** * *Basic:* Time-based temporal features.
    * *Lags:* 24h and 1-week historical shifts.
    * *Rolling:* 24h moving averages.
* **Hyperparameter Tuning:** Systematic variations of estimators and alpha values.
* **Time-Series Splits:** Validating across 3 different historical training windows.

### 2. Automated Evaluation & Artifacts
For every single run, the system automatically calculates and logs:
* **Metrics:** `RMSE` and `MAE`.
* **Artifacts:** High-resolution forecasting plots for both 1-week and 1-month horizons.

<p align="center">
  <img src="assets/Screenshot 2026-03-20 at 10.40.37.png" width="90%" style="border-radius: 10px; border: 1px solid #ddd;" alt="Forecasting Results">
  <br>
  <em>Figure 2: Actual vs. Predicted energy consumption (1-week vs 1-month).</em>
</p>

---

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have the PJME dataset (`PJME_hourly.csv`) in the root directory.

### Setup
```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

# Install dependencies
pip install pandas numpy xgboost scikit-learn matplotlib mlflow
