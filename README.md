# AML Detection Model

This project implements an AI-based Anti-Money Laundering (AML) detection model using supervised machine learning. The goal is to classify financial transactions as suspicious or non-suspicious based on key features extracted from transaction data. This can help banks or financial institutions flag potentially fraudulent activity in real-time or in post-processing.

---

## 📌 Project Overview

Money laundering is a critical problem in the financial sector, involving illegal processing of large sums of money obtained from criminal activity. Detecting such behavior is difficult due to the imbalance in datasets (i.e., very few fraudulent cases) and the subtlety of laundering patterns.

This project uses machine learning to:

- Train a binary classification model on transaction data
- Handle class imbalance
- Evaluate model performance using ROC and confusion matrix
- Output metrics that help in understanding prediction quality

---

## 📊 Dataset Used

For demonstration purposes, the current setup uses **synthetic data** generated using NumPy. In a real-world scenario, this would be replaced with a dataset containing:

- Transaction amount
- Transaction type
- Account balance changes
- Timestamps and transaction frequency
- Customer metadata (risk score, KYC compliance, etc.)

You can integrate your own dataset by modifying the `load_data()` function in `aml_detection.py`.

Kaggle – Synthetic Financial Dataset For Fraud Detection
📎 https://www.kaggle.com/datasets/ealaxi/paysim1

---

## ⚙️ Model Training

The model pipeline includes:

1. **Data Generation/Loading** – Generates a synthetic dataset or loads your custom dataset.
2. **Preprocessing** – Standardizes features and optionally applies SMOTE or undersampling to handle imbalance.
3. **Train/Test Split** – Splits the data into training and test sets using stratified sampling.
4. **Model Training** – Uses a supervised classifier.
5. **Evaluation** – Outputs classification report, confusion matrix, and ROC curve with AUC.

---

## 🧠 Model Used

We use the **Random Forest Classifier** from scikit-learn due to its:

- Robustness against overfitting
- Ability to handle non-linear feature interactions
- Interpretability via feature importance

This can be swapped with more complex models (e.g., XGBoost, LightGBM, or deep learning) as needed.

---


