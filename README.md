# 🕵️‍♂️ Financial Fraud Detection System

A machine learning pipeline designed to detect fraudulent financial transactions. This project tackles the challenge of extreme class imbalance by benchmarking **Weighted XGBoost** against standard sampling techniques like **SMOTE** and **Undersampling**.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)

---

## 🔍 Overview
Financial fraud poses a significant threat to the global economy. This project analyzes the **PaySim** dataset to identify patterns in fraudulent behavior. By comparing multiple modeling approaches, we determined that tree-based boosting methods (XGBoost) significantly outperform linear models (Logistic Regression) in distinguishing complex fraud patterns.

**Key Challenges:**
* **Extreme Class Imbalance:** Only **0.13%** of transactions are fraudulent.
* **High False Positive Rate:** Balancing the need to catch fraud (Recall) without overwhelming investigators with false alarms (Precision).

## 💡 Key Findings
Through rigorous Exploratory Data Analysis (EDA), we discovered distinct behavioral fingerprints of fraud:

1.  **Account Emptying Pattern:** There is a strong correlation between fraud and account depletion. In **8,034** confirmed fraud cases, the transaction amount exactly matched the originating account's old balance.
2.  **Transaction Type:** Fraudulent activity was found exclusively in `TRANSFER` and `CASH_OUT` transaction types.
3.  **Customer-to-Customer Risk:** Analysis of the `trans_2type` feature revealed that **100%** of the fraudulent transactions in this dataset occurred between Customer accounts (C2C), rather than Customer-to-Merchant (CM).

## 📊 Dataset
* **Source:** Synthetic Financial Datasets For Fraud Detection (PaySim)
* **Size:** ~6.3 million transactions
* **Features Used:** `type`, `amount`, `oldbalance_org`, and the engineered `trans_2type`.
* **Target:** `isFraud` (Binary: 0 = Legitimate, 1 = Fraud)

## 🛠 Methodology

### 1. Feature Engineering
* **`trans_2type`**: A new categorical feature was created to classify transactions based on the flow of funds (e.g., Customer-to-Customer vs. Customer-to-Merchant).
* **Selection**: Irrelevant features like `nameOrig` and `nameDest` (IDs) were dropped to prevent overfitting to specific accounts.

### 2. Handling Imbalance
We experimented with three strategies to handle the 0.13% fraud rate:
* **Class Weights:** Using XGBoost's `scale_pos_weight` parameter to heavily penalize missing a fraud case.
* **SMOTE (Oversampling):** Generating synthetic instances of fraud.
* **Random Undersampling:** Reducing the number of non-fraud cases.

### 3. Modeling
We implemented pipelines using `scikit-learn` and `imbalanced-learn` for:
* **XGBoost Classifier** (Gradient Boosting)
* **Logistic Regression** (Baseline)
* **Random Forest** (Bagging)

## 📈 Model Performance

We used the **F1-Score** as our primary metric to balance Precision and Recall.

| Model Strategy | Precision | Recall | F1-Score | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Weighted)** | **0.42** | **0.99** | **0.59** | **Best Performer.** Caught 99% of fraud with reasonable precision. |
| XGBoost + SMOTE | 0.28 | 1.00 | 0.43 | Perfect recall, but too many false alarms (low precision). |
| XGBoost + Undersampling | 0.08 | 1.00 | 0.15 | Poor precision; the model flagged too many safe transactions. |
| Logistic Regression | 0.81 | 0.14 | 0.24 | High precision but missed 86% of fraud cases (low recall). |

**Conclusion:** The **Weighted XGBoost** model proved superior, effectively learning the complex non-linear patterns of fraud (like account emptying) that linear models missed. Sampling techniques like SMOTE improved recall but degraded overall reliability by increasing false positives.

## 📂 Project Structure

```bash
Python-Fraud-Detection/
├── dataset/                 # Raw CSV data
├── source_code/             # Jupyter Notebooks
│   └── Fraud_detection.ipynb # Complete pipeline: EDA, Preprocessing, Modeling
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```
