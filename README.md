# 🔍 Customer Churn Risk Prediction

**Churn Prediction & Retention Intelligence using Logistic Regression and Random Forest**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project predicts which customers are at risk of churning using two complementary models — Logistic Regression for interpretability and Random Forest for accuracy. It assigns each customer a churn probability and risk tier (HIGH / MEDIUM / LOW), and generates targeted retention recommendations for the customer success team.

Reducing churn by even 5% can dramatically improve revenue. This model gives teams an early-warning system to intervene before customers leave.

---

## 🧠 How It Works

```
Customer Data (Tenure, Charges, Satisfaction, Contract, Behaviour)
      │
      ▼
Feature Engineering ──► Engagement Score, Value Score, Encoded categoricals
      │
      ▼
Logistic Regression ──► Interpretable baseline (AUC, accuracy)
      │
      ▼
Random Forest Classifier ──► Higher accuracy + feature importances
      │
      ▼
Churn Probability Scoring ──► 0.0–1.0 per customer
      │
      ▼
Risk Tier Assignment ──► 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
      │
      ▼
Retention Recommendations + CSV Export
```

---

## 📁 Project Structure

```
customer-churn-risk/
├── data/
│   └── customers.csv               # Input data (auto-generated if missing)
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   └── churn_risk.py               # Full prediction pipeline
├── outputs/
│   └── churn_predictions.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Features Used

| Feature | Description |
|---------|-------------|
| `Tenure_Months` | How long the customer has been active |
| `Monthly_Charges` | Current monthly billing amount |
| `Total_Charges` | Cumulative revenue from this customer |
| `Num_Products` | Number of products/services subscribed |
| `Support_Tickets` | Open or resolved support tickets |
| `Last_Login_Days` | Days since last platform login |
| `Satisfaction_Score` | Customer satisfaction rating (1–5) |
| `Contract_Enc` | Encoded contract type (Month-to-Month, Annual, etc.) |
| `Payment_Enc` | Encoded payment method |

---

## 🎯 Risk Tiers

| Tier | Churn Probability | Recommended Action |
|------|------------------|--------------------|
| 🔴 **HIGH** | ≥ 70% | Immediate personal outreach |
| 🟡 **MEDIUM** | 40–69% | Targeted retention offer |
| 🟢 **LOW** | < 40% | Monitor, standard comms |

---

## 📊 Model Comparison

| Model | Strength |
|-------|----------|
| Logistic Regression | Interpretable, fast, good baseline |
| Random Forest | Higher accuracy, feature importance ranking |

Both models are evaluated on accuracy and ROC-AUC score.

---

## 🚀 Getting Started

```bash
git clone https://github.com/Fariba6384/customer-churn-risk.git
cd customer-churn-risk
pip install -r requirements.txt
python src/churn_risk.py
```

If no `data/customers.csv` is found, a 600-customer synthetic dataset is auto-generated with realistic churn dynamics and saved for reuse.

---

## 👤 Author

**Fariba Seyedjafarrangraz**  
Business Transformation Analyst · Dalhousie University  
CBAP Certified · PhD · Halifax, Nova Scotia, Canada  
📧 rangrazfariba6384@gmail.com · 🐙 [github.com/Fariba6384](https://github.com/Fariba6384)
