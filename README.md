# Fraud Detection for E‑commerce and Bank Transactions

## Overview
This project focuses on improving the detection of fraud cases for e‑commerce and bank transactions by analyzing transaction data and user behavior. The goal is to build a machine learning model that accurately flags fraudulent activity while minimizing false positives, improving security, and protecting customer trust.

## Business Objective
- Reduce financial losses from chargebacks and fraudulent purchases.
- Flag high‑risk transactions in real time.
- Improve customer trust by minimizing false positives.

## Work Summary

### Exploratory Data Analysis (EDA)
A comprehensive univariate, bivariate, and multivariate analysis was performed to understand data distributions and fraud patterns.

**Key findings:**
- The dataset is highly imbalanced (legitimate transactions far exceed fraudulent ones).
- `purchase_value` is highly right‑skewed.
- Fraud rates vary significantly by **source**, **browser**, **country**, **purchase value**, and **age**.
- Fraud peaks during specific **hours** and **days**, indicating possible automated or patterned attacks.

**Notable observations:**
- **Direct** traffic has the highest fraud rate despite lower volume.
- **Opera/Firefox** show higher fraud rates than high‑volume browsers.
- “Red‑flag” countries (e.g., Mexico, Sweden) have low volume but very high fraud rates.
- Purchase values around **$86–$92** show a spike in fraud.
- The **41–64** age group shows elevated fraud susceptibility.

Placeholders from the report:
- ![Placeholder: Class Balance Pie Chart]
- ![Placeholder: Purchase Value Distribution Plot]
- ![Placeholder: Fraud Rate by Browser Plot]
- ![Placeholder: Fraud Rate by Country Plot]

### Geolocation Analysis
IP addresses were mapped to countries using the `IpAddress_to_Country` dataset. IPs were converted from float to integer to enable accurate range matching and identify high‑risk regions.

### Temporal Analysis
Fraud rates were evaluated by hour and day of week. Fraud peaked during early‑morning/late‑night hours and specific days, suggesting scheduled or automated activity.

## Feature Engineering
New features were created to capture fraud signals:

- **`purchase_time_hour`** — hour of purchase, capturing time‑of‑day risk.
- **`purchase_time_day_of_week`** — day of week, capturing weekly trends.
- **`purchase_time_since_signup_seconds`** — time between signup and purchase; spikes reveal likely bot behavior.

## Next Steps (Suggested)
- Train and evaluate classification models (e.g., tree‑based models).
- Address class imbalance (e.g., class weighting, sampling).
- Use precision/recall and ROC‑AUC as primary metrics.
- Validate the model for low false‑positive rates.

## Repository
**Repo:** `MYGBM/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions`
