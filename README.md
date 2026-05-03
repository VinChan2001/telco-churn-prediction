# Telco Customer Churn Prediction

This project builds a production-style customer churn prediction workflow using the IBM/Kaggle Telco Customer Churn dataset.

The goal is not only to train a model, but to connect model predictions to business value through threshold tuning, risk segmentation, and retention ROI analysis.

## Current Status

The project currently includes:

- Data preprocessing and cleaning
- Scikit-learn preprocessing pipeline
- Logistic Regression baseline and tuning
- Random Forest baseline
- Gradient Boosting baseline
- XGBoost baseline
- Threshold tuning for churn recall
- Precision-recall analysis
- Feature importance and permutation importance
- Business ROI simulation
- Saved XGBoost model artifact
- Batch scoring script
- Dashboard-ready scored customer outputs

## Selected Model

The current selected model is:

- Model: XGBoost
- Threshold: 0.30
- ROC AUC: ~0.854
- Churn recall: ~0.77
- Churn precision: ~0.54
- Churn F1-score: ~0.64

The threshold was lowered from the default 0.50 to 0.30 because churn prediction is a recall-sensitive business problem. Missing an actual churner may be more expensive than incorrectly flagging a non-churner for a low-cost retention intervention.

## Business Impact Simulation

Using the selected XGBoost model and 0.30 threshold:

- Customers flagged for retention: 530
- Actual churners caught: 288
- Estimated annual revenue at risk captured: ~$258K
- Retention offer cost per customer: $50
- Assumed retention success rate: 25%
- Estimated revenue saved: ~$64.6K
- Estimated net value: ~$38.1K
- Estimated ROI: ~1.44

## Project Structure

```text
telco-churn-prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── Telco_customer_churn.xlsx
│   └── processed/
│       ├── scored_churn_customers.csv
│       └── risk_segment_summary.csv
├── models/
│   └── xgb_churn_pipeline.joblib
├── notebooks/
│   └── 01_churn_modeling_and_roi.ipynb
├── src/
│   ├── config.py
│   └── score_customers.py
├── README.md
├── requirements.txt
└── .gitignore
