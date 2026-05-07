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
- Cloud Run synthetic customer generation
- Google Cloud Storage ingestion landing zone
- BigQuery CSV loading workflow

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
│       ├── risk_segment_summary.csv
│       ├── model_metrics.csv
│       ├── shap_feature_importance.csv
│       ├── sample_inference_customers.csv
│       └── synthetic_new_customers.csv
├── models/
│   └── xgb_churn_pipeline.joblib
├── notebooks/
│   └── 01_churn_modeling_and_roi.ipynb
├── src/
│   ├── config.py
│   ├── train_model.py
│   ├── score_customers.py
│   ├── generate_synthetic_customers.py
│   ├── cloud_run_faker_app.py
│   ├── bq_loader.py
│   ├── cloud_run_bq_loader_app.py
│   └── load_latest_to_bigquery.py
├── Makefile
├── Dockerfile
├── Dockerfile.bq-loader
├── cloudbuild-bq-loader.yaml
├── README.md
├── requirements.txt
└── .gitignore
```

## Run Full Local Pipeline

From the project root:

```bash
make pipeline
```

## Cloud Synthetic Customer Generation

This project includes an automated Cloud Run workflow that simulates new telco customer records arriving on a schedule.

### Architecture

```text
Cloud Scheduler
        ↓ every 6 hours
Cloud Run Faker service
        ↓
Generates synthetic customer records
        ↓
Uploads timestamped CSV files to Google Cloud Storage
        ↓
gs://telco-churn-vinay-2026/incoming/
```

## BigQuery Batch Loader

This project also includes a BigQuery loader that appends generated CSV files from Cloud Storage into:

```text
telco-churn-vinay-raw.telco_churn.synthetic_customers
```

### Loader Flow

```text
CSV file lands in Cloud Storage
        ↓
Cloud Run BigQuery loader receives the object event
        ↓
Loads the CSV with schema autodetection
        ↓
Rows are appended to the BigQuery synthetic_customers table
```

For a manual backfill or one-off load, run:

```bash
python -m src.load_latest_to_bigquery
```
