from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SCORED_CUSTOMERS_PATH = PROCESSED_DATA_DIR / "scored_churn_customers.csv"
RISK_SEGMENT_SUMMARY_PATH = PROCESSED_DATA_DIR / "risk_segment_summary.csv"
METRICS_PATH = PROCESSED_DATA_DIR / "model_metrics.csv"
MODEL_RESULTS_PATH = PROCESSED_DATA_DIR / "model_results.csv"
SHAP_IMPORTANCE_PATH = PROCESSED_DATA_DIR / "shap_feature_importance.csv"
SYNTHETIC_CUSTOMERS_PATH = PROCESSED_DATA_DIR / "synthetic_new_customers.csv"

# Model path
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_churn_pipeline.joblib"

# Churn decision threshold
CHURN_THRESHOLD = 0.30

# Business assumptions
RETENTION_OFFER_COST = 50
RETENTION_SUCCESS_RATE = 0.25