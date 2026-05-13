# Telco Customer Churn Prediction

This project builds a production-style customer churn prediction workflow using the IBM/Kaggle Telco Customer Churn dataset.

The goal is not only to train a churn model, but to connect model predictions to business value through threshold tuning, risk segmentation, retention ROI analysis, batch scoring, dashboard outputs, and a cloud-based synthetic ingestion pipeline using Google Cloud.

---

## Project Goal

Customer churn prediction is usually not just a classification problem. In a real business setting, the model needs to answer questions like:

- Which customers are most likely to churn?
- Which churn-risk threshold should the business use?
- How many customers should be targeted for retention?
- What is the expected ROI of a retention campaign?
- How can new customer records be ingested and made available for scoring or analysis?

This project is designed as a portfolio-quality churn prediction system that combines:

- Machine learning
- Business decisioning
- ROI simulation
- Batch scoring
- Streamlit dashboard outputs
- Google Cloud ingestion components
- BigQuery loading workflow

---

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
- Streamlit dashboard
- Local synthetic customer generation
- Cloud Run synthetic customer generation service
- Google Cloud Storage ingestion landing zone
- Cloud Scheduler job for automatic customer batch generation
- BigQuery dataset and table for synthetic customer records
- Python BigQuery loader utility
- Cloud Run BigQuery loader service
- Eventarc trigger from GCS to the BigQuery loader service
- Duplicate protection using a BigQuery `loaded_files` metadata table
- BigQuery model-ready input table
- BigQuery scored customer table
- Scoring idempotency using a BigQuery `scored_files` metadata table
- Cloud Run scorer service

---

## Selected Model

The current selected model is:

| Item | Value |
|---|---:|
| Model | XGBoost |
| Threshold | 0.30 |
| ROC AUC | ~0.854 |
| Churn recall | ~0.77 |
| Churn precision | ~0.54 |
| Churn F1-score | ~0.64 |

The threshold was lowered from the default `0.50` to `0.30` because churn prediction is a recall-sensitive business problem. Missing an actual churner may be more expensive than incorrectly flagging a non-churner for a low-cost retention intervention.

---

## Business Impact Simulation

Using the selected XGBoost model and `0.30` threshold:

| Metric | Value |
|---|---:|
| Customers flagged for retention | 530 |
| Actual churners caught | 288 |
| Estimated annual revenue at risk captured | ~$258K |
| Retention offer cost per customer | $50 |
| Assumed retention success rate | 25% |
| Estimated revenue saved | ~$64.6K |
| Estimated net value | ~$38.1K |
| Estimated ROI | ~1.44 |

This connects the model output to a business decision: targeting high-risk customers with a retention intervention.

---

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
│   ├── __init__.py
│   ├── config.py
│   ├── train_model.py
│   ├── score_customers.py
│   ├── generate_synthetic_customers.py
│   ├── gcs_utils.py
│   ├── cloud_run_faker_app.py
│   ├── bq_loader.py
│   ├── bq_model_input.py
│   ├── bq_scorer.py
│   ├── cloud_run_bq_loader_app.py
│   ├── cloud_run_scorer_app.py
│   ├── score_bq_customers.py
│   ├── refresh_bq_model_input_table.py
│   └── load_latest_to_bigquery.py
├── Makefile
├── Dockerfile
├── Dockerfile.bq-loader
├── Dockerfile.scorer
├── Dockerfile.dashboard
├── cloudbuild-bq-loader.yaml
├── cloudbuild-scorer.yaml
├── cloudbuild-dashboard.yaml
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Local ML Pipeline

From the project root, run:

```bash
make pipeline
```

This runs the local pipeline for model training/scoring artifacts depending on the Makefile configuration.

---

## Streamlit Dashboard

The project includes a Streamlit dashboard for viewing churn-risk outputs and business-facing summaries.

Deployed dashboard:

```text
https://telco-churn-dashboard-service-aecec5bsxa-ue.a.run.app
```

Run locally with:

```bash
streamlit run app/streamlit_app.py
```

By default, the dashboard reads from BigQuery:

```text
telco-churn-vinay-raw.telco_churn.scored_customers
```

If BigQuery access fails, it falls back to local processed outputs such as:

```text
data/processed/scored_churn_customers.csv
data/processed/risk_segment_summary.csv
data/processed/model_metrics.csv
data/processed/shap_feature_importance.csv
```

---

## Cloud Ingestion Pipeline

In addition to the local ML workflow, this project includes a cloud-based ingestion pipeline that simulates new customer records arriving automatically.

The cloud pipeline uses:

- Cloud Scheduler
- Cloud Run
- Google Cloud Storage
- Eventarc
- BigQuery

---

## Cloud Architecture

```text
Cloud Scheduler
        ↓ every 6 hours
Cloud Run Faker service
        ↓
Generates synthetic telco customer records
        ↓
Uploads timestamped CSV files to Google Cloud Storage
        ↓
Eventarc detects new GCS object creation
        ↓
Cloud Run BigQuery loader service
        ↓
Loads new CSV rows into BigQuery
        ↓
Refreshes model-ready BigQuery input table
        ↓
Scores unscored files with saved XGBoost pipeline
        ↓
Writes scored customers and scored file metadata to BigQuery
```

---

## Google Cloud Resources

| Resource | Name |
|---|---|
| GCP Project | `telco-churn-vinay-raw` |
| GCS Bucket | `telco-churn-vinay-2026` |
| GCS Landing Folder | `incoming/` |
| Cloud Run Service 1 | `telco-faker-service` |
| Cloud Run Service 2 | `telco-bq-loader-service` |
| Cloud Run Service 3 | `telco-bq-scorer-service` |
| Cloud Run Service 4 | `telco-churn-dashboard-service` |
| Cloud Scheduler Job | `telco-faker-every-6-hours` |
| Eventarc Trigger | `telco-gcs-to-bq-loader` |
| BigQuery Dataset | `telco_churn` |
| BigQuery Main Table | `synthetic_customers` |
| BigQuery Model Input Table | `synthetic_customers_model_input` |
| BigQuery Scored Table | `scored_customers` |
| BigQuery Loaded Metadata Table | `loaded_files` |
| BigQuery Scored Metadata Table | `scored_files` |

---

## Cloud Run Faker Service

The Faker service generates synthetic telco customer records and uploads them to GCS as timestamped CSV files.

Service:

```text
telco-faker-service
```

Main app file:

```text
src/cloud_run_faker_app.py
```

Dockerfile:

```text
Dockerfile
```

Example endpoint:

```text
https://telco-faker-service-516325234883.us-east1.run.app/generate?n=10
```

Example response:

```json
{
  "gcs_path": "gs://telco-churn-vinay-2026/incoming/synthetic_new_customers_20260507_222212.csv",
  "rows_generated": 10,
  "status": "success"
}
```

This service is triggered automatically every 6 hours by Cloud Scheduler.

---

## Cloud Scheduler

Cloud Scheduler calls the Faker service every 6 hours.

Job:

```text
telco-faker-every-6-hours
```

Schedule:

```text
0 */6 * * *
```

Target:

```text
https://telco-faker-service-516325234883.us-east1.run.app/generate?n=10
```

Meaning:

```text
Every 6 hours:
generate 10 new synthetic telco customer records
and upload them to GCS
```

---

## Google Cloud Storage Landing Zone

Generated CSV files land in:

```text
gs://telco-churn-vinay-2026/incoming/
```

Example files:

```text
synthetic_new_customers_20260506_183522.csv
synthetic_new_customers_20260507_222212.csv
```

GCS acts as the raw ingestion landing zone. It keeps the history of generated customer batches.

---

## BigQuery Dataset and Tables

Dataset:

```text
telco_churn
```

Main table:

```text
telco-churn-vinay-raw.telco_churn.synthetic_customers
```

Model input table:

```text
telco-churn-vinay-raw.telco_churn.synthetic_customers_model_input
```

Scored customers table:

```text
telco-churn-vinay-raw.telco_churn.scored_customers
```

Loaded-file metadata table:

```text
telco-churn-vinay-raw.telco_churn.loaded_files
```

Scored-file metadata table:

```text
telco-churn-vinay-raw.telco_churn.scored_files
```

The `synthetic_customers` table stores customer records loaded from GCS. This is the raw ingestion table and includes `source_file`, `ingested_at`, and `load_id` metadata for traceability.

The `synthetic_customers_model_input` table stores the raw feature columns expected by the saved scikit-learn/XGBoost pipeline plus ingestion metadata. It converts BigQuery boolean values back into the model's `Yes`/`No` string categories and normalizes service-dependent values such as `No phone service` and `No internet service`.

The `scored_customers` table stores churn probabilities, churn predictions, risk segments, estimated annual revenue, `scored_at`, and source metadata.

The `loaded_files` table tracks which files have already been processed so the loader does not append duplicate rows. It stores:

| Column | Purpose |
|---|---|
| `file_uri` | Full GCS URI for the loaded CSV file |
| `loaded_at` | Timestamp when the file was recorded as loaded |
| `row_count` | Number of rows appended from that file |

The `scored_files` table tracks which source files have already been scored so the scorer does not write duplicate scored rows.

---

## BigQuery Loader Utility

Core loader file:

```text
src/bq_loader.py
```

This file contains the reusable loading logic:

```text
GCS CSV file
        ↓
check loaded_files metadata table
        ↓
if file was already loaded, skip
        ↓
if new file, append rows to BigQuery
        ↓
record file_uri in loaded_files
        ↓
refresh synthetic_customers_model_input
        ↓
score files not present in scored_files
        ↓
write scored_customers and scored_files
```

This protects against accidental duplicate loading, keeps the model input table current, and scores each source file once.

---

## Manual BigQuery Load Script

Manual one-off loader:

```text
src/load_latest_to_bigquery.py
```

Run:

```bash
python -m src.load_latest_to_bigquery
```

This script finds the latest CSV file in the GCS `incoming/` folder and loads it into BigQuery.

This is useful for manual backfills or testing.

Refresh only the model input table:

```bash
python -m src.refresh_bq_model_input_table
```

Score all unscored BigQuery model-input rows:

```bash
python -m src.score_bq_customers
```

---

## Cloud Run BigQuery Loader Service

The BigQuery loader service receives GCS object events and loads the new CSV into BigQuery.

Service:

```text
telco-bq-loader-service
```

Main app file:

```text
src/cloud_run_bq_loader_app.py
```

Dockerfile:

```text
Dockerfile.bq-loader
```

Cloud Build config:

```text
cloudbuild-bq-loader.yaml
```

Health check:

```text
https://telco-bq-loader-service-516325234883.us-east1.run.app/
```

Expected response:

```json
{
  "service": "telco-bq-loader",
  "status": "ok"
}
```

---

## Cloud Run Scorer Service

The scorer service runs the saved model artifact against unscored rows in BigQuery and writes scored outputs back to BigQuery.

Service:

```text
telco-bq-scorer-service
```

Main app file:

```text
src/cloud_run_scorer_app.py
```

Dockerfile:

```text
Dockerfile.scorer
```

Cloud Build config:

```text
cloudbuild-scorer.yaml
```

Health check:

```text
https://telco-bq-scorer-service-516325234883.us-east1.run.app/
```

Run scoring:

```text
https://telco-bq-scorer-service-516325234883.us-east1.run.app/score
```

---

## Cloud Run Dashboard Service

The dashboard service hosts the Streamlit dashboard from `app/streamlit_app.py` and reads scored churn predictions from BigQuery.

Service:

```text
telco-churn-dashboard-service
```

Dashboard URL:

```text
https://telco-churn-dashboard-service-aecec5bsxa-ue.a.run.app
```

Dockerfile:

```text
Dockerfile.dashboard
```

Cloud Build config:

```text
cloudbuild-dashboard.yaml
```

Health check:

```text
https://telco-churn-dashboard-service-aecec5bsxa-ue.a.run.app/_stcore/health
```

---

## Eventarc Trigger

Eventarc connects GCS object creation events to the BigQuery loader service.

Trigger:

```text
telco-gcs-to-bq-loader
```

Event type:

```text
google.cloud.storage.object.v1.finalized
```

Bucket filter:

```text
telco-churn-vinay-2026
```

Meaning:

```text
Whenever a new file is created/finalized in the GCS bucket,
Eventarc sends that event to the BigQuery loader Cloud Run service.
```

The loader then extracts the bucket name and file name from the event payload and loads that exact file into BigQuery.

---

## Validation Completed

The automated pipeline was validated end-to-end.

Manual test:

```text
1. Triggered Faker Cloud Run endpoint manually
2. Confirmed a new CSV appeared in GCS
3. Eventarc triggered the BigQuery loader service automatically
4. BigQuery row count increased without manually running bq load
```

Validation result:

```text
Before: 30 rows
After new Faker run: 40 rows
```

This confirmed:

```text
Cloud Run Faker → GCS → Eventarc → Cloud Run BigQuery Loader → BigQuery
```

worked successfully.

Duplicate protection was also tested locally:

```text
First run with same file:
Loaded 10 rows and recorded file_uri

Second run with same file:
Skipped because file was already present in loaded_files
```

---

## Example BigQuery Queries

Count rows:

```sql
SELECT COUNT(*) AS row_count
FROM `telco-churn-vinay-raw.telco_churn.synthetic_customers`;
```

View sample records:

```sql
SELECT
  `Gender`,
  `Contract`,
  `Internet Service`,
  `Payment Method`,
  `Tenure Months`,
  `Monthly Charges`,
  `Total Charges`
FROM `telco-churn-vinay-raw.telco_churn.synthetic_customers`
LIMIT 10;
```

View loaded files:

```sql
SELECT *
FROM `telco-churn-vinay-raw.telco_churn.loaded_files`
ORDER BY loaded_at DESC;
```

View model-ready inference input:

```sql
SELECT *
FROM `telco-churn-vinay-raw.telco_churn.synthetic_customers_model_input`
LIMIT 10;
```

View scored customers:

```sql
SELECT
  churn_probability,
  predicted_churn,
  risk_segment,
  estimated_annual_revenue,
  source_file,
  scored_at
FROM `telco-churn-vinay-raw.telco_churn.scored_customers`
ORDER BY churn_probability DESC
LIMIT 10;
```

View scored files:

```sql
SELECT *
FROM `telco-churn-vinay-raw.telco_churn.scored_files`
ORDER BY scored_at DESC;
```

---

## Deployment Commands

Deploy Faker service from source:

```bash
gcloud run deploy telco-faker-service \
  --source . \
  --region us-east1 \
  --allow-unauthenticated
```

Build BigQuery loader image:

```bash
gcloud builds submit --config cloudbuild-bq-loader.yaml .
```

Deploy BigQuery loader service:

```bash
gcloud run deploy telco-bq-loader-service \
  --image us-east1-docker.pkg.dev/telco-churn-vinay-raw/cloud-run-source-deploy/telco-bq-loader-service:latest \
  --region us-east1 \
  --allow-unauthenticated
```

Build scorer image:

```bash
gcloud builds submit --config cloudbuild-scorer.yaml .
```

Deploy scorer service:

```bash
gcloud run deploy telco-bq-scorer-service \
  --image us-east1-docker.pkg.dev/telco-churn-vinay-raw/cloud-run-source-deploy/telco-bq-scorer-service:latest \
  --region us-east1 \
  --allow-unauthenticated
```

Build dashboard image:

```bash
gcloud builds submit --config cloudbuild-dashboard.yaml .
```

Deploy dashboard service:

```bash
gcloud run deploy telco-churn-dashboard-service \
  --image us-east1-docker.pkg.dev/telco-churn-vinay-raw/cloud-run-source-deploy/telco-churn-dashboard-service:latest \
  --region us-east1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars DASHBOARD_DATA_SOURCE=bigquery
```

---

## Current End-to-End System

The project now has two connected workflows.

### ML workflow

```text
Raw Telco dataset
        ↓
Preprocessing
        ↓
Model training and threshold tuning
        ↓
Business ROI simulation
        ↓
Batch scoring
        ↓
Dashboard outputs
```

### Cloud ingestion workflow

```text
Cloud Scheduler
        ↓
Cloud Run Faker service
        ↓
GCS incoming CSV
        ↓
Eventarc
        ↓
Cloud Run BigQuery loader
        ↓
BigQuery synthetic_customers table
        ↓
BigQuery synthetic_customers_model_input table
        ↓
Cloud Run scorer / BigQuery scoring utility
        ↓
BigQuery scored_customers table
        ↓
Streamlit dashboard
```

Together, these make the project more realistic than a notebook-only churn model.

---

## Next Improvements

Planned next steps:

- Add a separate snake_case analytics view if downstream tools need standard SQL identifiers
- Add tests for key utility functions
- Add CI/CD checks for formatting and basic pipeline validation

### Latest Validation

After adding row-level ingestion metadata, model-input refresh, and BigQuery scoring, the deployed Eventarc-triggered loader was tested successfully.

Validation result:

```text
synthetic_customers rows: 263
synthetic_customers_model_input rows: 263
scored_customers rows: 263
scored_files rows: 1
invalid model-input service combinations: 0
```

The latest live Cloud Run test generated a new two-row GCS file and confirmed that ingestion, model-input refresh, scoring, and scoring idempotency stayed aligned:

```text
synthetic_customers rows: 267
synthetic_customers_model_input rows: 267
scored_customers rows: 267
scored_files rows: 3
latest generated file: gs://telco-churn-vinay-2026/incoming/synthetic_new_customers_20260512_043838.csv
scorer idempotency check: 0 rows scored on rerun
```

Latest live dashboard deployment check:

```text
dashboard service: telco-churn-dashboard-service
dashboard revision: telco-churn-dashboard-service-00001-fft
dashboard URL: https://telco-churn-dashboard-service-aecec5bsxa-ue.a.run.app
health check: ok
synthetic_customers rows: 347
synthetic_customers_model_input rows: 347
scored_customers rows: 347
loaded_files rows: 31
scored_files rows: 11
invalid model-input service combinations: 0
null Total Charges in model-input table: 0
```
