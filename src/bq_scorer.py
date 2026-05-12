from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from google.cloud import bigquery

from src.bq_model_input import MODEL_INPUT_TABLE_ID, refresh_model_input_table
from src.config import CHURN_THRESHOLD, MODEL_PATH
from src.synthetic_customer_generator import FEATURE_COLUMNS


SCORED_CUSTOMERS_TABLE_ID = "telco-churn-vinay-raw.telco_churn.scored_customers"
SCORED_FILES_TABLE_ID = "telco-churn-vinay-raw.telco_churn.scored_files"

METADATA_COLUMNS = ["source_file", "ingested_at", "load_id"]
SCORING_COLUMNS = [
    "churn_probability",
    "predicted_churn",
    "risk_segment",
    "estimated_annual_revenue",
    "scored_at",
]


@dataclass
class ScoreResult:
    rows_scored: int
    files_scored: int
    scored_customers_table: str
    scored_files_table: str


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def _feature_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("Gender", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Senior Citizen", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Partner", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Dependents", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Tenure Months", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("Phone Service", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Multiple Lines", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Internet Service", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Online Security", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Online Backup", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Device Protection", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Tech Support", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Streaming TV", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Streaming Movies", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Contract", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Paperless Billing", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Payment Method", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Monthly Charges", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("Total Charges", "FLOAT", mode="NULLABLE"),
    ]


def ensure_scoring_tables(
    client: bigquery.Client | None = None,
) -> None:
    client = client or bigquery.Client()

    scored_customers_schema = [
        *_feature_schema(),
        bigquery.SchemaField("source_file", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("load_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("churn_probability", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("predicted_churn", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("risk_segment", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("estimated_annual_revenue", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("scored_at", "TIMESTAMP", mode="NULLABLE"),
    ]
    scored_files_schema = [
        bigquery.SchemaField("file_uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("scored_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("row_count", "INTEGER", mode="NULLABLE"),
    ]

    client.create_table(
        bigquery.Table(
            SCORED_CUSTOMERS_TABLE_ID,
            schema=scored_customers_schema,
        ),
        exists_ok=True,
    )
    client.create_table(
        bigquery.Table(
            SCORED_FILES_TABLE_ID,
            schema=scored_files_schema,
        ),
        exists_ok=True,
    )


def _query_unscored_rows(
    client: bigquery.Client,
    model_input_table_id: str = MODEL_INPUT_TABLE_ID,
) -> pd.DataFrame:
    selected_columns = [*FEATURE_COLUMNS, *METADATA_COLUMNS]
    selected_columns_sql = ", ".join(f"m.`{column}`" for column in selected_columns)

    query = f"""
    SELECT {selected_columns_sql}
    FROM `{model_input_table_id}` AS m
    WHERE m.source_file IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM `{SCORED_FILES_TABLE_ID}` AS s
          WHERE s.file_uri = m.source_file
      )
    ORDER BY m.ingested_at, m.source_file
    """

    rows = client.query(query).result()
    records = [dict(row) for row in rows]

    if not records:
        return pd.DataFrame(columns=selected_columns)

    return pd.DataFrame(records)[selected_columns]


def _risk_segment(probability: float) -> str:
    if probability <= 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


def _serialize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _insert_json_rows(
    client: bigquery.Client,
    table_id: str,
    rows: list[dict],
    chunk_size: int = 500,
) -> None:
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        errors = client.insert_rows_json(table_id, chunk)
        if errors:
            raise RuntimeError(f"Failed to insert rows into {table_id}: {errors}")


def _delete_scored_rows_for_files(
    client: bigquery.Client,
    file_uris: list[str],
) -> None:
    if not file_uris:
        return

    query = f"""
    DELETE FROM `{SCORED_CUSTOMERS_TABLE_ID}`
    WHERE source_file IN UNNEST(@file_uris)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("file_uris", "STRING", file_uris),
        ]
    )
    client.query(query, job_config=job_config).result()


def score_unscored_customers(
    client: bigquery.Client | None = None,
    refresh_input: bool = False,
) -> ScoreResult:
    client = client or bigquery.Client()

    if refresh_input:
        refresh_model_input_table(client)

    ensure_scoring_tables(client)

    unscored = _query_unscored_rows(client)
    if unscored.empty:
        return ScoreResult(
            rows_scored=0,
            files_scored=0,
            scored_customers_table=SCORED_CUSTOMERS_TABLE_ID,
            scored_files_table=SCORED_FILES_TABLE_ID,
        )

    model = load_model()
    features = unscored[FEATURE_COLUMNS].copy()
    probabilities = model.predict_proba(features)[:, 1]

    scored = unscored.copy()
    scored["churn_probability"] = probabilities
    scored["predicted_churn"] = (
        scored["churn_probability"] >= CHURN_THRESHOLD
    ).astype(int)
    scored["risk_segment"] = scored["churn_probability"].apply(_risk_segment)
    scored["estimated_annual_revenue"] = scored["Monthly Charges"] * 12
    scored["scored_at"] = pd.Timestamp.utcnow()

    source_files = sorted(scored["source_file"].dropna().unique().tolist())
    _delete_scored_rows_for_files(client, source_files)

    scored_rows = [
        {
            column: _serialize_value(row[column])
            for column in [*FEATURE_COLUMNS, *METADATA_COLUMNS, *SCORING_COLUMNS]
        }
        for _, row in scored.iterrows()
    ]
    _insert_json_rows(client, SCORED_CUSTOMERS_TABLE_ID, scored_rows)

    scored_file_rows = []
    for source_file, batch in scored.groupby("source_file", dropna=True):
        scored_file_rows.append(
            {
                "file_uri": source_file,
                "scored_at": pd.Timestamp.utcnow().isoformat(),
                "row_count": int(len(batch)),
            }
        )
    _insert_json_rows(client, SCORED_FILES_TABLE_ID, scored_file_rows)

    return ScoreResult(
        rows_scored=len(scored_rows),
        files_scored=len(scored_file_rows),
        scored_customers_table=SCORED_CUSTOMERS_TABLE_ID,
        scored_files_table=SCORED_FILES_TABLE_ID,
    )
