import hashlib
import uuid

from google.cloud import bigquery

from src.bq_model_input import (
    METADATA_COLUMN_NAMES,
    MODEL_INPUT_TABLE_ID,
    ensure_raw_table_metadata,
    refresh_model_input_table,
)
from src.bq_scorer import score_unscored_customers


LOADED_FILES_TABLE_ID = "telco-churn-vinay-raw.telco_churn.loaded_files"


def build_load_id(gcs_uri: str) -> str:
    """Build a stable load id from the source GCS URI."""

    digest = hashlib.sha256(gcs_uri.encode("utf-8")).hexdigest()
    return f"gcs_{digest[:24]}"


def has_file_already_loaded(client: bigquery.Client, gcs_uri: str) -> bool:
    """
    Checks whether a GCS file has already been loaded into BigQuery.
    """

    query = f"""
    SELECT COUNT(*) AS file_count
    FROM `{LOADED_FILES_TABLE_ID}`
    WHERE file_uri = @gcs_uri
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("gcs_uri", "STRING", gcs_uri)
        ]
    )

    result = client.query(query, job_config=job_config).result()

    for row in result:
        return row.file_count > 0

    return False


def record_loaded_file(
    client: bigquery.Client,
    gcs_uri: str,
    row_count: int,
) -> None:
    """
    Records a successfully loaded GCS file in the loaded_files metadata table.
    """

    query = f"""
    INSERT INTO `{LOADED_FILES_TABLE_ID}` (file_uri, loaded_at, row_count)
    VALUES (@gcs_uri, CURRENT_TIMESTAMP(), @row_count)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("gcs_uri", "STRING", gcs_uri),
            bigquery.ScalarQueryParameter("row_count", "INT64", row_count),
        ]
    )

    client.query(query, job_config=job_config).result()


def load_gcs_csv_to_bigquery(
    gcs_uri: str,
    table_id: str,
) -> None:
    """
    Loads a CSV file from Google Cloud Storage into a BigQuery table.

    Production-safety behavior:
    1. Skips the file if it already exists in loaded_files.
    2. Reuses the existing BigQuery table schema instead of autodetecting every file.
    3. Appends rows only after schema validation succeeds.
    4. Records successfully loaded files in loaded_files.
    5. Refreshes the model input table used for future inference.
    """

    client = bigquery.Client()

    if has_file_already_loaded(client, gcs_uri):
        print(f"Skipped {gcs_uri}. File was already loaded.")
        refresh_model_input_table(client)
        score_result = score_unscored_customers(client)
        print(f"Refreshed model input table: {MODEL_INPUT_TABLE_ID}")
        print(f"Scored rows: {score_result.rows_scored}")
        return

    table_before = ensure_raw_table_metadata(client, table_id)
    rows_before = table_before.num_rows

    feature_schema = [
        field for field in table_before.schema
        if field.name not in METADATA_COLUMN_NAMES
    ]
    staging_table_id = f"{table_id}_staging_{uuid.uuid4().hex[:12]}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=feature_schema,
        autodetect=False,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        load_job = client.load_table_from_uri(
            gcs_uri,
            staging_table_id,
            job_config=job_config,
        )

        load_job.result()

        staging_table = client.get_table(staging_table_id)
        loaded_row_count = staging_table.num_rows

        load_id = build_load_id(gcs_uri)
        feature_columns = [field.name for field in feature_schema]
        insert_columns = [*feature_columns, *METADATA_COLUMN_NAMES]
        insert_columns_sql = ", ".join(
            f"`{column}`" for column in insert_columns
        )
        select_feature_columns_sql = ", ".join(
            f"`{column}`" for column in feature_columns
        )

        insert_query = f"""
        INSERT INTO `{table_id}` ({insert_columns_sql})
        SELECT
            {select_feature_columns_sql},
            @gcs_uri AS source_file,
            CURRENT_TIMESTAMP() AS ingested_at,
            @load_id AS load_id
        FROM `{staging_table_id}`
        """
        insert_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("gcs_uri", "STRING", gcs_uri),
                bigquery.ScalarQueryParameter("load_id", "STRING", load_id),
            ]
        )
        client.query(insert_query, job_config=insert_job_config).result()

    finally:
        client.delete_table(staging_table_id, not_found_ok=True)

    table_after = client.get_table(table_id)
    rows_after = table_after.num_rows

    record_loaded_file(
        client=client,
        gcs_uri=gcs_uri,
        row_count=loaded_row_count,
    )

    refresh_model_input_table(client)
    score_result = score_unscored_customers(client)

    print(f"Loaded {gcs_uri} into {table_id}")
    print(f"Rows loaded from this file: {loaded_row_count}")
    print(f"Table now has {rows_after} rows")
    print(f"Refreshed model input table: {MODEL_INPUT_TABLE_ID}")
    print(f"Scored rows: {score_result.rows_scored}")
