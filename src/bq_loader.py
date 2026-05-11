from google.cloud import bigquery

from src.bq_model_input import MODEL_INPUT_TABLE_ID, refresh_model_input_table


LOADED_FILES_TABLE_ID = "telco-churn-vinay-raw.telco_churn.loaded_files"


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
        print(f"Refreshed model input table: {MODEL_INPUT_TABLE_ID}")
        return

    table_before = client.get_table(table_id)
    rows_before = table_before.num_rows

    # Use the existing BigQuery table schema.
    # This prevents autodetect from guessing inconsistent types across files.
    existing_schema = table_before.schema

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=existing_schema,
        autodetect=False,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    load_job = client.load_table_from_uri(
        gcs_uri,
        table_id,
        job_config=job_config,
    )

    load_job.result()

    table_after = client.get_table(table_id)
    rows_after = table_after.num_rows

    loaded_row_count = rows_after - rows_before

    record_loaded_file(
        client=client,
        gcs_uri=gcs_uri,
        row_count=loaded_row_count,
    )

    refresh_model_input_table(client)

    print(f"Loaded {gcs_uri} into {table_id}")
    print(f"Rows loaded from this file: {loaded_row_count}")
    print(f"Table now has {rows_after} rows")
    print(f"Refreshed model input table: {MODEL_INPUT_TABLE_ID}")
