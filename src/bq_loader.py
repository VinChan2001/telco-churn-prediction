from google.cloud import bigquery


def load_gcs_csv_to_bigquery(
    gcs_uri: str,
    table_id: str,
) -> None:
    """
    Loads a CSV file from Google Cloud Storage into a BigQuery table.

    Example:
    gcs_uri = "gs://telco-churn-vinay-2026/incoming/synthetic_new_customers_20260506_183522.csv"
    table_id = "telco-churn-vinay-raw.telco_churn.synthetic_customers"
    """

    client = bigquery.Client()

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    load_job = client.load_table_from_uri(
        gcs_uri,
        table_id,
        job_config=job_config,
    )

    load_job.result()

    table = client.get_table(table_id)

    print(f"Loaded {gcs_uri} into {table_id}")
    print(f"Table now has {table.num_rows} rows")