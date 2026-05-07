from google.cloud import storage

from src.bq_loader import load_gcs_csv_to_bigquery


BUCKET_NAME = "telco-churn-vinay-2026"
PREFIX = "incoming/"
TABLE_ID = "telco-churn-vinay-raw.telco_churn.synthetic_customers"


def get_latest_gcs_file(bucket_name: str, prefix: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))

    csv_blobs = [
        blob for blob in blobs
        if blob.name.endswith(".csv")
    ]

    if not csv_blobs:
        raise FileNotFoundError(f"No CSV files found in gs://{bucket_name}/{prefix}")

    latest_blob = max(csv_blobs, key=lambda blob: blob.time_created)

    return f"gs://{bucket_name}/{latest_blob.name}"


if __name__ == "__main__":
    latest_gcs_uri = get_latest_gcs_file(BUCKET_NAME, PREFIX)

    print(f"Latest file found: {latest_gcs_uri}")

    load_gcs_csv_to_bigquery(
        gcs_uri=latest_gcs_uri,
        table_id=TABLE_ID,
    )