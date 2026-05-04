from google.cloud import storage
from pathlib import Path


def upload_file_to_gcs(
    local_file_path: str,
    bucket_name: str,
    destination_blob_name: str
) -> None:
    """
    Uploads a local file to a Google Cloud Storage bucket.

    Example:
    local_file_path = "data/processed/synthetic_new_customers.csv"
    bucket_name = "your-bucket-name"
    destination_blob_name = "incoming/synthetic_new_customers.csv"
    """

    local_path = Path(local_file_path)

    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_file_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(str(local_path))

    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")