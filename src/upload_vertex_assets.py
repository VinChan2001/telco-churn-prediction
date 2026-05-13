import argparse
import json
from pathlib import Path

from google.cloud import storage

from src.config import CHURN_THRESHOLD, MODEL_PATH, RAW_DATA_PATH


DEFAULT_BUCKET = "telco-churn-vinay-2026"
DEFAULT_DATASET_BLOB = "training/Telco_customer_churn.xlsx"
DEFAULT_CHAMPION_MODEL_BLOB = "models/champion/xgb_churn_pipeline.joblib"
DEFAULT_CHAMPION_METADATA_BLOB = "models/champion/model_metadata.json"


def upload_file(bucket: storage.Bucket, source_path: Path, blob_name: str) -> str:
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_path)
    return f"gs://{bucket.name}/{blob_name}"


def upload_text(bucket: storage.Bucket, text: str, blob_name: str) -> str:
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type="application/json")
    return f"gs://{bucket.name}/{blob_name}"


def gcs_uri(bucket: storage.Bucket, blob_name: str) -> str:
    return f"gs://{bucket.name}/{blob_name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local training/champion assets used by Vertex AI."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--dataset-blob", default=DEFAULT_DATASET_BLOB)
    parser.add_argument(
        "--champion-model-blob",
        default=DEFAULT_CHAMPION_MODEL_BLOB,
    )
    parser.add_argument(
        "--champion-metadata-blob",
        default=DEFAULT_CHAMPION_METADATA_BLOB,
    )
    parser.add_argument(
        "--overwrite-champion",
        action="store_true",
        help="Overwrite the existing champion model and metadata with local files.",
    )
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    dataset_uri = upload_file(bucket, RAW_DATA_PATH, args.dataset_blob)
    champion_model_uri = gcs_uri(bucket, args.champion_model_blob)
    champion_model_blob = bucket.blob(args.champion_model_blob)
    if args.overwrite_champion or not champion_model_blob.exists():
        champion_model_uri = upload_file(
            bucket,
            MODEL_PATH,
            args.champion_model_blob,
        )

    champion_metadata = {
        "model_type": "xgboost_sklearn_pipeline",
        "threshold": CHURN_THRESHOLD,
        "source": "local_champion_model",
    }
    champion_metadata_uri = gcs_uri(bucket, args.champion_metadata_blob)
    champion_metadata_blob = bucket.blob(args.champion_metadata_blob)
    if args.overwrite_champion or not champion_metadata_blob.exists():
        champion_metadata_uri = upload_text(
            bucket,
            json.dumps(champion_metadata, indent=2, sort_keys=True),
            args.champion_metadata_blob,
        )

    print(f"Dataset: {dataset_uri}")
    print(f"Champion model: {champion_model_uri}")
    print(f"Champion metadata: {champion_metadata_uri}")


if __name__ == "__main__":
    main()
