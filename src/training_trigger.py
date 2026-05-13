import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from src.submit_vertex_pipeline import (
    DEFAULT_CANDIDATE_PREFIX,
    DEFAULT_CHAMPION_METADATA_GCS_URI,
    DEFAULT_CHAMPION_MODEL_GCS_URI,
    DEFAULT_PIPELINE_ROOT,
    DEFAULT_PROJECT,
    DEFAULT_REGION,
    DEFAULT_TEMPLATE_PATH,
    submit_pipeline,
)
from src.synthetic_customer_generator import FEATURE_COLUMNS


TRAINING_DATA_PREFIX = os.getenv("TRAINING_DATA_PREFIX", "training/incoming/")
TRAINING_RUNS_TABLE_ID = os.getenv(
    "TRAINING_RUNS_TABLE_ID",
    "telco-churn-vinay-raw.telco_churn.training_pipeline_runs",
)
MIN_TRAINING_ROWS = int(os.getenv("MIN_TRAINING_ROWS", "100"))
SUPPORTED_TRAINING_SUFFIXES = (".csv", ".xlsx")
TARGET_COLUMN = "Churn Value"
REQUIRED_TRAINING_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


@dataclass
class TrainingValidation:
    row_count: int
    missing_columns: list[str]


@dataclass
class TrainingTriggerResult:
    status: str
    message: str
    gcs_uri: str
    run_id: str | None = None
    row_count: int | None = None
    pipeline_job_name: str | None = None
    candidate_output_prefix: str | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def build_training_run_id(gcs_uri: str, generation: str | None = None) -> str:
    source = f"{gcs_uri}#{generation or ''}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{digest[:8]}"


def is_in_training_prefix(file_name: str, prefix: str = TRAINING_DATA_PREFIX) -> bool:
    return file_name.startswith(prefix)


def is_supported_training_file(file_name: str) -> bool:
    return file_name.lower().endswith(SUPPORTED_TRAINING_SUFFIXES)


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected GCS URI, got: {gcs_uri}")

    bucket_name, blob_name = gcs_uri.replace("gs://", "", 1).split("/", 1)
    return bucket_name, blob_name


def validate_training_dataframe(
    dataframe: pd.DataFrame,
    minimum_row_count: int = MIN_TRAINING_ROWS,
) -> TrainingValidation:
    missing_columns = [
        column for column in REQUIRED_TRAINING_COLUMNS
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Training file is missing required columns: "
            + ", ".join(missing_columns)
        )

    row_count = len(dataframe)
    if row_count < minimum_row_count:
        raise ValueError(
            f"Training file has {row_count} rows; minimum is {minimum_row_count}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Training target column contains missing values")

    target_values = set(dataframe[TARGET_COLUMN].unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError("Training target column must contain only 0/1 values")

    return TrainingValidation(row_count=row_count, missing_columns=[])


def load_training_dataframe(gcs_uri: str) -> pd.DataFrame:
    from google.cloud import storage

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    suffix = Path(blob_name).suffix.lower()

    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as data_file:
        data_path = Path(data_file.name)

    try:
        blob.download_to_filename(data_path)
        if suffix == ".csv":
            return pd.read_csv(data_path)
        if suffix == ".xlsx":
            return pd.read_excel(data_path)
    finally:
        data_path.unlink(missing_ok=True)

    raise ValueError(f"Unsupported training file type: {suffix}")


def ensure_training_runs_table(client: bigquery.Client) -> None:
    try:
        client.get_table(TRAINING_RUNS_TABLE_ID)
        return
    except NotFound:
        pass

    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("file_uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("file_generation", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("message", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("row_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("pipeline_job_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("candidate_output_prefix", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    client.create_table(bigquery.Table(TRAINING_RUNS_TABLE_ID, schema=schema))


def has_training_file_already_handled(
    client: bigquery.Client,
    gcs_uri: str,
    generation: str | None,
) -> bool:
    query = f"""
    SELECT COUNT(*) AS file_count
    FROM `{TRAINING_RUNS_TABLE_ID}`
    WHERE file_uri = @gcs_uri
      AND COALESCE(file_generation, '') = @generation
      AND status IN ('submitted', 'rejected')
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("gcs_uri", "STRING", gcs_uri),
            bigquery.ScalarQueryParameter(
                "generation",
                "STRING",
                generation or "",
            ),
        ]
    )

    result = client.query(query, job_config=job_config).result()
    for row in result:
        return row.file_count > 0

    return False


def record_training_run(
    client: bigquery.Client,
    run_id: str,
    gcs_uri: str,
    generation: str | None,
    status: str,
    message: str,
    row_count: int | None = None,
    pipeline_job_name: str | None = None,
    candidate_output_prefix: str | None = None,
) -> None:
    errors = client.insert_rows_json(
        TRAINING_RUNS_TABLE_ID,
        [
            {
                "run_id": run_id,
                "file_uri": gcs_uri,
                "file_generation": generation,
                "status": status,
                "message": message,
                "row_count": row_count,
                "pipeline_job_name": pipeline_job_name,
                "candidate_output_prefix": candidate_output_prefix,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    if errors:
        raise RuntimeError(f"Failed to record training run metadata: {errors}")


def trigger_training_pipeline_for_gcs_file(
    bucket_name: str,
    file_name: str,
    generation: str | None = None,
) -> TrainingTriggerResult:
    gcs_uri = f"gs://{bucket_name}/{file_name}"

    if not is_in_training_prefix(file_name):
        return TrainingTriggerResult(
            status="skipped",
            message=f"File is outside {TRAINING_DATA_PREFIX}",
            gcs_uri=gcs_uri,
        )

    if not is_supported_training_file(file_name):
        return TrainingTriggerResult(
            status="skipped",
            message="File is not a supported training format",
            gcs_uri=gcs_uri,
        )

    client = bigquery.Client()
    ensure_training_runs_table(client)

    if has_training_file_already_handled(client, gcs_uri, generation):
        return TrainingTriggerResult(
            status="skipped",
            message="Training file generation was already handled",
            gcs_uri=gcs_uri,
        )

    run_id = build_training_run_id(gcs_uri, generation)

    try:
        dataframe = load_training_dataframe(gcs_uri)
        validation = validate_training_dataframe(dataframe)
    except Exception as exc:
        message = f"Training file validation failed: {exc}"
        record_training_run(
            client=client,
            run_id=run_id,
            gcs_uri=gcs_uri,
            generation=generation,
            status="rejected",
            message=message,
        )
        return TrainingTriggerResult(
            status="rejected",
            message=message,
            gcs_uri=gcs_uri,
            run_id=run_id,
        )

    try:
        pipeline_job_name, candidate_output_prefix = submit_pipeline(
            project=os.getenv("VERTEX_PROJECT", DEFAULT_PROJECT),
            region=os.getenv("VERTEX_REGION", DEFAULT_REGION),
            pipeline_root=os.getenv("VERTEX_PIPELINE_ROOT", DEFAULT_PIPELINE_ROOT),
            template_path=os.getenv("VERTEX_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH),
            dataset_gcs_uri=gcs_uri,
            champion_model_gcs_uri=os.getenv(
                "CHAMPION_MODEL_GCS_URI",
                DEFAULT_CHAMPION_MODEL_GCS_URI,
            ),
            champion_metadata_gcs_uri=os.getenv(
                "CHAMPION_METADATA_GCS_URI",
                DEFAULT_CHAMPION_METADATA_GCS_URI,
            ),
            candidate_prefix=os.getenv(
                "CANDIDATE_MODEL_PREFIX",
                f"{DEFAULT_CANDIDATE_PREFIX}/training-trigger",
            ),
            n_iter=int(os.getenv("VERTEX_N_ITER", "12")),
            min_recall=float(os.getenv("VERTEX_MIN_RECALL", "0.75")),
            min_precision=float(os.getenv("VERTEX_MIN_PRECISION", "0.50")),
            run_id=run_id,
        )
    except Exception as exc:
        message = f"Pipeline submission failed: {exc}"
        record_training_run(
            client=client,
            run_id=run_id,
            gcs_uri=gcs_uri,
            generation=generation,
            status="failed",
            message=message,
            row_count=validation.row_count,
        )
        raise

    message = "Submitted Vertex AI champion/challenger pipeline"
    record_training_run(
        client=client,
        run_id=run_id,
        gcs_uri=gcs_uri,
        generation=generation,
        status="submitted",
        message=message,
        row_count=validation.row_count,
        pipeline_job_name=pipeline_job_name,
        candidate_output_prefix=candidate_output_prefix,
    )

    return TrainingTriggerResult(
        status="submitted",
        message=message,
        gcs_uri=gcs_uri,
        run_id=run_id,
        row_count=validation.row_count,
        pipeline_job_name=pipeline_job_name,
        candidate_output_prefix=candidate_output_prefix,
    )
