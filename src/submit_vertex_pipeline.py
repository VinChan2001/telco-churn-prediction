import argparse
from datetime import datetime, timezone


DEFAULT_PROJECT = "telco-churn-vinay-raw"
DEFAULT_REGION = "us-east1"
DEFAULT_PIPELINE_ROOT = "gs://telco-churn-vinay-2026/vertex/pipeline-root"
DEFAULT_TEMPLATE_PATH = "pipelines/vertex_champion_challenger_pipeline.json"
DEFAULT_DATASET_GCS_URI = "gs://telco-churn-vinay-2026/training/Telco_customer_churn.xlsx"
DEFAULT_CHAMPION_MODEL_GCS_URI = (
    "gs://telco-churn-vinay-2026/models/champion/xgb_churn_pipeline.joblib"
)
DEFAULT_CHAMPION_METADATA_GCS_URI = (
    "gs://telco-churn-vinay-2026/models/champion/model_metadata.json"
)
DEFAULT_CANDIDATE_PREFIX = "gs://telco-churn-vinay-2026/models/candidates"


def submit_pipeline(
    project: str = DEFAULT_PROJECT,
    region: str = DEFAULT_REGION,
    pipeline_root: str = DEFAULT_PIPELINE_ROOT,
    template_path: str = DEFAULT_TEMPLATE_PATH,
    dataset_gcs_uri: str = DEFAULT_DATASET_GCS_URI,
    champion_model_gcs_uri: str = DEFAULT_CHAMPION_MODEL_GCS_URI,
    champion_metadata_gcs_uri: str = DEFAULT_CHAMPION_METADATA_GCS_URI,
    candidate_prefix: str = DEFAULT_CANDIDATE_PREFIX,
    n_iter: int = 12,
    min_recall: float = 0.75,
    min_precision: float = 0.50,
    run_id: str | None = None,
) -> tuple[str, str]:
    from google.cloud import aiplatform

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate_output_prefix = f"{candidate_prefix}/{run_id}"

    aiplatform.init(project=project, location=region)

    job = aiplatform.PipelineJob(
        display_name=f"telco-churn-champion-challenger-{run_id}",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values={
            "dataset_gcs_uri": dataset_gcs_uri,
            "champion_model_gcs_uri": champion_model_gcs_uri,
            "champion_metadata_gcs_uri": champion_metadata_gcs_uri,
            "candidate_output_prefix": candidate_output_prefix,
            "promoted_model_gcs_uri": champion_model_gcs_uri,
            "promoted_metadata_gcs_uri": champion_metadata_gcs_uri,
            "n_iter": n_iter,
            "min_recall": min_recall,
            "min_precision": min_precision,
        },
    )
    job.submit()

    return job.resource_name, candidate_output_prefix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the compiled Vertex AI champion/challenger pipeline."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--pipeline-root", default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--template-path", default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--dataset-gcs-uri", default=DEFAULT_DATASET_GCS_URI)
    parser.add_argument(
        "--champion-model-gcs-uri",
        default=DEFAULT_CHAMPION_MODEL_GCS_URI,
    )
    parser.add_argument(
        "--champion-metadata-gcs-uri",
        default=DEFAULT_CHAMPION_METADATA_GCS_URI,
    )
    parser.add_argument("--candidate-prefix", default=DEFAULT_CANDIDATE_PREFIX)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--min-precision", type=float, default=0.50)
    args = parser.parse_args()

    resource_name, candidate_output_prefix = submit_pipeline(
        project=args.project,
        region=args.region,
        pipeline_root=args.pipeline_root,
        template_path=args.template_path,
        dataset_gcs_uri=args.dataset_gcs_uri,
        champion_model_gcs_uri=args.champion_model_gcs_uri,
        champion_metadata_gcs_uri=args.champion_metadata_gcs_uri,
        candidate_prefix=args.candidate_prefix,
        n_iter=args.n_iter,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
    )

    print(f"Submitted Vertex AI pipeline: {resource_name}")
    print(f"Candidate output prefix: {candidate_output_prefix}")


if __name__ == "__main__":
    main()
