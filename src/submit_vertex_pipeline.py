import argparse
from datetime import datetime, timezone

from google.cloud import aiplatform


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

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate_output_prefix = f"{args.candidate_prefix}/{run_id}"

    aiplatform.init(project=args.project, location=args.region)

    job = aiplatform.PipelineJob(
        display_name=f"telco-churn-champion-challenger-{run_id}",
        template_path=args.template_path,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "dataset_gcs_uri": args.dataset_gcs_uri,
            "champion_model_gcs_uri": args.champion_model_gcs_uri,
            "champion_metadata_gcs_uri": args.champion_metadata_gcs_uri,
            "candidate_output_prefix": candidate_output_prefix,
            "promoted_model_gcs_uri": args.champion_model_gcs_uri,
            "promoted_metadata_gcs_uri": args.champion_metadata_gcs_uri,
            "n_iter": args.n_iter,
            "min_recall": args.min_recall,
            "min_precision": args.min_precision,
        },
    )
    job.submit()

    print(f"Submitted Vertex AI pipeline: {job.resource_name}")
    print(f"Candidate output prefix: {candidate_output_prefix}")


if __name__ == "__main__":
    main()
