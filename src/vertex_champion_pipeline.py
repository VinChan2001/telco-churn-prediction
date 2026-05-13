from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "xgboost==3.0.4",
        "joblib==1.4.2",
        "openpyxl",
        "google-cloud-storage==3.4.1",
    ],
)
def tune_compare_promote_model(
    dataset_gcs_uri: str,
    champion_model_gcs_uri: str,
    champion_metadata_gcs_uri: str,
    candidate_output_prefix: str,
    promoted_model_gcs_uri: str,
    promoted_metadata_gcs_uri: str,
    n_iter: int,
    min_recall: float,
    min_precision: float,
    min_roc_auc_delta: float,
    random_state: int,
    retention_offer_cost: float,
    retention_success_rate: float,
    metrics: dsl.Output[dsl.Metrics],
) -> None:
    import json
    import math
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import storage
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    feature_columns = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
    ]
    columns_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Value",
        "Churn Score",
        "CLTV",
        "Churn Reason",
    ]

    def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Expected GCS URI, got: {gcs_uri}")
        bucket_name, blob_name = gcs_uri.replace("gs://", "", 1).split("/", 1)
        return bucket_name, blob_name

    def to_builtin(value):
        if isinstance(value, dict):
            return {key: to_builtin(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_builtin(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    def download_gcs_file(gcs_uri: str, suffix: str = "") -> Path:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        target = Path(tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name)
        blob.download_to_filename(target)
        return target

    def upload_file(source_path: Path, target_gcs_uri: str) -> None:
        bucket_name, blob_name = parse_gcs_uri(target_gcs_uri)
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(source_path)

    def upload_json(payload: dict, target_gcs_uri: str) -> None:
        bucket_name, blob_name = parse_gcs_uri(target_gcs_uri)
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(
            json.dumps(payload, indent=2, sort_keys=True),
            content_type="application/json",
        )

    def build_pipeline(params: dict | None = None) -> Pipeline:
        params = params or {}
        categorical_features = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()
        numeric_features = X_train.select_dtypes(
            exclude=["object"]
        ).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[("scaler", StandardScaler())]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            **params,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def evaluate_thresholds(model, threshold_grid: list[float]) -> dict:
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
        rows = []

        for threshold in threshold_grid:
            predictions = (probabilities >= threshold).astype(int)
            flagged = predictions == 1
            true_churn_captured = (predictions == 1) & (y_test.to_numpy() == 1)
            captured_revenue = (
                X_test.loc[true_churn_captured, "Monthly Charges"].sum() * 12
            )
            campaign_cost = flagged.sum() * retention_offer_cost
            estimated_revenue_saved = captured_revenue * retention_success_rate
            net_value = estimated_revenue_saved - campaign_cost

            rows.append(
                {
                    "threshold": threshold,
                    "roc_auc": roc_auc,
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    ),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                    "flagged_customers": int(flagged.sum()),
                    "captured_revenue": float(captured_revenue),
                    "estimated_revenue_saved": float(estimated_revenue_saved),
                    "campaign_cost": float(campaign_cost),
                    "net_value": float(net_value),
                }
            )

        valid_rows = [
            row for row in rows
            if row["recall"] >= min_recall and row["precision"] >= min_precision
        ]
        if not valid_rows:
            valid_rows = rows

        return max(
            valid_rows,
            key=lambda row: (
                row["net_value"],
                row["recall"],
                row["precision"],
                row["roc_auc"],
            ),
        )

    print("Downloading training dataset...", flush=True)
    dataset_path = download_gcs_file(dataset_gcs_uri, suffix=".xlsx")
    dataset = pd.read_excel(dataset_path)
    dataset["Total Charges"] = pd.to_numeric(
        dataset["Total Charges"],
        errors="coerce",
    )

    X = dataset.drop(columns=columns_to_drop, errors="ignore")
    X = X[feature_columns]
    y = dataset["Churn Value"]

    print("Creating train/test split...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    baseline_pipeline = build_pipeline()
    param_distributions = {
        "classifier__n_estimators": [150, 250, 350, 500],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__min_child_weight": [1, 3, 5, 8],
        "classifier__subsample": [0.70, 0.80, 0.90, 1.00],
        "classifier__colsample_bytree": [0.70, 0.80, 0.90, 1.00],
        "classifier__reg_alpha": [0.0, 0.01, 0.1, 0.5],
        "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    print(f"Running randomized search with {n_iter} trials...", flush=True)
    search = RandomizedSearchCV(
        estimator=baseline_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=random_state,
        n_jobs=1,
        verbose=2,
    )
    search.fit(X_train, y_train)
    candidate_model = search.best_estimator_

    print("Evaluating challenger model...", flush=True)
    threshold_grid = [round(value, 2) for value in np.arange(0.10, 0.71, 0.02)]
    candidate_metrics = evaluate_thresholds(candidate_model, threshold_grid)

    champion_exists = True
    champion_metrics = None
    try:
        print("Evaluating champion model...", flush=True)
        champion_path = download_gcs_file(champion_model_gcs_uri, suffix=".joblib")
        champion_model = joblib.load(champion_path)
        champion_metrics = evaluate_thresholds(champion_model, threshold_grid)
    except Exception as exc:
        print(f"Champion model unavailable or unevaluable: {exc}", flush=True)
        champion_exists = False

    candidate_model_uri = f"{candidate_output_prefix}/xgb_churn_pipeline.joblib"
    candidate_metadata_uri = f"{candidate_output_prefix}/model_metadata.json"
    candidate_model_path = Path(
        tempfile.NamedTemporaryFile(suffix=".joblib", delete=False).name
    )
    joblib.dump(candidate_model, candidate_model_path)
    upload_file(candidate_model_path, candidate_model_uri)

    if champion_exists:
        promote = (
            candidate_metrics["roc_auc"]
            >= champion_metrics["roc_auc"] + min_roc_auc_delta
            and candidate_metrics["recall"] >= min_recall
            and candidate_metrics["precision"] >= min_precision
            and candidate_metrics["net_value"] > champion_metrics["net_value"]
        )
    else:
        promote = True

    metadata = to_builtin({
        "model_type": "xgboost_sklearn_pipeline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_gcs_uri": dataset_gcs_uri,
        "feature_columns": feature_columns,
        "threshold": candidate_metrics["threshold"],
        "best_params": search.best_params_,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
        "promotion_rules": {
            "min_recall": min_recall,
            "min_precision": min_precision,
            "min_roc_auc_delta": min_roc_auc_delta,
            "requires_net_value_improvement": True,
        },
        "promoted": bool(promote),
        "candidate_model_gcs_uri": candidate_model_uri,
    })
    upload_json(metadata, candidate_metadata_uri)

    if promote:
        print("Promoting challenger to champion.", flush=True)
        upload_file(candidate_model_path, promoted_model_gcs_uri)
        upload_json(metadata, promoted_metadata_gcs_uri)
    else:
        print("Keeping existing champion model.", flush=True)

    metrics.log_metric("candidate_roc_auc", float(candidate_metrics["roc_auc"]))
    metrics.log_metric("candidate_precision", float(candidate_metrics["precision"]))
    metrics.log_metric("candidate_recall", float(candidate_metrics["recall"]))
    metrics.log_metric("candidate_f1", float(candidate_metrics["f1"]))
    metrics.log_metric("candidate_net_value", float(candidate_metrics["net_value"]))
    metrics.log_metric("candidate_threshold", float(candidate_metrics["threshold"]))
    metrics.log_metric("promoted", int(promote))

    if champion_metrics:
        metrics.log_metric("champion_roc_auc", float(champion_metrics["roc_auc"]))
        metrics.log_metric(
            "champion_precision",
            float(champion_metrics["precision"]),
        )
        metrics.log_metric("champion_recall", float(champion_metrics["recall"]))
        metrics.log_metric("champion_f1", float(champion_metrics["f1"]))
        metrics.log_metric(
            "champion_net_value",
            float(champion_metrics["net_value"]),
        )

    # KFP metrics require finite floats.
    for key, value in candidate_metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite candidate metric {key}: {value}")


@dsl.pipeline(
    name="telco-churn-champion-challenger",
    description="Tune an XGBoost churn model and promote it only if it beats the champion.",
)
def vertex_champion_challenger_pipeline(
    dataset_gcs_uri: str = "gs://telco-churn-vinay-2026/training/Telco_customer_churn.xlsx",
    champion_model_gcs_uri: str = "gs://telco-churn-vinay-2026/models/champion/xgb_churn_pipeline.joblib",
    champion_metadata_gcs_uri: str = "gs://telco-churn-vinay-2026/models/champion/model_metadata.json",
    candidate_output_prefix: str = "gs://telco-churn-vinay-2026/models/candidates/manual-run",
    promoted_model_gcs_uri: str = "gs://telco-churn-vinay-2026/models/champion/xgb_churn_pipeline.joblib",
    promoted_metadata_gcs_uri: str = "gs://telco-churn-vinay-2026/models/champion/model_metadata.json",
    n_iter: int = 12,
    min_recall: float = 0.75,
    min_precision: float = 0.50,
    min_roc_auc_delta: float = 0.0,
    random_state: int = 42,
    retention_offer_cost: float = 50.0,
    retention_success_rate: float = 0.25,
) -> None:
    tune_compare_promote_model(
        dataset_gcs_uri=dataset_gcs_uri,
        champion_model_gcs_uri=champion_model_gcs_uri,
        champion_metadata_gcs_uri=champion_metadata_gcs_uri,
        candidate_output_prefix=candidate_output_prefix,
        promoted_model_gcs_uri=promoted_model_gcs_uri,
        promoted_metadata_gcs_uri=promoted_metadata_gcs_uri,
        n_iter=n_iter,
        min_recall=min_recall,
        min_precision=min_precision,
        min_roc_auc_delta=min_roc_auc_delta,
        random_state=random_state,
        retention_offer_cost=retention_offer_cost,
        retention_success_rate=retention_success_rate,
    )
