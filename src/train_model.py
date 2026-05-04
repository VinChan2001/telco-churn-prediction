import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from config import (
    RAW_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    CHURN_THRESHOLD,
)


def load_data():
    """Load the raw Telco churn dataset."""
    df = pd.read_excel(RAW_DATA_PATH)
    return df


def clean_data(df):
    """Apply basic data cleaning."""
    df = df.copy()

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    return df


def split_features_target(df):
    """Create model features X and target y."""
    target = "Churn Value"

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

    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target]

    return X, y


def build_pipeline(X):
    """Build preprocessing + XGBoost pipeline."""
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate trained model and return metrics."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= CHURN_THRESHOLD).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "churn_precision": precision_score(y_test, predictions),
        "churn_recall": recall_score(y_test, predictions),
        "churn_f1": f1_score(y_test, predictions),
        "threshold": CHURN_THRESHOLD,
    }

    print("ROC AUC:", metrics["roc_auc"])
    print("Accuracy:", metrics["accuracy"])
    print("Churn Precision:", metrics["churn_precision"])
    print("Churn Recall:", metrics["churn_recall"])
    print("Churn F1:", metrics["churn_f1"])

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return metrics


def save_metrics(metrics):
    """Save model metrics to CSV."""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)


def main():
    """Run full training pipeline."""
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting features and target...")
    X, y = split_features_target(df)

    print("Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Building pipeline...")
    model = build_pipeline(X)

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("Saving model artifact...")
    joblib.dump(model, MODEL_PATH)

    print("Saving metrics...")
    save_metrics(metrics)

    print("Training pipeline complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()