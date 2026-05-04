import joblib
import pandas as pd

from config import (
    SYNTHETIC_CUSTOMERS_PATH,
    MODEL_PATH,
    SCORED_CUSTOMERS_PATH,
    RISK_SEGMENT_SUMMARY_PATH,
    CHURN_THRESHOLD,
)


def load_data():
    """Load new customer data for inference."""
    df = pd.read_csv(SYNTHETIC_CUSTOMERS_PATH)
    return df


def prepare_features(df):
    """Prepare model input features for inference."""
    X = df.copy()

    X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")

    return X


def score_customers(model, X):
    """Generate churn probabilities, predictions, revenue estimates, and risk segments."""
    scored = X.copy()

    scored["churn_probability"] = model.predict_proba(X)[:, 1]
    scored["predicted_churn"] = (
        scored["churn_probability"] >= CHURN_THRESHOLD
    ).astype(int)

    scored["estimated_annual_revenue"] = scored["Monthly Charges"] * 12

    scored["risk_segment"] = pd.cut(
        scored["churn_probability"],
        bins=[0, 0.30, 0.60, 1.00],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )

    return scored


def create_risk_summary(scored):
    """Create risk-segment summary for dashboard reporting."""
    summary = scored.groupby("risk_segment", observed=True).agg(
        customer_count=("risk_segment", "count"),
        avg_churn_probability=("churn_probability", "mean"),
        total_estimated_annual_revenue=("estimated_annual_revenue", "sum"),
        avg_monthly_charges=("Monthly Charges", "mean"),
    ).reset_index()

    return summary


def main():
    """Run batch churn scoring."""
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading raw data...")
    df = load_data()

    print("Preparing features...")
    X = prepare_features(df)

    print("Scoring customers...")
    scored = score_customers(model, X)

    print("Creating risk segment summary...")
    summary = create_risk_summary(scored)

    print("Saving outputs...")
    scored.to_csv(SCORED_CUSTOMERS_PATH, index=False)
    summary.to_csv(RISK_SEGMENT_SUMMARY_PATH, index=False)

    print("Batch scoring complete.")
    print(f"Scored customers saved to: {SCORED_CUSTOMERS_PATH}")
    print(f"Risk segment summary saved to: {RISK_SEGMENT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()