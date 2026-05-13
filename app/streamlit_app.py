import os
from typing import Any

import pandas as pd
import streamlit as st
from google.cloud import bigquery


SCORED_CUSTOMERS_TABLE = "telco-churn-vinay-raw.telco_churn.scored_customers"
RAW_CUSTOMERS_TABLE = "telco-churn-vinay-raw.telco_churn.synthetic_customers"
MODEL_INPUT_TABLE = "telco-churn-vinay-raw.telco_churn.synthetic_customers_model_input"
LOADED_FILES_TABLE = "telco-churn-vinay-raw.telco_churn.loaded_files"
SCORED_FILES_TABLE = "telco-churn-vinay-raw.telco_churn.scored_files"
HISTORICAL_SOURCE_FILE = "__historical_pre_source_file__"

LOCAL_SCORED_CUSTOMERS_PATH = "data/processed/scored_churn_customers.csv"
LOCAL_MODEL_METRICS_PATH = "data/processed/model_metrics.csv"
LOCAL_SHAP_IMPORTANCE_PATH = "data/processed/shap_feature_importance.csv"


st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    layout="wide",
)


def format_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(value):,}"


def format_currency(value: Any) -> str:
    if value is None or pd.isna(value):
        return "$0"
    return f"${float(value):,.0f}"


def format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "0.0%"
    return f"{float(value) * 100:.1f}%"


def format_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "Not available"

    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def risk_segment_from_probability(probability: float) -> str:
    if probability <= 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


def normalize_scored_customers(scored_customers: pd.DataFrame) -> pd.DataFrame:
    scored_customers = scored_customers.copy()

    if "churn_probability" in scored_customers.columns:
        scored_customers["churn_probability"] = pd.to_numeric(
            scored_customers["churn_probability"],
            errors="coerce",
        )

    if "predicted_churn" in scored_customers.columns:
        scored_customers["predicted_churn"] = (
            pd.to_numeric(scored_customers["predicted_churn"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    if "estimated_annual_revenue" not in scored_customers.columns:
        scored_customers["estimated_annual_revenue"] = (
            pd.to_numeric(scored_customers["Monthly Charges"], errors="coerce") * 12
        )

    if "risk_segment" not in scored_customers.columns:
        scored_customers["risk_segment"] = scored_customers[
            "churn_probability"
        ].apply(risk_segment_from_probability)

    if "scored_at" in scored_customers.columns:
        scored_customers["scored_at"] = pd.to_datetime(
            scored_customers["scored_at"],
            utc=True,
            errors="coerce",
        )

    return scored_customers


def query_bigquery_dataframe(query: str) -> pd.DataFrame:
    client = bigquery.Client()
    rows = client.query(query).result()
    records = [dict(row) for row in rows]
    return pd.DataFrame(records)


def load_scored_customers(
    data_source: str | None = None,
) -> tuple[pd.DataFrame, str, str | None]:
    data_source = (data_source or os.getenv("DASHBOARD_DATA_SOURCE", "bigquery")).lower()

    if data_source == "local":
        customers = pd.read_csv(LOCAL_SCORED_CUSTOMERS_PATH)
        return normalize_scored_customers(customers), "Local CSV", None

    query = f"""
    SELECT *
    FROM `{SCORED_CUSTOMERS_TABLE}`
    ORDER BY scored_at DESC, churn_probability DESC
    """

    try:
        customers = query_bigquery_dataframe(query)
        if not customers.empty:
            return normalize_scored_customers(customers), "BigQuery", None
    except Exception as exc:
        fallback_error = str(exc)
    else:
        fallback_error = "BigQuery returned no scored customers."

    customers = pd.read_csv(LOCAL_SCORED_CUSTOMERS_PATH)
    return normalize_scored_customers(customers), "Local CSV fallback", fallback_error


def load_pipeline_freshness(
    data_source: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    data_source = (data_source or os.getenv("DASHBOARD_DATA_SOURCE", "bigquery")).lower()
    if data_source == "local":
        return {}, "Pipeline freshness is unavailable in local CSV mode."

    query = f"""
    SELECT
      (SELECT COUNT(*) FROM `{RAW_CUSTOMERS_TABLE}`) AS raw_rows,
      (SELECT COUNT(*) FROM `{MODEL_INPUT_TABLE}`) AS model_input_rows,
      (SELECT COUNT(*) FROM `{SCORED_CUSTOMERS_TABLE}`) AS scored_rows,
      (SELECT COUNT(*) FROM `{LOADED_FILES_TABLE}`) AS loaded_files,
      (SELECT COUNT(*) FROM `{SCORED_FILES_TABLE}`) AS scored_files,
      (
        SELECT COUNT(DISTINCT source_file)
        FROM `{MODEL_INPUT_TABLE}`
        WHERE source_file IS NOT NULL
          AND source_file != @historical_source_file
      ) AS traceable_source_files,
      (
        SELECT COUNT(*)
        FROM `{MODEL_INPUT_TABLE}` AS m
        WHERE m.source_file IS NOT NULL
          AND m.source_file != @historical_source_file
          AND NOT EXISTS (
            SELECT 1
            FROM `{SCORED_FILES_TABLE}` AS s
            WHERE s.file_uri = m.source_file
          )
      ) AS unscored_traceable_rows,
      (
        SELECT COUNT(*)
        FROM `{MODEL_INPUT_TABLE}`
        WHERE `Total Charges` IS NULL
      ) AS null_total_charges,
      (
        SELECT COUNT(*)
        FROM `{MODEL_INPUT_TABLE}`
        WHERE (`Phone Service` = "No" AND `Multiple Lines` != "No phone service")
           OR (
             `Internet Service` = "No"
             AND (
               `Online Security` != "No internet service"
               OR `Online Backup` != "No internet service"
               OR `Device Protection` != "No internet service"
               OR `Tech Support` != "No internet service"
               OR `Streaming TV` != "No internet service"
               OR `Streaming Movies` != "No internet service"
             )
           )
      ) AS invalid_service_combos,
      (
        SELECT file_uri
        FROM `{LOADED_FILES_TABLE}`
        ORDER BY loaded_at DESC
        LIMIT 1
      ) AS latest_loaded_file,
      (
        SELECT loaded_at
        FROM `{LOADED_FILES_TABLE}`
        ORDER BY loaded_at DESC
        LIMIT 1
      ) AS latest_loaded_at,
      (
        SELECT row_count
        FROM `{LOADED_FILES_TABLE}`
        ORDER BY loaded_at DESC
        LIMIT 1
      ) AS latest_loaded_rows,
      (
        SELECT file_uri
        FROM `{SCORED_FILES_TABLE}`
        ORDER BY scored_at DESC
        LIMIT 1
      ) AS latest_scored_file,
      (
        SELECT scored_at
        FROM `{SCORED_FILES_TABLE}`
        ORDER BY scored_at DESC
        LIMIT 1
      ) AS latest_scored_at,
      (
        SELECT row_count
        FROM `{SCORED_FILES_TABLE}`
        ORDER BY scored_at DESC
        LIMIT 1
      ) AS latest_scored_rows
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "historical_source_file",
                "STRING",
                HISTORICAL_SOURCE_FILE,
            )
        ]
    )

    try:
        client = bigquery.Client()
        row = next(iter(client.query(query, job_config=job_config).result()))
        return dict(row), None
    except Exception as exc:
        return {}, str(exc)


@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str | None]:
    scored_customers, source_name, source_error = load_scored_customers()
    model_metrics = pd.read_csv(LOCAL_MODEL_METRICS_PATH)
    shap_importance = pd.read_csv(LOCAL_SHAP_IMPORTANCE_PATH)
    risk_summary = create_risk_summary(scored_customers)

    return (
        scored_customers,
        risk_summary,
        model_metrics,
        shap_importance,
        source_name,
        source_error,
    )


@st.cache_data(ttl=300)
def cached_pipeline_freshness() -> tuple[dict[str, Any], str | None]:
    return load_pipeline_freshness()


def create_risk_summary(scored_customers: pd.DataFrame) -> pd.DataFrame:
    if scored_customers.empty:
        return pd.DataFrame(
            columns=[
                "risk_segment",
                "customer_count",
                "avg_churn_probability",
                "total_estimated_annual_revenue",
                "avg_monthly_charges",
            ]
        )

    summary = scored_customers.groupby("risk_segment", observed=True).agg(
        customer_count=("risk_segment", "count"),
        avg_churn_probability=("churn_probability", "mean"),
        total_estimated_annual_revenue=("estimated_annual_revenue", "sum"),
        avg_monthly_charges=("Monthly Charges", "mean"),
    ).reset_index()

    return summary


def dashboard_summary(scored_customers: pd.DataFrame) -> dict[str, Any]:
    high_risk = scored_customers[scored_customers["risk_segment"] == "High Risk"]
    flagged = scored_customers[scored_customers["predicted_churn"] == 1]

    latest_scored_at = None
    if "scored_at" in scored_customers.columns:
        latest_scored_at = scored_customers["scored_at"].max()

    return {
        "total_customers": len(scored_customers),
        "high_risk_customers": len(high_risk),
        "flagged_customers": len(flagged),
        "avg_churn_probability": scored_customers["churn_probability"].mean(),
        "flagged_annual_revenue": flagged["estimated_annual_revenue"].sum(),
        "latest_scored_at": latest_scored_at,
    }


def pipeline_is_in_sync(freshness: dict[str, Any]) -> bool:
    if not freshness:
        return False

    return (
        freshness.get("raw_rows") == freshness.get("model_input_rows")
        and freshness.get("model_input_rows") == freshness.get("scored_rows")
        and freshness.get("unscored_traceable_rows") == 0
        and freshness.get("invalid_service_combos") == 0
        and freshness.get("null_total_charges") == 0
    )


def render_overview(
    scored_customers: pd.DataFrame,
    risk_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    source_name: str,
    source_error: str | None,
) -> None:
    summary = dashboard_summary(scored_customers)

    if source_error:
        st.warning(f"Using local fallback data. BigQuery issue: {source_error}")

    metric_row = st.columns(5)
    metric_row[0].metric("Data Source", source_name)
    metric_row[1].metric("Scored Customers", format_int(summary["total_customers"]))
    metric_row[2].metric("High-Risk Customers", format_int(summary["high_risk_customers"]))
    metric_row[3].metric("Predicted Churners", format_int(summary["flagged_customers"]))
    metric_row[4].metric("Latest Score", format_timestamp(summary["latest_scored_at"]))

    st.subheader("Model Performance")

    perf_row = st.columns(4)
    perf_row[0].metric("ROC AUC", f"{model_metrics.loc[0, 'roc_auc']:.3f}")
    perf_row[1].metric("Churn Recall", f"{model_metrics.loc[0, 'churn_recall']:.3f}")
    perf_row[2].metric("Churn Precision", f"{model_metrics.loc[0, 'churn_precision']:.3f}")
    perf_row[3].metric("Churn F1", f"{model_metrics.loc[0, 'churn_f1']:.3f}")

    st.subheader("Risk Segment Summary")
    formatted_summary = risk_summary.copy()
    if not formatted_summary.empty:
        formatted_summary["avg_churn_probability"] = formatted_summary[
            "avg_churn_probability"
        ].map(format_percent)
        formatted_summary["total_estimated_annual_revenue"] = formatted_summary[
            "total_estimated_annual_revenue"
        ].map(format_currency)
        formatted_summary["avg_monthly_charges"] = formatted_summary[
            "avg_monthly_charges"
        ].map(lambda value: f"${value:,.2f}")

    st.dataframe(formatted_summary, use_container_width=True, hide_index=True)

    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_count_chart = (
        risk_summary.set_index("risk_segment")
        .reindex(risk_order)["customer_count"]
        .fillna(0)
    )
    st.bar_chart(risk_count_chart)


def render_customers(scored_customers: pd.DataFrame) -> None:
    selected_segment = st.selectbox(
        "Risk segment",
        ["All"] + sorted(scored_customers["risk_segment"].dropna().unique().tolist()),
    )
    show_metadata = st.checkbox("Show pipeline metadata", value=False)

    if selected_segment == "All":
        filtered_customers = scored_customers
    else:
        filtered_customers = scored_customers[
            scored_customers["risk_segment"] == selected_segment
        ]

    customer_columns = [
        "churn_probability",
        "predicted_churn",
        "risk_segment",
        "Monthly Charges",
        "estimated_annual_revenue",
        "Contract",
        "Internet Service",
        "Tenure Months",
    ]
    if show_metadata:
        customer_columns.extend(["source_file", "load_id", "scored_at"])

    customer_columns = [
        column for column in customer_columns
        if column in filtered_customers.columns
    ]

    st.dataframe(
        filtered_customers[customer_columns].sort_values(
            "churn_probability",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    top_n = st.slider(
        "Top at-risk customers",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
    )
    top_at_risk = scored_customers.sort_values(
        "churn_probability",
        ascending=False,
    ).head(top_n)

    st.subheader("Retention Outreach Shortlist")
    st.dataframe(
        top_at_risk[customer_columns],
        use_container_width=True,
        hide_index=True,
    )


def render_business_impact(scored_customers: pd.DataFrame) -> None:
    flagged_customers = scored_customers[scored_customers["predicted_churn"] == 1]
    customers_flagged = len(flagged_customers)
    annual_revenue_flagged = flagged_customers["estimated_annual_revenue"].sum()

    assumption_col1, assumption_col2 = st.columns(2)
    retention_offer_cost = assumption_col1.number_input(
        "Retention offer cost per flagged customer ($)",
        min_value=0,
        value=50,
        step=5,
    )
    assumed_retention_success_rate = assumption_col2.slider(
        "Assumed retention success rate",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    estimated_campaign_cost = customers_flagged * retention_offer_cost
    estimated_revenue_saved = annual_revenue_flagged * assumed_retention_success_rate
    estimated_net_value = estimated_revenue_saved - estimated_campaign_cost

    biz_col1, biz_col2, biz_col3, biz_col4 = st.columns(4)
    biz_col1.metric("Customers Flagged", format_int(customers_flagged))
    biz_col2.metric("Flagged Annual Revenue", format_currency(annual_revenue_flagged))
    biz_col3.metric("Campaign Cost", format_currency(estimated_campaign_cost))
    biz_col4.metric("Estimated Net Value", format_currency(estimated_net_value))

    st.subheader("Top 20% High-Risk Targeting Scenario")

    top_20_cutoff = scored_customers["churn_probability"].quantile(0.80)
    top_20_customers = scored_customers[
        scored_customers["churn_probability"] >= top_20_cutoff
    ]

    top_20_customer_count = len(top_20_customers)
    top_20_annual_revenue = top_20_customers["estimated_annual_revenue"].sum()
    top_20_campaign_cost = top_20_customer_count * retention_offer_cost
    top_20_estimated_revenue_saved = (
        top_20_annual_revenue * assumed_retention_success_rate
    )
    top_20_estimated_net_value = (
        top_20_estimated_revenue_saved - top_20_campaign_cost
    )

    top20_col1, top20_col2, top20_col3, top20_col4 = st.columns(4)
    top20_col1.metric("Top 20% Customers", format_int(top_20_customer_count))
    top20_col2.metric("Top 20% Annual Revenue", format_currency(top_20_annual_revenue))
    top20_col3.metric("Top 20% Campaign Cost", format_currency(top_20_campaign_cost))
    top20_col4.metric("Top 20% Net Value", format_currency(top_20_estimated_net_value))


def render_model_drivers(shap_importance: pd.DataFrame) -> None:
    st.subheader("Global Model Drivers")

    top_shap_features = shap_importance.head(15).copy()
    st.dataframe(top_shap_features, use_container_width=True, hide_index=True)

    top_shap_chart = (
        top_shap_features.sort_values("mean_abs_shap_value", ascending=True)
        .set_index("feature")["mean_abs_shap_value"]
    )
    st.bar_chart(top_shap_chart)


def render_pipeline(freshness: dict[str, Any], freshness_error: str | None) -> None:
    if freshness_error:
        st.warning(f"Pipeline freshness unavailable: {freshness_error}")
        return

    if pipeline_is_in_sync(freshness):
        st.success("Live BigQuery pipeline is in sync.")
    else:
        st.warning("Pipeline counts need review.")

    status_row = st.columns(5)
    status_row[0].metric("Raw Rows", format_int(freshness.get("raw_rows")))
    status_row[1].metric("Model Input Rows", format_int(freshness.get("model_input_rows")))
    status_row[2].metric("Scored Rows", format_int(freshness.get("scored_rows")))
    status_row[3].metric("Loaded Files", format_int(freshness.get("loaded_files")))
    status_row[4].metric("Scored Files", format_int(freshness.get("scored_files")))

    quality_row = st.columns(4)
    quality_row[0].metric(
        "Unscored Traceable Rows",
        format_int(freshness.get("unscored_traceable_rows")),
    )
    quality_row[1].metric(
        "Invalid Service Combos",
        format_int(freshness.get("invalid_service_combos")),
    )
    quality_row[2].metric(
        "Null Total Charges",
        format_int(freshness.get("null_total_charges")),
    )
    quality_row[3].metric(
        "Traceable Source Files",
        format_int(freshness.get("traceable_source_files")),
    )

    latest = pd.DataFrame(
        [
            {
                "stage": "Latest loaded file",
                "file": freshness.get("latest_loaded_file"),
                "timestamp": format_timestamp(freshness.get("latest_loaded_at")),
                "rows": freshness.get("latest_loaded_rows"),
            },
            {
                "stage": "Latest scored file",
                "file": freshness.get("latest_scored_file"),
                "timestamp": format_timestamp(freshness.get("latest_scored_at")),
                "rows": freshness.get("latest_scored_rows"),
            },
        ]
    )
    st.dataframe(latest, use_container_width=True, hide_index=True)


def main() -> None:
    (
        scored_customers,
        risk_summary,
        model_metrics,
        shap_importance,
        source_name,
        source_error,
    ) = load_data()
    freshness, freshness_error = cached_pipeline_freshness()

    st.title("Telco Customer Churn Prediction Dashboard")
    st.caption("Cloud Run dashboard backed by BigQuery scored churn predictions.")

    overview_tab, customers_tab, impact_tab, drivers_tab, pipeline_tab = st.tabs(
        ["Overview", "Customers", "Business Impact", "Model Drivers", "Pipeline"]
    )

    with overview_tab:
        render_overview(
            scored_customers,
            risk_summary,
            model_metrics,
            source_name,
            source_error,
        )

    with customers_tab:
        render_customers(scored_customers)

    with impact_tab:
        render_business_impact(scored_customers)

    with drivers_tab:
        render_model_drivers(shap_importance)

    with pipeline_tab:
        render_pipeline(freshness, freshness_error)


if __name__ == "__main__":
    main()
