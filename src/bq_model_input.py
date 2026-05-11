from google.cloud import bigquery


RAW_SYNTHETIC_TABLE_ID = "telco-churn-vinay-raw.telco_churn.synthetic_customers"
MODEL_INPUT_TABLE_ID = "telco-churn-vinay-raw.telco_churn.synthetic_customers_model_input"


def _yes_no_expression(column_name: str) -> str:
    column = f"`{column_name}`"

    return f"""
        CASE
            WHEN {column} IS NULL THEN NULL
            WHEN LOWER(CAST({column} AS STRING)) IN ('true', 'yes', '1') THEN 'Yes'
            WHEN LOWER(CAST({column} AS STRING)) IN ('false', 'no', '0') THEN 'No'
            ELSE CAST({column} AS STRING)
        END
    """


def _no_phone_service_expression(column_name: str) -> str:
    return f"""
        CASE
            WHEN LOWER(CAST(`Phone Service` AS STRING)) IN ('false', 'no', '0')
                THEN 'No phone service'
            ELSE CAST(`{column_name}` AS STRING)
        END
    """


def _no_internet_service_expression(column_name: str) -> str:
    return f"""
        CASE
            WHEN CAST(`Internet Service` AS STRING) = 'No'
                THEN 'No internet service'
            ELSE CAST(`{column_name}` AS STRING)
        END
    """


def build_model_input_sql(
    raw_table_id: str = RAW_SYNTHETIC_TABLE_ID,
    model_input_table_id: str = MODEL_INPUT_TABLE_ID,
) -> str:
    """Build the query that materializes the model-ready BigQuery table."""

    return f"""
    CREATE OR REPLACE TABLE `{model_input_table_id}` AS
    SELECT
        CAST(`Gender` AS STRING) AS `Gender`,
        {_yes_no_expression("Senior Citizen")} AS `Senior Citizen`,
        {_yes_no_expression("Partner")} AS `Partner`,
        {_yes_no_expression("Dependents")} AS `Dependents`,
        SAFE_CAST(`Tenure Months` AS INT64) AS `Tenure Months`,
        {_yes_no_expression("Phone Service")} AS `Phone Service`,
        {_no_phone_service_expression("Multiple Lines")} AS `Multiple Lines`,
        CAST(`Internet Service` AS STRING) AS `Internet Service`,
        {_no_internet_service_expression("Online Security")} AS `Online Security`,
        {_no_internet_service_expression("Online Backup")} AS `Online Backup`,
        {_no_internet_service_expression("Device Protection")} AS `Device Protection`,
        {_no_internet_service_expression("Tech Support")} AS `Tech Support`,
        {_no_internet_service_expression("Streaming TV")} AS `Streaming TV`,
        {_no_internet_service_expression("Streaming Movies")} AS `Streaming Movies`,
        CAST(`Contract` AS STRING) AS `Contract`,
        {_yes_no_expression("Paperless Billing")} AS `Paperless Billing`,
        CAST(`Payment Method` AS STRING) AS `Payment Method`,
        SAFE_CAST(`Monthly Charges` AS FLOAT64) AS `Monthly Charges`,
        SAFE_CAST(`Total Charges` AS FLOAT64) AS `Total Charges`
    FROM `{raw_table_id}`
    """


def refresh_model_input_table(
    client: bigquery.Client | None = None,
    raw_table_id: str = RAW_SYNTHETIC_TABLE_ID,
    model_input_table_id: str = MODEL_INPUT_TABLE_ID,
) -> bigquery.Table:
    """Create or replace the BigQuery table used as model inference input."""

    client = client or bigquery.Client()
    query = build_model_input_sql(
        raw_table_id=raw_table_id,
        model_input_table_id=model_input_table_id,
    )

    client.query(query).result()

    return client.get_table(model_input_table_id)
