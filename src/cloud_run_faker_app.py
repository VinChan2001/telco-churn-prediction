from datetime import datetime

import pandas as pd
from faker import Faker
from flask import Flask, jsonify, request

from src.gcs_utils import upload_file_to_gcs


app = Flask(__name__)
fake = Faker()

BUCKET_NAME = "telco-churn-vinay-2026"


def generate_fake_telco_customers(n: int = 25) -> pd.DataFrame:
    rows = []

    for _ in range(n):
        tenure_months = fake.random_int(min=1, max=72)
        monthly_charges = round(fake.pyfloat(min_value=20, max_value=120, right_digits=2), 2)
        total_charges = round(tenure_months * monthly_charges, 2)

        rows.append(
            {
                "Gender": fake.random_element(["Male", "Female"]),
                "Senior Citizen": fake.random_element(["Yes", "No"]),
                "Partner": fake.random_element(["Yes", "No"]),
                "Dependents": fake.random_element(["Yes", "No"]),
                "Tenure Months": tenure_months,
                "Phone Service": fake.random_element(["Yes", "No"]),
                "Multiple Lines": fake.random_element(["Yes", "No", "No phone service"]),
                "Internet Service": fake.random_element(["DSL", "Fiber optic", "No"]),
                "Online Security": fake.random_element(["Yes", "No", "No internet service"]),
                "Online Backup": fake.random_element(["Yes", "No", "No internet service"]),
                "Device Protection": fake.random_element(["Yes", "No", "No internet service"]),
                "Tech Support": fake.random_element(["Yes", "No", "No internet service"]),
                "Streaming TV": fake.random_element(["Yes", "No", "No internet service"]),
                "Streaming Movies": fake.random_element(["Yes", "No", "No internet service"]),
                "Contract": fake.random_element(["Month-to-month", "One year", "Two year"]),
                "Paperless Billing": fake.random_element(["Yes", "No"]),
                "Payment Method": fake.random_element(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ]
                ),
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
            }
        )

    return pd.DataFrame(rows)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "telco-cloud-run-faker"})


@app.route("/generate", methods=["GET"])
def generate_and_upload():
    n = int(request.args.get("n", 25))

    df = generate_fake_telco_customers(n)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    local_path = f"/tmp/synthetic_new_customers_{timestamp}.csv"
    gcs_path = f"incoming/synthetic_new_customers_{timestamp}.csv"

    df.to_csv(local_path, index=False)

    upload_file_to_gcs(
        local_file_path=local_path,
        bucket_name=BUCKET_NAME,
        destination_blob_name=gcs_path,
    )

    return jsonify(
        {
            "status": "success",
            "rows_generated": len(df),
            "gcs_path": f"gs://{BUCKET_NAME}/{gcs_path}",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)