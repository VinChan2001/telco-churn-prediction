from datetime import datetime

from flask import Flask, jsonify, request

from src.gcs_utils import upload_file_to_gcs
from src.synthetic_customer_generator import generate_customers


app = Flask(__name__)

BUCKET_NAME = "telco-churn-vinay-2026"


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "telco-cloud-run-faker"})


@app.route("/generate", methods=["GET"])
def generate_and_upload():
    n = int(request.args.get("n", 25))

    df = generate_customers(n_customers=n)

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
