from flask import Flask, jsonify, request

from src.bq_loader import load_gcs_csv_to_bigquery


app = Flask(__name__)

TABLE_ID = "telco-churn-vinay-raw.telco_churn.synthetic_customers"


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "telco-bq-loader"})


@app.route("/", methods=["POST"])
def load_gcs_event_to_bigquery():
    """
    Receives a Cloud Storage event and loads the newly created CSV file into BigQuery.
    """

    event = request.get_json()

    if not event:
        return jsonify({"status": "error", "message": "No event payload received"}), 400

    bucket_name = event.get("bucket")
    file_name = event.get("name")

    if not bucket_name or not file_name:
        return jsonify(
            {
                "status": "error",
                "message": "Missing bucket or file name in event payload",
                "event": event,
            }
        ), 400

    if not file_name.endswith(".csv"):
        return jsonify(
            {
                "status": "skipped",
                "message": "File is not a CSV",
                "file_name": file_name,
            }
        )

    gcs_uri = f"gs://{bucket_name}/{file_name}"

    load_gcs_csv_to_bigquery(
        gcs_uri=gcs_uri,
        table_id=TABLE_ID,
    )

    return jsonify(
        {
            "status": "success",
            "loaded_file": gcs_uri,
            "table_id": TABLE_ID,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)