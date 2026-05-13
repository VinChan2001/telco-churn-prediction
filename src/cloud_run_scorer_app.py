from flask import Flask, jsonify

from src.bq_scorer import score_unscored_customers


app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "telco-bq-scorer"})


@app.route("/score", methods=["GET", "POST"])
def score_customers():
    result = score_unscored_customers(refresh_input=True)

    return jsonify(
        {
            "status": "success",
            "rows_scored": result.rows_scored,
            "files_scored": result.files_scored,
            "scored_customers_table": result.scored_customers_table,
            "scored_files_table": result.scored_files_table,
            "threshold": result.threshold,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
