from flask import Flask, jsonify, request

from src.training_trigger import trigger_training_pipeline_for_gcs_file


app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "telco-training-trigger"})


@app.route("/", methods=["POST"])
def trigger_training_pipeline():
    """
    Receives a Cloud Storage finalized event and submits a Vertex AI pipeline
    only for labeled training files under training/incoming/.
    """

    event = request.get_json()
    if not event:
        return jsonify({"status": "error", "message": "No event payload received"}), 400

    bucket_name = event.get("bucket")
    file_name = event.get("name")
    generation = event.get("generation")
    generation = str(generation) if generation is not None else None

    if not bucket_name or not file_name:
        return jsonify(
            {
                "status": "error",
                "message": "Missing bucket or file name in event payload",
                "event": event,
            }
        ), 400

    result = trigger_training_pipeline_for_gcs_file(
        bucket_name=bucket_name,
        file_name=file_name,
        generation=generation,
    )

    return jsonify(result.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
