"""WheezeDetect Flask Backend Application.

Provides REST endpoints for respiratory audio analysis and health status.
"""

import logging
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import (
    AudioProcessingError,
    AudioTooShortError,
    WheezeModelPipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("wheezedetect")

# Initialize Flask
app = Flask(__name__)

# Max upload size: 50 MB (matches frontend config)
MAX_FILE_SIZE_MB = 50
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Restrict CORS to local development origins (avoid open wildcards)
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
CORS(
    app,
    resources={r"/api/*": {"origins": [LOCAL_ORIGIN_REGEX, "null"]}},
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"]
)

# Initialize and load model pipeline once at startup
pipeline = WheezeModelPipeline()
try:
    pipeline.load()
    logger.info("Machine learning model pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Failed to load model pipeline at startup: {e}")


# Error Handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "success": False,
        "error": f"Uploaded file is too large. Maximum supported size is {MAX_FILE_SIZE_MB} MB."
    }), 413


@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        "success": False,
        "error": "Bad request."
    }), 400


@app.errorhandler(500)
def handle_server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "success": False,
        "error": "An internal server error occurred during audio processing."
    }), 500


# API Endpoints
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status endpoint."""
    loaded = pipeline.is_loaded()
    if loaded:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_name": pipeline.model_name,
            "sample_rate": pipeline.sample_rate,
            "threshold": pipeline.threshold
        }), 200
    else:
        return jsonify({
            "status": "degraded",
            "model_loaded": False,
            "error": "Model artifact is not loaded."
        }), 503


@app.route("/api/analyze", methods=["POST"])
def analyze_audio_endpoint():
    """Analyze uploaded audio for wheeze patterns.

    Expects multipart/form-data with a 'file' parameter containing a WAV file.
    """
    if not pipeline.is_loaded():
        return jsonify({
            "success": False,
            "error": "Machine learning model is currently unavailable."
        }), 503

    # 1. Validate file presence
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded. Please include a WAV audio file with the field name 'file'."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({
            "success": False,
            "error": "No audio file selected for analysis."
        }), 400

    filename = uploaded_file.filename.lower()
    if not filename.endswith(".wav"):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Only WAV (.wav) recordings are supported."
        }), 400

    # 2. Check for empty file content
    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size == 0:
        return jsonify({
            "success": False,
            "error": "The uploaded audio file is empty (0 bytes)."
        }), 400

    # 3. Process and analyze audio safely in-memory
    try:
        audio_stream = uploaded_file.stream
        result = pipeline.analyze_audio(audio_stream)
        logger.info(
            f"Successfully analyzed '{uploaded_file.filename}' "
            f"({result['audio_duration']}s, {result['segments_analyzed']} segments): "
            f"{result['prediction']} (score: {result['score']:.4f})"
        )
        return jsonify(result), 200

    except AudioTooShortError as e:
        logger.warning(f"Audio too short: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except AudioProcessingError as e:
        logger.error(f"Audio processing failure: {e}")
        return jsonify({
            "success": False,
            "error": f"Audio processing error: {str(e)}"
        }), 400

    except Exception as e:
        logger.error(f"Unexpected inference error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Audio analysis could not be completed. Please ensure the file is an uncorrupted WAV recording."
        }), 500


# Static Frontend Serving (Convenience for single-server local development)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@app.route("/", methods=["GET"])
def serve_frontend_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_frontend_static(filename):
    if os.path.isfile(os.path.join(FRONTEND_DIR, filename)):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting WheezeDetect server on http://{host}:{port}...")
    app.run(host=host, port=port, debug=False)
