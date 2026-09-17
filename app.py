"""Local development server. All calculation logic lives in api/upload.py
(the Vercel function) so both environments behave identically.

Run:  python3 app.py   →  http://localhost:5050
"""
import os
import sys

from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from upload import process_upload, MAX_FILE_SIZE  # noqa: E402

app = Flask(__name__, static_folder="public")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No se subió ningún archivo"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        return jsonify({"error": "Por favor suba un archivo Excel (.xlsx o .xls)"}), 400

    try:
        return jsonify(process_upload(file.read(), request.form))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
