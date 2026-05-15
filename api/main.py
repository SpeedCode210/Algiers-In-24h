from __future__ import annotations

import os
import sys

_API_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.abspath(os.path.join(_API_DIR, "..", "algiers-lib"))

for _path in (_LIB_DIR, _API_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Flask and blueprint imports (safe after path is configured)

from flask import Flask, jsonify
from flask_cors import CORS

from routes.landmarks import landmarks_bp
from routes.solve import solve_bp
from routes.docs import docs_bp

# App factory

app = Flask(__name__)

# Allow all origins during development so the Next.js dev server
# (typically localhost:3000) can reach this API (localhost:5000).
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Register blueprints — all routes are prefixed with /api
app.register_blueprint(landmarks_bp, url_prefix="/api")
app.register_blueprint(solve_bp,     url_prefix="/api")
app.register_blueprint(docs_bp)


# Health check

@app.route("/api/health", methods=["GET"])
def health():
    """Quick health check endpoint.

    Returns:
        200 JSON confirming the server is running.
    """
    return jsonify({"status": "ok", "message": "Algiers-in-24h API is running."}), 200


# Entry point

if __name__ == "__main__":
    # debug=True enables auto-reload on file changes during development.
    # Set debug=False before any public/jury demo (remind me guys)
    app.run(debug=True, host="0.0.0.0", port=5000)