from __future__ import annotations

import os

from flask import Blueprint, render_template_string, send_from_directory

docs_bp = Blueprint("docs", __name__)


@docs_bp.route("/api/openapi.json", methods=["GET"])
def openapi_spec():
    """Serves the bundled OpenAPI JSON spec."""
    spec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.json"))
    return send_from_directory(os.path.dirname(spec_path), os.path.basename(spec_path), mimetype="application/json")


@docs_bp.route("/api/docs", methods=["GET"])
def swagger_ui():
    """Serves a minimal Swagger UI that points to /api/openapi.json.

    Uses the Swagger UI bundle from a CDN so no extra Python dependency is required.
    """
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Algiers-in-24h API Docs</title>
        <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@4/swagger-ui.css" />
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
        <script>
          window.onload = function() {
            const ui = SwaggerUIBundle({
              url: '/api/openapi.json',
              dom_id: '#swagger-ui',
              presets: [SwaggerUIBundle.presets.apis],
              layout: "BaseLayout",
            });
            window.ui = ui;
          };
        </script>
      </body>
    </html>
    """
    return render_template_string(html)
