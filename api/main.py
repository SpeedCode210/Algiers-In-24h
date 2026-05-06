import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../algiers-lib'))

from flask import Flask
from flask_cors import CORS
from routes.landmarks import landmarks_bp
from routes.solve import solve_bp

app = Flask(__name__)
CORS(app)  # allows Next.js frontend to call this API

app.register_blueprint(landmarks_bp, url_prefix="/api")
app.register_blueprint(solve_bp,    url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True, port=5000)