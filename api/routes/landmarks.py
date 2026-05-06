from __future__ import annotations

from flask import Blueprint, jsonify

from services.problem_loader import (
    get_all_hotels,
    get_all_landmarks,
    get_unique_categories,
)
from utils.time import time_in_string

landmarks_bp = Blueprint("landmarks", __name__)


# GET /api/landmarks


@landmarks_bp.route("/landmarks", methods=["GET"])
def get_landmarks():
    """Returns all landmarks for map initialisation.

    Called once on page load. The frontend uses the response to place
    pins on the map and populate the hover detail cards.

    Response shape (200):
        {
            "count": 56,
            "landmarks": [
                {
                    "id":                     "casbah",
                    "name":                   "Casbah of Algiers",
                    "latitude":               36.7845,
                    "longitude":              3.0589,
                    "interest_score":         8.9,
                    "visit_duration_minutes": 120,
                    "category":               "historical",
                    "schedule": {
                        "saturday": [{"open": "00:00", "close": "23:59"}],
                        ...
                    }
                },
                ...
            ]
        }
    """
    try:
        landmarks = get_all_landmarks()
    except Exception as exc:
        return jsonify({"error": f"Failed to load landmarks: {exc}"}), 500

    serialised = []
    for lm in landmarks:
        schedule: dict[str, list[dict]] = {}
        for day, slots in lm.schedule.schedule.items():
            schedule[day.name.lower()] = [
                {
                    "open":  time_in_string(slot.open_time),
                    "close": time_in_string(slot.close_time),
                }
                for slot in slots
            ]
        serialised.append({
            "id":                     lm.id,
            "name":                   lm.name,
            "latitude":               lm.latitude,
            "longitude":              lm.longitude,
            "interest_score":         lm.interest_score,
            "visit_duration_minutes": lm.visit_duration,
            "category":               lm.category,
            "schedule":               schedule,
        })

    return jsonify({"count": len(serialised), "landmarks": serialised}), 200


# GET /api/hotels

@landmarks_bp.route("/hotels", methods=["GET"])
def get_hotels():
    """Returns all available hotels for the hotel selector.

    The user picks one hotel from this list. The selected hotel's id
    is then sent as hotel_id in every POST /api/solve request.

    Response shape (200):
        {
            "count": 10,
            "hotels": [
                {
                    "id":        "hotel_aurassi",
                    "name":      "Hotel El-Aurassi",
                    "latitude":  36.7692,
                    "longitude": 3.0564
                },
                ...
            ]
        }
    """
    try:
        hotels = get_all_hotels()
    except Exception as exc:
        return jsonify({"error": f"Failed to load hotels: {exc}"}), 500

    return jsonify({"count": len(hotels), "hotels": hotels}), 200


# GET /api/categories

@landmarks_bp.route("/categories", methods=["GET"])
def get_categories():
    """Returns all landmark categories for the preference weight panel.

    The user assigns a weight multiplier (0.0–5.0) to each category.
    Weights are sent as category_weights in /api/solve requests and
    are applied to landmark scores before the solver runs.

    Response shape (200):
        {
            "categories": [
                {
                    "id":             "historical",
                    "label":          "Historical",
                    "default_weight": 1.0
                },
                ...
            ]
        }
    """
    try:
        categories = get_unique_categories()
    except Exception as exc:
        return jsonify({"error": f"Failed to load categories: {exc}"}), 500

    return jsonify({"categories": categories}), 200