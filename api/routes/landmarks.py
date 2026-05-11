from __future__ import annotations
 
import os
 
from flask import Blueprint, jsonify
 
from models.landmark import loadAllHotels, loadLandmarks
from utils.time import time_in_string
 
landmarks_bp = Blueprint("landmarks", __name__)
 
# Absolute paths — resolved once at import time
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR      = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "algiers-lib"))
_LANDMARKS_CSV = os.path.join(_LIB_DIR, "data", "data.csv")
_HOTELS_CSV    = os.path.join(_LIB_DIR, "data", "hotel.csv")
 
 
# GET /api/landmarks
 
@landmarks_bp.route("/landmarks", methods=["GET"])
def get_landmarks():
    """Returns every landmark in the dataset for map initialisation.
 
    Called once on page load. The frontend uses the result to place
    pins on the map and populate the hover detail cards.
 
    Returns:
        200  { "count": int, "landmarks": [ LandmarkObject, ... ] }
        500  { "error": str }
 
    LandmarkObject shape::
 
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
                "friday":   [{"open": "09:00", "close": "12:00"},
                             {"open": "14:00", "close": "17:00"}]
            }
        }
    """
    try:
        landmarks = loadLandmarks(_LANDMARKS_CSV)
    except Exception as exc:
        return jsonify({"error": f"Failed to load landmarks: {exc}"}), 500
 
    result = []
    for lm in landmarks:
        schedule: dict = {}
        for day, slots in lm.schedule.schedule.items():
            schedule[day.name.lower()] = [
                {"open": time_in_string(s.open_time),
                 "close": time_in_string(s.close_time)}
                for s in slots
            ]
        result.append({
            "id":                     lm.id,
            "name":                   lm.name,
            "latitude":               lm.latitude,
            "longitude":              lm.longitude,
            "interest_score":         lm.interest_score,
            "visit_duration_minutes": lm.visit_duration,
            "category":               lm.category,
            "schedule":               schedule,
        })
 
    return jsonify({"count": len(result), "landmarks": result}), 200
 
 
# GET /api/hotels
 
@landmarks_bp.route("/hotels", methods=["GET"])
def get_hotels():
    """Returns all available hotels for the hotel selector.
 
    The user picks one hotel from this list. The chosen hotel's id
    must be sent as hotel_id in every POST /api/solve request.
 
    Returns:
        200  { "count": int, "hotels": [ HotelObject, ... ] }
        500  { "error": str }
 
    HotelObject shape::
 
        {
            "id":        "hotel_aurassi",
            "name":      "Hotel El-Aurassi",
            "latitude":  36.7692,
            "longitude": 3.0564
        }
    """
    try:
        hotels = loadAllHotels(_HOTELS_CSV)
    except Exception as exc:
        return jsonify({"error": f"Failed to load hotels: {exc}"}), 500
 
    result = [
        {
            "id":        h.id,
            "name":      h.name,
            "latitude":  h.latitude,
            "longitude": h.longitude,
        }
        for h in hotels
    ]
    return jsonify({"count": len(result), "hotels": result}), 200
 
 
# GET /api/categories
 
@landmarks_bp.route("/categories", methods=["GET"])
def get_categories():
    """Returns all distinct landmark categories for the preference panel.
 
    The user assigns a weight multiplier (0.4–1.6) per category.
    Weights are sent as category_weights in /api/solve requests and
    applied to landmark scores before the solver runs.
 
    Returns:
        200  { "categories": [ CategoryObject, ... ] }
        500  { "error": str }
 
    CategoryObject shape::
 
        {
            "id":             "historical",
            "label":          "Historical",
            "default_weight": 1.0
        }
    """
    try:
        landmarks = loadLandmarks(_LANDMARKS_CSV)
    except Exception as exc:
        return jsonify({"error": f"Failed to load categories: {exc}"}), 500
 
    seen:       set[str]  = set()
    categories: list[dict] = []
    for lm in landmarks:
        if lm.category not in seen:
            seen.add(lm.category)
            categories.append({
                "id":             lm.category,
                "label":          lm.category.replace("_", " ").title(),
                "default_weight": 1.0,
            })
 
    return jsonify({"categories": categories}), 200