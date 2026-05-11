from __future__ import annotations


import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../algiers-lib"))

from models.landmark import Day, Landmark, loadAllHotels, loadLandmarks
from models.problem import Problem

_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_LANDMARKS_PATH = os.path.join(_BASE_DIR, "../../algiers-lib/data/data.csv")
_HOTEL_PATH     = os.path.join(_BASE_DIR, "../../algiers-lib/data/hotel.csv")


_landmarks_cache: list[Landmark] | None     = None
_hotels_index:    dict[str, Landmark] | None = None


def build_problem(
    hotel_id: str,
    time_budget: int,
    tour_day: str,
    start_time: int = 540,
) -> Problem:
    """
    Build a user-configured Problem using the cached landmarks and hotel.
    Creates a new Problem instance with the user's specific parameters
    without reloading or recomputing the travel matrix from scratch.
    """
    landmarks = get_all_landmarks()
    hotels    = get_hotel()
 
    if hotel_id not in hotels:
        available = sorted(hotels.keys())
        raise KeyError(
            f"Hotel '{hotel_id}' not found. "
            f"Available IDs: {available}"
        )

    return Problem(
        hotel= hotels[hotel_id],
        landmarks=landmarks,
        time_budget=time_budget,
        tour_day=Day.from_string(tour_day),
        start_time=start_time,
    )


def get_all_landmarks() -> list:
    """
    Return all landmarks from the base problem.
    Used by the landmarks route to populate the map on page load.
    """
    global _landmarks_cache
    if _landmarks_cache is None:
        if not os.path.exists(_LANDMARKS_PATH):
            raise FileNotFoundError(
                f"data.csv not found at: {_LANDMARKS_PATH}"
            )
        _landmarks_cache = loadLandmarks(_LANDMARKS_PATH)
    return _landmarks_cache


def get_hotel():
    """
    Return the hotel landmark.
    Used by the landmarks route as the map starting point.
    """
    global _hotels_index
    if _hotels_index is None:
        if not os.path.exists(_HOTEL_PATH):
            raise FileNotFoundError(
                f"hotel.csv not found at: {_HOTEL_PATH}"
            )
        all_hotels     = loadAllHotels(_HOTEL_PATH)
        _hotels_index  = {h.id: h for h in all_hotels}
    return _hotels_index
