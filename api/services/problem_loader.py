from __future__ import annotations


import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../algiers-lib"))

from models.landmark import Day
from models.problem import Problem

_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_LANDMARKS_PATH = os.path.join(_BASE_DIR, "../../algiers-lib/data/data.csv")
_HOTEL_PATH     = os.path.join(_BASE_DIR, "../../algiers-lib/data/hotel.csv")


_BASE_PROBLEM: Problem | None = None


def get_base_problem() -> Problem:
    """
    Load and cache the base Problem instance from CSV files.
    Expensive on first call (reads CSVs + builds travel matrix).
    Subsequent calls return the cached instance instantly.
    """
    global _BASE_PROBLEM
    if _BASE_PROBLEM is None:
        _BASE_PROBLEM = Problem.LoadProblem(
            landmarks_path=_LANDMARKS_PATH,
            hotel_path=_HOTEL_PATH,
            time_budget=480,
            tour_day=Day.MONDAY,
        )
    return _BASE_PROBLEM


def build_problem(
    time_budget: int,
    tour_day: str,
    start_time: int = 540,
) -> Problem:
    """
    Build a user-configured Problem using the cached landmarks and hotel.
    Creates a new Problem instance with the user's specific parameters
    without reloading or recomputing the travel matrix from scratch.
    """
    base = get_base_problem()
    day  = Day.from_string(tour_day)

    return Problem(
        hotel=base.hotel,
        landmarks=base.landmarks,
        time_budget=time_budget,
        tour_day=day,
        start_time=start_time,
    )


def get_all_landmarks() -> list:
    """
    Return all landmarks from the base problem.
    Used by the landmarks route to populate the map on page load.
    """
    return get_base_problem().landmarks


def get_hotel():
    """
    Return the hotel landmark.
    Used by the landmarks route as the map starting point.
    """
    return get_base_problem().hotel


def get_all_categories() -> list[str]:
    """
    Return a sorted list of unique landmark categories.
    Used by the landmarks route to build the category filter panel.
    """
    landmarks = get_all_landmarks()
    return sorted({lm.category for lm in landmarks})