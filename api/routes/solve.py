from __future__ import annotations

import time
from typing import Any

from flask import Blueprint, jsonify, request

from schemas.request_schemas import (
    ALGORITHM_LABELS,
    COMPARISON_ALGORITHMS,
    ValidationError,
    validate_solve_all_request,
    validate_solve_request,
)
from services.problem_loader import build_problem
from utils.time import time_in_string

#  Optional service imports with graceful fallback 
try:
    from services.solver_service import run_solver
    _SOLVER_OK = True
except ImportError:
    _SOLVER_OK = False

try:
    from services.scoring_service import apply_category_weights
    _SCORING_OK = True
except ImportError:
    _SCORING_OK = False

try:
    from services.routing_service import get_route_geometry, get_leg_distances
    _ROUTING_OK = True
except ImportError:
    _ROUTING_OK = False

solve_bp = Blueprint("solve", __name__)


# Private helpers

def _error(message: str, code: int = 400):
    """Builds a JSON error response.

    Args:
        message: Human-readable error message.
        code: HTTP status code.

    Returns:
        Flask response tuple.
    """
    return jsonify({"error": message}), code


def _straight_line_geometry(ordered_stops, hotel) -> dict:
    """Builds a straight-line GeoJSON fallback when OSRM is unavailable.

    Args:
        ordered_stops: Landmarks in visit order.
        hotel: The hotel landmark (start and end point).

    Returns:
        GeoJSON LineString dict.
    """
    all_points = [hotel] + list(ordered_stops) + [hotel]
    return {
        "type": "LineString",
        "coordinates": [[lm.longitude, lm.latitude] for lm in all_points],
    }


def _get_road_data(tour, problem) -> tuple[dict, list[float]]:
    """Fetches road geometry and distances, falling back to straight lines.

    Args:
        tour: The solved Tour instance.
        problem: The Problem instance used during solving.

    Returns:
        Tuple of (GeoJSON geometry dict, list of distances in km per leg).
    """
    sim     = tour.simulation_cache()
    ordered = [entry.landmark for entry in sim.entries]

    if _ROUTING_OK and ordered:
        try:
            geometry      = get_route_geometry(ordered, problem.hotel)
            leg_distances = get_leg_distances(ordered, problem.hotel)
            return geometry, leg_distances
        except Exception:
            pass  # fall through to straight-line fallback

    return _straight_line_geometry(ordered, problem.hotel), [0.0] * len(ordered)


def _format_stops(sim_entries, problem, leg_distances: list[float]) -> list[dict]:
    """Converts simulation entries into JSON stop dicts.

    Args:
        sim_entries: ScheduleEntry list from tour.simulation_cache().
        problem: The Problem instance (for start_time).
        leg_distances: Real road distances per leg in km.

    Returns:
        Ordered list of stop dicts.
    """
    stops = []
    for i, entry in enumerate(sim_entries):
        prev_departure = (
            float(problem.start_time) if i == 0
            else float(sim_entries[i - 1].departure_time)
        )
        stops.append({
            "order":                    i + 1,
            "id":                       entry.landmark.id,
            "name":                     entry.landmark.name,
            "latitude":                 entry.landmark.latitude,
            "longitude":                entry.landmark.longitude,
            "category":                 entry.landmark.category,
            "interest_score":           entry.landmark.interest_score,
            "visit_duration_minutes":   entry.landmark.visit_duration,
            "arrival_time":             time_in_string(round(entry.arrival_time)),
            "visit_start_time":         time_in_string(entry.visit_start_time),
            "departure_time":           time_in_string(entry.departure_time),
            "waiting_minutes":          round(entry.waiting_time),
            "travel_from_prev_minutes": round(entry.arrival_time - prev_departure),
            "distance_from_prev_km":    leg_distances[i] if i < len(leg_distances) else 0.0,
        })
    return stops


def _build_result(algorithm: str, tour, problem, elapsed_ms: int) -> dict:
    """Assembles the full JSON result for one solver run.

    Args:
        algorithm: Algorithm ID (e.g. "grasp").
        tour: The solved Tour instance.
        problem: The Problem instance.
        elapsed_ms: Solver wall time in milliseconds.

    Returns:
        Fully formatted result dict ready for jsonify.
    """
    sim                     = tour.simulation_cache()
    geometry, leg_distances = _get_road_data(tour, problem)
    stops                   = _format_stops(sim.entries, problem, leg_distances)

    return {
        "algorithm":              algorithm,
        "algorithm_label":        ALGORITHM_LABELS.get(algorithm, algorithm),
        "total_score":            round(tour.total_score(), 2),
        "total_duration_minutes": round(sim.total_duration),
        "total_distance_km":      round(sum(leg_distances), 2),
        "num_landmarks":          len(tour),
        "is_valid":               sim.is_valid,
        "execution_time_ms":      elapsed_ms,
        "stops":                  stops,
        "road_geometry":          geometry,
        "hotel": {
            "id":        problem.hotel.id,
            "name":      problem.hotel.name,
            "latitude":  problem.hotel.latitude,
            "longitude": problem.hotel.longitude,
        },
    }


# POST /api/solve

@solve_bp.route("/solve", methods=["POST"])
def solve():
    """Runs one algorithm and returns the full optimised tour.

    Request body (JSON):
        algorithm        (str, required) — algorithm ID, e.g. "grasp"
        hotel_id         (str, required) — e.g. "hotel_aurassi"
        time_budget      (int, required) — minutes, 60-1440
        tour_day         (str, required) — e.g. "saturday"
        start_time       (str, optional) — "HH:MM", default "09:00"
        category_weights (obj, optional) — {"historical": 1.5, ...}
        algorithm_params (obj, optional) — {"iterations": 50, ...}

    Response (200):
        Full tour result with stops, road geometry, scores, timing.

    Response (400): Validation error with field name.
    Response (404): hotel_id not found.
    Response (503): Solver service not yet implemented.
    Response (500): Unexpected server error.
    """
    if not _SOLVER_OK:
        return _error(
            "Solver service not available — "
            "implement services/solver_service.py first.", 503
        )

    # Validate
    try:
        params = validate_solve_request(request.get_json(silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), 400

    # Build problem
    try:
        problem = build_problem(
            hotel_id    = params["hotel_id"],
            time_budget = params["time_budget"],
            tour_day    = params["tour_day"],
            start_time  = params["start_time"],
        )
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return _error(f"Failed to build problem: {exc}", 500)

    # Apply category weights
    if _SCORING_OK and params["category_weights"]:
        try:
            problem = apply_category_weights(problem, params["category_weights"])
        except Exception as exc:
            return _error(f"Scoring service error: {exc}", 500)

    # Run solver
    try:
        t0         = time.time()
        tour       = run_solver(params["algorithm"], problem, params["algorithm_params"])
        elapsed_ms = round((time.time() - t0) * 1000)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        return _error(f"Solver error: {exc}", 500)

    return jsonify(_build_result(params["algorithm"], tour, problem, elapsed_ms)), 200


# POST /api/solve/all

@solve_bp.route("/solve/all", methods=["POST"])
def solve_all():
    """Runs all comparison algorithms and returns a ranked leaderboard.

    Algorithms compared: greedy, sa, grasp, tabu, ga_penalty.
    All use the same configuration and default hyperparameters.

    Request body (JSON):
        hotel_id         (str, required)
        time_budget      (int, required)
        tour_day         (str, required)
        start_time       (str, optional) — default "09:00"
        category_weights (obj, optional)

    Response (200):
        {
            "rankings": [
                {
                    "rank": 1,
                    "algorithm": "grasp",
                    "algorithm_label": "GRASP",
                    "total_score": 42.5,
                    "num_landmarks": 6,
                    "total_duration_minutes": 387,
                    "execution_time_ms": 312
                },
                ...
            ],
            "best_algorithm": "grasp",
            "best_tour": { ...full tour result... },
            "all_results":   { "grasp": {...}, "sa": {...}, ... },
            "solver_errors": { "cplex": "CPLEX not installed" }  (if any)
        }
    """
    if not _SOLVER_OK:
        return _error(
            "Solver service not available — "
            "implement services/solver_service.py first.", 503
        )

    # Validate
    try:
        params = validate_solve_all_request(request.get_json(silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), 400

    # Build problem
    try:
        problem = build_problem(
            hotel_id    = params["hotel_id"],
            time_budget = params["time_budget"],
            tour_day    = params["tour_day"],
            start_time  = params["start_time"],
        )
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return _error(f"Failed to build problem: {exc}", 500)

    # Apply category weights
    if _SCORING_OK and params["category_weights"]:
        try:
            problem = apply_category_weights(problem, params["category_weights"])
        except Exception as exc:
            return _error(f"Scoring service error: {exc}", 500)

    # Run all algorithms
    all_results: dict[str, dict] = {}
    solver_errors: dict[str, str] = {}

    for algorithm in COMPARISON_ALGORITHMS:
        try:
            t0         = time.time()
            tour       = run_solver(algorithm, problem, {})
            elapsed_ms = round((time.time() - t0) * 1000)
            all_results[algorithm] = _build_result(algorithm, tour, problem, elapsed_ms)
        except Exception as exc:
            solver_errors[algorithm] = str(exc)

    if not all_results:
        return _error(f"All solvers failed: {solver_errors}", 500)

    # Rank by total score
    ranked = sorted(
        all_results.values(),
        key=lambda r: r["total_score"],
        reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    rankings = [
        {
            "rank":                   r["rank"],
            "algorithm":              r["algorithm"],
            "algorithm_label":        r["algorithm_label"],
            "total_score":            r["total_score"],
            "num_landmarks":          r["num_landmarks"],
            "total_duration_minutes": r["total_duration_minutes"],
            "execution_time_ms":      r["execution_time_ms"],
        }
        for r in ranked
    ]

    response: dict[str, Any] = {
        "rankings":       rankings,
        "best_algorithm": ranked[0]["algorithm"],
        "best_tour":      ranked[0],
        "all_results":    all_results,
    }
    if solver_errors:
        response["solver_errors"] = solver_errors

    return jsonify(response), 200