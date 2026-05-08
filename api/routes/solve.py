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
 
# Service imports with graceful fallback
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
    from services.routing_service import get_leg_distances, get_route_geometry
    _ROUTING_OK = True
except ImportError:
    _ROUTING_OK = False
 
solve_bp = Blueprint("solve", __name__)
 
 
# Private helpers
 
def _err(message: str, code: int = 400):
    """Returns a JSON error response tuple.
 
    Args:
        message: Human-readable error description.
        code: HTTP status code.
 
    Returns:
        Flask (Response, int) tuple.
    """
    return jsonify({"error": message}), code
 
 
def _straight_line_fallback(ordered_stops, hotel) -> dict:
    """Builds a straight-line GeoJSON when OSRM is unavailable.
 
    Args:
        ordered_stops: Landmarks in visit order.
        hotel: Start/end hotel Landmark.
 
    Returns:
        GeoJSON LineString dict.
    """
    coords = (
        [[hotel.longitude, hotel.latitude]]
        + [[lm.longitude, lm.latitude] for lm in ordered_stops]
        + [[hotel.longitude, hotel.latitude]]
    )
    return {"type": "LineString", "coordinates": coords}
 
 
def _fetch_road_data(tour, problem) -> tuple[dict, list[float]]:
    """Fetches road geometry and leg distances, with straight-line fallback.
 
    Args:
        tour: Solved Tour instance.
        problem: Problem instance used during solving.
 
    Returns:
        (GeoJSON geometry dict, list of km distances per leg).
    """
    sim     = tour.simulation_cache()
    ordered = [entry.landmark for entry in sim.entries]
 
    if _ROUTING_OK and ordered:
        try:
            return (
                get_route_geometry(ordered, problem.hotel),
                get_leg_distances(ordered, problem.hotel),
            )
        except Exception:
            pass  # fall through to straight-line
 
    return _straight_line_fallback(ordered, problem.hotel), [0.0] * len(ordered)
 
 
def _format_stops(sim_entries, problem, leg_distances: list[float]) -> list[dict]:
    """Converts simulation schedule entries into JSON stop dicts.
 
    Args:
        sim_entries: List of ScheduleEntry from tour.simulation_cache().
        problem: The Problem instance (for start_time reference).
        leg_distances: Real road distances per leg in km.
 
    Returns:
        Ordered list of stop dicts for the itinerary panel.
    """
    stops = []
    for i, entry in enumerate(sim_entries):
        prev_dep = (
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
            "travel_from_prev_minutes": round(entry.arrival_time - prev_dep),
            "distance_from_prev_km":    (
                leg_distances[i] if i < len(leg_distances) else 0.0
            ),
        })
    return stops
 
 
def _build_result(
    algorithm: str,
    tour,
    problem,
    elapsed_ms: int,
) -> dict:
    """Assembles the complete JSON result for one solver run.
 
    Args:
        algorithm: Algorithm ID string (e.g. "grasp").
        tour: Solved Tour instance.
        problem: Problem instance used during solving.
        elapsed_ms: Solver wall-clock time in milliseconds.
 
    Returns:
        Fully formatted result dict ready to jsonify.
    """
    sim                     = tour.simulation_cache()
    geometry, leg_distances = _fetch_road_data(tour, problem)
 
    return {
        "algorithm":              algorithm,
        "algorithm_label":        ALGORITHM_LABELS.get(algorithm, algorithm),
        "total_score":            round(tour.total_score(), 2),
        "total_duration_minutes": round(sim.total_duration),
        "total_distance_km":      round(sum(leg_distances), 2),
        "num_landmarks":          len(tour),
        "is_valid":               sim.is_valid,
        "execution_time_ms":      elapsed_ms,
        "hotel": {
            "id":        problem.hotel.id,
            "name":      problem.hotel.name,
            "latitude":  problem.hotel.latitude,
            "longitude": problem.hotel.longitude,
        },
        "stops":        _format_stops(sim.entries, problem, leg_distances),
        "road_geometry": geometry,
    }
 
 
# POST /api/solve
 
@solve_bp.route("/solve", methods=["POST"])
def solve():
    """Runs one algorithm and returns the optimised tour.
 
    Request body (JSON):
        algorithm        str  required   e.g. "grasp"
        hotel_id         str  required   e.g. "hotel_aurassi"
        time_budget      int  required   minutes, 60–1440
        tour_day         str  required   e.g. "saturday"
        start_time       str  optional   "HH:MM", default "09:00"
        category_weights obj  optional   {"historical": 1.5, ...}
        algorithm_params obj  optional   {"iterations": 50, ...}
 
    Responses:
        200  Full tour result (stops, road_geometry, scores, timing).
        400  { "error": str, "field": str }   validation failure.
        404  { "error": str }                 hotel_id not found.
        503  { "error": str }                 service not yet implemented.
        500  { "error": str }                 unexpected server error.
    """
    if not _SOLVER_OK:
        return _err(
            "Solver service unavailable — "
            "implement services/solver_service.py.", 503
        )
 
    # 1. Validate
    try:
        params = validate_solve_request(request.get_json(silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), 400
 
    # 2. Build Problem
        problem = build_problem(
            hotel_id    = params["hotel_id"],
            time_budget = params["time_budget"],
            tour_day    = params["tour_day"],
            start_time  = params["start_time"],
        )
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return _err(f"Problem construction failed: {exc}", 500)
 
    # 3. Apply category weights 
    if _SCORING_OK and params["category_weights"]:
        try:
            problem = apply_category_weights(
                problem, params["category_weights"]
            )
        except Exception as exc:
            return _err(f"Scoring service error: {exc}", 500)
 
    # 4. Run solver 
    try:
        t0         = time.time()
        tour       = run_solver(
            params["algorithm"],
            problem,
            params["algorithm_params"],
        )
        elapsed_ms = round((time.time() - t0) * 1000)
    except ValueError as exc:
        return _err(str(exc), 400)
    except Exception as exc:
        return _err(f"Solver error: {exc}", 500)
 
    # 5. Return 
    return jsonify(
        _build_result(params["algorithm"], tour, problem, elapsed_ms)
    ), 200
 
 
# POST /api/solve/all
 
@solve_bp.route("/solve/all", methods=["POST"])
def solve_all():
    """Runs all comparison algorithms and returns a ranked leaderboard.
 
    Runs: greedy, sa, grasp, tabu, ga_penalty (defined in COMPARISON_ALGORITHMS).
    All use the same problem configuration with default hyperparameters.
 
    Request body (JSON):
        hotel_id         str  required
        time_budget      int  required   minutes, 60–1440
        tour_day         str  required   e.g. "saturday"
        start_time       str  optional   "HH:MM", default "09:00"
        category_weights obj  optional   {"historical": 1.5, ...}
 
    Response (200)::
 
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
            "best_tour":   { ...full result... },
            "all_results": { "grasp": {...}, "sa": {...}, ... },
            "solver_errors": { "cplex": "not installed" }
        }
 
    Responses:
        200  Ranked comparison result.
        400  Validation error.
        404  hotel_id not found.
        503  Solver service unavailable.
        500  All solvers failed.
    """
    if not _SOLVER_OK:
        return _err(
            "Solver service unavailable — "
            "implement services/solver_service.py.", 503
        )
 
    # 1. Validate 
    try:
        params = validate_solve_all_request(request.get_json(silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), 400
 
    # 2. Build Problem 
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
        return _err(f"Problem construction failed: {exc}", 500)
 
    # 3. Apply category weights 
    if _SCORING_OK and params["category_weights"]:
        try:
            problem = apply_category_weights(
                problem, params["category_weights"]
            )
        except Exception as exc:
            return _err(f"Scoring service error: {exc}", 500)
 
    # 4. Run all comparison algorithms 
    all_results:   dict[str, dict] = {}
    solver_errors: dict[str, str]  = {}
 
    for algorithm in COMPARISON_ALGORITHMS:
        try:
            t0         = time.time()
            tour       = run_solver(algorithm, problem, {})
            elapsed_ms = round((time.time() - t0) * 1000)
            all_results[algorithm] = _build_result(
                algorithm, tour, problem, elapsed_ms
            )
        except Exception as exc:
            solver_errors[algorithm] = str(exc)
 
    if not all_results:
        return _err(f"All solvers failed: {solver_errors}", 500)
 
    # 5. Rank and return 
    ranked = sorted(
        all_results.values(),
        key=lambda r: r["total_score"],
        reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
 
    response: dict[str, Any] = {
        "rankings": [
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
        ],
        "best_algorithm": ranked[0]["algorithm"],
        "best_tour":      ranked[0],
        "all_results":    all_results,
    }
    if solver_errors:
        response["solver_errors"] = solver_errors
 
    return jsonify(response), 200