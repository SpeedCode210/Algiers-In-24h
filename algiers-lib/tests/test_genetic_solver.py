from __future__ import annotations

import os
import sys
import time
from typing import Any

# Ensure the library root is on the path so solver imports resolve.
_LIBRARY_ROOT = os.path.normpath(
     os.path.abspath(os.path.join(os.path.dirname(__file__), "../../algiers-lib"))
 )
if _LIBRARY_ROOT not in {os.path.normpath(os.path.abspath(path)) for path in sys.path}:
     sys.path.insert(0, _LIBRARY_ROOT)
     
from models.problem import Problem
from models.tour import Tour
from utils.time import time_in_string
from solvers.genetic_mutation import Mutation
from solvers.genetic_crossover import Crossover
from solvers.genetic_fitness import (FeasibilityFitnessFunction, ScoreFitnessFunction)
from solvers.genetic_selection import Selection 
from solvers.genetic_augmented_representation import AugmentedRepresentation
from solvers.genetic_solver import GeneticSolver , TailoredGeneticSolver


def load_dataset_problem() -> Problem:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    landmarks = loadLandmarks(str(data_dir / "data.csv"))
    hotel = loadHotel(str(data_dir / "hotel.csv"))

    return Problem(hotel=hotel, landmarks=landmarks, time_budget=500, tour_day=Day.MONDAY)


def format_route(tour: Tour) -> str:
    if not tour.visited_landmarks:
        return "Hotel -> Hotel"
    return "Hotel -> " + " -> ".join(
        landmark.name for landmark in tour.visited_landmarks
    ) + " -> Hotel"


def format_time_window_feedback(entry, tour_day: Day) -> str:
    day_slots = entry.landmark.schedule.get_slots(tour_day)
    if not day_slots:
        return "closed on tour day"

    matching_slot = next(
        (
            slot for slot in day_slots
            if slot.contains(entry.visit_start_time, entry.landmark.visit_duration)
        ),
        None,
    )

    if matching_slot is None:
        return "no valid slot for this visit"

    arrival = round(entry.arrival_time)
    if arrival < matching_slot.open_time:
        wait = int(entry.visit_start_time - entry.arrival_time)
        return (
            f"waited {wait} min until slot {time_in_string(matching_slot.open_time)}-"
            f"{time_in_string(matching_slot.close_time)}"
        )

    if matching_slot.contains(arrival, entry.landmark.visit_duration):
        return f"arrived inside slot {time_in_string(matching_slot.open_time)}-{time_in_string(matching_slot.close_time)}"

    return f"started in slot {time_in_string(matching_slot.open_time)}-{time_in_string(matching_slot.close_time)}"


def format_route_details(tour: Tour) -> str:
    simulation = tour.simulation_cache()
    if not simulation.entries:
        return "No visited landmarks."

    lines = []
    for entry in simulation.entries:
        landmark = entry.landmark
        day_slots = landmark.schedule.get_slots(tour.problem.tour_day)
        slot_info = ", ".join(
            f"{slot.open_time}-{slot.close_time}"
            for slot in day_slots
        ) or "closed"

        lines.append(
            f"{landmark.name} | arrival: {entry.arrival_time} "
            f"| start: {entry.visit_start_time} "
            f"| depart: {entry.departure_time} "
            f"| slots: {slot_info} "
            f"| window: {format_time_window_feedback(entry, tour.problem.tour_day)}"
        )

    return "\n".join(lines)


def test_genetic_solver_runs_with_penalty_fitness() -> None:
    problem = load_dataset_problem()
    solver = GeneticSolver(
        problem=problem,
        fitness_function=ScoreFitnessFunction(),
        regenerations=100,
        population_size=20,
        mutation_rate=0.1,
        crossover_method="order",
    )

def _make_grasp(problem: Problem, params: dict):
    return GraspSolver(
        problem,
        iterations=int(params.get("iterations", 50)),
        alpha=float(params.get("alpha", 0.3)),
        max_local_search_iters=int(params.get("max_local_search_iters", 30)),
    )


def test_genetic_solver_runs_with_infeasibility_fitness() -> None:
    problem = load_dataset_problem()
    solver = GeneticSolver(
        problem=problem,
        fitness_function=ScoreFitnessFunction(),
        regenerations=1,
        population_size=2,
        mutation_rate=0.1,
        crossover_method="order",
    )

def _make_ga(problem: Problem, params: dict):
    return GeneticSolver(
        problem,
        fitness_function=ScoreFitnessFunction(),
        regenerations=int(params.get("regenerations", 1000)),
        population_size=int(params.get("population_size", 1075)),
        mutation_rate=float(params.get("mutation_rate", 0.8)),
        crossover_method="order",
    )

def _make_ga_tailored(problem: Problem, params: dict):
    return TailoredGeneticSolver(
        problem,
        fitness_function=FeasibilityFitnessFunction(),
        regenerations=int(params.get("regenerations", 1000)),
        population_size=int(params.get("population_size", 1075)),
        mutation_rate=float(params.get("mutation_rate", 0.8)),
        crossover_method="tailored",
    )

def _make_cplex(problem: Problem, params: dict):
    return CPLEXSolver(problem)


_SOLVER_REGISTRY: dict[str, Any] = {
    "greedy":         _make_greedy,
    "greedy_ratio":   _make_greedy_ratio,
    "greedy_nearest": _make_greedy_nearest,
    "greedy_random":  _make_greedy_random,
    "sa":             _make_sa,
    "grasp":          _make_grasp,
    "tabu":           _make_tabu,
    "ga":             _make_ga,
    "ga_tailored":    _make_ga_tailored,
    "cplex":          _make_cplex,
}


def run_solver(algorithm: str, problem: Problem, params: dict) -> Tour:
    """Instantiate and run the requested solver, returning the best Tour.

    Args:
        algorithm: Frontend algorithm ID (e.g. ``"grasp"``).
        problem:   Fully configured Problem instance (scores already adjusted).
        params:    Algorithm hyperparameters from the request body. May be
                   empty — each factory supplies sensible defaults.

    Returns:
        Best Tour found by the solver.

    Raises:
        ValueError: If ``algorithm`` is not in the registry.
        RuntimeError: If the solver raises an unexpected exception.
    """
    factory = _SOLVER_REGISTRY.get(algorithm)
    if factory is None:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Valid options: {sorted(_SOLVER_REGISTRY)}"
        )

    solver = factory(problem, params or {})
    try:
        return solver.solve()
    except Exception as exc:
        raise RuntimeError(
            f"Solver '{algorithm}' raised an unexpected exception."
        ) from exc


def run_all_solvers(
    problem: Problem,
) -> tuple[list[dict], dict[str, str]]:
    """Run every algorithm listed in COMPARISON_ALGORITHMS.

    Runs each solver sequentially with default hyperparameters, times
    each run, and collects results.  Solver failures are caught and
    reported in the errors dict rather than crashing the whole request.

    Args:
        problem: Fully configured Problem instance.

    Returns:
        A tuple of:
            - ``results``: list of dicts, each containing ``algorithm``,
              ``tour``, and ``elapsed_ms``.  Sorted by total_score desc.
            - ``errors``: dict mapping algorithm ID to error message for
              any solver that raised an exception.
    """
    results: list[dict] = []
    errors:  dict[str, str] = {}

    for algorithm in COMPARISON_ALGORITHMS:
        if algorithm not in _SOLVER_REGISTRY:
            errors[algorithm] = f"Algorithm '{algorithm}' not in registry."
            continue
        try:
            t0   = time.time()
            tour = run_solver(algorithm, problem, {})
            elapsed_ms = round((time.time() - t0) * 1000)
            results.append({
                "algorithm":  algorithm,
                "tour":       tour,
                "elapsed_ms": elapsed_ms,
            })
        except Exception as exc:
            errors[algorithm] = str(exc)
    results.sort(key=lambda r: r["tour"].total_score(), reverse=True)
    return results, errors