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

from solvers.greedy_solver import GreedySolver
from solvers.greedy_for_app import RandomGreedy, TimeGreedy
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from solvers.grasp_solver import GraspSolver
from solvers.tabu_solver import TabuSolver
from solvers.genetic_solver import GeneticSolver, TailoredGeneticSolver
from solvers.genetic_fitness import (
    PenaltyFitnessFunction,
    FeasibilityFitnessFunction,
)
from solvers.cplex_solver import CPLEXSolver

from schemas.request_schemas import COMPARISON_ALGORITHMS

def _make_greedy(problem: Problem, params: dict):
    return GreedySolver(problem, use_ratio=False)

def _make_greedy_ratio(problem: Problem, params: dict):
    return GreedySolver(problem, use_ratio=True)

def _make_greedy_nearest(problem: Problem, params: dict):
    return TimeGreedy(problem)

def _make_greedy_random(problem: Problem, params: dict):
    return RandomGreedy(problem)

def _make_sa(problem: Problem, params: dict):
    return SimulatedAnnealingSolver(
        problem,
        initial_temperature=float(params.get("initial_temperature", 10)),
        cooling_rate=float(params.get("cooling_rate", 0.95)),
        max_iterations=int(params.get("max_iterations", 10_000)),
    )

def _make_grasp(problem: Problem, params: dict):
    return GraspSolver(
        problem,
        iterations=int(params.get("iterations", 50)),
        alpha=float(params.get("alpha", 0.3)),
        max_local_search_iters=int(params.get("max_local_search_iters", 30)),
    )

def _make_tabu(problem: Problem, params: dict):
    return TabuSolver(
        problem,
        max_iterations=int(params.get("max_iterations", 200)),
        tabu_tenure=int(params.get("tabu_tenure", 20)),
        plateau_threshold=int(params.get("plateau_threshold", 10)),
        oscillation_slack=float(params.get("oscillation_slack", 240.0)),
    )

def _make_ga(problem: Problem, params: dict):
    return GeneticSolver(
        problem,
        fitness_function=PenaltyFitnessFunction(),
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
    return solver.solve()


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