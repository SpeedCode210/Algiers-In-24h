"""
Solver registry: defines which solver variants to run on each dataset group.

Each registry entry is a dict with:
    "name"    : str          -- display name
    "factory" : callable     -- factory(problem) -> solver instance
    "stochastic" : bool      -- True if solver is non-deterministic (needs multiple runs)
                               False for deterministic solvers (Greedy, CPLEX)

Stochastic solvers are run NUM_RUNS times (see runner.py) and results are
aggregated (mean score, std, best, worst).  Deterministic solvers run once.

CPLEX configuration:
    - Algiers    : 60 s time limit (regular solver, no ground truth)
    - Solomon-50 : 600 s time limit (ground truth on screened instances)
    - Solomon-100: NO CPLEX (uses Righini & Salani ground truth files)
    - Solomon-200: 60 s time limit
"""

from __future__ import annotations
from typing import Callable, Any

from models.problem import Problem
from solvers.greedy_solver import GreedySolver
from solvers.greedy_for_app import RandomGreedy
from solvers.grasp_solver import GraspSolver
from solvers.simulated_annealing_solver import (
    SimulatedAnnealingSolver, AcceptanceFunction, DecayFunction
)
from solvers.tabu_solver import TabuSolver
from solvers.genetic_solver import GeneticSolver, TailoredGeneticSolver
from solvers.genetic_fitness import (
    FeasibilityFitnessFunction, ScoreFitnessFunction
)

try:
    from solvers.cplex_solver import CPLEXSolver
    _HAS_CPLEX = True
except Exception:
    _HAS_CPLEX = False


# ---------------------------------------------------------------------------
# Helper type
# ---------------------------------------------------------------------------
SolverEntry = dict[str, Any]


def _make(name: str, factory: Callable[[Problem], Any],
          stochastic: bool = False, **kwargs) -> SolverEntry:
    entry = {"name": name, "factory": factory, "stochastic": stochastic}
    entry.update(kwargs)
    return entry


def _stochastic(name: str, factory: Callable[[Problem], Any], **kwargs) -> SolverEntry:
    """Shorthand for a stochastic solver entry."""
    return _make(name, factory, stochastic=True, **kwargs)


def _deterministic(name: str, factory: Callable[[Problem], Any], **kwargs) -> SolverEntry:
    """Shorthand for a deterministic solver entry."""
    return _make(name, factory, stochastic=False, **kwargs)


# ---------------------------------------------------------------------------
# A. ALGIERS -- full parameter sensitivity (3-4 variants per solver family)
# ---------------------------------------------------------------------------
ALGIERS_VARIANTS: list[SolverEntry] = [
    # Greedy (deterministic)
    _deterministic("Greedy-Score", lambda p: GreedySolver(p, use_ratio=False)),
    _deterministic("Greedy-Ratio", lambda p: GreedySolver(p, use_ratio=True)),

    # GRASP (stochastic) -- vary alpha and iteration count
    _stochastic("GRASP-a0.1-30it",  lambda p: GraspSolver(p, iterations=30,  alpha=0.1)),
    _stochastic("GRASP-a0.3-50it",  lambda p: GraspSolver(p, iterations=50,  alpha=0.3)),
    _stochastic("GRASP-a0.5-50it",  lambda p: GraspSolver(p, iterations=50,  alpha=0.5)),
    

    # Simulated Annealing (stochastic) -- vary acceptance, temperature, reheating
    _stochastic("SA-Boltzmann", lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.BOLTZMANN,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA-Cauchy",    lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.CAUCHY,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA-HighTemp",  lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.BOLTZMANN,
        initial_temperature=150, cooling_rate=0.99,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA-Reheat",    lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.BOLTZMANN,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0.95, max_iterations=80_000)),

    # Tabu (stochastic) -- vary tenure, iterations, oscillation slack
    _deterministic("Tabu-Default",    lambda p: TabuSolver(p)),
    _stochastic("Tabu-randTenure", lambda p: TabuSolver(p, tabu_tenure=None)),
    _deterministic("Tabu-LongTenure", lambda p: TabuSolver(p, tabu_tenure=40)),
    _deterministic("Tabu-LargeSlack", lambda p: TabuSolver(p, oscillation_slack=480.0)),

    # Genetic (stochastic) -- two fitness functions
    _stochastic("Genetic-Tailored",   lambda p: TailoredGeneticSolver(p, FeasibilityFitnessFunction())),
    _stochastic("Genetic-Score",      lambda p: GeneticSolver(p, ScoreFitnessFunction())),
]

if _HAS_CPLEX:
    ALGIERS_VARIANTS.append(
        _deterministic("CPLEX-60s", lambda p: CPLEXSolver(p, time_limit=60))
    )


# ---------------------------------------------------------------------------
# A2. ALGIERS -- one best variant per solver family (for benchmark comparison)
# ---------------------------------------------------------------------------
ALGIERS_BEST_VARIANTS: list[SolverEntry] = [
    _deterministic("Greedy-Ratio",  lambda p: GreedySolver(p, use_ratio=True)),
    _deterministic("Greedy-Score",  lambda p: GreedySolver(p, use_ratio=False)),
    _stochastic("Greedy-Random",    lambda p: RandomGreedy(p)),
    _stochastic("GRASP",       lambda p: GraspSolver(p, iterations=50, alpha=0.3)),
    _stochastic("SA-Boltzmann", lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.BOLTZMANN,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA-Cauchy",    lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.CAUCHY,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA",           lambda p: SimulatedAnnealingSolver(
        p, initial_temperature=70, cooling_rate=0.97, max_iterations=60_000)),
    _deterministic("Tabu",          lambda p: TabuSolver(p)),
    _stochastic("Tabu-Random",      lambda p: TabuSolver(p, tabu_tenure=None)),
    _stochastic("Genetic-Tailored", lambda p: TailoredGeneticSolver(
        p, FeasibilityFitnessFunction())),
    _stochastic("Genetic-Score", lambda p: GeneticSolver(
        p, ScoreFitnessFunction()), time_limit=200),
]

if _HAS_CPLEX:
    ALGIERS_BEST_VARIANTS.append(
        _deterministic("CPLEX", lambda p: CPLEXSolver(p, time_limit=60))
    )


# ---------------------------------------------------------------------------
# B. SOLOMON-50 -- Boltzmann + Cauchy (instead of SA), Tabu + Tabu-Random,
#    Genetic-Tailored + Genetic-Score(200s), CPLEX (ground truth)
# ---------------------------------------------------------------------------
SOLOMON_50_VARIANTS: list[SolverEntry] = [
    _deterministic("Greedy-Ratio",  lambda p: GreedySolver(p, use_ratio=True)),
    _deterministic("Greedy-Score",  lambda p: GreedySolver(p, use_ratio=False)),
    _stochastic("Greedy-Random",    lambda p: RandomGreedy(p)),
    _stochastic("GRASP",       lambda p: GraspSolver(p, iterations=50, alpha=0.3)),
    _stochastic("SA-Boltzmann", lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.BOLTZMANN,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA-Cauchy",    lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.CAUCHY,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _deterministic("Tabu",          lambda p: TabuSolver(p)),
    _stochastic("Tabu-Random",      lambda p: TabuSolver(p, tabu_tenure=None)),
    _stochastic("Genetic-Tailored", lambda p: TailoredGeneticSolver(
        p, FeasibilityFitnessFunction())),
    _stochastic("Genetic-Score", lambda p: GeneticSolver(
        p, ScoreFitnessFunction()), time_limit=200),
]

if _HAS_CPLEX:
    SOLOMON_50_VARIANTS.append(
        _deterministic("CPLEX", lambda p: CPLEXSolver(p, time_limit=600))
    )


# ---------------------------------------------------------------------------
# C. SOLOMON-100 -- one best variant per family (NO CPLEX)
#    Uses Righini & Salani bestPossible ground truth files.
# ---------------------------------------------------------------------------
SOLOMON_100_VARIANTS: list[SolverEntry] = [
    _deterministic("Greedy-Ratio",  lambda p: GreedySolver(p, use_ratio=True)),
    _deterministic("Greedy-Score",  lambda p: GreedySolver(p, use_ratio=False)),
    _stochastic("Greedy-Random",    lambda p: RandomGreedy(p)),
    _stochastic("GRASP",    lambda p: GraspSolver(p, iterations=50, alpha=0.3)),
    _stochastic("SA-Boltzmann", lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.BOLTZMANN,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _stochastic("SA-Cauchy",    lambda p: SimulatedAnnealingSolver(
        p, acceptance_criterion=AcceptanceFunction.CAUCHY,
        initial_temperature=70, cooling_rate=0.97,
        reheating_rate=0, max_iterations=80_000)),
    _deterministic("Tabu",     lambda p: TabuSolver(p)),
    _stochastic("Tabu-Random",     lambda p: TabuSolver(p, tabu_tenure=None)),
    _stochastic("Genetic-Tailored",   lambda p: TailoredGeneticSolver(
        p, FeasibilityFitnessFunction())),
    _stochastic("Genetic-Score",      lambda p: GeneticSolver(
        p, ScoreFitnessFunction())),
]


# ---------------------------------------------------------------------------
# D. SOLOMON-200 -- single best variant per family + CPLEX 60s
# ---------------------------------------------------------------------------
SOLOMON_200_VARIANTS: list[SolverEntry] = [
    _deterministic("Greedy",   lambda p: GreedySolver(p, use_ratio=True)),
    _stochastic("GRASP",    lambda p: GraspSolver(p, iterations=30, alpha=0.3)),
    _stochastic("SA",       lambda p: SimulatedAnnealingSolver(
        p, initial_temperature=70, cooling_rate=0.97, max_iterations=40_000)),
    _stochastic("Tabu",     lambda p: TabuSolver(p, max_iterations=150)),
    _stochastic("Genetic-Tailored",   lambda p: TailoredGeneticSolver(
        p, FeasibilityFitnessFunction(), regenerations=50)),
    _stochastic("Genetic-Score",      lambda p: GeneticSolver(
        p, ScoreFitnessFunction())),
]

if _HAS_CPLEX:
    SOLOMON_200_VARIANTS.append(
        _deterministic("CPLEX-60s", lambda p: CPLEXSolver(p, time_limit=60))
    )
