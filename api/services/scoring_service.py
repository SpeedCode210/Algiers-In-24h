from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../algiers-lib"))

from dataclasses import replace
from models.landmark import Landmark
from models.problem import Problem

# ---------------------------------------------------------------------------
# Weight formula
# ---------------------------------------------------------------------------

_N_CATEGORIES = 5      # total number of categories
_COEF         = 1.2    # amplification coefficient


def _rank_to_weight(rank: int, n: int = _N_CATEGORIES, coef: float = _COEF) -> float:
    """
    Convert a category rank (1 = best, n = worst) to a score multiplier.

    Formula: W = ((n - r) / (n - 1) - 0.5) * coef + 1

    Args:
        rank:  Category rank assigned by the user (1 to n).
        n:     Total number of categories.
        coef:  Amplification coefficient controlling spread.

    Returns:
        Float weight multiplier to apply to interest_score.
    """
    return ((n - rank) / (n - 1) - 0.5) * coef + 1.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_category_weights(
    problem: Problem,
    category_ranks: dict[str, int],
) -> Problem:
    """
    Build a new Problem with landmark scores adjusted by user category preferences.

    The user ranks categories from 1 (most preferred) to 5 (least preferred).
    Each rank is converted to a weight multiplier via _rank_to_weight() and
    applied to the interest_score of every landmark in that category.

    Categories not present in category_ranks are left at weight 1.0 (neutral).
    The original Problem and its Landmark objects are never modified

    Args:
        problem:         The base Problem instance to adjust.
        category_ranks:  Dict mapping category name to rank (1–5).
                         e.g. {"historical": 1, "religious": 2}

    Returns:
        A new Problem instance with adjusted landmark scores,
        sharing the same hotel and travel matrix as the original.
    """
    # convert ranks to weights
    weights: dict[str, float] = {
        category: _rank_to_weight(rank)
        for category, rank in category_ranks.items()
    }

    # rebuild landmarks with adjusted scores — frozen dataclass requires replace()
    adjusted_landmarks: list[Landmark] = []
    for lm in problem.landmarks:
        weight = weights.get(lm.category, 1.0)
        adjusted_score = round(lm.interest_score * weight, 4)
        adjusted_landmarks.append(replace(lm, interest_score=adjusted_score))

    return Problem(
        hotel=problem.hotel,
        landmarks=adjusted_landmarks,
        time_budget=problem.time_budget,
        tour_day=problem.tour_day,
        start_time=problem.start_time,
    )


def compute_weights_from_ranks(category_ranks: dict[str, int]) -> dict[str, float]:
    """
    Utility function that exposes the rank-to-weight conversion.
    Useful for the frontend to preview weights before running a solver.

    Args:
        category_ranks: Dict mapping category name to rank (1–5).

    Returns:
        Dict mapping category name to computed weight multiplier.
    """
    return {
        category: round(_rank_to_weight(rank), 4)
        for category, rank in category_ranks.items()
    }