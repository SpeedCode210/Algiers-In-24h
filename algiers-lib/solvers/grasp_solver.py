from __future__ import annotations

import random
from typing import Optional

from models.landmark import Landmark
from models.problem import Problem
from models.tour import Tour
from .solver import Solver


class GraspSolver(Solver):
    """Solves the TCOP using the GRASP metaheuristic.

    Each GRASP iteration consists of two phases:

    1. **Randomized greedy construction**: builds a tour by repeatedly
       picking from a Restricted Candidate List (RCL) — the top-alpha
       fraction of candidates by score-to-time ratio — rather than
       always picking the single best candidate. This produces diverse
       starting solutions across iterations.

    2. **Local search**: improves the constructed tour using three
       neighbourhood operators (replace, insert, swap) until no
       improvement is found or the iteration cap is reached.

    The globally best valid tour across all iterations is returned.

    Attributes:
        problem: The problem instance to solve.
        iterations: Number of construction + local search cycles.
        alpha: Greediness parameter in [0.0, 1.0]. 0.0 = pure greedy,
            1.0 = pure random. Values around 0.2–0.4 work best.
        max_local_search_iters: Hard cap on local search outer cycles.
            Prevents performance blow-up on large instances.
    """

    def __init__(
        self,
        problem: Problem,
        iterations: int = 50,
        alpha: float = 0.3,
        max_local_search_iters: int = 30,
    ) -> None:
        """Initialises the GRASP solver.

        Args:
            problem: The fully loaded Problem instance.
            iterations: Number of construction + local search cycles.
                More iterations improve quality at the cost of runtime.
                Recommended range: 20–100.
            alpha: Controls randomness during construction. Must be in
                [0.0, 1.0]. Recommended starting value: 0.3.
            max_local_search_iters: Hard cap on the number of outer
                improvement cycles in local search. Prevents performance
                blow-up on datasets with many landmarks. Default: 30.

        Raises:
            ValueError: If alpha is outside [0.0, 1.0], iterations < 1,
                or max_local_search_iters < 1.
        """
        super().__init__(problem)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}.")
        if max_local_search_iters < 1:
            raise ValueError(
                f"max_local_search_iters must be >= 1, got {max_local_search_iters}."
            )

        self.iterations: int = iterations
        self.alpha: float = alpha
        self.max_local_search_iters: int = max_local_search_iters

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tour:
        """Runs GRASP and returns the best tour found.

        Executes `self.iterations` cycles of randomized construction
        followed by local search. Tracks and returns the globally best
        valid tour across all iterations.

        Returns:
            The best valid Tour found. Returns an empty tour only if no
            valid solution was found in any iteration, which should not
            happen on a well-formed dataset.
        """
        best_tour: Tour = self.problem.create_empty_tour()

        for _ in range(self.iterations):
            tour = self._construction_phase()
            tour = self._local_search(tour)

            if tour.is_valid() and tour.total_score() > best_tour.total_score():
                best_tour = tour

        return best_tour

    # ------------------------------------------------------------------
    # Phase 1 — Randomized greedy construction
    # ------------------------------------------------------------------

    def _construction_phase(self) -> Tour:
        """Builds a feasible tour using randomized greedy insertion.

        At each step, scores all feasible unvisited landmarks by their
        interest-score-to-time-cost ratio, builds a Restricted Candidate
        List (RCL) from the top-scoring ones, and picks one randomly.

        Landmarks that make the tour infeasible when added are permanently
        rejected for the rest of this construction phase. This guarantees
        the loop terminates in at most N iterations (one per landmark).

        Returns:
            A feasible Tour. May be empty on very tight time budgets.
        """
        tour = self.problem.create_empty_tour()
        permanently_rejected: set[str] = set()

        while True:
            candidates = [
                lm for lm in self.problem.feasible_candidates(tour)
                if lm.id not in permanently_rejected
            ]
            if not candidates:
                break

            scored = self._score_candidates(candidates, tour)
            if not scored:
                break

            rcl = self._build_rcl(scored)
            chosen = random.choice(rcl)

            tour.add_landmark(chosen)
            if not tour.is_valid():
                tour.remove_landmark(chosen)
                permanently_rejected.add(chosen.id)

        return tour

    def _score_candidates(
        self,
        candidates: list[Landmark],
        tour: Tour,
    ) -> list[tuple[float, Landmark]]:
        """Scores each candidate by its interest-score-to-time-cost ratio.

        The greedy criterion is::

            ratio = interest_score / (travel_time_to_here + visit_duration)

        Higher ratio = more interest collected per minute spent.
        Travel time is measured from the last visited landmark,
        or from the hotel if the tour is empty.

        Args:
            candidates: Unvisited landmarks open on the tour day.
            tour: The current partial tour.

        Returns:
            List of (ratio, Landmark) tuples sorted descending by ratio.
        """
        current_position = (
            tour.visited_landmarks[-1]
            if tour.visited_landmarks
            else self.problem.hotel
        )

        scored: list[tuple[float, Landmark]] = []
        for landmark in candidates:
            travel = self.problem.travel_time(current_position, landmark)
            time_cost = travel + landmark.visit_duration
            if time_cost <= 0:
                continue
            ratio = landmark.interest_score / time_cost
            scored.append((ratio, landmark))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def _build_rcl(
        self,
        scored: list[tuple[float, Landmark]],
    ) -> list[Landmark]:
        """Builds the Restricted Candidate List from scored candidates.

        Includes all landmarks whose ratio falls within the alpha-defined
        range of the best::

            threshold = best - alpha × (best - worst)
            RCL = { c : ratio(c) >= threshold }

        alpha = 0 → RCL contains only the best (pure greedy).
        alpha = 1 → RCL contains all candidates (pure random).

        Args:
            scored: Non-empty list of (ratio, Landmark) pairs sorted
                descending. Must contain at least one element.

        Returns:
            Non-empty list of Landmark objects eligible for selection.
        """
        best_score = scored[0][0]
        worst_score = scored[-1][0]
        threshold = best_score - self.alpha * (best_score - worst_score)
        rcl = [lm for ratio, lm in scored if ratio >= threshold]

        if not rcl:  # floating-point safety guard
            rcl = [scored[0][1]]

        return rcl

    # ------------------------------------------------------------------
    # Phase 2 — Local search
    # ------------------------------------------------------------------

    def _local_search(self, tour: Tour) -> Tour:
        """Improves a tour using three neighbourhood operators.

        Applies replace, insert, and swap operators in priority order
        until no operator finds improvement or the iteration cap is hit.

        Operator priority rationale:
        - Replace and insert directly increase score (primary goal) and
          run first so they dominate the search when possible.
        - Swap only reorders landmarks to free time budget, enabling
          future inserts. It runs last to avoid monopolising the loop.

        The ``max_local_search_iters`` cap is critical for performance:
        without it, a long chain of duration-reducing swaps can make
        the algorithm appear to hang on datasets of 20+ landmarks.

        Args:
            tour: The tour to improve. Modified in-place.

        Returns:
            The improved Tour (always at least as good as the input).
        """
        for _ in range(self.max_local_search_iters):

            if self._try_replace(tour):
                continue

            if self._try_insert(tour):
                continue

            if self._try_swap(tour):
                continue

            # No operator improved anything — local optimum reached
            break

        return tour

    def _try_swap(self, tour: Tour) -> bool:
        """Tries pairwise swaps to improve time-window alignment.

        A swap changes the visiting order of two landmarks without
        changing which ones are visited. It does not improve score
        directly but can free time budget, enabling future insertions.

        Uses first-improvement strategy: accepts the first swap that
        strictly reduces total tour duration. This avoids the O(N²)
        best-improvement scan that caused performance blow-up in the
        previous version.

        Args:
            tour: The tour to improve in-place.

        Returns:
            True if a duration-reducing swap was found and applied.
        """
        n = len(tour)
        if n < 2:
            return False

        current_duration = tour.simulation_cache().total_duration

        for i in range(n):
            for j in range(i + 1, n):
                candidate = tour.copy()
                candidate.swap_by_index(i, j)

                if not candidate.is_valid():
                    continue

                if candidate.simulation_cache().total_duration < current_duration:
                    tour.swap_by_index(i, j)
                    return True

        return False

    def _try_replace(self, tour: Tour) -> bool:
        """Tries replacing each visited landmark with each unvisited one.

        Evaluates all (visited × unvisited) pairs and applies the
        replacement with the greatest score improvement.

        Args:
            tour: The tour to improve in-place.

        Returns:
            True if an improving replacement was found and applied.
        """
        unvisited = self.problem.unvisited_landmarks(tour)
        if not unvisited or not tour.visited_landmarks:
            return False

        best_gain = 0.0
        best_old: Optional[Landmark] = None
        best_new: Optional[Landmark] = None

        for old in list(tour.visited_landmarks):
            for new in unvisited:
                candidate = tour.copy()
                candidate.replace_landmark(old, new)

                if not candidate.is_valid():
                    continue

                gain = candidate.total_score() - tour.total_score()
                if gain > best_gain:
                    best_gain = gain
                    best_old = old
                    best_new = new

        if best_old is not None:
            tour.replace_landmark(best_old, best_new)  # type: ignore[arg-type]
            return True

        return False

    def _try_insert(self, tour: Tour) -> bool:
        """Tries inserting each unvisited landmark at every position.

        Evaluates all (candidate × position) pairs and applies the
        insertion with the greatest score gain.

        Args:
            tour: The tour to improve in-place.

        Returns:
            True if an improving insertion was found and applied.
        """
        candidates = self.problem.feasible_candidates(tour)
        if not candidates:
            return False

        n = len(tour)
        best_gain = 0.0
        best_landmark: Optional[Landmark] = None
        best_position: Optional[int] = None

        for landmark in candidates:
            for position in range(n + 1):
                candidate = tour.copy()
                candidate.add_landmark(landmark, position)

                if not candidate.is_valid():
                    continue

                gain = candidate.total_score() - tour.total_score()
                if gain > best_gain:
                    best_gain = gain
                    best_landmark = landmark
                    best_position = position

        if best_landmark is not None:
            tour.add_landmark(best_landmark, best_position)
            return True

        return False