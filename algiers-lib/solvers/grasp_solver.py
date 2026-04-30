from __future__ import annotations

import random
from typing import Optional

from models.landmark import Landmark
from models.problem import Problem
from models.tour import Tour
from .solver import Solver


class GraspSolver(Solver):

    def __init__(
        self,
        problem: Problem,
        iterations: int = 100,
        alpha: float = 0.3,
    ) -> None:
        """Initialises the GRASP solver.

        Args:
            problem: The fully loaded Problem instance.
            iterations: How many construction + local search cycles to run.
                More iterations = better quality but longer runtime.
            alpha: Controls randomness during construction. Must be in
                [0.0, 1.0]. Recommended starting value: 0.3.

        Raises:
            ValueError: If alpha is outside [0.0, 1.0] or iterations < 1.
        """
        super().__init__(problem)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}.")

        self.iterations: int = iterations
        self.alpha: float = alpha

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tour:
        """Runs GRASP and returns the best tour found.

        Executes `self.iterations` cycles of randomized construction
        followed by local search. Tracks and returns the globally best
        valid tour across all iterations.

        Returns:
            The best valid Tour found. Returns an empty tour if no valid
            solution was found in any iteration (should not happen in
            practice with a well-formed dataset).
        """
        best_tour: Tour = self.problem.create_empty_tour()

        for _ in range(self.iterations):

            # Phase 1 — build a randomized greedy solution
            tour = self._construction_phase()

            # Phase 2 — improve it with local search
            tour = self._local_search(tour)

            # Keep the best valid tour found so far
            if tour.is_valid() and tour.total_score() > best_tour.total_score():
                best_tour = tour

        return best_tour

    # ------------------------------------------------------------------
    # Phase 1 — Randomized greedy construction
    # ------------------------------------------------------------------

    def _construction_phase(self) -> Tour:
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
        """Scores each candidate landmark by its value-per-time-cost ratio.

        The greedy criterion is::

            ratio = interest_score / (travel_time_to_here + visit_duration)

        A higher ratio means more interest collected per minute spent.
        Candidates that would make the tour infeasible are excluded.

        Args:
            candidates: Unvisited landmarks open on the tour day.
            tour: The current partial tour (used to find the last position).

        Returns:
            List of (score, Landmark) tuples sorted descending by score.
            May be empty if every candidate makes the tour infeasible.
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

        # Sort best first
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def _build_rcl(
        self,
        scored: list[tuple[float, Landmark]],
    ) -> list[Landmark]:
        """Builds the Restricted Candidate List from scored candidates.

        Includes all landmarks whose score falls within alpha × score
        range of the best candidate::

            threshold = best - alpha × (best - worst)
            RCL = { c : score(c) >= threshold }

        When alpha = 0 the RCL contains only the single best candidate.
        When alpha = 1 the RCL contains every candidate.

        Args:
            scored: Non-empty list of (score, Landmark) pairs sorted
                descending. Assumes at least one element.

        Returns:
            Non-empty list of Landmark objects eligible for selection.
        """
        best_score = scored[0][0]
        worst_score = scored[-1][0]
        threshold = best_score - self.alpha * (best_score - worst_score)

        rcl = [landmark for score, landmark in scored if score >= threshold]

        # Guarantee at least one element (handles floating-point edge cases)
        if not rcl:
            rcl = [scored[0][1]]

        return rcl

    # ------------------------------------------------------------------
    # Phase 2 — Local search
    # ------------------------------------------------------------------

    def _local_search(self, tour: Tour) -> Tour:
        """Improves a tour using three neighbourhood move operators.

        Repeatedly applies swap, replace, and insert moves until no
        operator produces any improvement. This is a best-improvement
        strategy: in each pass all moves are evaluated and the best
        improving move is applied.

        Args:
            tour: The starting tour produced by the construction phase.

        Returns:
            An improved Tour (always at least as good as the input).
        """
        improved = True

        while improved:
            improved = False
            if self._try_replace(tour):
                improved = True
                continue
            if self._try_insert(tour):
                improved = True
                continue
            if self._try_swap(tour):  
                improved = True

        return tour

    def _try_swap(self, tour: Tour) -> bool:
        """Tries all pairwise swaps to improve time-window alignment.

        A swap doesn't change the score directly but can reorder the route
        to better satisfy time windows, potentially enabling subsequent
        insertions. Accepts any swap that keeps the tour valid and maximises
        remaining time budget (enabling future inserts).

        Args:
            tour: The tour to improve in-place.

        Returns:
            True if a beneficial swap was found and applied.
        """
        n = len(tour)
        if n < 2:
            return False

        best_remaining = float('inf')
        best_i: Optional[int] = None
        best_j: Optional[int] = None

        for i in range(n):
            for j in range(i + 1, n):
                candidate = tour.copy()
                candidate.swap_by_index(i, j)
                if not candidate.is_valid():
                    continue
                # Prefer the ordering that leaves the most time budget free
                remaining = candidate.simulation_cache().total_duration
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_i = i
                    best_j = j

        if best_i is not None:
            tour.swap_by_index(best_i, best_j)  # type: ignore[arg-type]
            return True

        return False

    def _try_replace(self, tour: Tour) -> bool:
        """Tries replacing each visited landmark with each unvisited one.

        A replace swaps a landmark in the tour for one currently outside
        it at the same position. This explores a different subset of
        landmarks without changing the tour length.

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

        Tests adding a new landmark at every possible index in the route.
        This can increase the total score if there is remaining time in
        the budget.

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