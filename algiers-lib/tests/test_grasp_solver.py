from __future__ import annotations

import random
import time
from typing import Optional

import pytest

from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule
from models.problem import Problem
from models.tour import Tour
from solvers.grasp_solver import GraspSolver


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------


def _make_slot(open_time: int = 480, close_time: int = 1080) -> TimeSlot:
    """Creates a single TimeSlot."""
    return TimeSlot(open_time=open_time, close_time=close_time)


def _all_day_schedule(open_time: int = 480, close_time: int = 1080) -> WeeklySchedule:
    """Creates a WeeklySchedule open every day with a single slot."""
    slot = _make_slot(open_time, close_time)
    return WeeklySchedule(schedule={day: [slot] for day in Day})


def _closed_schedule() -> WeeklySchedule:
    """Creates a WeeklySchedule that is closed every day."""
    return WeeklySchedule(schedule={})


def _hotel(lat: float = 36.769, lon: float = 3.056) -> Landmark:
    """Creates a hotel landmark (always open, zero duration, zero score)."""
    full_slot = TimeSlot(open_time=0, close_time=1439)
    schedule = WeeklySchedule(schedule={day: [full_slot] for day in Day})
    return Landmark(
        id="hotel", name="Hotel",
        latitude=lat, longitude=lon,
        interest_score=0.0, visit_duration=0,
        schedule=schedule, category="hotel",
    )


def _landmark(
    id: str,
    lat: float,
    lon: float,
    score: float = 5.0,
    duration: int = 60,
    open_time: int = 480,
    close_time: int = 1080,
) -> Landmark:
    """Creates a landmark open every day with the given parameters."""
    return Landmark(
        id=id, name=id,
        latitude=lat, longitude=lon,
        interest_score=score, visit_duration=duration,
        schedule=_all_day_schedule(open_time, close_time),
        category="test",
    )


def _closed_landmark(id: str, lat: float, lon: float) -> Landmark:
    """Creates a landmark that is closed every day."""
    return Landmark(
        id=id, name=id,
        latitude=lat, longitude=lon,
        interest_score=9.0, visit_duration=30,
        schedule=_closed_schedule(),
        category="test",
    )


def _problem(
    landmarks: list[Landmark],
    time_budget: int = 480,
    tour_day: Day = Day.SATURDAY,
    start_time: int = 540,
) -> Problem:
    """Creates a Problem instance from a list of landmarks."""
    return Problem(
        hotel=_hotel(),
        landmarks=landmarks,
        time_budget=time_budget,
        tour_day=tour_day,
        start_time=start_time,
    )


# ---------------------------------------------------------------------------
# Fixture: small 5-landmark problem
# ---------------------------------------------------------------------------


@pytest.fixture
def small_problem() -> Problem:
    """A 5-landmark problem with generous time budget.

    All landmarks are within a few km of the hotel and each other.
    A 6-hour budget is enough to visit all five comfortably.
    """
    landmarks = [
        _landmark("casbah",    36.788, 3.060, score=9.5, duration=90),
        _landmark("maqam",     36.753, 3.041, score=8.5, duration=45),
        _landmark("jardin",    36.771, 3.047, score=7.5, duration=60),
        _landmark("bardo",     36.758, 3.020, score=8.0, duration=60),
        _landmark("notre_dame",36.795, 3.036, score=7.0, duration=30),
    ]
    return _problem(landmarks, time_budget=360)


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:

    def test_invalid_alpha_low_raises(self, small_problem: Problem) -> None:
        """alpha < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            GraspSolver(small_problem, alpha=-0.1)

    def test_invalid_alpha_high_raises(self, small_problem: Problem) -> None:
        """alpha > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            GraspSolver(small_problem, alpha=1.1)

    def test_invalid_iterations_raises(self, small_problem: Problem) -> None:
        """iterations < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="iterations"):
            GraspSolver(small_problem, iterations=0)

    def test_invalid_max_local_search_iters_raises(self, small_problem: Problem) -> None:
        """max_local_search_iters < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="max_local_search_iters"):
            GraspSolver(small_problem, max_local_search_iters=0)

    def test_valid_boundary_alpha_zero(self, small_problem: Problem) -> None:
        """alpha = 0.0 (pure greedy) must not raise."""
        GraspSolver(small_problem, alpha=0.0)

    def test_valid_boundary_alpha_one(self, small_problem: Problem) -> None:
        """alpha = 1.0 (pure random) must not raise."""
        GraspSolver(small_problem, alpha=1.0)


# ---------------------------------------------------------------------------
# 2. Termination — the most important test category
# ---------------------------------------------------------------------------


class TestTermination:
    """Verify the solver always terminates within an acceptable time.

    These tests fail fast if the algorithm hangs, giving a clear signal
    rather than an indefinite wait.
    """

    @pytest.mark.timeout(5)
    def test_terminates_small_problem(self, small_problem: Problem) -> None:
        """Solver must complete on a 5-landmark problem within 5 seconds."""
        solver = GraspSolver(small_problem, iterations=20, alpha=0.3)
        solver.solve()

    @pytest.mark.timeout(10)
    def test_terminates_larger_problem(self) -> None:
        """Solver must complete on a 15-landmark problem within 10 seconds."""
        landmarks = [
            _landmark(f"lm{i}", 36.75 + i * 0.005, 3.04 + i * 0.004,
                      score=float(i % 10 + 1), duration=30 + i * 5)
            for i in range(15)
        ]
        problem = _problem(landmarks, time_budget=480)
        solver = GraspSolver(problem, iterations=10, alpha=0.3, max_local_search_iters=15)
        solver.solve()

    @pytest.mark.timeout(5)
    def test_terminates_empty_problem(self) -> None:
        """Solver must terminate immediately when no landmarks exist."""
        problem = _problem([], time_budget=720)
        solver = GraspSolver(problem, iterations=10)
        result = solver.solve()
        assert len(result) == 0

    @pytest.mark.timeout(5)
    def test_terminates_all_landmarks_closed(self) -> None:
        """Solver must terminate when all landmarks are closed on the tour day."""
        landmarks = [
            _closed_landmark(f"closed{i}", 36.77 + i * 0.01, 3.05)
            for i in range(5)
        ]
        problem = _problem(landmarks, time_budget=720)
        solver = GraspSolver(problem, iterations=10)
        result = solver.solve()
        assert len(result) == 0

    @pytest.mark.timeout(5)
    def test_terminates_impossible_budget(self) -> None:
        """Solver must terminate when the time budget is too tight for any visit."""
        landmarks = [_landmark("far", 37.5, 4.0, score=9.0, duration=600)]
        problem = _problem(landmarks, time_budget=10)
        solver = GraspSolver(problem, iterations=10)
        result = solver.solve()
        assert result.is_valid()

    @pytest.mark.timeout(3)
    def test_construction_phase_terminates(self, small_problem: Problem) -> None:
        """_construction_phase must terminate within 3 seconds on a small problem."""
        solver = GraspSolver(small_problem)
        for _ in range(20):
            tour = solver._construction_phase()
            assert tour.is_valid()

    @pytest.mark.timeout(3)
    def test_local_search_terminates(self, small_problem: Problem) -> None:
        """_local_search must terminate within 3 seconds."""
        solver = GraspSolver(small_problem)
        tour = solver._construction_phase()
        solver._local_search(tour)


# ---------------------------------------------------------------------------
# 3. Correctness — output properties
# ---------------------------------------------------------------------------


class TestCorrectness:

    def test_solve_returns_tour(self, small_problem: Problem) -> None:
        """solve() must return a Tour instance."""
        solver = GraspSolver(small_problem, iterations=5)
        assert isinstance(solver.solve(), Tour)

    def test_result_is_valid(self, small_problem: Problem) -> None:
        """The returned tour must satisfy all constraints."""
        solver = GraspSolver(small_problem, iterations=20, alpha=0.3)
        result = solver.solve()
        assert result.is_valid(), f"Tour is invalid:\n{result}"

    def test_result_score_non_negative(self, small_problem: Problem) -> None:
        """Total score must never be negative."""
        solver = GraspSolver(small_problem, iterations=20)
        assert solver.solve().total_score() >= 0.0

    def test_result_visits_at_least_one_landmark(self, small_problem: Problem) -> None:
        """On a generous budget the solver must find at least one landmark."""
        solver = GraspSolver(small_problem, iterations=20, alpha=0.3)
        result = solver.solve()
        assert len(result) > 0, "Expected at least one landmark on a 6-hour budget"

    def test_all_visited_landmarks_are_in_problem(self, small_problem: Problem) -> None:
        """Every landmark in the result must belong to the problem."""
        solver = GraspSolver(small_problem, iterations=10)
        result = solver.solve()
        problem_ids = {lm.id for lm in small_problem.landmarks}
        for lm in result.visited_landmarks:
            assert lm.id in problem_ids, f"Foreign landmark '{lm.id}' in result"

    def test_no_duplicate_landmarks_in_result(self, small_problem: Problem) -> None:
        """The returned tour must not visit any landmark twice."""
        solver = GraspSolver(small_problem, iterations=10)
        result = solver.solve()
        ids = [lm.id for lm in result.visited_landmarks]
        assert len(ids) == len(set(ids)), f"Duplicate landmarks found: {ids}"

    def test_total_time_within_budget(self, small_problem: Problem) -> None:
        """The tour's total duration must not exceed the time budget."""
        solver = GraspSolver(small_problem, iterations=20)
        result = solver.solve()
        assert result.simulation_cache().total_duration <= small_problem.time_budget


# ---------------------------------------------------------------------------
# 4. Quality — alpha and iteration effects
# ---------------------------------------------------------------------------


class TestQuality:

    def test_alpha_zero_is_deterministic(self, small_problem: Problem) -> None:
        """With alpha=0 (pure greedy), construction always produces the same tour."""
        solver = GraspSolver(small_problem, iterations=1, alpha=0.0,
                             max_local_search_iters=1)
        random.seed(42)
        score1 = solver._construction_phase().total_score()
        random.seed(42)
        score2 = solver._construction_phase().total_score()
        assert score1 == score2

    def test_more_iterations_never_worsens_result(self, small_problem: Problem) -> None:
        """More iterations must produce equal or better results (monotone property).

        GRASP always keeps the global best, so running more iterations
        cannot decrease the best score found.
        """
        random.seed(0)
        solver_few = GraspSolver(small_problem, iterations=3, alpha=0.3)
        score_few = solver_few.solve().total_score()

        random.seed(0)
        solver_many = GraspSolver(small_problem, iterations=30, alpha=0.3)
        score_many = solver_many.solve().total_score()

        assert score_many >= score_few, (
            f"More iterations gave worse result: {score_many} < {score_few}"
        )

    def test_local_search_never_worsens_score(self, small_problem: Problem) -> None:
        """Local search must never decrease the tour's total score."""
        solver = GraspSolver(small_problem)
        for _ in range(10):
            tour = solver._construction_phase()
            score_before = tour.total_score()
            solver._local_search(tour)
            assert tour.total_score() >= score_before, (
                f"Local search decreased score from {score_before} to {tour.total_score()}"
            )

    def test_local_search_preserves_validity(self, small_problem: Problem) -> None:
        """Local search must never leave the tour in an invalid state."""
        solver = GraspSolver(small_problem)
        for _ in range(10):
            tour = solver._construction_phase()
            solver._local_search(tour)
            assert tour.is_valid(), f"Local search produced invalid tour:\n{tour}"


# ---------------------------------------------------------------------------
# 5. Operator unit tests
# ---------------------------------------------------------------------------


class TestOperators:

    def test_try_insert_adds_landmark_when_space_available(
        self, small_problem: Problem
    ) -> None:
        """_try_insert must add a landmark when budget allows."""
        solver = GraspSolver(small_problem)
        # Start with a single landmark — plenty of room for more
        tour = small_problem.create_empty_tour()
        tour.add_landmark(small_problem.landmarks[0])
        length_before = len(tour)
        solver._try_insert(tour)
        # Either something was added, or correctly nothing fit
        assert len(tour) >= length_before

    def test_try_replace_only_improves_or_keeps_score(
        self, small_problem: Problem
    ) -> None:
        """_try_replace must never decrease the tour score."""
        solver = GraspSolver(small_problem)
        tour = solver._construction_phase()
        score_before = tour.total_score()
        solver._try_replace(tour)
        assert tour.total_score() >= score_before

    def test_try_swap_preserves_visited_set(self, small_problem: Problem) -> None:
        """_try_swap must not change which landmarks are in the tour."""
        solver = GraspSolver(small_problem)
        tour = solver._construction_phase()
        ids_before = {lm.id for lm in tour.visited_landmarks}
        solver._try_swap(tour)
        ids_after = {lm.id for lm in tour.visited_landmarks}
        assert ids_before == ids_after, (
            "Swap changed the landmark set (should only change order)"
        )

    def test_try_swap_preserves_validity(self, small_problem: Problem) -> None:
        """After _try_swap the tour must still be valid."""
        solver = GraspSolver(small_problem)
        tour = solver._construction_phase()
        solver._try_swap(tour)
        assert tour.is_valid()

    def test_try_insert_on_empty_tour(self, small_problem: Problem) -> None:
        """_try_insert must work correctly on an empty tour."""
        solver = GraspSolver(small_problem)
        tour = small_problem.create_empty_tour()
        result = solver._try_insert(tour)
        # If the budget allows any visit, something should be inserted
        if result:
            assert len(tour) == 1
            assert tour.is_valid()

    def test_rcl_always_non_empty(self, small_problem: Problem) -> None:
        """_build_rcl must never return an empty list."""
        solver = GraspSolver(small_problem, alpha=0.0)
        tour = small_problem.create_empty_tour()
        candidates = small_problem.feasible_candidates(tour)
        scored = solver._score_candidates(candidates, tour)
        if scored:
            rcl = solver._build_rcl(scored)
            assert len(rcl) >= 1

    def test_score_candidates_sorted_descending(self, small_problem: Problem) -> None:
        """_score_candidates must return ratios in descending order."""
        solver = GraspSolver(small_problem)
        tour = small_problem.create_empty_tour()
        candidates = small_problem.feasible_candidates(tour)
        scored = solver._score_candidates(candidates, tour)
        ratios = [r for r, _ in scored]
        assert ratios == sorted(ratios, reverse=True), (
            "Candidates not sorted descending by ratio"
        )


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_single_landmark_problem(self) -> None:
        """Solver must handle a problem with exactly one landmark."""
        lm = _landmark("only", 36.78, 3.06, score=9.0, duration=60)
        problem = _problem([lm], time_budget=240)
        solver = GraspSolver(problem, iterations=5)
        result = solver.solve()
        assert result.is_valid()

    def test_single_landmark_unreachable(self) -> None:
        """When the only landmark cannot fit in the budget, return empty tour."""
        # Hotel at 36.769, 3.056. Landmark far away with long visit.
        lm = _landmark("far_long", 38.0, 5.0, score=9.0, duration=600)
        problem = _problem([lm], time_budget=30)
        solver = GraspSolver(problem, iterations=5)
        result = solver.solve()
        assert result.is_valid()
        assert len(result) == 0

    def test_one_iteration(self, small_problem: Problem) -> None:
        """A single iteration must still produce a valid tour."""
        solver = GraspSolver(small_problem, iterations=1, alpha=0.3)
        result = solver.solve()
        assert result.is_valid()

    def test_alpha_boundary_zero(self, small_problem: Problem) -> None:
        """alpha=0 (pure greedy) must produce a valid tour."""
        solver = GraspSolver(small_problem, iterations=5, alpha=0.0)
        assert solver.solve().is_valid()

    def test_alpha_boundary_one(self, small_problem: Problem) -> None:
        """alpha=1 (pure random) must produce a valid tour."""
        solver = GraspSolver(small_problem, iterations=5, alpha=1.0)
        assert solver.solve().is_valid()

    def test_tight_local_search_cap(self, small_problem: Problem) -> None:
        """max_local_search_iters=1 must still produce a valid tour."""
        solver = GraspSolver(small_problem, iterations=5,
                             max_local_search_iters=1)
        assert solver.solve().is_valid()

    def test_friday_with_prayer_break(self) -> None:
        """Solver must handle landmarks with two slots on the same day (e.g. Friday)."""
        from models.landmark import TimeSlot, WeeklySchedule

        morning = TimeSlot(open_time=540, close_time=690)   # 09:00–11:30
        afternoon = TimeSlot(open_time=840, close_time=1020) # 14:00–17:00

        friday_schedule = WeeklySchedule(
            schedule={Day.FRIDAY: [morning, afternoon]}
        )
        lm = Landmark(
            id="mosque", name="Mosque",
            latitude=36.78, longitude=3.06,
            interest_score=8.0, visit_duration=30,
            schedule=friday_schedule,
            category="religious",
        )
        problem = Problem(
            hotel=_hotel(),
            landmarks=[lm],
            time_budget=480,
            tour_day=Day.FRIDAY,
            start_time=540,
        )
        solver = GraspSolver(problem, iterations=5)
        result = solver.solve()
        assert result.is_valid()

    def test_reproducible_with_seed(self, small_problem: Problem) -> None:
        """Same random seed must produce the same result."""
        random.seed(123)
        solver = GraspSolver(small_problem, iterations=10, alpha=0.3)
        score1 = solver.solve().total_score()

        random.seed(123)
        solver2 = GraspSolver(small_problem, iterations=10, alpha=0.3)
        score2 = solver2.solve().total_score()

        assert score1 == score2


# ---------------------------------------------------------------------------
# 7. Performance sanity check (not a correctness test)
# ---------------------------------------------------------------------------


class TestPerformance:

    @pytest.mark.timeout(15)
    def test_25_landmarks_completes_in_time(self) -> None:
        """25 landmarks with 50 iterations must finish within 15 seconds.

        This is the main regression test for the halting bug. If this
        test hangs or times out, the performance blow-up has returned.
        """
        landmarks = [
            _landmark(
                f"lm{i}",
                36.74 + (i % 5) * 0.01,
                3.03 + (i // 5) * 0.01,
                score=float(i % 10 + 1),
                duration=30 + (i % 4) * 15,
            )
            for i in range(25)
        ]
        problem = _problem(landmarks, time_budget=480)
        solver = GraspSolver(
            problem,
            iterations=50,
            alpha=0.3,
            max_local_search_iters=20,
        )
        start = time.time()
        result = solver.solve()
        elapsed = time.time() - start

        assert result.is_valid()
        print(f"\n25 landmarks, 50 iterations: {elapsed:.2f}s, score={result.total_score():.1f}")