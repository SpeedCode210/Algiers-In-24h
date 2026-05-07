from __future__ import annotations

import random

import pytest

from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule
from models.problem import Problem
from models.tour import Tour
from solvers.tabu_solver import MoveType, OscillationPhase, TabuMove, TabuSolver


def _make_slot(open_time: int = 480, close_time: int = 1080) -> TimeSlot:
    """Create a single TimeSlot."""
    return TimeSlot(open_time=open_time, close_time=close_time)


def _all_day_schedule(open_time: int = 480, close_time: int = 1080) -> WeeklySchedule:
    """Create a WeeklySchedule open every day with a single slot."""
    slot = _make_slot(open_time, close_time)
    return WeeklySchedule(schedule={day: [slot] for day in Day})


def _closed_schedule() -> WeeklySchedule:
    """Create a WeeklySchedule that is closed every day."""
    return WeeklySchedule(schedule={})


def _hotel(lat: float = 36.769, lon: float = 3.056) -> Landmark:
    """Create a hotel landmark (always open, zero duration, zero score)."""
    full_slot = TimeSlot(open_time=0, close_time=1439)
    schedule = WeeklySchedule(schedule={day: [full_slot] for day in Day})
    return Landmark(
        id="hotel",
        name="Hotel",
        latitude=lat,
        longitude=lon,
        interest_score=0.0,
        visit_duration=0,
        schedule=schedule,
        category="hotel",
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
    """Create a landmark open every day with the given parameters."""
    return Landmark(
        id=id,
        name=id,
        latitude=lat,
        longitude=lon,
        interest_score=score,
        visit_duration=duration,
        schedule=_all_day_schedule(open_time, close_time),
        category="test",
    )


def _closed_landmark(id: str, lat: float, lon: float) -> Landmark:
    """Create a landmark that is closed every day."""
    return Landmark(
        id=id,
        name=id,
        latitude=lat,
        longitude=lon,
        interest_score=9.0,
        visit_duration=30,
        schedule=_closed_schedule(),
        category="test",
    )


def _problem(
    landmarks: list[Landmark],
    time_budget: int = 480,
    tour_day: Day = Day.SATURDAY,
    start_time: int = 540,
) -> Problem:
    """Create a Problem instance from a list of landmarks."""
    return Problem(
        hotel=_hotel(),
        landmarks=landmarks,
        time_budget=time_budget,
        tour_day=tour_day,
        start_time=start_time,
    )


@pytest.fixture
def small_problem() -> Problem:
    """A 5-landmark problem with a generous time budget."""
    landmarks = [
        _landmark("casbah", 36.788, 3.060, score=9.5, duration=90),
        _landmark("maqam", 36.753, 3.041, score=8.5, duration=45),
        _landmark("jardin", 36.771, 3.047, score=7.5, duration=60),
        _landmark("bardo", 36.758, 3.020, score=8.0, duration=60),
        _landmark("notre_dame", 36.795, 3.036, score=7.0, duration=30),
    ]
    return _problem(landmarks, time_budget=360)


# ---------------------------------------------------------------------------
# 1. Termination and basic validity
# ---------------------------------------------------------------------------


class TestTermination:

    @pytest.mark.timeout(5)
    def test_terminates_small_problem(self, small_problem: Problem) -> None:
        solver = TabuSolver(
            small_problem,
            max_iterations=50,
            tabu_tenure=7,
            plateau_threshold=5,
            expansion_iterations=3,
        )
        solver.solve()

    @pytest.mark.timeout(5)
    def test_terminates_empty_problem(self) -> None:
        problem = _problem([], time_budget=720)
        solver = TabuSolver(problem, max_iterations=10, tabu_tenure=5)
        result = solver.solve()
        assert len(result) == 0

    @pytest.mark.timeout(5)
    def test_terminates_all_landmarks_closed(self) -> None:
        landmarks = [_closed_landmark(f"closed{i}", 36.77 + i * 0.01, 3.05) for i in range(5)]
        problem = _problem(landmarks, time_budget=720)
        solver = TabuSolver(problem, max_iterations=10, tabu_tenure=5)
        result = solver.solve()
        assert len(result) == 0

    @pytest.mark.timeout(5)
    def test_terminates_impossible_budget(self) -> None:
        landmarks = [_landmark("far", 37.5, 4.0, score=9.0, duration=600)]
        problem = _problem(landmarks, time_budget=10)
        solver = TabuSolver(problem, max_iterations=10, tabu_tenure=5)
        result = solver.solve()
        assert result.is_valid()


class TestCorrectness:

    def test_solve_returns_tour(self, small_problem: Problem) -> None:
        solver = TabuSolver(small_problem, max_iterations=20, tabu_tenure=7)
        assert isinstance(solver.solve(), Tour)

    def test_result_is_valid(self, small_problem: Problem) -> None:
        solver = TabuSolver(small_problem, max_iterations=30, tabu_tenure=7)
        result = solver.solve()
        assert result.is_valid(), f"Tour is invalid:\n{result}"

    def test_no_duplicate_landmarks_in_result(self, small_problem: Problem) -> None:
        solver = TabuSolver(small_problem, max_iterations=20, tabu_tenure=7)
        result = solver.solve()
        ids = [lm.id for lm in result.visited_landmarks]
        assert len(ids) == len(set(ids)), f"Duplicate landmarks found: {ids}"

    def test_all_visited_landmarks_are_in_problem(self, small_problem: Problem) -> None:
        solver = TabuSolver(small_problem, max_iterations=20, tabu_tenure=7)
        result = solver.solve()
        problem_ids = {lm.id for lm in small_problem.landmarks}
        for lm in result.visited_landmarks:
            assert lm.id in problem_ids, f"Foreign landmark '{lm.id}' in result"


# ---------------------------------------------------------------------------
# 2. Neighbor move semantics
# ---------------------------------------------------------------------------


class TestMoveSemantics:

    def test_remove_neighbors_use_remove_move_type(self) -> None:
        landmarks = [
            _landmark("a", 36.77, 3.05, score=4.0, duration=30),
            _landmark("b", 36.771, 3.051, score=5.0, duration=30),
            _landmark("c", 36.772, 3.052, score=6.0, duration=30),
        ]
        problem = _problem(landmarks, time_budget=600)
        tour = problem.create_empty_tour()
        tour.add_landmark(landmarks[0])
        tour.add_landmark(landmarks[1])

        solver = TabuSolver(problem, max_iterations=1, tabu_tenure=5)
        neighbors = solver._get_neighbors(tour, effective_budget=float(problem.time_budget), phase=OscillationPhase.INTENSIFICATION)

        removed_moves = [
            move for neighbor, move in neighbors
            if len(neighbor.visited_landmarks) == len(tour.visited_landmarks) - 1
        ]
        assert removed_moves, "Expected at least one removal neighbor"
        assert all(move.move_type == MoveType.REMOVE for move in removed_moves)

    def test_insert_neighbors_use_insert_move_type(self) -> None:
        landmarks = [
            _landmark("a", 36.77, 3.05, score=4.0, duration=30),
            _landmark("b", 36.771, 3.051, score=5.0, duration=30),
            _landmark("c", 36.772, 3.052, score=6.0, duration=30),
        ]
        problem = _problem(landmarks, time_budget=600)
        tour = problem.create_empty_tour()
        tour.add_landmark(landmarks[0])
        tour.add_landmark(landmarks[1])

        solver = TabuSolver(problem, max_iterations=1, tabu_tenure=5)
        neighbors = solver._get_neighbors(tour, effective_budget=float(problem.time_budget), phase=OscillationPhase.EXPANSION)

        inserted_moves = [
            move for neighbor, move in neighbors
            if len(neighbor.visited_landmarks) == len(tour.visited_landmarks) + 1
        ]
        assert inserted_moves, "Expected at least one insertion neighbor"
        assert all(move.move_type == MoveType.INSERT for move in inserted_moves)

    def test_recover_adds_insert_not_remove_to_tabu(
        self, small_problem: Problem
    ) -> None:
        """_recover_tour must tabu INSERT (not REMOVE) for removed landmarks.

        This is the critical regression test for the bug where REMOVE was
        incorrectly added instead of INSERT, allowing immediate re-insertion
        of just-removed landmarks.
        """
        solver = TabuSolver(small_problem, tabu_tenure=10)

        # Force an infeasible tour by appending all landmarks directly
        infeasible = small_problem.create_empty_tour()
        for lm in small_problem.landmarks:
            infeasible.visited_landmarks.append(lm)
            infeasible._invalidate_cache()

        original_ids = {lm.id for lm in infeasible.visited_landmarks}
        recovered = solver._recover_tour(infeasible, iteration=5)
        recovered_ids = {lm.id for lm in recovered.visited_landmarks}
        removed_ids = original_ids - recovered_ids

        for removed_id in removed_ids:
            insert_move = TabuMove(MoveType.INSERT, (removed_id,))
            remove_move = TabuMove(MoveType.REMOVE, (removed_id,))

            assert insert_move in solver.tabu_end, (
                f"INSERT({removed_id}) not in tabu after recovery — "
                f"re-insertion not prevented."
            )
            assert remove_move not in solver.tabu_end, (
                f"REMOVE({removed_id}) incorrectly in tabu — "
                f"landmark is no longer in the tour, this entry is useless."
            )


# ---------------------------------------------------------------------------
# 3. Feasibility checks
# ---------------------------------------------------------------------------


class TestFeasibility:

    def test_is_feasible_under_rejects_invalid_schedule(self) -> None:
        random.seed(0)
        tight_slot = WeeklySchedule(schedule={Day.SATURDAY: [TimeSlot(540, 570)]})
        far_landmark = Landmark(
            id="far",
            name="far",
            latitude=38.769,
            longitude=3.056,
            interest_score=8.0,
            visit_duration=30,
            schedule=tight_slot,
            category="test",
        )
        problem = Problem(
            hotel=_hotel(),
            landmarks=[far_landmark],
            time_budget=600,
            tour_day=Day.SATURDAY,
            start_time=540,
        )
        tour = problem.create_empty_tour()
        tour.add_landmark(far_landmark)

        solver = TabuSolver(problem, max_iterations=1, tabu_tenure=5)
        assert not tour.is_valid(), "Sanity check: tour should be invalid due to schedule"
        assert not solver._is_feasible_under(tour, budget=600)


# ---------------------------------------------------------------------------
# 4. Tabu reverse-move behavior
# ---------------------------------------------------------------------------


class TestTabuReverseMove:

    def test_reverse_move_is_tabued_after_remove(self, monkeypatch: pytest.MonkeyPatch) -> None:
        morning_slot = WeeklySchedule(schedule={Day.SATURDAY: [TimeSlot(540, 600)]})
        afternoon_slot = WeeklySchedule(schedule={Day.SATURDAY: [TimeSlot(720, 1020)]})

        landmark_a = Landmark(
            id="a",
            name="a",
            latitude=36.770,
            longitude=3.057,
            interest_score=10.0,
            visit_duration=30,
            schedule=morning_slot,
            category="test",
        )
        landmark_b = Landmark(
            id="b",
            name="b",
            latitude=36.771,
            longitude=3.058,
            interest_score=5.0,
            visit_duration=30,
            schedule=afternoon_slot,
            category="test",
        )

        problem = Problem(
            hotel=_hotel(),
            landmarks=[landmark_a, landmark_b],
            time_budget=600,
            tour_day=Day.SATURDAY,
            start_time=540,
        )

        tour = problem.create_empty_tour()
        tour.add_landmark(landmark_a)
        tour.add_landmark(landmark_b)
        assert tour.is_valid()

        def _fixed_random_tour() -> Tour:
            return tour.copy()

        monkeypatch.setattr(problem, "random_tour", _fixed_random_tour)

        solver = TabuSolver(problem, max_iterations=1, tabu_tenure=5)
        solver.solve()

        expected_reverse = TabuMove(MoveType.INSERT, ("b",))
        assert expected_reverse in solver.tabu_end
