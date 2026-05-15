import pytest
#test_grasp_solver.py
from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule
from models.problem import Problem
from models.tour import Tour
from solvers.grasp_solver import GraspSolver


def _make_slot(open_time: int = 480, close_time: int = 1080) -> TimeSlot:
    """Create a single TimeSlot."""
    return TimeSlot(open_time=open_time, close_time=close_time)


def _all_day_schedule(open_time: int = 480, close_time: int = 1080) -> WeeklySchedule:
    """Create a WeeklySchedule open every day with a single slot."""
    slot = _make_slot(open_time, close_time)
    return WeeklySchedule(schedule={day: [slot] for day in Day})


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


class TestGraspSolver:

    def test_solve_returns_valid_tour_with_non_negative_score(self, small_problem: Problem) -> None:
        """Verify that the GRASP solver produces a valid tour and non-negative score."""
        solver = GraspSolver(small_problem, iterations=10)
        tour = solver.solve()
        
        assert isinstance(tour, Tour)
        
        sim_result = tour.simulate()
        assert sim_result.is_valid, f"Tour is invalid:\n{tour}"
        assert tour.total_score() >= 0.0

    def test_terminates_impossible_budget(self) -> None:
        """Include a test case for a time budget that is too small to visit any landmarks."""
        landmarks = [_landmark("far", 37.5, 4.0, score=9.0, duration=600)]
        problem = _problem(landmarks, time_budget=10) # Too small
        
        solver = GraspSolver(problem, iterations=5)
        tour = solver.solve()
        
        assert len(tour.visited_landmarks) == 0
        assert tour.is_valid()

    def test_time_window_feasibility(self) -> None:
        """Verify the solver skips a high-score landmark if it's impossible to reach before it closes."""
        # Landmark is very close (fast travel) but closes in 10 minutes (550).
        # We start at 540, travel takes some time, duration is 30 mins, so it's impossible to finish before 550.
        close_early = _landmark("closes_early", 36.770, 3.057, score=100.0, duration=30, open_time=480, close_time=550)
        problem = _problem([close_early], time_budget=600, start_time=540)
        
        solver = GraspSolver(problem, iterations=5)
        tour = solver.solve()
        
        assert len(tour.visited_landmarks) == 0

    def test_multi_run_feasibility(self, small_problem: Problem) -> None:
        """Use a loop to run GRASP 50 times. Assert that tour.simulate().is_valid is true for every single run."""
        for _ in range(50):
            solver = GraspSolver(small_problem, iterations=2)
            tour = solver.solve()
            assert tour.simulate().is_valid
