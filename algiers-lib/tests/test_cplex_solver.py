import pytest

from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule
from models.problem import Problem
from models.tour import Tour
from solvers.cplex_solver import CPLEXSolver


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
    """A 5-landmark problem with a generous time budget (small enough for CPLEX Community limit)."""
    landmarks = [
        _landmark("casbah", 36.788, 3.060, score=9.5, duration=90),
        _landmark("maqam", 36.753, 3.041, score=8.5, duration=45),
        _landmark("jardin", 36.771, 3.047, score=7.5, duration=60),
        _landmark("bardo", 36.758, 3.020, score=8.0, duration=60),
        _landmark("notre_dame", 36.795, 3.036, score=7.0, duration=30),
    ]
    return _problem(landmarks, time_budget=360)


class TestCPLEXSolver:

    def test_solve_returns_valid_tour(self, small_problem: Problem) -> None:
        """Verify that the CPLEX solver produces a valid tour on a small instance."""
        try:
            solver = CPLEXSolver(small_problem, time_limit=10.0)
            tour = solver.solve()
            
            assert isinstance(tour, Tour)
            
            sim_result = tour.simulate()
            assert sim_result.is_valid, f"Tour is invalid:\n{tour}"
            assert tour.total_score() >= 0.0
        except ImportError:
            pytest.skip("docplex is not installed")

    def test_terminates_impossible_budget(self) -> None:
        """Include a test case for a time budget that is too small to visit any landmarks."""
        landmarks = [_landmark("far", 37.5, 4.0, score=9.0, duration=600)]
        problem = _problem(landmarks, time_budget=10) # Too small
        
        try:
            solver = CPLEXSolver(problem, time_limit=10.0)
            tour = solver.solve()
            
            assert len(tour.visited_landmarks) == 0
            assert tour.is_valid()
        except ImportError:
            pytest.skip("docplex is not installed")

    def test_time_window_feasibility(self) -> None:
        """Verify the solver skips a high-score landmark if it's impossible to reach before it closes."""
        close_early = _landmark("closes_early", 36.770, 3.057, score=100.0, duration=30, open_time=480, close_time=550)
        problem = _problem([close_early], time_budget=600, start_time=540)
        
        try:
            solver = CPLEXSolver(problem, time_limit=10.0)
            tour = solver.solve()
            assert len(tour.visited_landmarks) == 0
        except ImportError:
            pytest.skip("docplex is not installed")

    def test_global_optimality(self) -> None:
        """Create a 'Toy Problem' with 3 landmarks where the optimal path is manually known. Assert CPLEX finds exactly that score."""
        hotel_mock = _hotel(lat=0.0, lon=0.0)
        l1 = _landmark("L1", 0.001, 0.001, score=10.0, duration=10)
        l2 = _landmark("L2", 0.002, 0.002, score=20.0, duration=10)
        l3 = _landmark("L3", 0.003, 0.003, score=30.0, duration=10)
        
        problem = Problem(
            hotel=hotel_mock,
            landmarks=[l1, l2, l3],
            time_budget=60,
            tour_day=Day.SATURDAY,
            start_time=540,
        )
        
        try:
            solver = CPLEXSolver(problem, time_limit=10.0)
            tour = solver.solve()
            assert tour.total_score() == 60.0
            assert len(tour.visited_landmarks) == 3
        except ImportError:
            pytest.skip("docplex is not installed")

    def test_big_m_integrity(self) -> None:
        """Use landmarks with very large coordinates/distances to ensure the BIG_M constant doesn't cause precision issues."""
        hotel_mock = _hotel(lat=0.0, lon=0.0)
        l1 = _landmark("L1", 89.0, 179.0, score=10.0, duration=10)
        l2 = _landmark("L2", -89.0, -179.0, score=10.0, duration=10)
        
        problem = Problem(
            hotel=hotel_mock,
            landmarks=[l1, l2],
            time_budget=1000000, 
            tour_day=Day.SATURDAY,
            start_time=540,
        )
        
        try:
            solver = CPLEXSolver(problem, time_limit=10.0)
            tour = solver.solve()
            assert tour.is_valid()
        except ImportError:
            pytest.skip("docplex is not installed")
