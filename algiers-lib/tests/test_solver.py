import pytest
from models.landmark import Landmark, Day, TimeSlot, WeeklySchedule
from models.problem import Problem
from models.tour import Tour
from solvers.solver import Solver
from solvers.greedy_solver import GreedySolver
from solvers.greedy_for_app import RandomGreedy, TimeGreedy
from solvers.grasp_solver import GraspSolver
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver, AcceptanceFunction
from plots import compare_solvers


def _build_landmark(landmark_id, name, lat, lon, score, duration, schedule):
    return Landmark(
        id=landmark_id, name=name, latitude=lat, longitude=lon,
        interest_score=score, visit_duration=duration,
        schedule=schedule, category="Tourist Site"
    )


def _build_hotel(schedule):
    return Landmark(
        id="hotel", name="Hotel", latitude=36.7, longitude=3.1,
        interest_score=0, visit_duration=0,
        schedule=schedule, category="Hotel"
    )


@pytest.fixture
def sample_problem():
    schedule = WeeklySchedule()
    schedule.schedule[Day.MONDAY] = [TimeSlot(540, 720)]
    hotel = _build_hotel(schedule)
    landmarks = [
        _build_landmark("1", "Museum", 36.5, 3.0, 8.0, 30, schedule),
        _build_landmark("2", "Park", 36.6, 3.05, 5.0, 30, schedule),
        _build_landmark("3", "Beach", 36.4, 3.1, 9.0, 45, schedule),
    ]
    return Problem(hotel=hotel, landmarks=landmarks, time_budget=180,
                   tour_day=Day.MONDAY, start_time=540)


@pytest.fixture
def medium_problem():
    schedule = WeeklySchedule()
    schedule.schedule[Day.MONDAY] = [TimeSlot(540, 960)]
    hotel = _build_hotel(schedule)
    landmarks = [
        _build_landmark(
            str(i), f"Landmark {i}",
            36.5 + (i % 4) * 0.02, 3.0 + (i // 4) * 0.02,
            float(10 - i), 20 + (i % 3) * 10, schedule
        )
        for i in range(6)
    ]
    return Problem(hotel=hotel, landmarks=landmarks, time_budget=360,
                   tour_day=Day.MONDAY, start_time=540)



class TestSolverBase:

    def test_solver_is_abstract(self, sample_problem):
        with pytest.raises(TypeError):
            Solver(sample_problem)


class TestGreedySolver:

    def test_greedy_solver_runs(self, sample_problem):
        solver = GreedySolver(sample_problem)
        result_tour = solver.solve()
        
        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert all(landmark in sample_problem.landmarks 
                   for landmark in result_tour.visited_landmarks)


class TestSimulatedAnnealingSolver:

    def test_sa_solver_runs(self, sample_problem):
        solver = SimulatedAnnealingSolver(sample_problem, max_iterations=50)
        result_tour = solver.solve()
        
        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert all(landmark in sample_problem.landmarks 
                   for landmark in result_tour.visited_landmarks)

    def test_sa_solver_acceptance_functions(self, sample_problem):
        solver = SimulatedAnnealingSolver(
            sample_problem, acceptance_criterion=AcceptanceFunction.CAUCHY,
            max_iterations=10
        )
        assert solver.acceptance_criterion == AcceptanceFunction.CAUCHY


class TestGraspSolver:

    def test_grasp_solver_runs(self, sample_problem):
        solver = GraspSolver(sample_problem, iterations=10, alpha=0.3, 
                            max_local_search_iters=5)
        result_tour = solver.solve()
        
        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert all(landmark in sample_problem.landmarks 
                   for landmark in result_tour.visited_landmarks)

    def test_grasp_solver_parameters(self, sample_problem):
        solver = GraspSolver(sample_problem, alpha=0.0, iterations=5, 
                            max_local_search_iters=3)
        assert solver.alpha == 0.0
        assert solver.iterations == 5
        assert solver.max_local_search_iters == 3


class TestRandomGreedySolver:

    def test_random_greedy_runs(self, sample_problem):
        solver = RandomGreedy(sample_problem)
        result_tour = solver.solve()

        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert all(landmark in sample_problem.landmarks
                   for landmark in result_tour.visited_landmarks)

    def test_random_greedy_no_duplicates(self, sample_problem):
        solver = RandomGreedy(sample_problem)
        result_tour = solver.solve()

        ids = [lm.id for lm in result_tour.visited_landmarks]
        assert len(ids) == len(set(ids))

    def test_random_greedy_medium_problem(self, medium_problem):
        solver = RandomGreedy(medium_problem)
        result_tour = solver.solve()

        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()

    def test_random_greedy_all_landmarks_closed(self):
        monday_schedule = WeeklySchedule()
        monday_schedule.schedule[Day.MONDAY] = [TimeSlot(540, 720)]
        hotel = _build_hotel(monday_schedule)

        tuesday_schedule = WeeklySchedule()
        tuesday_schedule.schedule[Day.TUESDAY] = [TimeSlot(540, 720)]
        landmarks = [
            _build_landmark("1", "Museum", 36.5, 3.0, 8.0, 30, tuesday_schedule),
            _build_landmark("2", "Park", 36.6, 3.05, 5.0, 30, tuesday_schedule),
        ]
        problem = Problem(hotel=hotel, landmarks=landmarks, time_budget=180,
                          tour_day=Day.MONDAY, start_time=540)

        result_tour = RandomGreedy(problem).solve()

        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert len(result_tour.visited_landmarks) == 0


class TestTimeGreedySolver:

    def test_time_greedy_runs(self, sample_problem):
        solver = TimeGreedy(sample_problem)
        result_tour = solver.solve()

        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert all(landmark in sample_problem.landmarks
                   for landmark in result_tour.visited_landmarks)

    def test_time_greedy_no_duplicates(self, sample_problem):
        solver = TimeGreedy(sample_problem)
        result_tour = solver.solve()

        ids = [lm.id for lm in result_tour.visited_landmarks]
        assert len(ids) == len(set(ids))

    def test_time_greedy_medium_problem(self, medium_problem):
        solver = TimeGreedy(medium_problem)
        result_tour = solver.solve()

        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()

    def test_time_greedy_prefers_nearest_landmark(self):
        schedule = WeeklySchedule()
        schedule.schedule[Day.MONDAY] = [TimeSlot(540, 960)]
        hotel = _build_hotel(schedule)
        near = _build_landmark("near", "Near", 36.701, 3.1,  5.0, 10, schedule)
        far  = _build_landmark("far",  "Far",  36.9,   3.1, 10.0, 10, schedule)
        problem = Problem(hotel=hotel, landmarks=[near, far],
                          time_budget=600, tour_day=Day.MONDAY, start_time=540)

        result_tour = TimeGreedy(problem).solve()

        assert result_tour.is_valid()
        if len(result_tour.visited_landmarks) >= 2:
            assert result_tour.visited_landmarks[0].id == "near"

    def test_time_greedy_all_landmarks_closed(self):
        monday_schedule = WeeklySchedule()
        monday_schedule.schedule[Day.MONDAY] = [TimeSlot(540, 720)]
        hotel = _build_hotel(monday_schedule)

        tuesday_schedule = WeeklySchedule()
        tuesday_schedule.schedule[Day.TUESDAY] = [TimeSlot(540, 720)]
        landmarks = [
            _build_landmark("1", "Museum", 36.5, 3.0, 8.0, 30, tuesday_schedule),
            _build_landmark("2", "Park", 36.6, 3.05, 5.0, 30, tuesday_schedule),
        ]
        problem = Problem(hotel=hotel, landmarks=landmarks, time_budget=180,
                          tour_day=Day.MONDAY, start_time=540)

        result_tour = TimeGreedy(problem).solve()

        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert len(result_tour.visited_landmarks) == 0


class TestSolverArrayInterface:
    
    @pytest.mark.parametrize(
        "solver_class,config",
        [
            (GreedySolver, {'use_ratio': False}),
            (RandomGreedy, {}),
            (TimeGreedy, {}),
            (SimulatedAnnealingSolver, {'max_iterations': 50}),
            (GraspSolver, {'iterations': 10, 'alpha': 0.3, 'max_local_search_iters': 5}),
        ],
        ids=['Greedy', 'RandomGreedy', 'TimeGreedy', 'SimulatedAnnealing', 'GRASP']
    )
    def test_solver_array_produces_valid_tours(self, sample_problem, solver_class, config):
        solver = solver_class(sample_problem, **config)
        result_tour = solver.solve()
        
        assert isinstance(result_tour, Tour)
        assert result_tour.is_valid()
        assert all(landmark in sample_problem.landmarks 
                   for landmark in result_tour.visited_landmarks)

    def test_compare_solvers_accepts_solver_array(self, medium_problem):
        solver_classes = [GreedySolver, SimulatedAnnealingSolver, GraspSolver]
        solver_config = {
            'GreedySolver': {'use_ratio': False},
            'SimulatedAnnealingSolver': {'max_iterations': 50},
            'GraspSolver': {'iterations': 10, 'alpha': 0.3, 'max_local_search_iters': 5},
        }
        
        results_df = compare_solvers(medium_problem, solver_classes,
                                    solver_kwargs=solver_config, num_runs=2)
        
        assert not results_df.empty
        assert list(results_df['Solver']) == ['GreedySolver', 'SimulatedAnnealingSolver', 'GraspSolver']
        assert all(results_df['Tour Quality'] >= 0)
        assert all(results_df['Execution Time (s)'] >= 0)
        assert all(results_df['Landmarks Visited'] >= 0)

    def test_solver_array_quality_metrics(self, medium_problem):
        all_solvers = [
            GreedySolver(medium_problem),
            RandomGreedy(medium_problem),
            TimeGreedy(medium_problem),
            SimulatedAnnealingSolver(medium_problem, max_iterations=30),
            GraspSolver(medium_problem, iterations=10, alpha=0.3, 
                       max_local_search_iters=5),
        ]
        
        for current_solver in all_solvers:
            tour = current_solver.solve()
            solver_name = current_solver.__class__.__name__
            assert tour.is_valid(), f"{solver_name} must produce valid tour"
            assert tour.total_score() >= 0, f"{solver_name} score must be non-negative"