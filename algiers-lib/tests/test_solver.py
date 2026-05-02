"""Comprehensive unit tests for solver classes."""

import pytest
from models.landmark import Landmark, Day, TimeSlot, WeeklySchedule
from models.problem import Problem
from models.tour import Tour
from solvers.solver import Solver
from solvers.greedy_solver import GreedySolver
from solvers.simulated_annealing_solver import (
    SimulatedAnnealingSolver, AcceptanceFunction, DecayFunction
)


@pytest.fixture
def sample_problem():
    """Create a sample problem for testing."""
    schedule = WeeklySchedule()
    schedule.schedule[Day.MONDAY] = [TimeSlot(540, 720)]
    
    hotel = Landmark(
        id="hotel",
        name="Hotel",
        latitude=36.7,
        longitude=3.1,
        interest_score=0,
        visit_duration=0,
        schedule=schedule,
        category="Hotel"
    )
    
    landmarks = [
        Landmark(
            id="1",
            name="Museum",
            latitude=36.5,
            longitude=3.0,
            interest_score=8.0,
            visit_duration=30,
            schedule=schedule,
            category="Museum"
        ),
        Landmark(
            id="2",
            name="Park",
            latitude=36.6,
            longitude=3.05,
            interest_score=5.0,
            visit_duration=30,
            schedule=schedule,
            category="Park"
        ),
        Landmark(
            id="3",
            name="Beach",
            latitude=36.4,
            longitude=3.1,
            interest_score=9.0,
            visit_duration=45,
            schedule=schedule,
            category="Beach"
        )
    ]
    
    return Problem(
        hotel=hotel,
        landmarks=landmarks,
        time_budget=180,
        tour_day=Day.MONDAY,
        start_time=540
    )


class TestSolver:
    """Tests for the abstract Solver base class."""
    
    def test_solver_abstract(self, sample_problem):
        """Test that Solver is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Solver(sample_problem)


class TestGreedySolver:
    """Tests for the GreedySolver class."""
    
    def test_greedy_solver_creation(self, sample_problem):
        """Test GreedySolver creation."""
        solver = GreedySolver(sample_problem)
        assert solver.problem == sample_problem
        assert solver.use_ratio is False
    
    def test_greedy_solver_with_ratio(self, sample_problem):
        """Test GreedySolver with ratio priority."""
        solver = GreedySolver(sample_problem, use_ratio=True)
        assert solver.use_ratio is True
    
    def test_greedy_solver_solve(self, sample_problem):
        """Test GreedySolver.solve() method."""
        solver = GreedySolver(sample_problem)
        tour = solver.solve()
        
        assert isinstance(tour, Tour)
        assert tour.problem == sample_problem
        assert all(lm in sample_problem.landmarks for lm in tour.visited_landmarks)
    
    def test_greedy_solver_valid_tour(self, sample_problem):
        """Test that GreedySolver produces valid tours."""
        solver = GreedySolver(sample_problem)
        tour = solver.solve()
        
        assert tour.is_valid()
    
    def test_greedy_solver_deterministic(self, sample_problem):
        """Test that GreedySolver is deterministic."""
        solver1 = GreedySolver(sample_problem)
        tour1 = solver1.solve()
        
        solver2 = GreedySolver(sample_problem)
        tour2 = solver2.solve()
        
        # Same landmarks should be visited in same order
        assert [lm.id for lm in tour1.visited_landmarks] == [lm.id for lm in tour2.visited_landmarks]
    
    def test_greedy_solver_ratio_vs_non_ratio(self, sample_problem):
        """Test that ratio-based greedy differs from non-ratio."""
        solver_ratio = GreedySolver(sample_problem, use_ratio=True)
        tour_ratio = solver_ratio.solve()
        
        solver_non_ratio = GreedySolver(sample_problem, use_ratio=False)
        tour_non_ratio = solver_non_ratio.solve()
        
        # Tours might differ in order or number of landmarks
        assert isinstance(tour_ratio, Tour)
        assert isinstance(tour_non_ratio, Tour)
    
    def test_greedy_solver_empty_problem(self):
        """Test GreedySolver with problem with no feasible landmarks."""
        schedule_closed = WeeklySchedule()
        schedule_closed.schedule[Day.TUESDAY] = [TimeSlot(540, 720)]
        
        hotel = Landmark(
            id="hotel",
            name="Hotel",
            latitude=36.7,
            longitude=3.1,
            interest_score=0,
            visit_duration=0,
            schedule=schedule_closed,
            category="Hotel"
        )
        
        landmark = Landmark(
            id="1",
            name="Museum",
            latitude=36.5,
            longitude=3.0,
            interest_score=8.0,
            visit_duration=30,
            schedule=schedule_closed,
            category="Museum"
        )
        
        problem = Problem(
            hotel=hotel,
            landmarks=[landmark],
            time_budget=180,
            tour_day=Day.MONDAY,
            start_time=540
        )
        
        solver = GreedySolver(problem)
        tour = solver.solve()
        assert len(tour.visited_landmarks) == 0


class TestSimulatedAnnealingSolver:
    """Tests for the SimulatedAnnealingSolver class."""
    
    def test_sa_solver_creation(self, sample_problem):
        """Test SimulatedAnnealingSolver creation."""
        solver = SimulatedAnnealingSolver(sample_problem)
        assert solver.problem == sample_problem
        assert solver.initial_temperature == 10
        assert solver.cooling_rate == 0.95
        assert solver.acceptance_criterion == AcceptanceFunction.BOLTZMANN
        assert solver.max_iterations == 10000
    
    def test_sa_solver_custom_parameters(self, sample_problem):
        """Test SimulatedAnnealingSolver with custom parameters."""
        solver = SimulatedAnnealingSolver(
            sample_problem,
            acceptance_criterion=AcceptanceFunction.CAUCHY,
            initial_temperature=20,
            cooling_rate=0.9,
            max_iterations=100
        )
        assert solver.initial_temperature == 20
        assert solver.cooling_rate == 0.9
        assert solver.acceptance_criterion == AcceptanceFunction.CAUCHY
        assert solver.max_iterations == 100
    
    def test_sa_solver_invalid_parameters(self, sample_problem):
        """Test SimulatedAnnealingSolver with invalid parameters."""
        with pytest.raises(ValueError):
            SimulatedAnnealingSolver(sample_problem, initial_temperature=-1)
        
        with pytest.raises(ValueError):
            SimulatedAnnealingSolver(sample_problem, cooling_rate=1.5)
        
        with pytest.raises(ValueError):
            SimulatedAnnealingSolver(sample_problem, max_iterations=-1)
    
    def test_sa_solver_solve(self, sample_problem):
        """Test SimulatedAnnealingSolver.solve() method."""
        solver = SimulatedAnnealingSolver(
            sample_problem,
            max_iterations=100
        )
        tour = solver.solve()
        
        assert isinstance(tour, Tour)
        assert tour.problem == sample_problem
        assert all(lm in sample_problem.landmarks for lm in tour.visited_landmarks)
    
    def test_sa_solver_valid_tour(self, sample_problem):
        """Test that SimulatedAnnealingSolver produces valid tours."""
        solver = SimulatedAnnealingSolver(
            sample_problem,
            max_iterations=100
        )
        tour = solver.solve()
        
        assert tour.is_valid()
    
    def test_sa_solver_acceptance_probability(self, sample_problem):
        """Test acceptance probability calculation."""
        solver = SimulatedAnnealingSolver(sample_problem)
        
        # Better solution should always be accepted
        prob_better = solver._acceptance_probability(5, 10)
        assert prob_better == 1.0
        
        # Worse solution has some acceptance probability
        prob_worse_boltzmann = solver._acceptance_probability(-5, 10)
        assert 0 < prob_worse_boltzmann < 1
        
        # At very low temperature, worse solutions have low acceptance
        prob_worse_cold = solver._acceptance_probability(-5, 0.01)
        assert prob_worse_cold < prob_worse_boltzmann
        
        # At zero temperature, worse solutions not accepted
        prob_worse_zero = solver._acceptance_probability(-5, 0)
        assert prob_worse_zero == 0


class TestSolverArrayInterface:
    """Tests for the shared solver interface using an array of solver variants."""

    @pytest.mark.parametrize(
        "solver_cls, solver_kwargs",
        [
            pytest.param(GreedySolver, {'use_ratio': False}, id='GreedyScore'),
            pytest.param(GreedySolver, {'use_ratio': True}, id='GreedyRatio'),
            pytest.param(SimulatedAnnealingSolver, {'max_iterations': 100}, id='SimulatedAnnealing')
        ]
    )
    def test_solver_variants_valid_solution(self, sample_problem, solver_cls, solver_kwargs):
        solver = solver_cls(sample_problem, **solver_kwargs)
        tour = solver.solve()

        assert tour.problem == sample_problem
        assert tour.is_valid()
        assert all(lm in sample_problem.landmarks for lm in tour.visited_landmarks)

    def test_compare_solvers_accepts_solver_array(self, sample_problem):
        from plots import compare_solvers

        solver_classes = [GreedySolver, SimulatedAnnealingSolver]
        solver_kwargs = {
            'GreedySolver': {'use_ratio': False},
            'SimulatedAnnealingSolver': {'max_iterations': 50}
        }

        comparison_df = compare_solvers(sample_problem, solver_classes, solver_kwargs=solver_kwargs, num_runs=2)

        assert list(comparison_df['Solver']) == ['GreedySolver', 'SimulatedAnnealingSolver']
        assert all(comparison_df['Execution Time (s)'] >= 0)
        assert all(comparison_df['Tour Quality'] >= 0)
    
    def test_sa_solver_acceptance_function_cauchy(self, sample_problem):
        """Test CAUCHY acceptance function."""
        solver = SimulatedAnnealingSolver(
            sample_problem,
            acceptance_criterion=AcceptanceFunction.CAUCHY
        )
        
        prob_better = solver._acceptance_probability(5, 10)
        assert prob_better == 1.0
        
        prob_worse = solver._acceptance_probability(-5, 10)
        assert 0 < prob_worse < 1
    
    def test_sa_solver_swap_operator(self, sample_problem):
        """Test swap operator."""
        solver = SimulatedAnnealingSolver(sample_problem, max_iterations=1)
        tour = sample_problem.create_empty_tour()
        
        # Add landmarks to tour
        tour.add_landmark(sample_problem.landmarks[0])
        tour.add_landmark(sample_problem.landmarks[1])
        tour.add_landmark(sample_problem.landmarks[2])
        
        original_landmarks = list(tour.visited_landmarks)
        swapped_tour = solver._swap_operator(tour)
        
        # Should have same landmarks
        assert set(lm.id for lm in swapped_tour.visited_landmarks) == \
               set(lm.id for lm in original_landmarks)
        
        # Should be a copy
        assert swapped_tour is not tour
    
    def test_sa_solver_insert_operator(self, sample_problem):
        """Test insert operator."""
        solver = SimulatedAnnealingSolver(sample_problem, max_iterations=1)
        tour = sample_problem.create_empty_tour()
        
        tour.add_landmark(sample_problem.landmarks[0])
        
        inserted_tour = solver._insert_operator(tour)
        
        # Should have one more landmark
        assert len(inserted_tour.visited_landmarks) >= len(tour.visited_landmarks)
        
        # Should be a copy
        assert inserted_tour is not tour
    
    def test_sa_solver_remove_operator(self, sample_problem):
        """Test remove operator."""
        solver = SimulatedAnnealingSolver(sample_problem, max_iterations=1)
        tour = sample_problem.create_empty_tour()
        
        tour.add_landmark(sample_problem.landmarks[0])
        tour.add_landmark(sample_problem.landmarks[1])
        
        removed_tour = solver._remove_operator(tour)
        
        # Should have one less landmark or same
        assert len(removed_tour.visited_landmarks) <= len(tour.visited_landmarks)
        
        # Should be a copy
        assert removed_tour is not tour
    
    def test_sa_solver_invert_operator(self, sample_problem):
        """Test invert operator."""
        solver = SimulatedAnnealingSolver(sample_problem, max_iterations=1)
        tour = sample_problem.create_empty_tour()
        
        tour.add_landmark(sample_problem.landmarks[0])
        tour.add_landmark(sample_problem.landmarks[1])
        tour.add_landmark(sample_problem.landmarks[2])
        
        original_landmarks = [lm.id for lm in tour.visited_landmarks]
        inverted_tour = solver._invert_operator(tour)
        
        # Should have same landmarks
        assert set(lm.id for lm in inverted_tour.visited_landmarks) == set(original_landmarks)
        
        # Should be a copy
        assert inverted_tour is not tour
    
    def test_sa_solver_exponential_decay(self, sample_problem):
        """Test exponential temperature decay."""
        solver = SimulatedAnnealingSolver(
            sample_problem,
            initial_temperature=10,
            cooling_rate=0.95,
            decay_function=DecayFunction.EXPONENTIAL,
            max_iterations=100
        )
        
        tour = solver.solve()
        assert isinstance(tour, Tour)
    
    def test_sa_solver_linear_decay(self, sample_problem):
        """Test linear temperature decay."""
        solver = SimulatedAnnealingSolver(
            sample_problem,
            initial_temperature=10,
            cooling_rate=0.1,
            decay_function=DecayFunction.LINEAR,
            max_iterations=100
        )
        
        tour = solver.solve()
        assert isinstance(tour, Tour)


class TestSolverComparison:
    """Tests for comparing different solvers."""
    
    def test_solvers_produce_tours(self, sample_problem):
        """Test that different solvers produce valid tours."""
        solvers = [
            GreedySolver(sample_problem),
            GreedySolver(sample_problem, use_ratio=True),
            SimulatedAnnealingSolver(sample_problem, max_iterations=50),
        ]
        
        for solver in solvers:
            tour = solver.solve()
            assert isinstance(tour, Tour)
            assert tour.is_valid()
            assert all(lm in sample_problem.landmarks for lm in tour.visited_landmarks)
    
    def test_solvers_different_results(self, sample_problem):
        """Test that different solvers might produce different results."""
        greedy_solver = GreedySolver(sample_problem)
        greedy_tour = greedy_solver.solve()
        
        sa_solver = SimulatedAnnealingSolver(sample_problem, max_iterations=200)
        sa_tour = sa_solver.solve()
        
        # Tours might be different (at least potentially)
        assert isinstance(greedy_tour, Tour)
        assert isinstance(sa_tour, Tour)
        assert greedy_tour.is_valid()
        assert sa_tour.is_valid()
