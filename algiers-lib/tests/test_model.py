"""Comprehensive unit tests for models (Problem, Tour, Landmark, etc.)."""

import pytest
from models.landmark import (
    Landmark, Day, TimeSlot, WeeklySchedule, loadLandmarks, loadHotel
)
from models.problem import Problem
from models.tour import Tour, ScheduleEntry, SimulationResult


class TestDay:
    """Tests for Day enum."""
    
    def test_day_values(self):
        """Test Day enum values."""
        assert Day.SUNDAY == 0
        assert Day.MONDAY == 1
        assert Day.SATURDAY == 6
    
    def test_day_from_string(self):
        """Test Day.from_string() method."""
        assert Day.from_string("monday") == Day.MONDAY
        assert Day.from_string("FRIDAY") == Day.FRIDAY
        assert Day.from_string("Sunday") == Day.SUNDAY
    
    def test_day_from_string_invalid(self):
        """Test Day.from_string() with invalid input."""
        with pytest.raises(ValueError):
            Day.from_string("invalid_day")


class TestTimeSlot:
    """Tests for TimeSlot class."""
    
    def test_timeslot_creation(self):
        """Test TimeSlot creation."""
        slot = TimeSlot(open_time=540, close_time=600)
        assert slot.open_time == 540
        assert slot.close_time == 600
    
    def test_timeslot_invalid(self):
        """Test TimeSlot with invalid times."""
        with pytest.raises(ValueError):
            TimeSlot(open_time=600, close_time=540)
        
        with pytest.raises(ValueError):
            TimeSlot(open_time=600, close_time=600)
    
    def test_timeslot_contains(self):
        """Test TimeSlot.contains() method."""
        slot = TimeSlot(open_time=540, close_time=600)
        
        assert slot.contains(550, 40) is True
        assert slot.contains(560, 30) is True
        assert slot.contains(540, 60) is True
        assert slot.contains(530, 30) is False
        assert slot.contains(550, 60) is False


class TestWeeklySchedule:
    """Tests for WeeklySchedule class."""
    
    def test_weekly_schedule_creation(self):
        """Test WeeklySchedule creation."""
        schedule = WeeklySchedule()
        assert schedule.schedule == {}
    
    def test_weekly_schedule_is_open_on(self):
        """Test is_open_on() method."""
        schedule = WeeklySchedule()
        schedule.schedule[Day.MONDAY] = [TimeSlot(540, 600)]
        
        assert schedule.is_open_on(Day.MONDAY) is True
        assert schedule.is_open_on(Day.TUESDAY) is False
    
    def test_weekly_schedule_get_slots(self):
        """Test get_slots() method."""
        schedule = WeeklySchedule()
        slots = [TimeSlot(540, 600), TimeSlot(650, 720)]
        schedule.schedule[Day.MONDAY] = slots
        
        assert schedule.get_slots(Day.MONDAY) == slots
        assert schedule.get_slots(Day.TUESDAY) == []
    
    def test_weekly_schedule_earliest_valid_start(self):
        """Test earliest_valid_start() method."""
        schedule = WeeklySchedule()
        schedule.schedule[Day.MONDAY] = [
            TimeSlot(540, 600),
            TimeSlot(650, 720)
        ]
        
        assert schedule.earliest_valid_start(Day.MONDAY, 550, 40) == 550
        assert schedule.earliest_valid_start(Day.MONDAY, 560, 50) == 650
        assert schedule.earliest_valid_start(Day.MONDAY, 690, 30) == 690
        assert schedule.earliest_valid_start(Day.MONDAY, 700, 100) is None
        assert schedule.earliest_valid_start(Day.TUESDAY, 550, 40) is None


class TestLandmark:
    """Tests for Landmark class."""
    
    @pytest.fixture
    def sample_landmark(self):
        """Create a sample landmark for testing."""
        schedule = WeeklySchedule()
        schedule.schedule[Day.MONDAY] = [TimeSlot(540, 720)]
        
        return Landmark(
            id="1",
            name="Test Landmark",
            latitude=36.5,
            longitude=3.0,
            interest_score=8.5,
            visit_duration=30,
            schedule=schedule,
            category="Museum"
        )
    
    def test_landmark_creation(self, sample_landmark):
        """Test Landmark creation."""
        assert sample_landmark.id == "1"
        assert sample_landmark.name == "Test Landmark"
        assert sample_landmark.interest_score == 8.5
        assert sample_landmark.visit_duration == 30
        assert sample_landmark.category == "Museum"
    
    def test_landmark_coordinates(self, sample_landmark):
        """Test coordinates property."""
        coords = sample_landmark.coordinates
        assert coords == (36.5, 3.0)
    
    def test_landmark_str(self, sample_landmark):
        """Test string representation."""
        str_repr = str(sample_landmark)
        assert "Test Landmark" in str_repr
        assert "Museum" in str_repr
        assert "8.5" in str_repr
        assert "30" in str_repr
    
    def test_landmark_frozen(self, sample_landmark):
        """Test that Landmark is immutable (frozen)."""
        with pytest.raises(AttributeError):
            sample_landmark.name = "New Name"


class TestProblem:
    """Tests for Problem class."""
    
    @pytest.fixture
    def sample_problem(self):
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
                visit_duration=45,
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
            )
        ]
        
        return Problem(
            hotel=hotel,
            landmarks=landmarks,
            time_budget=180,
            tour_day=Day.MONDAY,
            start_time=540
        )
    
    def test_problem_creation(self, sample_problem):
        """Test Problem creation."""
        assert sample_problem.hotel.name == "Hotel"
        assert len(sample_problem.landmarks) == 2
        assert sample_problem.time_budget == 180
        assert sample_problem.tour_day == Day.MONDAY
        assert sample_problem.start_time == 540
    
    def test_problem_travel_time(self, sample_problem):
        """Test travel_time() method."""
        hotel = sample_problem.hotel
        museum = sample_problem.landmarks[0]
        
        travel_time = sample_problem.travel_time(hotel, museum)
        assert travel_time >= 0
        
        # Travel time should be symmetric
        travel_back = sample_problem.travel_time(museum, hotel)
        assert travel_back >= 0
    
    def test_problem_travel_time_same_location(self, sample_problem):
        """Test travel_time() between same location."""
        hotel = sample_problem.hotel
        travel_time = sample_problem.travel_time(hotel, hotel)
        assert travel_time == 0.0
    
    def test_problem_create_empty_tour(self, sample_problem):
        """Test create_empty_tour() method."""
        tour = sample_problem.create_empty_tour()
        assert isinstance(tour, Tour)
        assert len(tour.visited_landmarks) == 0
        assert tour.problem == sample_problem
    
    def test_problem_unvisited_landmarks(self, sample_problem):
        """Test unvisited_landmarks() method."""
        tour = sample_problem.create_empty_tour()
        unvisited = sample_problem.unvisited_landmarks(tour)
        assert len(unvisited) == 2
        
        tour.add_landmark(sample_problem.landmarks[0])
        unvisited = sample_problem.unvisited_landmarks(tour)
        assert len(unvisited) == 1
        assert sample_problem.landmarks[1] in unvisited
    
    def test_problem_feasible_candidates(self, sample_problem):
        """Test feasible_candidates() method."""
        tour = sample_problem.create_empty_tour()
        candidates = sample_problem.feasible_candidates(tour)
        assert len(candidates) == 2
    
    def test_problem_random_tour(self, sample_problem):
        """Test random_tour() method."""
        tour = sample_problem.random_tour()
        assert isinstance(tour, Tour)
        assert tour.problem == sample_problem
    
    def test_problem_repr(self, sample_problem):
        """Test string representation."""
        repr_str = repr(sample_problem)
        assert "landmarks=2" in repr_str
        assert "budget=180" in repr_str
        assert "MONDAY" in repr_str


class TestTour:
    """Tests for Tour class."""
    
    @pytest.fixture
    def sample_problem(self):
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
                visit_duration=45,
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
            )
        ]
        
        return Problem(
            hotel=hotel,
            landmarks=landmarks,
            time_budget=300,
            tour_day=Day.MONDAY,
            start_time=540
        )
    
    def test_tour_creation(self, sample_problem):
        """Test Tour creation."""
        tour = Tour(sample_problem)
        assert tour.problem == sample_problem
        assert len(tour.visited_landmarks) == 0
    
    def test_tour_add_landmark(self, sample_problem):
        """Test add_landmark() method."""
        tour = Tour(sample_problem)
        landmark = sample_problem.landmarks[0]
        
        tour.add_landmark(landmark)
        assert landmark in tour.visited_landmarks
        assert len(tour.visited_landmarks) == 1
    
    def test_tour_add_landmark_duplicate(self, sample_problem):
        """Test adding duplicate landmark."""
        tour = Tour(sample_problem)
        landmark = sample_problem.landmarks[0]
        
        tour.add_landmark(landmark)
        with pytest.raises(ValueError):
            tour.add_landmark(landmark)
    
    def test_tour_add_landmark_at_position(self, sample_problem):
        """Test add_landmark() at specific position."""
        tour = Tour(sample_problem)
        lm1 = sample_problem.landmarks[0]
        lm2 = sample_problem.landmarks[1]
        
        tour.add_landmark(lm1)
        tour.add_landmark(lm2, position=0)
        
        assert tour.visited_landmarks[0] == lm2
        assert tour.visited_landmarks[1] == lm1
    
    def test_tour_remove_landmark(self, sample_problem):
        """Test remove_landmark() method."""
        tour = Tour(sample_problem)
        landmark = sample_problem.landmarks[0]
        
        tour.add_landmark(landmark)
        assert landmark in tour
        
        tour.remove_landmark(landmark)
        assert landmark not in tour
    
    def test_tour_remove_landmark_not_in_tour(self, sample_problem):
        """Test removing landmark not in tour."""
        tour = Tour(sample_problem)
        landmark = sample_problem.landmarks[0]
        
        with pytest.raises(ValueError):
            tour.remove_landmark(landmark)
    
    def test_tour_total_score(self, sample_problem):
        """Test total_score() method."""
        tour = Tour(sample_problem)
        assert tour.total_score() == 0
        
        tour.add_landmark(sample_problem.landmarks[0])
        assert tour.total_score() == 8.0
        
        tour.add_landmark(sample_problem.landmarks[1])
        assert tour.total_score() == 13.0
    
    def test_tour_swap_landmarks(self, sample_problem):
        """Test swap_landmarks() method."""
        tour = Tour(sample_problem)
        lm1 = sample_problem.landmarks[0]
        lm2 = sample_problem.landmarks[1]
        
        tour.add_landmark(lm1)
        tour.add_landmark(lm2)
        
        tour.swap_landmarks(lm1, lm2)
        assert tour.visited_landmarks[0] == lm2
        assert tour.visited_landmarks[1] == lm1
    
    def test_tour_swap_by_index(self, sample_problem):
        """Test swap_by_index() method."""
        tour = Tour(sample_problem)
        lm1 = sample_problem.landmarks[0]
        lm2 = sample_problem.landmarks[1]
        
        tour.add_landmark(lm1)
        tour.add_landmark(lm2)
        
        tour.swap_by_index(0, 1)
        assert tour.visited_landmarks[0] == lm2
        assert tour.visited_landmarks[1] == lm1
    
    def test_tour_swap_by_index_invalid(self, sample_problem):
        """Test swap_by_index() with invalid indices."""
        tour = Tour(sample_problem)
        tour.add_landmark(sample_problem.landmarks[0])
        
        with pytest.raises(IndexError):
            tour.swap_by_index(0, 5)
    
    def test_tour_copy(self, sample_problem):
        """Test copy() method."""
        tour = Tour(sample_problem)
        tour.add_landmark(sample_problem.landmarks[0])
        tour.add_landmark(sample_problem.landmarks[1])
        
        tour_copy = tour.copy()
        assert tour_copy.problem == tour.problem
        assert len(tour_copy.visited_landmarks) == len(tour.visited_landmarks)
        assert tour_copy.visited_landmarks is not tour.visited_landmarks
    
    def test_tour_contains(self, sample_problem):
        """Test __contains__() method."""
        tour = Tour(sample_problem)
        landmark = sample_problem.landmarks[0]
        
        assert landmark not in tour
        tour.add_landmark(landmark)
        assert landmark in tour
    
    def test_tour_len(self, sample_problem):
        """Test __len__() method."""
        tour = Tour(sample_problem)
        assert len(tour) == 0
        
        tour.add_landmark(sample_problem.landmarks[0])
        assert len(tour) == 1
    
    def test_tour_simulate(self, sample_problem):
        """Test simulate() method."""
        tour = Tour(sample_problem)
        result = tour.simulate()
        
        assert isinstance(result, SimulationResult)
        assert result.is_valid is True
        assert result.total_duration >= 0
    
    def test_tour_is_valid(self, sample_problem):
        """Test is_valid() method."""
        tour = Tour(sample_problem)
        assert tour.is_valid() is True
        
        tour.add_landmark(sample_problem.landmarks[0])
        assert tour.is_valid() is True
    
    def test_tour_cache_invalidation(self, sample_problem):
        """Test that cache is invalidated on modifications."""
        tour = Tour(sample_problem)
        
        # Get cached result
        result1 = tour.simulation_cache()
        
        # Add landmark and verify cache is invalidated
        tour.add_landmark(sample_problem.landmarks[0])
        result2 = tour.simulation_cache()
        
        assert result1 is not result2
    
    def test_tour_str(self, sample_problem):
        """Test string representation."""
        tour = Tour(sample_problem)
        tour.add_landmark(sample_problem.landmarks[0])
        
        str_repr = str(tour)
        assert "Hotel" in str_repr
        assert "Museum" in str_repr
        assert "Valid" in str_repr
