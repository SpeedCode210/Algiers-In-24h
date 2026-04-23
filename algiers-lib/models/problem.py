from __future__ import annotations

from typing import Optional , TYPE_CHECKING
import random

from models.landmark import Day, Landmark, loadLandmarks, loadHotel
from utils.distance import travel_time_minutes
if TYPE_CHECKING:
    from models.tour import Tour 

class Problem:
    """Represents the Orienteering Problem instance with landmarks, hotel, and constraints.

    This class encapsulates the problem data including travel times, time budget, and tour day.
    It provides methods to query feasible landmarks and create tours.
    """

    def __init__(self, hotel: Landmark, landmarks: list[Landmark], time_budget: int, tour_day: Day, start_time: int = 540,) -> None:
        """Initialize the Problem instance.

        Args:
            hotel (Landmark): The starting and ending point of tours.
            landmarks (list[Landmark]): List of available landmarks to visit.
            time_budget (int): Maximum allowed time for the tour in minutes.
            tour_day (Day): The day of the week for the tour.
            start_time (int): Start time in minutes from midnight. Defaults to 540 (9 AM).
        """

        self.hotel: Landmark = hotel
        self.landmarks: list[Landmark] = landmarks
        self.time_budget: int = time_budget
        self.tour_day: Day = tour_day
        self.start_time: int = start_time

        self._travel_matrix: dict[tuple[str, str], float] = {}
        self._precompute_travel_matrix()


    def _precompute_travel_matrix(self) -> None:
        """Precompute travel times between all pairs of locations for efficiency."""

        all_locations: list[Landmark] = [self.hotel] + self.landmarks
        for origin in all_locations:
            for destination in all_locations:
                if origin.id == destination.id:
                    continue
                key = (origin.id, destination.id)
                self._travel_matrix[key] = travel_time_minutes(origin.coordinates, destination.coordinates)


    def travel_time(self, origin: Landmark, destination: Landmark) -> float:
        """Get the precomputed travel time between two landmarks.

        Args:
            origin (Landmark): Starting landmark.
            destination (Landmark): Destination landmark.

        Returns:
            float: Travel time in minutes.
        """

        if origin.id == destination.id:
            return 0.0
        return self._travel_matrix[(origin.id, destination.id)]
    

    def create_empty_tour(self) -> Tour:
        """Create an empty tour for this problem.

        Returns:
            Tour: A new empty Tour instance.
        """

        from models.tour import Tour
        return Tour(problem=self)
    
    def random_tour(self) -> Tour:
        """Generate valid random tour potentially used for initial solutions """

        tour = self.create_empty_tour()
        candidates = self.feasible_candidates(tour)
        random.shuffle(candidates)
        
        for landmark in candidates:
            tour.add_landmark(landmark)
            if not tour.is_valid():
                tour.remove_landmark(landmark)
        
        return tour


    def unvisited_landmarks(self, tour: Tour) -> list[Landmark]:
        """Get landmarks not yet visited in the given tour.

        Args:
            tour (Tour): The current tour.

        Returns:
            list[Landmark]: List of unvisited landmarks.
        """

        visited_ids = {lm.id for lm in tour.visited_landmarks}
        return [lm for lm in self.landmarks if lm.id not in visited_ids]
    
    
    def feasible_candidates(self, tour: Tour) -> list[Landmark]:
        """Get landmarks that are unvisited and open on the tour day.

        Args:
            tour (Tour): The current tour.

        Returns:
            list[Landmark]: List of feasible candidate landmarks.
        """

        return [lm for lm in self.unvisited_landmarks(tour) if lm.schedule.is_open_on(self.tour_day)]
    
    
    def __repr__(self) -> str:
        """Developer-facing summary of the problem instance."""
        return (
            f"Problem("
            f"landmarks={len(self.landmarks)}, "
            f"budget={self.time_budget} min, "
            f"day={self.tour_day.name}, "
            f"hotel='{self.hotel.name}')"
        )
    

    @classmethod
    def LoadProblem(cls, landmarks_path: str, hotel_path: str, time_budget: int, tour_day: Day, start_time: int = 540,) -> "Problem":
        """Load a Problem instance from CSV files.

        Args:
            landmarks_path (str): Path to the landmarks CSV file.
            hotel_path (str): Path to the hotel CSV file.
            time_budget (int): Time budget in minutes.
            tour_day (Day): Day of the tour.
            start_time (int): Start time in minutes. Defaults to 540.

        Returns:
            Problem: The loaded Problem instance.
        """

        hotel = loadHotel(hotel_path)
        landmarks = loadLandmarks(landmarks_path)
        return cls(
            hotel=hotel,
            landmarks=landmarks,
            time_budget=time_budget,
            tour_day=tour_day,
            start_time=start_time,
        )