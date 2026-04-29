from __future__ import annotations
from dataclasses import dataclass , field 
from typing import Optional

from models.landmark import Landmark
from utils.time import time_in_string
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.problem import Problem

@dataclass
class ScheduleEntry: 
    """Represents a scheduled visit to a landmark in the tour.

    Attributes:
        landmark (Landmark): The landmark being visited.
        arrival_time (float): The time of arrival at the landmark.
        visit_start_time (float): The time when the visit starts.
        departure_time (float): The time when the visit ends (computed).
    """

    landmark: Landmark
    arrival_time: float
    visit_start_time: float 
    departure_time: float = field(init = False)

    def __post_init__(self) -> None:
        self.departure_time = self.visit_start_time + self.landmark.visit_duration

    @property
    def waiting_time(self) -> float:
        return self.visit_start_time - self.arrival_time
    

@dataclass
class SimulationResult:
    """Result of simulating a tour's schedule.

    Attributes:
        total_duration (float): Total time for the tour including return to hotel.
        is_valid (bool): Whether the tour fits within time budget and constraints.
        entries (list[ScheduleEntry]): List of scheduled visits.
    """

    total_duration: float
    is_valid: bool
    entries: list[ScheduleEntry] = field(default_factory=list)
    


class Tour:
    """Represents a tour visiting a sequence of landmarks starting and ending at the hotel.

    The tour maintains a list of visited landmarks and caches simulation results for efficiency.
    """

    def __init__(self, problem: Problem , visited_landmarks: Optional[list["Landmark"]] = None ) -> None:
        """Initialize a Tour.

        Args:
            problem (Problem): The problem instance this tour belongs to.
            visited_landmarks (Optional[list[Landmark]]): Initial list of landmarks in the tour.
        """
        self.problem =  problem
        self.visited_landmarks = visited_landmarks if visited_landmarks is not None else []
        self._cache: Optional[SimulationResult] = None #for caching the last simulation result
    
    def simulate(self) -> SimulationResult:
        """Simulate the tour schedule, checking validity and computing timings.

        Returns:
            SimulationResult: The result of the simulation including validity and schedule.
        """

        entries: list[ScheduleEntry] = []
        current_position: Landmark = self.problem.hotel
        current_time: float = self.problem.start_time

        for landmark in self.visited_landmarks:

            travel_time = self.problem.travel_time(current_position, landmark)
            arrival_time = current_time + travel_time
            visit_start_time = landmark.schedule.earliest_valid_start(
                self.problem.tour_day, arrival_time , landmark.visit_duration)
            if visit_start_time is None:

                return_travel = self.problem.travel_time(current_position, self.problem.hotel)
                return SimulationResult(
                    total_duration=float(arrival_time + return_travel - self.problem.start_time),
                    is_valid=False,
                    entries=entries
                )
            
            entry = ScheduleEntry(landmark=landmark,
                                arrival_time=float(arrival_time),
                                visit_start_time=visit_start_time)
            
            entries.append(entry)
            current_time = float(entry.departure_time)
            current_position = landmark

        return_travel_time = self.problem.travel_time(current_position, self.problem.hotel)
        total_duration = float(( current_time + return_travel_time ) - self.problem.start_time)

        if total_duration > self.problem.time_budget:

            return SimulationResult( total_duration=total_duration , is_valid=False , entries=entries ) 
        
        return SimulationResult(total_duration=total_duration , is_valid=True , entries=entries)
    
    def simulation_cache(self) -> SimulationResult:
        """Get the cached simulation result, computing it if necessary.

        Returns:
            SimulationResult: The cached or newly computed simulation result.
        """

        if self._cache is None:
            self._cache = self.simulate()

        return self._cache
    
    def _invalidate_cache(self) -> None: #After each mutation
        """Invalidate the simulation cache after tour modifications."""

        self._cache = None

    def is_valid(self) -> bool:
        """Check if the tour is valid according to time and schedule constraints.

        Returns:
            bool: True if the tour is valid, False otherwise.
        """

        return self.simulation_cache().is_valid
    
    def total_score(self) -> float:
        """Calculate the total interest score of the visited landmarks.

        Returns:
            float: The sum of interest scores of all visited landmarks.
        """

        return sum(lm.interest_score for lm in self.visited_landmarks)
    
    def add_landmark(self , landmark: Landmark, position: Optional[int] = None) -> None:
        """Add a landmark to the tour at the specified position.

        Args:
            landmark (Landmark): The landmark to add.
            position (Optional[int]): Position to insert at, or append if None.

        Raises:
            ValueError: If the landmark is already in the tour.
        """

        if landmark  in self.visited_landmarks:
            raise ValueError(f"{landmark.name} is already in the tour.")
        
        if position is None:
            self.visited_landmarks.append(landmark)

        else:
            self.visited_landmarks.insert(position , landmark)

        self._invalidate_cache()

    def remove_landmark(self , landmark: Landmark) -> None:
        """Remove a landmark from the tour.

        Args:
            landmark (Landmark): The landmark to remove.

        Raises:
            ValueError: If the landmark is not in the tour.
        """

        if landmark not in self.visited_landmarks:
            raise ValueError(f"{landmark.name} does not exist in the tour.")
        
        self.visited_landmarks.remove(landmark)
        self._invalidate_cache()

    def swap_landmarks(self, lm1: Landmark , lm2: Landmark) -> None:
        """Swap two landmarks in the tour.

        Args:
            lm1 (Landmark): First landmark to swap.
            lm2 (Landmark): Second landmark to swap.

        Raises:
            ValueError: If either landmark is not in the tour.
        """

        if lm1 not in self.visited_landmarks:
            raise ValueError(f"{lm1.name} does not exist in the tour.")
        
        if lm2 not in self.visited_landmarks:
            raise ValueError(f"{lm2.name} does not exist in the tour.")
        
        i = self.visited_landmarks.index(lm1)
        j = self.visited_landmarks.index(lm2)
        self.visited_landmarks[i], self.visited_landmarks[j] = self.visited_landmarks[j], self.visited_landmarks[i]
        self._invalidate_cache()

    def swap_by_index(self, i: int, j: int) -> None:
        """Swap landmarks at the given indices.

        Args:
            i (int): Index of first landmark.
            j (int): Index of second landmark.

        Raises:
            IndexError: If indices are out of range.
        """

        if not (0 <= i < len(self.visited_landmarks)):
            raise IndexError(f"Indices {i} is out of range.")
        
        if not (0 <= j < len(self.visited_landmarks)):
            raise IndexError(f"Indices {j} is out of range.")
        
        self.visited_landmarks[i], self.visited_landmarks[j] = self.visited_landmarks[j], self.visited_landmarks[i]
        self._invalidate_cache()

    def replace_landmark(self, old: Landmark , new: Landmark) -> None:
        """Replace one landmark with another in the tour.

        Args:
            old (Landmark): The landmark to replace.
            new (Landmark): The new landmark.

        Raises:
            ValueError: If old is not in the tour or new is already in the tour.
        """

        if old not in self.visited_landmarks:
            raise ValueError(f"{old.name} does not exist in the tour.")
        
        if new in self.visited_landmarks:
            raise ValueError(f"{new.name} is already in the tour.")
        
        index = self.visited_landmarks.index(old)
        self.visited_landmarks[index] = new
        self._invalidate_cache()

    def copy(self) -> Tour:
        """Create a shallow copy of the tour.

        Returns:
            Tour: A new Tour instance with the same landmarks.
        """

        return Tour(self.problem, list(self.visited_landmarks))
    
    def __contains__(self, landmark: Landmark) -> bool:
        """Check if a landmark is in the tour.

        Args:
            landmark (Landmark): The landmark to check.

        Returns:
            bool: True if the landmark is in the tour.
        """

        return landmark in self.visited_landmarks
    
    def __len__(self) -> int:
        """Get the number of landmarks in the tour.

        Returns:
            int: The number of visited landmarks.
        """

        return len(self.visited_landmarks)
    
    def __str__(self) -> str:
        """Return a string representation of the tour schedule."""

        simulation = self.simulation_cache()
        tour_details = [f"Start: Hotel {self.problem.hotel.name} at {time_in_string(self.problem.start_time)}"]

        for entry in simulation.entries:
            wait_str = f" | wait: {time_in_string(round(entry.waiting_time))}" if entry.waiting_time > 0 else ""
            tour_details.append(f" {entry.landmark.name} | arrival: {time_in_string(round(entry.arrival_time))}{wait_str} | start visit: {time_in_string(entry.visit_start_time)} | departure: {time_in_string(entry.departure_time)} ")

        tour_details.append(f"end: Hotel {self.problem.hotel.name}")
        tour_details.append(f"Valid: {simulation.is_valid} | Total duration: {simulation.total_duration:.1f} min | Score: {self.total_score()}")
        return "\n".join(tour_details)




