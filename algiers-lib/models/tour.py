from __future__ import annotations
from dataclasses import dataclass , field 
from typing import Optional
import math 

from models.landmark import Landmark
from utils.time import time_in_string
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.problem import Problem

@dataclass
class ScheduleEntry: 

    landmark: Landmark
    arrival_time: float
    visit_start_time: int 
    departure_time: int = field(init = False)

    def __post_init__(self) -> None:
        self.departure_time = self.visit_start_time + self.landmark.visit_duration

    @property
    def waiting_time(self) -> float:
        return self.visit_start_time - self.arrival_time
    

@dataclass
class SimulationResult:

    total_duration: float
    is_valid: bool
    entries: list[ScheduleEntry] = field(default_factory=list)
    


class Tour:

    def __init__(self, problem: Problem , visited_landmarks: Optional[list["Landmark"]] = None ) -> None:

        self.problem =  problem
        self.visited_landmarks = visited_landmarks if visited_landmarks is not None else []
        self._cache: Optional[SimulationResult] = None #for caching the last simulation result
    
    def simulate(self) -> SimulationResult:

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

        if self._cache is None:
            self._cache = self.simulate()

        return self._cache
    
    def _invalidate_cache(self) -> None: #After each mutation

        self._cache = None

    def is_valid(self) -> bool:

        return self.simulation_cache().is_valid
    
    def total_score(self) -> float:

        return sum(lm.interest_score for lm in self.visited_landmarks)
    
    def add_landmark(self , landmark: Landmark, position: Optional[int] = None) -> None:

        if landmark  in self.visited_landmarks:
            raise ValueError(f"{landmark.name} is already in the tour.")
        
        if position is None:
            self.visited_landmarks.append(landmark)

        else:
            self.visited_landmarks.insert(position , landmark)

        self._invalidate_cache()

    def remove_landmark(self , landmark: Landmark) -> None:

        if landmark not in self.visited_landmarks:
            raise ValueError(f"{landmark.name} does not exist in the tour.")
        
        self.visited_landmarks.remove(landmark)
        self._invalidate_cache()

    def swap_landmarks(self, lm1: Landmark , lm2: Landmark) -> None:

        if lm1 not in self.visited_landmarks:
            raise ValueError(f"{lm1.name} does not exist in the tour.")
        
        if lm2 not in self.visited_landmarks:
            raise ValueError(f"{lm2.name} does not exist in the tour.")
        
        i = self.visited_landmarks.index(lm1)
        j = self.visited_landmarks.index(lm2)
        self.visited_landmarks[i], self.visited_landmarks[j] = self.visited_landmarks[j], self.visited_landmarks[i]
        self._invalidate_cache()

    def swap_by_index(self, i: int, j: int) -> None:

        if not (0 <= i < len(self.visited_landmarks)):
            raise IndexError(f"Indices {i} is out of range.")
        
        if not (0 <= j < len(self.visited_landmarks)):
            raise IndexError(f"Indices {j} is out of range.")
        
        self.visited_landmarks[i], self.visited_landmarks[j] = self.visited_landmarks[j], self.visited_landmarks[i]
        self._invalidate_cache()

    def replace_landmark(self, old: Landmark , new: Landmark) -> None:

        if old not in self.visited_landmarks:
            raise ValueError(f"{old.name} does not exist in the tour.")
        
        if new in self.visited_landmarks:
            raise ValueError(f"{new.name} is already in the tour.")
        
        index = self.visited_landmarks.index(old)
        self.visited_landmarks[index] = new
        self._invalidate_cache()

    def copy(self) -> Tour:

        return Tour(self.problem, list(self.visited_landmarks))
    
    def __contains__(self, landmark: Landmark) -> bool:

        return landmark in self.visited_landmarks
    
    def __len__(self) -> int:

        return len(self.visited_landmarks)
    
    def __str__(self) -> str:

        simulation = self.simulation_cache()
        tour_details = [f"Start: Hotel {self.problem.hotel.name} at {time_in_string(self.problem.start_time)}"]

        for entry in simulation.entries:
            wait_str = f" | wait: {time_in_string(round(entry.waiting_time))}" if entry.waiting_time > 0 else ""
            tour_details.append(f" {entry.landmark.name} | arrival: {time_in_string(round(entry.arrival_time))}{wait_str} | start visit: {time_in_string(entry.visit_start_time)} | departure: {time_in_string(entry.departure_time)} ")

        tour_details.append(f"end: Hotel {self.problem.hotel.name}")
        tour_details.append(f"Valid: {simulation.is_valid} | Total duration: {simulation.total_duration:.1f} min | Score: {self.total_score()}")
        return "\n".join(tour_details)


        

            


      

        



