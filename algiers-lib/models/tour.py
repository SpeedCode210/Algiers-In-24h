from dataclasses import dataclass , field 
from typing import Optional
import math 

from landmark import Landmark
from problem import Problem 

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
    def __init__(self, problem: "Problem" , visited_landmarks: Optional[list["Landmark"]] = None ) -> None:
        self.problem =  problem
        self.visited_landmarks = visited_landmarks if visited_landmarks is not None else []
    
    def simulate(self) -> SimulationResult:
        entries: list = []
        current_position: Landmark = self.problem.hotel
        current_time: float = self.problem.start_time

        for landmark in self.visited_landmarks:

            travel_time = self.problem.travel_time(current_position, landmark)
            arrival_time = current_time + travel_time
            visit_start_time = landmark.schedule.earliest_valid_start(
                self.problem.tour_day, math.ceil(arrival_time) , landmark.visit_duration)

            if visit_start_time is None:

                return SimulationResult(total_duration=float(current_time - self.problem.start_time),
                                        is_valid=False,
                                        entries=entries) 
            
            entry = ScheduleEntry(landmark=landmark,
                                arrival_time=float(arrival_time),
                                visit_start_time=visit_start_time,)
            
            entries.append(entry)
            current_time = float(entry.departure_time)
            current_position = landmark

        return_travel_time = self.problem.travel_time(current_position, self.problem.hotel)
        total_duration = float(( current_time + return_travel_time ) - self.problem.start_time)

        if total_duration > self.problem.time_budget:
            return SimulationResult( total_duration=total_duration , is_valid=False , entries=entries ) 
        
        return SimulationResult(total_duration=total_duration , is_valid=True , entries=entries)
    

        


        

            


      

        



