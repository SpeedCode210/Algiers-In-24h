from dataclasses import dataclass , field 
from typing import Optional

from landmark import Landmark
from problem import Problem 

@dataclass
class ScheduleEntry: 

    landmark: Landmark
    arrival_time: int
    waiting_time: int
    departure_time: int 
    visit_start_time: int = field(init = False)

    def __post_init__(self) -> None:
        self.departure_time = self.visit_start_time + self.landmark.visit_duration

    @property
    def waiting_time(self) -> int:
        return self.visit_start_time - self.arrival_time
    


      

        



