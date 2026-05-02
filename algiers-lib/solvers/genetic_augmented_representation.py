from __future__ import annotations

from typing import Optional
from models.landmark import Landmark
from models.problem import Problem
from models.tour import Tour


class AugmentedRepresentation:
    """Augmented representation for genetic algorithm optimization of tour routes.
 
    Extends a tour representation with a timeline that includes flexibility
    (max_shift) values for each landmark visit. These values are used by the
    tailored crossover and mutation operators to determine valid cut points and
    insertion positions without re-simulating the full tour.
 
    Attributes:
        landmarks: Ordered list of landmarks in the tour.
        problem: The optimization problem instance.
        timeline: List of tuples, one per landmark, each containing:
            - arrival_time (float): Time the tourist arrives at the landmark.
            - wait (float): Idle time before the visit window opens.
            - start_time (float): Actual visit start time.
            - departure_time (float): Time the tourist leaves the landmark.
            - max_shift (float): Maximum delay that can be applied to start_time
              while still respecting all downstream time windows and the budget.
    """

    def __init__(
        self,
        landmarks: list[Landmark],
        problem: Optional[Problem] = None,
    ) -> None:
        """Initialize an augmented representation.
 
        Args:
            landmarks: Ordered list of landmarks in the tour route.
            problem: The optimization problem instance. Optional for basic
                operations that do not require problem context.
        """
        self.landmarks = landmarks
        self.problem = problem
        self.timeline: list[tuple[float, float, float, float, float]] = []

    @classmethod
    def from_tour(cls, tour: Tour) -> "AugmentedRepresentation":
        """Create an augmented representation from a valid tour.
 
        Runs the tour simulation to obtain the scheduled timeline, then performs
        a backward pass to compute the max_shift for each landmark. The max_shift
        at position i represents how much the visit start at i can be delayed
        while guaranteeing that all subsequent visits and the return to the hotel
        still fit within their time windows and the overall time budget.
 
        Args:
            tour: A valid tour to convert. Must have a non-empty simulation result.
 
        Returns:
            AugmentedRepresentation with a fully computed timeline. If the tour
            has no visited landmarks, the timeline will be empty.
        """
        simulation = tour.simulation_cache()
        landmarks = [entry.landmark for entry in simulation.entries]
        augmented = cls(landmarks, tour.problem)

        if not simulation.entries:
            return augmented

        timeline: list[list[float]] = []
        for entry in simulation.entries:
            arrival = entry.arrival_time
            start = entry.visit_start_time
            end = entry.departure_time
            wait = start - arrival
            timeline.append([arrival, wait, start, end, 0.0])

        max_shifts: list[float] = [0.0] * len(timeline)

        for index in range(len(timeline) - 1, -1, -1):
            entry = simulation.entries[index]
            day_slots = entry.landmark.schedule.get_slots(tour.problem.tour_day)
            start_time = timeline[index][2]
            visit_duration = entry.landmark.visit_duration

            if index == len(timeline) - 1:
                next_term = tour.problem.time_budget - simulation.total_duration
            else:
                next_wait = float(timeline[index + 1][1])
                next_term = next_wait + max_shifts[index + 1]

            if day_slots:
                max_shift = cls._compute_closing_term(
                    day_slots, start_time, visit_duration, next_term
                )
            else:
                max_shift = next_term

            max_shifts[index] = max_shift
            timeline[index][4] = max_shift

        augmented.timeline = [tuple(entry) for entry in timeline]
        return augmented

    @staticmethod
    def _compute_closing_term(# this must be consulted 
        day_slots: list,
        start_time: float,
        visit_duration: float,
        next_limit: float,
    ) -> float:
        """Compute the maximum allowable shift for a single landmark visit.
 
        Finds the tightest constraint between the landmark's own closing time
        and the downstream propagated limit. The result is the largest delay
        that can be applied to start_time without violating the slot's closing
        time, capped by the downstream constraint next_limit.
 
        Args:
            day_slots: Available time slots for the landmark on the tour day.
            start_time: Currently scheduled visit start time.
            visit_duration: Duration of the landmark visit in minutes.
            next_limit: Maximum allowable delay propagated from downstream
                constraints (subsequent landmarks and time budget).
 
        Returns:
            Maximum time shift in minutes that can be applied to start_time.
            Returns next_limit if the closing slack of every valid slot exceeds
            next_limit, meaning the downstream constraint is the binding one.
        """
        candidates = [
            float(slot.close_time - start_time - visit_duration)
            for slot in day_slots
            if slot.open_time <= start_time and (start_time + visit_duration) <= slot.close_time
        ]

        valid_candidates = [candidate for candidate in candidates if candidate <= next_limit]
        if valid_candidates:
            return max(valid_candidates)

        return next_limit