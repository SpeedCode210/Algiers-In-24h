from __future__ import annotations

import math
import random
from typing import Optional
from models.landmark import Landmark
from models.problem import Problem
from models.tour import Tour


class AugmentedRepresentation:
    """Augmented representation for genetic algorithm optimization of tour routes.

    This class extends tour representations with timeline information that includes
    flexibility calculations for genetic crossover operations.
    """

    def __init__(
        self,
        landmarks: list[Landmark],
        problem: Optional[Problem] = None,
    ) -> None:
        """Initialize an augmented representation.

        Args:
            landmarks: List of landmarks in the tour route.
            problem: The optimization problem instance. Optional for basic operations.
        """
        self.landmarks = landmarks
        self.problem = problem
        self.timeline: list[tuple[float, float, float, float, float]] = []

    @classmethod
    def from_tour(cls, tour: Tour) -> "AugmentedRepresentation":
        """Create an augmented representation from a tour with timeline calculations.

        Computes the timeline with flexibility information for each landmark visit,
        including maximum allowable shifts for genetic crossover operations.

        Args:
            tour: The tour to convert to augmented representation.

        Returns:
            Augmented representation with computed timeline.
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
        """Compute the maximum allowable shift for a landmark visit.

        Calculates how much a visit can be delayed while still respecting
        time windows and subsequent constraints.

        Args:
            day_slots: Available time slots for the landmark on the tour day.
            start_time: Current planned start time of the visit.
            visit_duration: Duration of the landmark visit.
            next_limit: Maximum allowable delay from subsequent constraints.

        Returns:
            Maximum time shift allowed for this visit.
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