from __future__ import annotations

import math
from models.tour import Tour

NEGATIVE_INFINITY = float("-inf")


class FitnessFunction:
    """Abstract base class for tour fitness evaluation.

    All concrete fitness functions must inherit from this class and implement
    the ``fitness`` method. The optional ``_evaluate_tour`` helper performs a
    forward simulation that tolerates invalid visits, making it useful for
    penalty-based subclasses.
    """

    def fitness(self, tour: Tour) -> float:
        """Compute the fitness score of a tour.

        Higher values indicate better tours. Subclasses define the exact
        scoring formula.

        Args:
            tour: The tour to evaluate.

        Returns:
            A scalar fitness value.

        Raises:
            NotImplementedError: If called on the base class directly.
        """
        raise NotImplementedError("Fitness function class must implement fitness().")

    def _evaluate_tour(self, tour: Tour) -> tuple[int, float]:
        """Simulate the tour and count constraint violations.

        Performs a forward pass through the tour's landmarks. If a landmark
        has no valid time window at its arrival time, it is counted as invalid
        and the visit is still assumed to start at the ceiled arrival time so
        that the simulation can continue.

        Args:
            tour: The tour to simulate.

        Returns:
            A tuple of:
                - invalid_count (int): Number of landmarks visited outside their
                  time windows.
                - total_duration (float): Total elapsed time from the start of
                  the tour to the return to the hotel, in minutes.
        """
        current_position = tour.problem.hotel
        current_time = float(tour.problem.start_time)
        invalid_count = 0

        for landmark in tour.visited_landmarks:
            travel_time = tour.problem.travel_time(current_position, landmark)
            arrival_time = current_time + travel_time
            visit_start_time = landmark.schedule.earliest_valid_start(
                tour.problem.tour_day,
                math.ceil(arrival_time),
                landmark.visit_duration,
            )

            if visit_start_time is None:
                invalid_count += 1
                # this might be the problem 
                visit_start_time = math.ceil(arrival_time)

            current_time = float(visit_start_time + landmark.visit_duration)
            current_position = landmark

        return_travel_time = tour.problem.travel_time(current_position, tour.problem.hotel)
        total_duration = float((current_time + return_travel_time) - tour.problem.start_time)
        return invalid_count, total_duration


class ScoreFitnessFunction(FitnessFunction):
    """Fitness function that returns just the total score .
    this uses the evaluate 
    """

    def fitness(self, tour: Tour) -> float:
        """Compute fitness with penalties for violations and overtime.

        Args:
            tour: The tour to evaluate.

        Returns:
            Total interest score of all visited landmarks.
        """
        invalid_count, total_duration = self._evaluate_tour(tour)
        return total_duration



# this is used in the Tailored genetic solver 
class FeasibilityFitnessFunction(FitnessFunction):
    """Fitness function for use when all tours in the population are guaranteed feasible.

    Scores tours based on total interest reward plus a small bonus for
    finishing early, encouraging efficient use of the time budget without
    sacrificing reward.
    """
    def fitness(self, tour: Tour) -> float:
        """Compute fitness as reward plus a time-efficiency bonus.

        Args:
            tour: The tour to evaluate. Must be a valid tour.

        Returns:
            Total interest score of all visited landmarks plus a normalized
            bonus for remaining time budget.
        """
        invalid_count, total_duration = self._evaluate_tour(tour)
        total_reward = sum(
            float(landmark.interest_score) for landmark in tour.visited_landmarks
        )
        return total_reward + (tour.problem.time_budget - total_duration) / tour.problem.time_budget 
