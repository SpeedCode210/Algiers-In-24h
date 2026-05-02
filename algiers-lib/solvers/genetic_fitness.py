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


class PenaltyFitnessFunction(FitnessFunction):
    """Fitness function that penalizes time-window violations and overtime.
 
    Computes fitness as the total interest score of visited landmarks minus
    penalties for visiting landmarks outside their time windows and for
    exceeding the overall time budget.
 
    Attributes:
        invalid_penalty: Score deducted per landmark visited outside its
            time window.
        overtime_penalty: Score deducted per minute the tour exceeds the
            time budget.
    """
    def __init__(self, invalid_penalty: float = 2.0, overtime_penalty: float = 1.0) -> None:
        """Initialize the penalty fitness function.
 
        Args:
            invalid_penalty: Penalty applied per time-window violation.
                Defaults to 2.0.
            overtime_penalty: Penalty applied per minute over the time budget.
                Defaults to 1.0.
        """
        self.invalid_penalty = invalid_penalty
        self.overtime_penalty = overtime_penalty

    def fitness(self, tour: Tour) -> float:
        """Compute fitness with penalties for violations and overtime.
 
        Args:
            tour: The tour to evaluate.
 
        Returns:
            Fitness score rounded to the nearest integer. Can be negative if
            penalties outweigh the interest score.
        """
        invalid_count, total_duration = self._evaluate_tour(tour)
        raw_score = (
            tour.total_score()
            - self.invalid_penalty * invalid_count
            - self.overtime_penalty * max(total_duration - tour.problem.time_budget, 0)
        )
        return int(round(raw_score))


class InfeasibilityFitnessFunction(PenaltyFitnessFunction):
    """Fitness function that strongly penalizes infeasible tours.
 
    Extends PenaltyFitnessFunction by returning a heavily negative score
    for tours with time-window violations, proportional to the gap between
    the tour's interest score and the maximum possible interest score. Empty
    tours receive negative infinity. Feasible tours are scored identically
    to PenaltyFitnessFunction.
    """
    def fitness(self, tour: Tour) -> float:
        """Compute fitness with strong infeasibility penalization.
 
        Args:
            tour: The tour to evaluate.
 
        Returns:
            ``float('-inf')`` for empty tours, a large negative value for
            infeasible tours scaled by the interest gap, or the penalty-based
            score for feasible tours.
        """
        invalid_count= self._evaluate_tour(tour)[0]
        if len(tour.visited_landmarks) == 0:
            return NEGATIVE_INFINITY
        if invalid_count > 0:
            total_possible_interest = sum(
                float(landmark.interest_score) for landmark in tour.problem.landmarks
            )
            return tour.total_score() - total_possible_interest*invalid_count
        return super().fitness(tour)


class FeasibilityFitnessFunction(FitnessFunction):
    """Fitness function for use when all tours in the population are guaranteed feasible.
 
    Scores tours based on total interest reward plus a small bonus for
    finishing early, encouraging efficient use of the time budget without
    sacrificing reward.
    """
    # This is used when we are sure that the tours are always feasible 
    def fitness(self, tour: Tour) -> float:
        """Compute fitness as reward plus a time-efficiency bonus.
 
        Args:
            tour: The tour to evaluate. Must be a valid tour.
 
        Returns:
            Total interest score of all visited landmarks plus a normalized
            bonus for remaining time budget.
        """
        total_duration = self._evaluate_tour(tour)[1]
        total_reward = sum(
            float(landmark.interest_score) for landmark in tour.visited_landmarks
        )
        return total_reward + 2*(tour.problem.time_budget-total_duration)/tour.problem.time_budget # this is another interesting one , must be tested **5/total_duration**2+1 
