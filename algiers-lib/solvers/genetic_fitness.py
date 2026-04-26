from __future__ import annotations

import math
from models.tour import Tour

NEGATIVE_INFINITY = float("-inf")


class FitnessFunction:

    def fitness(self, tour: Tour) -> float:
        raise NotImplementedError("Fitness function class must implement fitness().")

    def _evaluate_tour(self, tour: Tour) -> tuple[int, float]:
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

    def __init__(self, invalid_penalty: float = 2.0, overtime_penalty: float = 1.0) -> None:
        self.invalid_penalty = invalid_penalty
        self.overtime_penalty = overtime_penalty

    def fitness(self, tour: Tour) -> int:
        invalid_count, total_duration = self._evaluate_tour(tour)
        raw_score = (
            tour.total_score()
            - self.invalid_penalty * invalid_count
            - self.overtime_penalty * max(total_duration - tour.problem.time_budget, 0)
        )
        return int(round(raw_score))


class InfeasibilityFitnessFunction(PenaltyFitnessFunction):
    def fitness(self, tour: Tour) -> float:
        invalid_count, total_duration = self._evaluate_tour(tour)
        if len(tour.visited_landmarks) == 0:
            return NEGATIVE_INFINITY
        if invalid_count > 0:
            total_possible_interest = sum(
                float(landmark.interest_score) for landmark in tour.problem.landmarks
            )
            return tour.total_score() - total_possible_interest
        return super().fitness(tour)


class FeasibilityFitnessFunction(FitnessFunction):
    # This is used when we are sure that the tours are always feasible 
    def fitness(self, tour: Tour) -> float:
        total_duration = tour.simulation_cache().total_duration
        total_reward = sum(
            float(landmark.interest_score) for landmark in tour.visited_landmarks
        )
        return total_reward + 2*(tour.problem.time_budget-total_duration)/tour.problem.time_budget # this is another interesting one , must be tested **5/total_duration**2+1 
