from __future__ import annotations

import math
import random
from typing import Optional
from models.landmark import Landmark
from models.tour import Tour
from .genetic_augmented_representation import AugmentedRepresentation


class Mutation:
    """Mutation operator for the genetic algorithm.
 
    Applies stochastic modifications to a tour to introduce diversity into
    the population. Supports a tailored insertion strategy that uses timeline
    flexibility (max_shift) to find valid insertion positions, and a random
    deletion strategy.
 
    Attributes:
        insertion_probability: Probability of performing an insertion mutation
            rather than a deletion when the tour is non-empty.
    """

    def __init__(self, insertion_probability: float = 0.5) -> None:
        """Initialize the mutation operator.
 
        Args:
            insertion_probability: Probability in [0, 1] of inserting a landmark
                rather than deleting one when the tour is non-empty. Defaults to 0.5.
 
        Raises:
            ValueError: If insertion_probability is not in [0, 1].
        """
        if not 0.0 <= insertion_probability <= 1.0:
            raise ValueError("Insertion probability must be between 0 and 1.")

        self.insertion_probability = insertion_probability

    def mutate(self, tour: Tour) -> Tour:
        """Apply a single mutation to a copy of the given tour.
 
        If the tour is empty, always performs an insertion. Otherwise, performs
        a tailored insertion with probability ``insertion_probability`` and a
        deletion otherwise.
 
        Args:
            tour: The tour to mutate. A copy is made before any modification.
 
        Returns:
            A mutated Tour. The original tour is not modified.
        """
        mutated_tour = tour.copy()

        if len(mutated_tour.visited_landmarks) == 0:
            return self._insert(mutated_tour)

        if random.random() < self.insertion_probability:
            return self._tailored_insert(mutated_tour)

        return self._delete(mutated_tour)

    def _tailored_insert(self, tour: Tour) -> Tour:
        """Insert a landmark at a time-feasible position using max_shift guidance.
 
        Builds an augmented representation of the tour and searches for positions
        where a new landmark can be inserted before an existing one without
        violating that landmark's allowable start window. For each such position,
        the candidate with the best interest-to-slack ratio is selected. One
        insertion is then chosen at random from all feasible position-candidate
        pairs found.
 
        Falls back to random insertion if no feasible position is found or if
        the tour has no timeline.
 
        Args:
            tour: The tour to insert into. Modified in place.
 
        Returns:
            The modified tour with one landmark inserted, or the original tour
            if no feasible insertion was found.
        """
        available_landmarks = tour.problem.feasible_candidates(tour)
        if not available_landmarks:
            return tour

        augmented = AugmentedRepresentation.from_tour(tour)
        if not augmented.timeline:
            return self._insert(tour)

        best_choices = []

        for selected_index, selected_landmark in enumerate(augmented.landmarks):
            _, _, start_time, _, max_shift = augmented.timeline[selected_index]
            if max_shift <= 0:
                continue

            target_deadline = start_time + max_shift
            if selected_index == 0:
                prev_departure = tour.problem.start_time
                prev_landmark = tour.problem.hotel
            else:
                prev_departure = augmented.timeline[selected_index - 1][3]
                prev_landmark = augmented.landmarks[selected_index - 1]

            best_score_for_selected = -1.0
            best_candidate_for_selected = None

            for candidate in available_landmarks:
                travel_to_candidate = tour.problem.travel_time(prev_landmark, candidate)
                candidate_arrival = prev_departure + travel_to_candidate
                candidate_start = candidate.schedule.earliest_valid_start(
                    tour.problem.tour_day,
                    candidate_arrival,
                    candidate.visit_duration,
                )
                if candidate_start is None:
                    continue

                candidate_finish = candidate_start + candidate.visit_duration
                travel_to_selected = tour.problem.travel_time(candidate, selected_landmark)
                arrival_at_selected = candidate_finish + travel_to_selected

                if arrival_at_selected > target_deadline:
                    continue

                diff = target_deadline - arrival_at_selected

                score = candidate.interest_score / (diff +1)
                if score > best_score_for_selected:
                    best_score_for_selected = score
                    best_candidate_for_selected = candidate

            if best_candidate_for_selected is not None:
                best_choices.append((selected_index, best_candidate_for_selected))

        if not best_choices:
            return tour

        # Randomly choose one of the best choices
        insert_index, candidate_landmark = random.choice(best_choices)
        tour.add_landmark(candidate_landmark, position=insert_index)
        return tour

    def _delete(self, tour: Tour) -> Tour:
        """Remove a randomly selected landmark from the tour.
 
        Args:
            tour: The tour to modify. Modified in place.
 
        Returns:
            The tour with one landmark removed, or unchanged if empty.
        """
        if len(tour.visited_landmarks) == 0:
            return tour

        delete_index = random.randrange(len(tour.visited_landmarks))
        tour.remove_landmark(tour.visited_landmarks[delete_index])
        return tour

    def _insert(self, tour: Tour) -> Tour:
        """Insert a randomly selected feasible landmark at a random position.
 
        If no feasible candidates are available, falls back to deletion if the
        tour is non-empty, or returns the tour unchanged if empty.
 
        Args:
            tour: The tour to modify. Modified in place.
 
        Returns:
            The tour with one landmark inserted, or the result of a deletion
            fallback if no candidates are available.
        """
        available_landmarks = tour.problem.feasible_candidates(tour)
        if not available_landmarks:
            return self._delete(tour) if tour.visited_landmarks else tour

        new_landmark = random.choice(available_landmarks)
        insert_index = random.randrange(len(tour.visited_landmarks) + 1)
        tour.add_landmark(new_landmark, position=insert_index)
        return tour
