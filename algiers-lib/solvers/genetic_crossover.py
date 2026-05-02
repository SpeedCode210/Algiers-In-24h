from __future__ import annotations
import random
from typing import Optional
from models.tour import Tour
from .genetic_augmented_representation import AugmentedRepresentation

class Crossover:

    def __init__(self, method: str = "order") -> None:
        self.method = method

    def crossover(
        self,
        parent1: Tour | AugmentedRepresentation,
        parent2: Tour | AugmentedRepresentation,
    ) -> tuple[Tour | AugmentedRepresentation, Tour | AugmentedRepresentation]:
        if self.method == "order":
            return self.order_crossover(parent1, parent2)
        if self.method == "tailored":
            first = (
                parent1
                if isinstance(parent1, AugmentedRepresentation)
                else AugmentedRepresentation.from_tour(parent1)
            )
            second = (
                parent2
                if isinstance(parent2, AugmentedRepresentation)
                else AugmentedRepresentation.from_tour(parent2)
            )
            return self.tailored_crossover(first, second)
        raise NotImplementedError(
            f"Crossover method '{self.method}' is not implemented."
        )

    def order_crossover(self, parent1: Tour, parent2: Tour) -> tuple[Tour, Tour]:
        if parent1.problem is not parent2.problem:
            raise ValueError("Both parents must belong to the same problem instance.")

        if len(parent1.visited_landmarks) == 0 and len(parent2.visited_landmarks) == 0:
            return Tour(parent1.problem, []), Tour(parent1.problem, [])

        if len(parent1.visited_landmarks) == 0 or len(parent2.visited_landmarks) == 0:
            raise ValueError(
                "Order crossover requires both parents to contain at least one landmark."
            )

        source_parent, donor_parent = (
            (parent1, parent2)
            if len(parent1.visited_landmarks) <= len(parent2.visited_landmarks)
            else (parent2, parent1)
        )

        source_length = len(source_parent.visited_landmarks)
        donor_length = len(donor_parent.visited_landmarks)

        if source_length < 2:
            start = end = 0
        else:
            start, end = sorted(random.sample(range(source_length), 2))

        child1 = self._build_order_child(
            segment_parent=source_parent,
            fill_parent=donor_parent,
            child_length=donor_length,
            start=start,
            end=end,
        )
        child2 = self._build_order_child(
            segment_parent=donor_parent,
            fill_parent=source_parent,
            child_length=source_length,
            start=start,
            end=end,
        )

        return child1, child2

    def tailored_crossover(
        self,
        parent1: AugmentedRepresentation,
        parent2: AugmentedRepresentation,
    ) -> tuple[AugmentedRepresentation, AugmentedRepresentation]:
        if parent1.problem is None or parent2.problem is None:
            raise ValueError(
                "Augmented representations must include problem context for tailored crossover."
            )
        if parent1.problem is not parent2.problem:
            raise ValueError(
                "Both augmented parent representations must belong to the same problem instance."
            )
        cut1: list[tuple[int, int]] = []
        cut2: list[tuple[int, int]] = []
        # found the problme , you must take into account the equality of the landmarks , that is a special case that must be treated by its own I guess , we will be back anytime , 
        for i in range(len(parent1.landmarks)):
            for j in range(len(parent2.landmarks)):
                departure_at_i = parent1.timeline[i][3]
                travel_estimate = parent1.problem.travel_time(
                    parent1.landmarks[i], parent2.landmarks[j]
                )
                next_start = parent2.timeline[j][2]
                next_max_shift = parent2.timeline[j][4]
                #next_wait = parent2.timeline[j][1]
                """max shift represent the time by which we can delay the start of the visit of the landmakr at j , so if we arrive by a time that
                that is less than the next_start + next_max_shift , then we are okay , we can start at that time"""
                if departure_at_i + travel_estimate <= next_start + next_max_shift  :
                    cut1.append((i, j))

        for j in range(len(parent2.landmarks)):
            for i in range(len(parent1.landmarks)):
                departure_at_j = parent2.timeline[j][3]
                travel_estimate = parent2.problem.travel_time(
                    parent2.landmarks[j], parent1.landmarks[i]
                )
                next_start = parent1.timeline[i][2]
                next_max_shift = parent1.timeline[i][4]
  

                if departure_at_j + travel_estimate <= next_start + next_max_shift:
                    cut2.append((j, i))

        combined = [("cut1", pair) for pair in cut1] + [("cut2", pair) for pair in cut2]
        if not combined:
            child1= parent1.problem.random_tour()
            child2 = parent2.problem.random_tour()
            return child1, child2

        if len(combined) == 1:
            origin, pair = combined[0]
            if origin == "cut1":
                i, j = pair
                child1 = self._build_child_from_cut(parent1, parent2, i, j)
                child2 = parent1.problem.random_tour()
            else:
                j, i = pair
                child1 = self._build_child_from_cut(parent2, parent1, j, i)
                child2 = parent1.problem.random_tour()
            return child1, child2

        selected = random.sample(combined, k=2)

        children: list[AugmentedRepresentation] = []
        for origin, pair in selected:
            if origin == "cut1":
                i, j = pair
                children.append(self._build_child_from_cut(parent1, parent2, i, j))
            else:
                j, i = pair
                children.append(self._build_child_from_cut(parent2, parent1, j, i))
        return children[0], children[1]

    def _build_child_from_cut(
        self,
        source: AugmentedRepresentation,
        donor: AugmentedRepresentation,
        source_index: int,
        donor_index: int,
    ) -> AugmentedRepresentation:
        source_prefix = source.landmarks[: source_index + 1]
        donor_tail = [
            landmark
            for landmark in donor.landmarks[donor_index:]
            if landmark not in source_prefix
        ]
        child_landmarks = source_prefix + donor_tail
        problem = source.problem or donor.problem

        if problem is None:
            return AugmentedRepresentation(child_landmarks)

        child_tour = Tour(problem, child_landmarks)
        return AugmentedRepresentation.from_tour(child_tour)

    def _build_order_child(
        self,
        segment_parent: Tour,
        fill_parent: Tour,
        child_length: int,
        start: int,
        end: int,
    ) -> Tour:
        segment = segment_parent.visited_landmarks[start : end + 1]
        child_genome: list[Optional[Tour]] = [None] * child_length
        child_genome[start : end + 1] = segment

        fill_order: list[Tour] = []
        fill_genome = fill_parent.visited_landmarks
        index = end + 1

        while len(fill_order) < child_length - len(segment):
            candidate = fill_genome[index % len(fill_genome)]
            if candidate not in segment and candidate not in fill_order:
                fill_order.append(candidate)
            index += 1

        fill_positions = list(range(end + 1, child_length)) + list(range(0, start))
        for position, landmark in zip(fill_positions, fill_order):
            child_genome[position] = landmark

        if any(landmark is None for landmark in child_genome):
            raise ValueError(
                "Order crossover could not fill the child genome because donor parent lacks enough unique landmarks."
            )

        return Tour(segment_parent.problem, list(child_genome))