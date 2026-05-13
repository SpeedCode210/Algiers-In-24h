from __future__ import annotations
import random
from typing import Optional
from models.tour import Tour
from .genetic_augmented_representation import AugmentedRepresentation

class Crossover:
    """Crossover operator for the genetic algorithm.

    Supports multiple crossover strategies for combining two parent tours into
    two child tours. The strategy is selected at construction time and applied
    uniformly through the ``crossover`` dispatch method.

    Attributes:
        method: Name of the crossover strategy in use. Supported values are
            ``"order"`` and ``"tailored"``.
    """

    def __init__(self, method: str = "order") -> None:
        """Initialize the crossover operator.

        Args:
            method: Crossover strategy to use. ``"order"`` applies order-based
                crossover on raw Tour objects. ``"tailored"`` applies the
                time-window-aware crossover using AugmentedRepresentation.
        """
        self.method = method

    def crossover(
        self,
        parent1: Tour | AugmentedRepresentation,
        parent2: Tour | AugmentedRepresentation,
    ) -> tuple[Tour | AugmentedRepresentation, Tour | AugmentedRepresentation]:
        """Dispatch crossover to the configured strategy.

        Converts parents to AugmentedRepresentation automatically when the
        tailored method is selected and raw Tours are provided.

        Args:
            parent1: First parent tour or augmented representation.
            parent2: Second parent tour or augmented representation.

        Returns:
            A tuple of two child individuals produced by the crossover.

        Raises:
            NotImplementedError: If the configured method is not supported.
        """
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
        """Perform order-based crossover (OX) on two parent tours.

        Selects a random segment from the shorter parent and fills the remaining
        positions with landmarks from the longer parent in their original order,
        skipping duplicates. Two children are produced symmetrically.

        Args:
            parent1: First parent tour.
            parent2: Second parent tour.

        Returns:
            A tuple of two child Tours.

        Raises:
            ValueError: If the parents belong to different problem instances.
        """
        if parent1.problem is not parent2.problem:
            raise ValueError("Both parents must belong to the same problem instance.")

        if len(parent1.visited_landmarks) == 0 and len(parent2.visited_landmarks) == 0:
            return Tour(parent1.problem, []), Tour(parent1.problem, [])

        if len(parent1.visited_landmarks) == 0 or len(parent2.visited_landmarks) == 0:
            return parent1 , parent2

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
        if source_length < 2:
            start = end = 0
        else:
            start, end = sorted(random.sample(range(source_length), 2))
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
        """Perform time-window-aware crossover using augmented representations.

        Searches for valid cut points between the two parents by checking whether
        the departure time at position i in one parent allows the tourist to reach
        position j in the other parent within its allowable start window
        (start_time + max_shift). Two cuts are selected at random from all valid
        candidates and used to splice the parents into two children.

        If no valid cut points are found, two random tours are returned as
        fallback children. If only one valid cut is found, the second child is
        a random tour.

        Args:
            parent1: First parent as an augmented representation.
            parent2: Second parent as an augmented representation.

        Returns:
            A tuple of two child AugmentedRepresentations (or Tours as fallback).

        Raises:
            ValueError: If either parent lacks a problem instance, or if they
                belong to different problem instances.
        """
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

        for i in range(len(parent1.landmarks)):
            for j in range(len(parent2.landmarks)):
                departure_at_i = parent1.timeline[i][3]
                travel_estimate = parent1.problem.travel_time(
                    parent1.landmarks[i], parent2.landmarks[j]
                )
                next_start = parent2.timeline[j][2]
                next_max_shift = parent2.timeline[j][4]
                # max_shift represents the time by which we can delay the start
                # of the visit to the landmark at j. We can proceed if we arrive
                # by next_start + next_max_shift.
                if departure_at_i + travel_estimate <= next_start + next_max_shift:
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
        """Build a child by splicing a source prefix with a donor tail.

        Takes landmarks 0 through source_index (inclusive) from the source, then
        appends landmarks from donor_index onward from the donor, excluding any
        landmark already present in the source prefix to avoid duplicates.

        Args:
            source: The parent providing the prefix.
            donor: The parent providing the tail.
            source_index: Last index (inclusive) of the source prefix.
            donor_index: First index (inclusive) of the donor tail.

        Returns:
            AugmentedRepresentation of the child tour built from the splice.
        """
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
        """Build a single order-crossover child.

        Copies the segment [start, end] from segment_parent into the child genome,
        then fills remaining positions in wrap-around order using landmarks from
        fill_parent, skipping any landmark already in the segment.

        Args:
            segment_parent: Parent whose segment is copied into the child.
            fill_parent: Parent whose landmarks fill the remaining positions.
            child_length: Total number of landmarks in the child genome.
            start: Start index of the copied segment (inclusive).
            end: End index of the copied segment (inclusive).

        Returns:
            A new Tour built from the assembled child genome.

        Raises:
            ValueError: If the child genome cannot be fully filled because
                fill_parent does not have enough unique landmarks.
        """
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