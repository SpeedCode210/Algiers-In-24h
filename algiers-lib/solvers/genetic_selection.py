from __future__ import annotations

import random
from models.tour import Tour
from .genetic_fitness import FitnessFunction


class Selection:
    """Selection operator for the genetic algorithm.

    Provides multiple parent selection strategies for choosing which tours
    in the population reproduce. All strategies return exactly two parents.

    Attributes:
        fitness_function: The fitness function used to rank tours.
    """

    def __init__(self, fitness_function: FitnessFunction) -> None:
        """Initialize the selection operator.

        Args:
            fitness_function: Fitness function used to evaluate and rank tours.
        """
        self.fitness_function = fitness_function

    def tournament_selection(self, population: list[Tour], k: int) -> tuple[Tour, Tour]:
        """Select two parents via tournament selection.

        Randomly samples k tours from the population and returns the two
        highest-scoring ones as parents.

        Args:
            population: The current population of tours.
            k: Tournament size. Must be at least 2 and at most len(population).

        Returns:
            A tuple of (best, second_best) from the tournament sample.

        Raises:
            ValueError: If k is less than 2 or greater than the population size.
        """
        if k > len(population):
            raise ValueError(
                f"Tournament size k={k} cannot be larger than population size {len(population)}."
            )
        if k < 2:
            raise ValueError("Tournament size must be at least 2.")

        candidates = random.sample(population, k)
        ordered = sorted(
            candidates,
            key=self.fitness_function.fitness,
            reverse=True,
        )
        return ordered[0], ordered[1]

    def random_selection(self, population: list[Tour]) -> tuple[Tour, Tour]:
        """Select two parents uniformly at random without replacement.

        Args:
            population: The current population of tours.

        Returns:
            A tuple of two distinct randomly selected tours.

        Raises:
            ValueError: If the population contains fewer than two tours.
        """
        if len(population) < 2:
            raise ValueError("Population must contain at least two tours.")

        parent1, parent2 = random.sample(population, 2)
        return parent1, parent2

    def probability_selection(self, population: list[Tour], p: float) -> tuple[Tour, Tour]:
        """Select parents with a bias toward the best tour.

        With probability ``1 - p``, always returns the best and second-best
        tours. With probability ``p``, returns the best tour paired with a
        randomly chosen alternative from the rest of the population.

        Args:
            population: The current population of tours.
            p: Probability of pairing the best tour with a random alternative
               instead of the second-best. Must be in [0, 1].

        Returns:
            A tuple of two parent tours.

        Raises:
            ValueError: If the population has fewer than two tours, or if p
                is not in [0, 1].
        """
        if len(population) < 2:
            raise ValueError("Population must contain at least two tours.")
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be between 0 and 1.")

        ordered = sorted(
            population,
            key=self.fitness_function.fitness,
            reverse=True,
        )
        best, second_best = ordered[0], ordered[1]

        if random.random() < 1.0 - p:
            return best, second_best

        alternatives = [tour for tour in population if tour is not best]
        return best, random.choice(alternatives)

    def fitness_proportionate_selection(self, population: list[Tour]) -> tuple[Tour, Tour]:
        """Select parents proportionally to their fitness (roulette wheel selection).

        Translates all fitness values to be strictly positive before computing
        selection probabilities, so tours with higher fitness are more likely
        to be chosen. The second parent is chosen from the remaining tours
        after removing the first parent's weight.

        Note:
            This method cannot be used with fitness functions that return
            ``float('-inf')``, as the translation step does not handle it.
            Negative finite values are handled correctly via shifting.

        Args:
            population: The current population of tours.

        Returns:
            A tuple of two parent tours selected proportionally to fitness.

        Raises:
            ValueError: If the population contains fewer than two tours.
        """
        if len(population) < 2:
            raise ValueError("Population must contain at least two tours.")

        fitness_values = [self.fitness_function.fitness(tour) for tour in population]
        minimum_fitness = min(fitness_values)
        shift = -minimum_fitness + 1 if minimum_fitness < 0 else 0
        translated = [fitness + shift for fitness in fitness_values]
        total_translated = sum(translated)

        if total_translated <= 0:
            return self.random_selection(population)

        parent1 = self._weighted_choice(population, translated)
        parent1_index = population.index(parent1)

        second_weights = list(translated)
        second_weights[parent1_index] = 0.0
        if sum(second_weights) <= 0:
            alternatives = [tour for tour in population if tour is not parent1]
            return parent1, random.choice(alternatives)

        parent2 = self._weighted_choice(population, second_weights)
        return parent1, parent2

    def _weighted_choice(self, population: list[Tour], weights: list[float]) -> Tour:
        """Select a single tour using weighted random sampling.
 
        Performs a linear scan with cumulative weights to select one tour.
        Each tour's probability of selection is proportional to its weight.
 
        Args:
            population: The list of tours to choose from.
            weights: Non-negative weights corresponding to each tour. Must sum
                to a positive value.
 
        Returns:
            A single selected tour.
 
        Raises:
            ValueError: If weights sum to zero or a negative value.
        """
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive value for weighted selection.")

        threshold = random.random() * total_weight
        cumulative = 0.0
        for tour, weight in zip(population, weights):
            cumulative += weight
            if threshold < cumulative:
                return tour

        return population[-1]
