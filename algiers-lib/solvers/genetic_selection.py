from __future__ import annotations

import random
from models.tour import Tour
from .genetic_fitness import FitnessFunction


class Selection:

    def __init__(self, fitness_function: FitnessFunction) -> None:
        self.fitness_function = fitness_function

    def tournament_selection(self, population: list[Tour], k: int) -> tuple[Tour, Tour]:
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
        if len(population) < 2:
            raise ValueError("Population must contain at least two tours.")

        parent1, parent2 = random.sample(population, 2)
        return parent1, parent2

    def probability_selection(self, population: list[Tour], p: float) -> tuple[Tour, Tour]:
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

    # this can not be used when the fitness function can take negative infinity. 
    # negative values are handeled accordingly with the negative values
    def fitness_proportionate_selection(self, population: list[Tour]) -> tuple[Tour, Tour]:
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
