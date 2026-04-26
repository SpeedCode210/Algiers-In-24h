from __future__ import annotations

import random
from models.problem import Problem
from models.tour import Tour
from models.landmark import Landmark
from .solver import Solver
from .genetic_augmented_representation import AugmentedRepresentation
from .genetic_crossover import Crossover
from .genetic_mutation import Mutation
from .genetic_fitness import FitnessFunction
from .genetic_selection import Selection

class GeneticSolver(Solver):  
    def __init__(
        self,
        problem: Problem,
        fitness_function: FitnessFunction,
        regenerations: int = 100,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        insertion_probability: float = 0.5,
        crossover_method: str = "order",
        elitism: bool = False,
        culling: bool = False,
        elite_proportion: float = 0.1,
        culling_proportion: float = 0.1,
    ) -> None:
        super().__init__(problem)
        if regenerations < 1:
            raise ValueError("Regenerations must be at least 1.")
        if population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0 and 1.")
        if not 0.0 <= elite_proportion <= 1.0:
            raise ValueError("Elite proportion must be between 0 and 1.")
        if not 0.0 <= culling_proportion <= 1.0:
            raise ValueError("Culling proportion must be between 0 and 1.")

        self.fitness_function = fitness_function
        self.regenerations = regenerations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection = Selection(fitness_function)
        self.crossover = Crossover(method=crossover_method)
        self.mutation = Mutation(insertion_probability=insertion_probability)
        self.elitism = elitism
        self.culling = culling
        self.elite_proportion = elite_proportion
        self.culling_proportion = culling_proportion

    def solve(self) -> Tour:
        population = [self.problem.random_tour() for _ in range(self.population_size)]

        best = max(population, key=self.fitness_function.fitness)
        best_fitness = self.fitness_function.fitness(best)
        unchanged_generations = 0
        for _ in range(self.regenerations):
            next_population: list[Tour] = []
            # this implemnts the ellitism that is ensuring that the fitness is always increasing
            if self.elitism:
                elite_count = int(self.population_size * self.elite_proportion)
                if elite_count > 0:
                    elites = sorted(
                        population,
                        key=self.fitness_function.fitness,
                        reverse=True,
                    )[:elite_count]
                    next_population.extend(elites)

            while len(next_population) < self.population_size:
                parent1, parent2 = self.selection.tournament_selection(
                    population, k=min(3, len(population))
                )
                child1, child2 = self.crossover.crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child1 = self.mutation.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutation.mutate(child2)

                next_population.extend([child1, child2])

            next_population = next_population[: self.population_size]
            #this implemnts culling that is we eliminate the worst performing children , they die ! this is not good for the OPTW problem .
            if self.culling:
                cull_count = int(self.population_size * self.culling_proportion)
                if cull_count > 0:
                    ordered_next = sorted(
                        next_population,
                        key=self.fitness_function.fitness,
                        reverse=True,
                    )
                    survivors = ordered_next[: self.population_size - cull_count]
                    replacements = [self.problem.random_tour() for _ in range(cull_count)]
                    next_population = survivors + replacements
                    random.shuffle(next_population)

            population = next_population
            current_best = max(population, key=self.fitness_function.fitness)
            current_fitness = self.fitness_function.fitness(current_best)
            if current_fitness > best_fitness:
                best = current_best
                best_fitness = current_fitness
                unchanged_generations = 0
            else:
                unchanged_generations += 1

            if unchanged_generations >= 100:
                break

        return best


class TailoredGeneticSolver(Solver):
    def __init__(
        self,
        problem: Problem,
        fitness_function: FitnessFunction,
        regenerations: int = 100,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        insertion_probability: float = 0.5,
        crossover_method: str = "tailored",
        elitism: bool = False,
        culling: bool = False,
        elite_proportion: float = 0.1,
        culling_proportion: float = 0.1,
        patience : int =100,
    ) -> None:
        super().__init__(problem)
        if regenerations < 1:
            raise ValueError("Regenerations must be at least 1.")
        if population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0 and 1.")
        if not 0.0 <= elite_proportion <= 1.0:
            raise ValueError("Elite proportion must be between 0 and 1.")
        if not 0.0 <= culling_proportion <= 1.0:
            raise ValueError("Culling proportion must be between 0 and 1.")
        if crossover_method != "tailored" and crossover_method != "tailored_mutation_motivated":
            raise ValueError("TailoredGeneticSolver only supports the tailored or tailored_mutation_motivated method.")
        if patience < 1:
            raise ValueError("Patience must be at least 1.")

        self.fitness_function = fitness_function
        self.regenerations = regenerations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection = Selection(fitness_function)
        self.crossover = Crossover(method=crossover_method)
        self.mutation = Mutation(insertion_probability=insertion_probability)
        self.elitism = elitism
        self.culling = culling
        self.elite_proportion = elite_proportion
        self.culling_proportion = culling_proportion
        self.patience = patience

    def solve(self) -> Tour:
        population = [self._generate_valid_route() for _ in range(self.population_size)]#this works fine 
        best = max(population, key=self.fitness_function.fitness)
        best_fitness = self.fitness_function.fitness(best)
        unchanged_generations = 0
          
        for _ in range(self.regenerations):
            
            next_population: list[Tour] = []
            if self.elitism:
                elite_count = int(self.population_size * self.elite_proportion)
                if elite_count > 0:
                    elites = sorted(
                        population,
                        key=self.fitness_function.fitness,
                        reverse=True,
                    )[:elite_count]
                    next_population.extend(elites)

            while len(next_population) < self.population_size:
                parent1, parent2 = self.selection.tournament_selection(
                    population, k=min(3, len(population))
                )
                #the crossover operation always gives a path with maximum limited size . this is why it needs to be enhanced 
                child1, child2 = self.crossover.crossover(parent1, parent2)
                child1 = self._normalize_child(child1)
                child2 = self._normalize_child(child2)
                # I do not think there is any problem here but the model must be checked 
                if random.random() < self.mutation_rate:
                    child1 = self.mutation.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutation.mutate(child2)

                next_population.extend([child1, child2])

            next_population = next_population[: self.population_size]
            if self.culling:
                cull_count = int(self.population_size * self.culling_proportion)
                if cull_count > 0:
                    ordered_next = sorted(
                        next_population,
                        key=self.fitness_function.fitness,
                        reverse=True,
                    )
                    survivors = ordered_next[: self.population_size - cull_count]
                    replacements = [
                        self.problem.random_tour() for _ in range(cull_count)
                    ]
                    next_population = survivors + replacements
                    random.shuffle(next_population)

            population = next_population
            current_best = max(population, key=self.fitness_function.fitness)
            current_fitness = self.fitness_function.fitness(current_best)
            if current_fitness > best_fitness:
                best = current_best
                best_fitness = current_fitness
                unchanged_generations = 0
            else:
                unchanged_generations += 1

            if unchanged_generations >= self.patience:
                break
            

        return best

    def augmented_representation(self, tour: Tour) -> AugmentedRepresentation:
        return AugmentedRepresentation.from_tour(tour)

    def _generate_valid_route(self) -> Tour:
        random_tour = self.problem.random_tour()
        return random_tour

    def _normalize_child(
        self, individual: Tour | AugmentedRepresentation
    ) -> Tour:
        if isinstance(individual, AugmentedRepresentation):
            return Tour(self.problem, list(individual.landmarks))
        return individual