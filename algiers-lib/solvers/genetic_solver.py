from __future__ import annotations

import random
from models.problem import Problem
from models.tour import Tour
from .solver import Solver
from .genetic_augmented_representation import AugmentedRepresentation
from .genetic_crossover import Crossover
from .genetic_mutation import Mutation
from .genetic_fitness import FitnessFunction
from .genetic_selection import Selection

class GeneticSolver(Solver):  
    """Standard genetic algorithm solver for the Orienteering Problem with Time Windows.
 
    Uses order-based crossover on raw Tour objects and supports optional elitism
    and culling strategies. The population is initialized with random tours and
    evolved over a fixed number of generations, with early stopping when fitness
    has not improved for 100 consecutive generations.
 
    Attributes:
        fitness_function: Fitness function used to evaluate and rank tours.
        regenerations: Maximum number of generations to run.
        population_size: Number of tours in the population at each generation.
        mutation_rate: Probability of applying mutation to each child.
        selection: Selection operator used to pick parents.
        crossover: Crossover operator used to produce children.
        mutation: Mutation operator applied to children.
        elitism: Whether to carry the top-performing tours into the next generation.
        culling: Whether to replace the lowest-performing tours with random ones
            each generation.
        elite_proportion: Fraction of the population preserved via elitism.
        culling_proportion: Fraction of the population replaced via culling.
    """
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
        patience : int =100,
    ) -> None:
        """Initialize the genetic solver.
 
        Args:
            problem: The OPTW problem instance to solve.
            fitness_function: Fitness function for evaluating tours.
            regenerations: Maximum number of generations. Defaults to 100.
            population_size: Number of individuals per generation. Defaults to 20.
            mutation_rate: Probability in [0, 1] of mutating each child.
                Defaults to 0.1.
            insertion_probability: Probability of insertion vs deletion in the
                mutation operator. Defaults to 0.5.
            crossover_method: Crossover strategy to pass to the Crossover operator.
                Defaults to ``"order"``.
            elitism: If True, the top ``elite_proportion`` of the population is
                carried unchanged into the next generation. Defaults to False.
            culling: If True, the bottom ``culling_proportion`` of the next
                generation is replaced with fresh random tours. Defaults to False.
            elite_proportion: Fraction of population size to preserve as elites.
                Defaults to 0.1.
            culling_proportion: Fraction of population size to cull each generation.
                Defaults to 0.1.
 
        Raises:
            ValueError: If any parameter is outside its valid range.
        """
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
        """Run the genetic algorithm and return the best tour found.
 
        Initializes the population with random tours, then evolves it for up
        to ``regenerations`` generations using tournament selection, crossover,
        and mutation. Terminates early if fitness does not improve for 100
        consecutive generations.
 
        Returns:
            The best Tour found across all generations.
        """
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

            if unchanged_generations >= self.patience:
                break

        return best


class TailoredGeneticSolver(Solver):
    """Time-window-aware genetic algorithm solver for the OPTW.
 
    Uses the tailored crossover operator, which exploits the augmented
    representation's max_shift values to find valid splice points between
    parent tours. This guarantees that children produced by crossover always
    respect the time windows of their inherited landmarks, provided the parents
    are valid.
 
    Attributes:
        fitness_function: Fitness function used to evaluate and rank tours.
        regenerations: Maximum number of generations to run.
        population_size: Number of tours in the population at each generation.
        mutation_rate: Probability of applying mutation to each child.
        selection: Selection operator used to pick parents.
        crossover: Crossover operator configured for the tailored method.
        mutation: Mutation operator applied to children.
        elitism: Whether to carry the top-performing tours into the next generation.
        culling: Whether to replace the lowest-performing tours with random ones.
        elite_proportion: Fraction of the population preserved via elitism.
        culling_proportion: Fraction of the population replaced via culling.
        patience: Number of generations without improvement before early stopping.
    """
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
        """Initialize the tailored genetic solver.
 
        Args:
            problem: The OPTW problem instance to solve.
            fitness_function: Fitness function for evaluating tours.
            regenerations: Maximum number of generations. Defaults to 100.
            population_size: Number of individuals per generation. Defaults to 20.
            mutation_rate: Probability in [0, 1] of mutating each child.
                Defaults to 0.1.
            insertion_probability: Probability of insertion vs deletion in the
                mutation operator. Defaults to 0.5.
            crossover_method: Must be ``"tailored"`` or
                ``"tailored_mutation_motivated"``. Defaults to ``"tailored"``.
            elitism: If True, the top ``elite_proportion`` of the population is
                carried unchanged into the next generation. Defaults to False.
            culling: If True, the bottom ``culling_proportion`` of the next
                generation is replaced with fresh random tours. Defaults to False.
            elite_proportion: Fraction of population size to preserve as elites.
                Defaults to 0.1.
            culling_proportion: Fraction of population size to cull each generation.
                Defaults to 0.1.
            patience: Number of consecutive generations without improvement before
                the algorithm terminates early. Defaults to 100.
 
        Raises:
            ValueError: If any parameter is outside its valid range, or if
                crossover_method is not a supported tailored variant.
        """
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
        """Run the tailored genetic algorithm and return the best tour found.
 
        Initializes the population with random tours, then evolves it using
        tailored crossover and mutation. Children produced by crossover are
        normalized to plain Tour objects before mutation is applied. Terminates
        early if fitness does not improve for ``patience`` consecutive generations.
 
        Returns:
            The best Tour found across all generations.
        """
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
        """Convert a tour to its augmented representation.
 
        Args:
            tour: A valid tour to convert.
 
        Returns:
            AugmentedRepresentation with a fully computed timeline.
        """
        return AugmentedRepresentation.from_tour(tour)

    def _generate_valid_route(self) -> Tour:
        """Generate a random tour for population initialization.
 
        Returns:
            A randomly generated Tour from the problem instance.
        """
        random_tour = self.problem.random_tour()
        return random_tour

    def _normalize_child(
        self, individual: Tour | AugmentedRepresentation
    ) -> Tour:
        """Convert a child individual to a plain Tour if needed.
 
        The tailored crossover returns AugmentedRepresentation objects.
        This method unwraps them so that the rest of the solve loop always
        works with Tour instances.
 
        Args:
            individual: A Tour or AugmentedRepresentation produced by crossover.
 
        Returns:
            A Tour built from the individual's landmark list.
        """
        if isinstance(individual, AugmentedRepresentation):
            return Tour(self.problem, list(individual.landmarks))
        return individual