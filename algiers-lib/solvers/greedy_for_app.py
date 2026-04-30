import random
from models.problem import Problem
from models.tour import Tour
from solvers.solver import Solver


class RandomGreedy(Solver):
    """Greedy solver variant that selects the next feasible landmark randomly.

    The solver iteratively shuffles feasible candidates and tries each one
    until a valid tour extension is found.
    """

    def __init__(self, problem: Problem) -> None:
        """Initialize the random greedy solver.

        Args:
            problem (Problem): The problem instance to solve.
        """
        super().__init__(problem)

    def solve(self) -> Tour:
        """Construct a tour by randomly selecting feasible landmarks.

        Returns:
            Tour: The tour created with the random greedy strategy.
        """
        tour = self.problem.create_empty_tour()

        while True:
            candidates = self.problem.feasible_candidates(tour)
            random.shuffle(candidates)
            added = False

            for can in candidates:
                tour.add_landmark(can)
                if tour.is_valid():
                    added = True
                    break
                tour.remove_landmark(can)

            if not added:
                break

        return tour


class TimeGreedy(Solver):
    """Greedy solver variant that selects the next feasible landmark with minimum travel time.

    The solver chooses the current feasible candidate that is nearest to the
    current tour endpoint and adds it if the tour remains valid.
    """

    def __init__(self, problem: Problem) -> None:
        """Initialize the time-based greedy solver.

        Args:
            problem (Problem): The problem instance to solve.
        """
        super().__init__(problem)

    def solve(self) -> Tour:
        """Construct a tour by selecting the nearest feasible landmark.

        Returns:
            Tour: The tour created with the time-based greedy strategy.
        """
        tour = self.problem.create_empty_tour()

        while True:
            cur = tour.visited_landmarks[-1] if tour.visited_landmarks else self.problem.hotel
            candidates = self.problem.feasible_candidates(tour)
            candidates.sort(key=lambda lm: self.problem.travel_time(cur, lm))
            added = False

            for best in candidates:
                tour.add_landmark(best)
                if tour.is_valid():
                    added = True
                    break
                tour.remove_landmark(best)

            if not added:
                break

        return tour