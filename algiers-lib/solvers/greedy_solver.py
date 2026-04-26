
from models.landmark import Landmark
from models.problem import Problem
from models.tour import Tour
from .solver import Solver


class GreedySolver(Solver):
    """Greedy solver for the Orienteering Problem.

    This solver constructs a tour by greedily selecting the next landmark
    that maximizes a priority function, either based on interest score alone
    or a ratio of interest score to travel time.
    """

    def __init__(self, problem: Problem, use_ratio: bool = False) -> None:
        """Initialize the GreedySolver.

        Args:
            problem (Problem): The problem instance to solve.
            use_ratio (bool, optional): If True, use interest score / travel time as priority.
                                        If False, use (interest_score, -travel_time). Defaults to False.
        """
        super().__init__(problem)
        self.use_ratio = use_ratio

    def _priority(self, candidate: Landmark, curr: Landmark):
        """Calculate the priority of a candidate landmark from the current position.

        Args:
            candidate (Landmark): The candidate landmark to evaluate.
            curr (Landmark): The current landmark in the tour.

        Returns:
            float or tuple: The priority value. If use_ratio is True, returns float (score/travel).
                            If False, returns tuple (score, -travel) for lexicographic ordering.
        """
        travel= self.problem.travel_time(curr, candidate)
        if self.use_ratio:
            if travel > 0:
                return candidate.interest_score / travel
            #impossible case if the data is consistent
            else:
                return float('inf')
        
        #maximize interest score, if equality minimize travel time
        return (candidate.interest_score, -travel)

    def solve(self) -> Tour:
        """Solve the problem using the greedy algorithm.

        Returns:
            Tour: The constructed tour.
        """
        tour = self.problem.create_empty_tour()
        while True:
            if tour.visited_landmarks:
                cur = tour.visited_landmarks[-1]
            else:
                cur = self.problem.hotel
            candidates = self.problem.feasible_candidates(tour)

            invalid: set[str] = set()
            
            added = False
            while candidates:
                best = max(candidates, key=lambda lm: self._priority(lm, cur))
                tour.add_landmark(best)
                if tour.is_valid():
                    added = True
                    break
                tour.remove_landmark(best)
                invalid.add(best.id)
                candidates = [c for c in candidates if c.id not in invalid]
                
            if not added:
                break
            
        return tour