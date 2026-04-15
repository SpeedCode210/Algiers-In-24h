from models.landmark import Landmark
from models.problem import Problem 
from models.tour import Tour

class GreedySolver:
    def __init__(self, problem: Problem, use_ratio: bool = False) -> None:
        self.problem = problem
        self.use_ratio = use_ratio

    def _priority(self, candidate: Landmark, curr: Landmark):
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