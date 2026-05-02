from __future__ import annotations
 
from typing import Optional
from models.landmark import Landmark
from models.tour import Tour
from models.problem import Problem
from dataclasses import dataclass
import random
from .solver import Solver
from enum import Enum

class MoveType(Enum):
    SWAP = "swap"
    INSERT = "insert"
    REMOVE = "remove"
    REPLACE = "replace"

@dataclass(frozen=True)
class TabuMove:
    move_type: MoveType
    landmark_ids: tuple[str, ...]

class OscillationPhase(Enum):
    INTENSIFICATION = "intensification"   # normal search: hard budget constraint
    EXPANSION       = "expansion"         # relaxed budget: diversification
    RECOVERY        = "recovery"          # tighten back: remove weak landmarks invalidating the tour
 

class TabuSolver(Solver):
    """
    Strategy Overview:
    ------------------
    The solver employs a 'Strategic Oscillation' strategy combined with memory-based
    neighborhood search to escape local optima and navigate complex temporal constraints.

    Key Mechanisms:
    1. Neighborhood Search (Intensification):
       Explores local changes through four move types: SWAP, INSERT, REMOVE, and REPLACE.
       To maintain efficiency, it uses 'Successive Filtering'—ranking landmarks by 
       'weakness' (induced travel cost vs. interest score) and 'fitness' to prune 
       the search space to only the most promising candidates.

    2. Tabu Memory & Aspiration:
       Prevents cycling by tracking recent moves in a 'Tabu List' for a duration 
       defined by `tabu_tenure`. It includes an 'Aspiration Criterion' that allows 
       a Tabu move if it results in a new global best score.

    3. Strategic Oscillation (Diversification):
       The solver oscillates between three distinct phases based on search progress:
       - INTENSIFICATION: Standard search within strict time budget constraints.
       - EXPANSION: Triggered by plateaus. The time budget is relaxed by `oscillation_slack`,
         allowing the search to pass through infeasible regions to reach distant 
         optimal clusters.
       - RECOVERY: A cleaning phase that aggressively removes the weakest landmarks
         to bring an expanded tour back into feasibility under the hard budget.

    4. Adaptive Plateau Management:
       Monitors `iterations_since_improvement`. If the search flatlines beyond 
       `plateau_threshold`, it forces a shift from exploitation to exploration.

    Attributes:
        problem (Problem): The orienteering problem instance to solve.
        max_iterations (int): Maximum search depth.
        tabu_tenure (int): The number of iterations a move remains forbidden.
        n_remove/insert_candidates (int): Beam-search width for candidate filtering.
        oscillation_slack (float): The temporal 'buffer' allowed during Expansion.
    """

    def __init__(self,
                  problem: Problem, 
                  max_iterations: int = 200, 
                  tabu_tenure: Optional[int] = 20,
                  n_remove_candidates: int = 5,
                  n_insert_candidates: int = 10,
                  plateau_threshold: int = 10,
                  oscillation_slack: float = 240.0,
                  expansion_iterations: int = 7
                  ) -> None:
        
        super().__init__(problem)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure if tabu_tenure is not None else random.randint(5, 12) 
        self.tabu_end: dict[TabuMove , int] = {}
        self.best_solution = None
        self.best_score = -float('inf')
        self.n_remove_candidates = n_remove_candidates
        self.n_insert_candidates = n_insert_candidates
        self.plateau_threshold = plateau_threshold
        self.oscillation_slack = oscillation_slack
        self.expansion_iterations = expansion_iterations


    def _weakness_score(self, landmark: Landmark, tour: Tour) -> float:
        """
        Score how weak a visited landmark is  higher means more worth removing.
        Criterion: induced travel cost / interest score.
        Induced cost = travel(prev→lm) + travel(lm→next) - travel(prev→next)
        i.e. how much extra travel time this landmark adds to the route.
        """
        visited = tour.visited_landmarks
        idx = visited.index(landmark)
        prev = self.problem.hotel if idx == 0 else visited[idx - 1]
        nxt = self.problem.hotel if idx == len(visited) - 1 else visited[idx + 1]

        induced_cost = (self.problem.travel_time(prev, landmark) 
            + self.problem.travel_time(landmark, nxt)
            - self.problem.travel_time(prev, nxt))
        
        if landmark.interest_score == 0:
            return float('inf')
 
        return induced_cost / landmark.interest_score

    def _fitness_score(self, candidate: Landmark, removed: Landmark, tour: Tour) -> float:
        """
        Score how well an unvisited landmark fits as a replacement for `removed`.
        Criterion: interest score / induced travel cost at the same position.
        """
        visited = tour.visited_landmarks
        idx = visited.index(removed)
 
        prev = self.problem.hotel if idx == 0 else visited[idx - 1]
        nxt = self.problem.hotel if idx == len(visited) - 1 else visited[idx + 1]
 
        induced_cost = (
            self.problem.travel_time(prev, candidate)
            + self.problem.travel_time(candidate, nxt)
            - self.problem.travel_time(prev, nxt)
        )
 
        return candidate.interest_score / (induced_cost + 1e-9)

    def _removal_candidates(self, tour: Tour) -> list[Landmark]:
        """
        Return the top-k weakest visited landmarks ranked by weakness score.
        """
        visited = tour.visited_landmarks
        ranked = sorted(visited, key=lambda lm: self._weakness_score(lm, tour), reverse=True)
        return ranked[:self.n_remove_candidates]
    
    def _insertion_candidates(self, removed: Landmark, tour: Tour, effective_budget: float) -> list[Landmark]:
        """
        Top-k best fitting unvisited landmarks for the position of `removed`.
        Pre-filtered: skip candidates whose induced cost exceeds available slack
        under the given effective_budget.
        """
        unvisited = self.problem.feasible_candidates(tour)
        visited = tour.visited_landmarks
        idx = visited.index(removed)

        prev = self.problem.hotel if idx == 0 else visited[idx - 1]
        nxt = self.problem.hotel if idx == len(visited) - 1 else visited[idx + 1]

        simulation = tour.simulation_cache()
        slack = effective_budget - simulation.total_duration

        feasible: list[Landmark] = []
        for lm in unvisited:
            induced_cost = (
                self.problem.travel_time(prev, lm)
                + self.problem.travel_time(lm, nxt)
                - self.problem.travel_time(prev, nxt)
            )
            if induced_cost <= slack:
                feasible.append(lm)

        ranked = sorted(feasible, key=lambda lm: self._fitness_score(lm, removed, tour), reverse=True)
        return ranked[:self.n_insert_candidates]
        
    def _get_neighbors(self, tour: Tour, effective_budget: float, ) -> list[tuple[Tour, TabuMove]]:
        neighbors = []
        visited = tour.visited_landmarks
        remove_candidates = self._removal_candidates(tour)
        
        for lm in remove_candidates:

            neighbor = tour.copy()
            neighbor.remove_landmark(lm)
            move = TabuMove(MoveType.INSERT, (lm.id,))
            neighbors.append((neighbor, move))

        for i in range(len(visited)): #swap
            for j in range( i+1 , len(visited)):
                neighbor = tour.copy()
                neighbor.swap_by_index(i , j)
                move = TabuMove(MoveType.SWAP , tuple(sorted([visited[i].id,visited[j].id])))
                neighbors.append((neighbor,move))

        for weak in remove_candidates: #successive filtering
            insertion_candidates = self._insertion_candidates(weak,tour, effective_budget)

            for strong in insertion_candidates: #replace
                neighbor = tour.copy()
                neighbor.replace_landmark(old=weak,new=strong)
                move = TabuMove(MoveType.REPLACE, tuple(sorted([weak.id, strong.id])))
                neighbors.append((neighbor, move))
            
            for strong in insertion_candidates:
                if strong in tour:
                    continue
                for position in range(len(visited) + 1):
                    neighbor = tour.copy()
                    neighbor.add_landmark(strong, position)
                    move = TabuMove(MoveType.REMOVE, (strong.id,))
                    neighbors.append((neighbor, move))

        return neighbors
    
    def _is_feasible_under(self, tour: Tour, budget: float) -> bool:
        """Check if the tour is valid under a given budget"""
        sim = tour.simulation_cache()
        return (
            sim.total_duration <= budget
            and len(sim.entries) == len(tour.visited_landmarks)
        )
    
    def _recover_tour(self, tour:Tour, iteration:int) -> Tour:
        """remove the weakest landmarks one by one
        until the tour is feasible under the hard budget again."""
        recovered = tour.copy()
        hard_budget = float(self.problem.time_budget)

        while not  self._is_feasible_under(recovered , hard_budget):
            if not recovered.visited_landmarks:
                break

            weakest = max(recovered.visited_landmarks,
                           key=lambda lm:self._weakness_score(lm,recovered))
            recovered.remove_landmark(weakest)
            move = TabuMove(MoveType.INSERT, (weakest.id,))
            self.tabu_end[move] = iteration + self.tabu_tenure

        return recovered
    
    def solve(self) -> Tour:
        
        current_tour: Tour = self.problem.random_tour()
        self.best_solution: Tour = current_tour.copy()
        self.best_score: float = current_tour.total_score()

        hard_budget      = float(self.problem.time_budget)
        soft_budget      = hard_budget + self.oscillation_slack
 
        phase            = OscillationPhase.INTENSIFICATION
        effective_budget = hard_budget
 
        iterations_since_improvement = 0
        expansion_counter            = 0

        for iteration in range(self.max_iterations):

            if phase == OscillationPhase.INTENSIFICATION:

                if iterations_since_improvement >= self.plateau_threshold:

                    phase            = OscillationPhase.EXPANSION
                    effective_budget = soft_budget
                    expansion_counter = 0

            elif phase == OscillationPhase.EXPANSION:

                expansion_counter += 1

                if expansion_counter >= self.expansion_iterations:
                    
                    phase            = OscillationPhase.RECOVERY
                    effective_budget = hard_budget
                    current_tour     = self._recover_tour(current_tour,iteration)
                    iterations_since_improvement = 0
                    continue  # skip neighbor search this iteration
 
            elif phase == OscillationPhase.RECOVERY:

                if self._is_feasible_under(current_tour, hard_budget):

                    # fully recovered
                    phase = OscillationPhase.INTENSIFICATION
            
            neighbors = self._get_neighbors(current_tour, effective_budget)

            best_neighbor: Optional[Tour] = None
            best_neighbor_score:float = -1.0
            best_neighbor_move: Optional[TabuMove] = None

            for neighbor , move in neighbors:

                if phase == OscillationPhase.EXPANSION:
                    if not self._is_feasible_under(neighbor, soft_budget):
                        continue
                else:
                    if not neighbor.is_valid():
                        continue

                is_tabu = move in self.tabu_end and self.tabu_end[move] > iteration
                score = neighbor.total_score()

                if is_tabu and score <= self.best_score: #aspiration criterion
                    continue
                
                if score >= best_neighbor_score:

                    best_neighbor = neighbor
                    best_neighbor_score = score
                    best_neighbor_move = move

            if best_neighbor is None:
                break

            current_tour = best_neighbor
            self.tabu_end[best_neighbor_move] = iteration + self.tabu_tenure

            if (self._is_feasible_under(current_tour, hard_budget)
                and current_tour.total_score() > self.best_score):

                self.best_solution = current_tour.copy()
                self.best_score    = self.best_solution.total_score()
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1

            self.tabu_end = {move: expiry for move, expiry in self.tabu_end.items() if expiry > iteration}

        return self.best_solution