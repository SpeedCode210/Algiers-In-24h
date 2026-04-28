from __future__ import annotations
 
from typing import Optional, TYPE_CHECKING
from models.landmark import Landmark
from models.tour import Tour
from models.problem import Problem
from dataclasses import dataclass , field 
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

class TabuSolver(Solver):

    def __init__(self,
                  problem: Problem, 
                  max_iterations: int = 200, 
                  tabu_tenure: Optional[int] = None,
                  n_remove_candidates: int = 5,
                  n_insert_candidates: int = 10
                  ) -> None:
        
        super().__init__(problem)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure if tabu_tenure is not None else random.randint(5, 12) 
        self.tabu_end: dict[TabuMove , int] = {}
        self.best_solution = None
        self.best_score = -float('inf')
        self.n_remove_candidates = n_remove_candidates
        self.n_insert_candidates = n_insert_candidates


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
    
    def _insertion_candidates(self, removed: Landmark, tour: Tour) -> list[Landmark]:
        """
        Return the top-k best fitting unvisited landmarks to replace `removed`,
        pre-filtered by feasibility: skip candidates whose induced cost
        exceeds the tour's remaining slack.
        """
        unvisited = self.problem.feasible_candidates(tour)
        visited = tour.visited_landmarks
        idx = visited.index(removed)

        prev = self.problem.hotel if idx == 0 else visited[idx - 1]
        nxt = self.problem.hotel if idx == len(visited) - 1 else visited[idx + 1]

        slack = tour.slack

        feasible = []
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
        
    def _get_neighbors(self, tour: Tour) -> list[tuple[Tour, TabuMove]]:
        neighbors = []
        visited = tour.visited_landmarks
        remove_candidates = self._removal_candidates(tour)
        
        for lm in remove_candidates:

            neighbor = tour.copy()
            neighbor.remove_landmark(lm)
            move = TabuMove(MoveType.REMOVE, (lm.id,))
            neighbors.append((neighbor, move))

        for i in range(len(visited)): #swap
            for j in range( i+1 , len(visited)):
                neighbor = tour.copy()
                neighbor.swap_by_index(i , j)
                move = TabuMove(MoveType.SWAP , tuple(sorted([visited[i].id,visited[j].id])))
                neighbors.append((neighbor,move))

        for weak in remove_candidates: #successive filtering
            insertion_candidates = self._insertion_candidates(weak,tour)

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
                    move = TabuMove(MoveType.INSERT, (strong.id,))
                    neighbors.append((neighbor, move))

        return neighbors
    
    def solve(self) -> Tour:
        
        current_tour: Tour = self.problem.random_tour()
        self.best_solution: Tour = current_tour.copy()
        self.best_score: float = current_tour.total_score()

        for iteration in range(self.max_iterations):
            
            neighbors = self._get_neighbors(current_tour)

            best_neighbor: Optional[Tour] = None
            best_neighbor_score:float = -1.0
            best_neighbor_move: Optional[TabuMove] = None

            for neighbor , move in neighbors:

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

            if current_tour.total_score() > self.best_score:
                self.best_solution = current_tour.copy()
                self.best_score = self.best_solution.total_score()

            self.tabu_end = {move: expiry for move, expiry in self.tabu_end.items() if expiry > iteration}

        return self.best_solution