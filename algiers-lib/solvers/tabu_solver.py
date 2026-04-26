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

    def __init__(self, problem: Problem, max_iterations: int = 300, tabu_tenure: int = 7) -> None:
        
        super().__init__(problem)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure 
        self.tabu_end: dict[TabuMove , int] = {}
        self.best_solution = None
        self.best_score = -float('inf')


    def _get_neighbors(self, tour: Tour) -> list[tuple[Tour, TabuMove]]:
        neighbors = []
        visited = tour.visited_landmarks
        unvisited = self.problem.feasible_candidates(tour)

        for lm in unvisited: #add
            for position in range(len(visited) + 1):
                neighbor = tour.copy()
                neighbor.add_landmark(lm , position)
                move = TabuMove(MoveType.INSERT , (lm.id,))
                neighbors.append((neighbor,move))
        
        for lm in visited: #remove
           
            neighbor = tour.copy()
            neighbor.remove_landmark(lm)
            move = TabuMove(MoveType.REMOVE , (lm.id,))
            neighbors.append((neighbor,move))

        for i in range(len(visited)): #swap
            for j in range( i+1 , len(visited)):
                neighbor = tour.copy()
                neighbor.swap_by_index(i , j)
                move = TabuMove(MoveType.SWAP , tuple(sorted([visited[i].id,visited[j].id])))
                neighbors.append((neighbor,move))

        for old in visited: #replace
            for new in unvisited:
                neighbor = tour.copy()
                neighbor.replace_landmark(old=old , new= new)
                move = TabuMove(MoveType.REPLACE , tuple(sorted([old.id ,new.id])))
                neighbors.append((neighbor,move))
            
        return neighbors
    
    def solve(self) -> Tour:
        
        current_tour: Tour = self.problem.random_tour()

        for iteration in range(self.max_iterations):
            
            neighbors = self._get_neighbors(current_tour)

            best_neighbor: Optional[Tour] = None
            best_neighbor_score:float = -1.0
            best_neighbor_move: Optional[TabuMove] = None

            for neighbor , move in neighbors:

                if not neighbor.is_valid():
                    continue

                is_tabu = move in self.tabu_end
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