from abc import ABC, abstractmethod
from models.problem import Problem
from models.tour import Tour

class Solver(ABC):
	"""
	Abstract base class for all solvers.
	Each solver must implement the solve() method.
	"""
	def __init__(self, problem: Problem) -> None:
		self.problem = problem

	@abstractmethod
	def solve(self) -> Tour:
		"""
		Solve the given problem and return a Tour.
		"""
		pass