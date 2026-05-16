"""
Benchmark script to compare all solvers on the dataset.
Measures execution time, total score, duration, and validity.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.problem import Problem
from models.landmark import Day, loadLandmarks, loadHotel
from models.tour import Tour

# Import all solvers
from solvers.greedy_solver import GreedySolver
from solvers.genetic_solver import GeneticSolver, TailoredGeneticSolver
from solvers.grasp_solver import GraspSolver
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from solvers.genetic_fitness import FeasibilityFitnessFunction
from solvers.tabu_solver import TabuSolver
# Try to import cplex solver
from solvers.cplex_solver import CPLEXSolver
HAS_CPLEX = True


class SolverBenchmark:
    """Benchmark harness for comparing solver performance."""
    
    def __init__(self, problem: Problem):
        """Initialize benchmark with a problem instance."""
        self.problem = problem
        self.results: Dict[str, Dict] = {}
    
    def run_solver(self, solver_name: str, solver) -> Tuple[Tour, float]:
        """
        Run a single solver and measure execution time.
        
        Args:
            solver_name: Name of the solver for logging
            solver: Solver instance to run
            
        Returns:
            Tuple of (Tour result, execution time in seconds)
        """
        print(f"  Running {solver_name}...", end=" ", flush=True)
        start_time = time.time()
        try:
            tour = solver.solve()
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.2f}s)")
            return tour, elapsed
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            traceback.print_exc()
            return None, time.time() - start_time
    
    def analyze_tour(self, tour: Tour) -> Dict:
        """Analyze a tour and extract metrics."""
        if tour is None:
            return {
                "valid": False,
                "landmarks_visited": 0,
                "total_score": 0,
                "total_duration": 0,
                "time_remaining": 0,
            }
        
        simulation = tour.simulation_cache()
        return {
            "valid": simulation.is_valid,
            "landmarks_visited": len(tour.visited_landmarks),
            "total_score": tour.total_score(),
            "total_duration": simulation.total_duration,
            "time_remaining": self.problem.time_budget - simulation.total_duration,
        }
    
    def benchmark_all(self) -> None:
        """Run all solvers and collect results."""
        print("\n" + "="*70)
        print("SOLVER BENCHMARK - Algiers In 24h")
        print("="*70)
        print(f"Problem: {len(self.problem.landmarks)} landmarks, "
              f"Time budget: {self.problem.time_budget} min")
        print(f"Tour day: {self.problem.tour_day.name}")
        print("="*70 + "\n")
        
        # Greedy Solver
        print("1. Greedy Solver")
        solver = GreedySolver(self.problem, use_ratio=True)
        tour, elapsed = self.run_solver("Greedy (ratio)", solver)
        self.results["Greedy (score/time)"] = {
            "tour": tour,
            "time": elapsed,
            "metrics": self.analyze_tour(tour)
        }
        
        # GRASP Solver
        print("\n2. GRASP Solver")
        solver = GraspSolver(self.problem, iterations=50, alpha=0.3)
        tour, elapsed = self.run_solver("GRASP", solver)
        self.results["GRASP"] = {
            "tour": tour,
            "time": elapsed,
            "metrics": self.analyze_tour(tour)
        }

        # Genetic Algorithm Solver
        print("\n3. Genetic Algorithm Solver")
        solver = TailoredGeneticSolver(
            self.problem,
            fitness_function=FeasibilityFitnessFunction()   
        )
        tour, elapsed = self.run_solver("Genetic Algorithm", solver)
        self.results["Genetic Algorithm"] = {
            "tour": tour,
            "time": elapsed,
            "metrics": self.analyze_tour(tour)
        }
        
        # Simulated Annealing Solver
        print("\n4. Simulated Annealing Solver")
        solver = SimulatedAnnealingSolver(
            self.problem,
            initial_temperature=70,
            cooling_rate=0.97,
            max_iterations=240000,
            reheating_rate=0.95
        )
        tour, elapsed = self.run_solver("Simulated Annealing", solver)
        self.results["Simulated Annealing"] = {
            "tour": tour,
            "time": elapsed,
            "metrics": self.analyze_tour(tour)
        }
        
        
        # CPLEX Solver (if available)
        if HAS_CPLEX:
            print("\n5. CPLEX Solver (Exact)")
            solver = CPLEXSolver(self.problem, time_limit=30)
            tour, elapsed = self.run_solver("CPLEX", solver)
            self.results["CPLEX"] = {
                "tour": tour,
                "time": elapsed,
                "metrics": self.analyze_tour(tour)
            }

        # Tabu Solver
        print("\n5. Tabu Solver")
        solver = TabuSolver(self.problem)
        tour, elapsed = self.run_solver("Tabu", solver)
        self.results["Tabu"] = {
            "tour": tour,
            "time": elapsed,
            "metrics": self.analyze_tour(tour)
        }
    
    def print_results(self) -> None:
        """Print formatted results table."""
        print("\n" + "="*110)
        print("RESULTS SUMMARY")
        print("="*110)
        
        # Header
        header = (
            f"{'Solver':<25} | "
            f"{'Time (s)':>8} | "
            f"{'Score':>8} | "
            f"{'Duration':>10} | "
            f"{'Landmarks':>9} | "
            f"{'Valid':>5} | "
            f"{'Time Left':>10}"
        )
        print(header)
        print("-" * 110)
        
        # Sort by score (descending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["metrics"]["total_score"],
            reverse=True
        )
        
        for solver_name, result in sorted_results:
            metrics = result["metrics"]
            exec_time = result["time"]
            
            valid_str = "✓" if metrics["valid"] else "✗"
            
            print(
                f"{solver_name:<25} | "
                f"{exec_time:>8.3f} | "
                f"{metrics['total_score']:>8.1f} | "
                f"{metrics['total_duration']:>10.1f} | "
                f"{metrics['landmarks_visited']:>9} | "
                f"{valid_str:>5} | "
                f"{metrics['time_remaining']:>10.1f}"
            )
        
        print("="*110)
        
        # Detailed metrics
        print("\nDETAILED METRICS:\n")
        for solver_name, result in sorted_results:
            metrics = result["metrics"]
            exec_time = result["time"]
            print(f"{solver_name}:")
            print(f"  Execution Time:     {exec_time:.3f} seconds")
            print(f"  Tour Valid:         {'Yes' if metrics['valid'] else 'No'}")
            print(f"  Total Score:        {metrics['total_score']:.1f}")
            print(f"  Total Duration:     {metrics['total_duration']:.1f} minutes")
            print(f"  Time Remaining:     {metrics['time_remaining']:.1f} minutes")
            print(f"  Landmarks Visited:  {metrics['landmarks_visited']}")
            print()
    
    def print_tour_details(self, solver_name: str) -> None:
        """Print detailed tour information for a specific solver."""
        if solver_name not in self.results:
            print(f"Solver '{solver_name}' not found in results.")
            return
        
        result = self.results[solver_name]
        tour = result["tour"]
        
        if tour is None:
            print(f"No valid tour for {solver_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"TOUR DETAILS: {solver_name}")
        print(f"{'='*60}\n")
        
        simulation = tour.simulation_cache()
        
        if not tour.visited_landmarks:
            print("No landmarks visited.")
            return
        
        print(f"{'Time':<10} {'Landmark':<30} {'Score':>8}")
        print("-" * 60)
        
        current_time = self.problem.start_time
        print(f"{self._minutes_to_hm(current_time):<10} {self.problem.hotel.name:<30}")
        
        for entry in simulation.entries:
            lm = entry.landmark
            print(f"{self._minutes_to_hm(entry.arrival_time):<10} {lm.name:<30} {lm.interest_score:>8.1f}")
        
        if simulation.entries:
            last_entry = simulation.entries[-1]
            return_time = last_entry.departure_time + self.problem.travel_time(
                last_entry.landmark, self.problem.hotel
            )
            print(f"{self._minutes_to_hm(return_time):<10} {self.problem.hotel.name:<30}")
        
        print(f"\nTotal Duration: {simulation.total_duration:.1f} minutes")
        print(f"Total Score: {tour.total_score():.1f}")
        print(f"Valid: {'Yes' if simulation.is_valid else 'No'}")
    
    @staticmethod
    def _minutes_to_hm(minutes: float) -> str:
        """Convert minutes since midnight to HH:MM format."""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"


def main():
    """Main benchmark function."""
    # Load data
    print("Loading data...")
    try:
        hotel = loadHotel("data/hotel.csv")
        landmarks = loadLandmarks("data/data.csv")
        print(f"✓ Loaded {len(landmarks)} landmarks and hotel")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Create problem instance (assuming Saturday, 480 minutes = 8 hours)
    problem = Problem(
        hotel=hotel,
        landmarks=landmarks,
        time_budget=480,  # 8 hours
        tour_day=Day.SATURDAY
    )
    
    # Run benchmark
    benchmark = SolverBenchmark(problem)
    benchmark.benchmark_all()
    benchmark.print_results()
    
    # Print details for top solver
    if benchmark.results:
        top_solver = sorted(
            benchmark.results.items(),
            key=lambda x: x[1]["metrics"]["total_score"],
            reverse=True
        )[0][0]
        benchmark.print_tour_details(top_solver)


if __name__ == "__main__":
    main()
