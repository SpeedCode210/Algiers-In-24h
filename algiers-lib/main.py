# Entry point for the experiments script, in order to compare the approaches on different datasets and generate diagrams for the reports.

import os
import sys
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config import solver_configs

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.problem import Problem
from models.landmark import Day
from solvers.greedy_solver import GreedySolver
from solvers.simulated_annealing_solver import (
    SimulatedAnnealingSolver, AcceptanceFunction, DecayFunction
)
from plots import (
    compare_solvers, visualize_tour, plot_solver_comparison,
    plot_solver_distributions, generate_performance_report,
    benchmark_solver, plot_convergence, save_metrics_json,
    generate_html_report, plot_quality_distribution
)


class ExperimentsRunner:
    """Runner for comprehensive solver performance experiments."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize the experiments runner.
        
        Args:
            data_dir (str): Directory containing data files.
            output_dir (str): Directory to save results and visualizations.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.tours = {}
        self.metrics = {}
    
    def load_problem(
        self,
        landmarks_file: str,
        hotel_file: str,
        time_budget: int = 480,
        tour_day: Day = Day.MONDAY,
        start_time: int = 540,
        problem_name: str = "problem"
    ) -> Problem:
        """
        Load a problem instance from files.
        
        Args:
            landmarks_file (str): Path to landmarks CSV file.
            hotel_file (str): Path to hotel CSV file.
            time_budget (int): Time budget in minutes.
            tour_day (Day): Day of the tour.
            start_time (int): Start time in minutes.
            problem_name (str): Name of the problem.
        
        Returns:
            Problem: The loaded problem instance.
        """
        landmarks_path = self.data_dir / landmarks_file
        hotel_path = self.data_dir / hotel_file
        
        print(f"Loading problem '{problem_name}'...")
        print(f"  Landmarks: {landmarks_path}")
        print(f"  Hotel: {hotel_path}")
        
        problem = Problem.LoadProblem(
            str(landmarks_path),
            str(hotel_path),
            time_budget,
            tour_day,
            start_time
        )
        
        print(f"  Loaded {len(problem.landmarks)} landmarks")
        print(f"  Time budget: {time_budget} minutes")
        print(f"  Tour day: {tour_day.name}\n")
        
        return problem
    
    def run_solver(
        self,
        problem: Problem,
        solver_class: type,
        solver_kwargs: dict = None,
        num_runs: int = 3,
        solver_name: str = None
    ) -> tuple[dict, object]:
        """
        Run a solver on a problem and collect metrics.
        
        Args:
            problem (Problem): The problem to solve.
            solver_class (type): The solver class.
            solver_kwargs (dict): Arguments for solver initialization.
            num_runs (int): Number of runs to average.
            solver_name (str): Name for the solver (defaults to class name).
        
        Returns:
            tuple[dict, object]: A result summary dict and the raw metrics object.
        """
        if solver_kwargs is None:
            solver_kwargs = {}
        
        if solver_name is None:
            solver_name = solver_class.__name__
        
        print(f"Running {solver_name}...")
        
        solver = solver_class(problem, **solver_kwargs)
        metrics = benchmark_solver(solver, num_runs=num_runs)
        
        result_dict = {
            'Solver': solver_name,
            'Tour Quality': metrics.tour_quality,
            'Landmarks Visited': len(metrics.tour.visited_landmarks),
            'Execution Time (s)': metrics.execution_time,
            'Tour': metrics.tour
        }
        
        time_seconds = metrics.execution_time
        if time_seconds < 1:
            time_text = f"{time_seconds * 1000:.1f} ms"
        else:
            time_text = f"{time_seconds:.3f} s"

        print(f"  Quality: {metrics.tour_quality:.2f}")
        print(f"  Landmarks: {len(metrics.tour.visited_landmarks)}")
        print(f"  Time: {time_text}\n")
        print(f"  Finished {solver_name}.\n")
        
        return result_dict, metrics
    
    def run_comparison_experiment(
        self,
        problem: Problem,
        experiment_name: str = "comparison"
    ) -> pd.DataFrame:
        """
        Run a comparison experiment with multiple solvers.
        
        Args:
            problem (Problem): The problem to solve.
            experiment_name (str): Name for this experiment.
        
        Returns:
            pd.DataFrame: Comparison results.
        """
        print(f"Running comparison experiment: '{experiment_name}'\n")
        print("Comparing solver performance, then generating easy-to-read output and reports.\n")
        
        results = []
    

        metrics_list = []
        for solver_class, kwargs, solver_name, tour_key in solver_configs:
            result, metrics = self.run_solver(
                problem,
                solver_class,
                kwargs,
                num_runs=3,
                solver_name=solver_name
            )
            results.append(result)
            metrics_list.append(metrics)
            self.tours[experiment_name + tour_key] = result['Tour']

        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.drop('Tour', axis=1)

        # Calculate time budget utilization
        time_budgets = []
        for tour in [r['Tour'] for r in results]:
            simulation = tour.simulate()
            if simulation.is_valid:
                utilization = (simulation.total_duration / problem.time_budget) * 100
            else:
                utilization = 0
            time_budgets.append(utilization)
        
        df['Time Budget Used (%)'] = time_budgets
        self.results[experiment_name] = df
        self.metrics[experiment_name] = metrics_list

        print(f"Comparison complete: {len(results)} solvers evaluated.\n")
        return df
    
    def run_sensitivity_analysis(
        self,
        problem: Problem,
        parameter_name: str = "temperature"
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on solver parameters.
        
        Args:
            problem (Problem): The problem to solve.
            parameter_name (str): Which parameter to analyze.
        
        Returns:
            pd.DataFrame: Sensitivity analysis results.
        """
        print(f"Running sensitivity analysis on {parameter_name}...\n")
        
        results = []
        
        if parameter_name == "temperature":
            temperatures = [1, 5, 10, 20, 50]
            for temp in temperatures:
                result = self.run_solver(
                    problem,
                    SimulatedAnnealingSolver,
                    {
                        'initial_temperature': temp,
                        'cooling_rate': 0.95,
                        'max_iterations': 500
                    },
                    num_runs=2,
                    solver_name=f"SA (temp={temp})"
                )
                result['Parameter Value'] = temp
                results.append(result)
        
        elif parameter_name == "iterations":
            iterations = [100, 500, 1000, 2000]
            for iters in iterations:
                result = self.run_solver(
                    problem,
                    SimulatedAnnealingSolver,
                    {
                        'initial_temperature': 10,
                        'cooling_rate': 0.95,
                        'max_iterations': iters
                    },
                    num_runs=2,
                    solver_name=f"SA (iters={iters})"
                )
                result['Parameter Value'] = iters
                results.append(result)
        
        df = pd.DataFrame(results)
        df = df.drop('Tour', axis=1)
        
        return df
    
    def generate_visualizations(self, experiment_name: str):
        """
        Generate visualization plots for an experiment.
        
        Args:
            experiment_name (str): Name of the experiment to visualize.
        """
        if experiment_name not in self.results:
            print(f"Experiment '{experiment_name}' not found")
            return
        
        print(f"Generating visualizations for '{experiment_name}'...\n")
        
        df = self.results[experiment_name]
        
        # Main comparison plot (2x2 grid)
        fig = plot_solver_comparison(df, figsize=(14, 10))
        plot_path = self.output_dir / f"{experiment_name}_comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close(fig)

        if experiment_name in self.metrics:
            fig = plot_solver_distributions(self.metrics[experiment_name], figsize=(16, 10))
            plot_path = self.output_dir / f"{experiment_name}_distributions.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {plot_path}")
            plt.close(fig)
            print("  Distribution visuals created for time, quality, and duration.")
        
        # Individual tour visualizations
        for tour_key, tour in self.tours.items():
            if experiment_name in tour_key:
                fig = visualize_tour(
                    tour,
                    title=f"{experiment_name.title()} - {tour_key.split('_', 1)[1].replace('_', ' ').title()}"
                )
                plot_path = self.output_dir / f"{experiment_name}_{tour_key.split('_', 1)[1]}_tour.png"
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {plot_path}")
                plt.close(fig)
        
        print()
    
    def generate_report(self, experiment_name: str):
        """
        Generate text and HTML reports for an experiment.
        
        Args:
            experiment_name (str): Name of the experiment to report on.
        """
        if experiment_name not in self.results:
            print(f"Experiment '{experiment_name}' not found")
            return
        
        print(f"Generating reports for '{experiment_name}'...\n")
        
        df = self.results[experiment_name]
        
        # Text report
        report = generate_performance_report(df)
        print(report)
        
        # Save text report
        report_path = self.output_dir / f"{experiment_name}_report.txt"
        generate_performance_report(df, str(report_path))
        print(f"✓ Report saved to: {report_path}\n")
        
        # HTML report
        html_path = self.output_dir / f"{experiment_name}_report.html"
        generate_html_report(df, str(html_path))
        print(f"✓ HTML report saved to: {html_path}\n")
        print("Your experiment report is ready and available in the results folder.\n")
    
    def run_full_experiment_suite(self):
        """Run a full suite of experiments."""
        print("=" * 80)
        print("ORIENTEERING PROBLEM SOLVER EXPERIMENTS")
        print("=" * 80 + "\n")
        
        # Load problem
        try:
            problem = self.load_problem(
                landmarks_file="data.csv",
                hotel_file="hotel.csv",
                time_budget=480,
                tour_day=Day.MONDAY,
                start_time=540,
                problem_name="Algiers Tour Planning"
            )
        except Exception as e:
            print(f"Error loading problem: {e}")
            print("Using demo problem instead...\n")
            from models.landmark import Landmark, TimeSlot, WeeklySchedule
            
            schedule = WeeklySchedule()
            schedule.schedule[Day.MONDAY] = [TimeSlot(540, 720)]
            
            hotel = Landmark(
                id="hotel",
                name="Hotel Central",
                latitude=36.7,
                longitude=3.1,
                interest_score=0,
                visit_duration=0,
                schedule=schedule,
                category="Hotel"
            )
            
            landmarks = [
                Landmark(
                    id=str(i),
                    name=f"Landmark {i}",
                    latitude=36.5 + (i % 5) * 0.05,
                    longitude=3.0 + (i // 5) * 0.05,
                    interest_score=float(9 - i % 9),
                    visit_duration=30 + (i % 5) * 10,
                    schedule=schedule,
                    category="Tourist Site"
                )
                for i in range(10)
            ]
            
            problem = Problem(
                hotel=hotel,
                landmarks=landmarks,
                time_budget=480,
                tour_day=Day.MONDAY,
                start_time=540
            )
        
        # Run comparison experiment
        comp_df = self.run_comparison_experiment(problem, experiment_name="main_comparison")
        
        # Generate visualizations and reports
        self.generate_visualizations("main_comparison")
        self.generate_report("main_comparison")
        
        print("=" * 80)
        print("EXPERIMENTS COMPLETED")
        print("=" * 80)


def main():
    """Main entry point for experiments."""
    runner = ExperimentsRunner(
        data_dir="data",
        output_dir="results"
    )
    
    runner.run_full_experiment_suite()


if __name__ == "__main__":
    main()
