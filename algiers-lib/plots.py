"""Performance analysis and visualization module for the Orienteering Problem solvers."""

import time
from typing import Callable, List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from models.problem import Problem
from models.tour import Tour
from models.landmark import Landmark
from solvers.solver import Solver
import json
from pathlib import Path


class PerformanceMetrics:
    """Container for performance metrics of a solver run."""
    
    def __init__(
        self,
        solver_name: str,
        tour: Tour,
        execution_time: float,
        num_landmarks: int,
        iteration_times: List[float] = None,
        memory_used: float = 0
    ):
        self.solver_name = solver_name
        self.tour = tour
        self.execution_time = execution_time
        self.num_landmarks = num_landmarks
        self.iteration_times = iteration_times or []
        self.memory_used = memory_used
        self.tour_quality = self._calculate_tour_quality()
    
    def _calculate_tour_quality(self) -> float:
        """Calculate tour quality as total interest score."""
        return sum(lm.interest_score for lm in self.tour.visited_landmarks)
    
    @property
    def average_iteration_time(self) -> float:
        """Get average time per iteration."""
        if not self.iteration_times:
            return 0
        return np.mean(self.iteration_times)
    
    @property
    def quality_per_second(self) -> float:
        """Get quality points per second of execution."""
        if self.execution_time == 0:
            return 0
        return self.tour_quality / self.execution_time
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'solver_name': self.solver_name,
            'tour_quality': float(self.tour_quality),
            'landmarks_visited': len(self.tour.visited_landmarks),
            'execution_time': float(self.execution_time),
            'average_iteration_time': float(self.average_iteration_time),
            'quality_per_second': float(self.quality_per_second),
            'memory_used_mb': float(self.memory_used),
            'total_duration': float(self.tour.simulate().total_duration),
            'is_valid': self.tour.is_valid()
        }
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(solver={self.solver_name}, "
            f"landmarks_visited={len(self.tour.visited_landmarks)}, "
            f"quality={self.tour_quality:.2f}, "
            f"time={self.execution_time:.4f}s, "
            f"quality/s={self.quality_per_second:.2f})"
        )


def benchmark_solver(
    solver: Solver,
    num_runs: int = 3
) -> PerformanceMetrics:
    """
    Benchmark a solver by running it multiple times and measuring performance.
    
    Args:
        solver (Solver): The solver instance to benchmark.
        num_runs (int): Number of runs to average over.
    
    Returns:
        PerformanceMetrics: Aggregated performance metrics.
    """
    execution_times = []
    best_tour = None
    best_quality = -1
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        tour = solver.solve()
        end_time = time.perf_counter()
        
        execution_times.append(end_time - start_time)
        
        quality = sum(lm.interest_score for lm in tour.visited_landmarks)
        if quality > best_quality:
            best_quality = quality
            best_tour = tour
    
    avg_time = np.mean(execution_times)
    metrics = PerformanceMetrics(
        solver_name=solver.__class__.__name__,
        tour=best_tour,
        execution_time=avg_time,
        num_landmarks=len(solver.problem.landmarks)
    )
    
    return metrics


def _human_time_label(seconds: float) -> str:
    """Return a human-friendly duration string."""
    if seconds < 0:
        return "0 s"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def compare_solvers(
    problem: Problem,
    solver_classes: List[type],
    solver_kwargs: Dict[str, Dict[str, Any]] = None,
    num_runs: int = 3
) -> pd.DataFrame:
    """
    Compare multiple solvers on the same problem.
    
    Args:
        problem (Problem): The problem instance to solve.
        solver_classes (List[type]): List of solver classes to compare.
        solver_kwargs (Dict[str, Dict[str, Any]]): Keyword arguments for each solver.
        num_runs (int): Number of runs per solver.
    
    Returns:
        pd.DataFrame: Comparison results with columns for solver name, quality, and time.
    """
    if solver_kwargs is None:
        solver_kwargs = {cls.__name__: {} for cls in solver_classes}
    
    results = []
    
    for solver_class in solver_classes:
        kwargs = solver_kwargs.get(solver_class.__name__, {})
        solver = solver_class(problem, **kwargs)
        metrics = benchmark_solver(solver, num_runs)
        
        results.append({
            'Solver': metrics.solver_name,
            'Tour Quality': metrics.tour_quality,
            'Landmarks Visited': len(metrics.tour.visited_landmarks),
            'Execution Time (s)': metrics.execution_time,
            'Time Budget Used (%)': (
                (metrics.tour.simulate().total_duration / problem.time_budget * 100)
                if metrics.tour.simulate().is_valid else 0
            )
        })
    
    return pd.DataFrame(results)


def visualize_tour(
    tour: Tour,
    title: str = "Tour Visualization",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize a tour on a map with landmarks and path.
    
    Args:
        tour (Tour): The tour to visualize.
        title (str): Title for the plot.
        figsize (Tuple[int, int]): Figure size.
    
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot hotel
    hotel = tour.problem.hotel
    ax.plot(hotel.coordinates[1], hotel.coordinates[0], 'r*', markersize=20, label='Hotel')
    
    # Plot visited landmarks
    visited_lats = [lm.coordinates[0] for lm in tour.visited_landmarks]
    visited_lons = [lm.coordinates[1] for lm in tour.visited_landmarks]
    ax.plot(visited_lons, visited_lats, 'go', markersize=10, label='Visited Landmarks')
    
    # Plot unvisited landmarks
    unvisited = tour.problem.unvisited_landmarks(tour)
    unvisited_lats = [lm.coordinates[0] for lm in unvisited]
    unvisited_lons = [lm.coordinates[1] for lm in unvisited]
    ax.plot(unvisited_lons, unvisited_lats, 'bx', markersize=8, label='Unvisited Landmarks')
    
    # Draw tour path
    path_lats = [hotel.coordinates[0]] + visited_lats + [hotel.coordinates[0]]
    path_lons = [hotel.coordinates[1]] + visited_lons + [hotel.coordinates[1]]
    ax.plot(path_lons, path_lats, 'g--', alpha=0.6, linewidth=1.5)
    
    # Add landmark labels
    for i, lm in enumerate(tour.visited_landmarks, 1):
        ax.annotate(f'{i}', (lm.coordinates[1], lm.coordinates[0]), fontsize=8)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_solver_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Create a comprehensive comparison visualization of solver performance.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame from compare_solvers().
        figsize (Tuple[int, int]): Figure size.
    
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Solver Performance Comparison', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
    
    # Quality comparison
    bars1 = axes[0, 0].bar(comparison_df['Solver'], comparison_df['Tour Quality'], color=colors)
    axes[0, 0].set_title('Tour Quality (Interest Score)', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Execution time comparison
    execution_times = comparison_df['Execution Time (s)']
    use_milliseconds = execution_times.max() < 1.0
    if use_milliseconds:
        plot_values = execution_times * 1000
        y_label = 'Time (milliseconds)'
        value_label = lambda v: f'{v:.1f} ms'
    else:
        plot_values = execution_times
        y_label = 'Time (seconds)'
        value_label = lambda v: f'{v:.3f} s'

    bars2 = axes[0, 1].bar(comparison_df['Solver'], plot_values, color=colors)
    axes[0, 1].set_title('Execution Time', fontweight='bold')
    axes[0, 1].set_ylabel(y_label)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                value_label(height), ha='center', va='bottom', fontsize=9)
    
    # Time budget utilization
    bars3 = axes[1, 0].bar(comparison_df['Solver'], comparison_df['Time Budget Used (%)'], color=colors)
    axes[1, 0].set_title('Time Budget Utilization', fontweight='bold')
    axes[1, 0].set_ylabel('Usage (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axhline(y=100, color='r', linestyle='--', linewidth=1, alpha=0.5, label='100% Budget')
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Landmarks visited
    bars4 = axes[1, 1].bar(comparison_df['Solver'], comparison_df['Landmarks Visited'], color=colors)
    axes[1, 1].set_title('Landmarks Visited', fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_solver_distributions(
    metrics_list: List[PerformanceMetrics],
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot histograms and distribution analysis for solver metrics.

    Args:
        metrics_list (List[PerformanceMetrics]): List of solver performance metrics.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    if not metrics_list:
        raise ValueError("metrics_list must contain at least one PerformanceMetrics item")

    solver_names = sorted(set(m.solver_name for m in metrics_list))
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Solver Distribution and Cost/Time Analysis', fontsize=16, fontweight='bold')

    # Execution time histogram / bar chart
    ax_time = axes[0, 0]
    execution_values = [m.execution_time for m in metrics_list]
    time_scale_ms = max(execution_values) < 1.0
    if time_scale_ms:
        samples_sets = [[t * 1000 for t in [m.execution_time for m in metrics_list if m.solver_name == name]] for name in solver_names]
        xlabel = 'Execution Time (milliseconds)'
        value_formatter = lambda v: f'{v:.1f} ms'
    else:
        samples_sets = [[m.execution_time for m in metrics_list if m.solver_name == name] for name in solver_names]
        xlabel = 'Execution Time (seconds)'
        value_formatter = lambda v: f'{v:.3f} s'

    for name, samples in zip(solver_names, samples_sets):
        if len(samples) > 1:
            ax_time.hist(samples, alpha=0.5, bins=8, label=name)
        elif samples:
            ax_time.bar(name, samples[0], alpha=0.8)
    ax_time.set_title('Execution Time Distribution')
    ax_time.set_ylabel('Number of runs')
    ax_time.set_xlabel(xlabel)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=8)

    # Tour quality histogram
    ax_quality = axes[0, 1]
    for name in solver_names:
        qualities = [m.tour_quality for m in metrics_list if m.solver_name == name]
        if len(qualities) > 1:
            ax_quality.hist(qualities, alpha=0.5, bins=8, label=name)
        else:
            ax_quality.bar(name, qualities[0], alpha=0.8)
    ax_quality.set_title('Tour Quality Distribution')
    ax_quality.set_ylabel('Number of runs')
    ax_quality.set_xlabel('Tour Quality (interest score)')
    ax_quality.grid(True, alpha=0.3)
    ax_quality.legend(fontsize=8)

    # Total duration box plot
    ax_duration = axes[1, 0]
    durations = [
        [m.tour.simulate().total_duration for m in metrics_list if m.solver_name == name]
        for name in solver_names
    ]
    ax_duration.boxplot(durations, labels=solver_names, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='navy'))
    ax_duration.set_title('Total Duration Distribution')
    ax_duration.set_ylabel('Duration (minutes)')
    ax_duration.grid(True, alpha=0.3)

    # Quality vs execution time scatter plot
    ax_scatter = axes[1, 1]
    for name in solver_names:
        solver_samples = [m for m in metrics_list if m.solver_name == name]
        times = [m.execution_time for m in solver_samples]
        if time_scale_ms:
            times = [t * 1000 for t in times]
        ax_scatter.scatter(
            times,
            [m.tour_quality for m in solver_samples],
            s=80,
            alpha=0.7,
            label=name
        )
    ax_scatter.set_title('Quality vs Execution Time')
    ax_scatter.set_xlabel('Execution Time' + (' (milliseconds)' if time_scale_ms else ' (seconds)'))
    ax_scatter.set_ylabel('Tour Quality')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_convergence(
    solver_class: type,
    problem: Problem,
    solver_kwargs: Dict[str, Any] = None,
    num_runs: int = 5,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot convergence behavior by running solver multiple times.
    
    Args:
        solver_class (type): The solver class to analyze.
        problem (Problem): The problem instance.
        solver_kwargs (Dict[str, Any]): Solver initialization kwargs.
        num_runs (int): Number of runs to track.
        figsize (Tuple[int, int]): Figure size.
    
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    if solver_kwargs is None:
        solver_kwargs = {}
    
    qualities = []
    times = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        solver = solver_class(problem, **solver_kwargs)
        tour = solver.solve()
        end_time = time.perf_counter()
        
        quality = sum(lm.interest_score for lm in tour.visited_landmarks)
        qualities.append(quality)
        times.append(end_time - start_time)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Quality progression
    ax1.plot(range(1, num_runs + 1), qualities, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.fill_between(range(1, num_runs + 1), qualities, alpha=0.3, color='steelblue')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Tour Quality')
    ax1.set_title(f'{solver_class.__name__} - Quality Over Multiple Runs')
    ax1.grid(True, alpha=0.3)
    
    # Execution time progression
    ax2.plot(range(1, num_runs + 1), times, 's-', linewidth=2, markersize=8, color='orange')
    ax2.fill_between(range(1, num_runs + 1), times, alpha=0.3, color='orange')
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title(f'{solver_class.__name__} - Execution Time Over Multiple Runs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_performance_report(
    comparison_df: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Generate a comprehensive text report of solver performance comparison.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame from compare_solvers().
        output_path (str): Optional path to save the report.
    
    Returns:
        str: The generated report text.
    """
    report = "=" * 90 + "\n"
    report += "SOLVER PERFORMANCE COMPARISON REPORT\n"
    report += "=" * 90 + "\n"
    report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "=" * 90 + "\n\n"
    
    report += "DETAILED METRICS\n"
    report += "-" * 90 + "\n"
    report += comparison_df.to_string(index=False) + "\n\n"
    
    report += "=" * 90 + "\n"
    report += "ANALYSIS SUMMARY\n"
    report += "=" * 90 + "\n\n"
    
    best_quality_idx = comparison_df['Tour Quality'].idxmax()
    best_quality_solver = comparison_df.loc[best_quality_idx, 'Solver']
    best_quality = comparison_df.loc[best_quality_idx, 'Tour Quality']
    report += f"[BEST QUALITY] {best_quality_solver}\n"
    report += f"   Score: {best_quality:.2f}\n\n"
    
    fastest_idx = comparison_df['Execution Time (s)'].idxmin()
    fastest_solver = comparison_df.loc[fastest_idx, 'Solver']
    fastest_time = comparison_df.loc[fastest_idx, 'Execution Time (s)']
    report += f"[FASTEST] {fastest_solver}\n"
    report += f"   Time: {fastest_time:.6f}s\n\n"
    
    most_landmarks_idx = comparison_df['Landmarks Visited'].idxmax()
    most_landmarks_solver = comparison_df.loc[most_landmarks_idx, 'Solver']
    most_landmarks = comparison_df.loc[most_landmarks_idx, 'Landmarks Visited']
    report += f"[MOST LANDMARKS] {most_landmarks_solver}\n"
    report += f"   Count: {int(most_landmarks)}\n\n"
    
    # Calculate efficiency
    comparison_df['Efficiency'] = comparison_df['Tour Quality'] / (comparison_df['Execution Time (s)'] + 0.0001)
    best_efficiency_idx = comparison_df['Efficiency'].idxmax()
    best_efficiency_solver = comparison_df.loc[best_efficiency_idx, 'Solver']
    best_efficiency = comparison_df.loc[best_efficiency_idx, 'Efficiency']
    report += f"[BEST EFFICIENCY] {best_efficiency_solver}\n"
    report += f"   Ratio: {best_efficiency:.2f}\n\n"
    
    report += "=" * 90 + "\n"
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


def save_metrics_json(
    metrics_list: List[PerformanceMetrics],
    output_path: str
) -> None:
    """
    Save performance metrics to JSON file for further analysis.
    
    Args:
        metrics_list (List[PerformanceMetrics]): List of metrics to save.
        output_path (str): Path to save JSON file.
    """
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': [m.to_dict() for m in metrics_list]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def plot_quality_distribution(
    metrics_list: List[PerformanceMetrics],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of tour qualities across different solver runs.
    
    Args:
        metrics_list (List[PerformanceMetrics]): List of metrics from multiple runs.
        figsize (Tuple[int, int]): Figure size.
    
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    solver_names = list(set(m.solver_name for m in metrics_list))
    
    for solver_name in solver_names:
        qualities = [m.tour_quality for m in metrics_list if m.solver_name == solver_name]
        ax.hist(qualities, alpha=0.6, label=solver_name, bins=10)
    
    ax.set_xlabel('Tour Quality')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Tour Qualities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_html_report(
    comparison_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate an HTML report with embedded visualizations.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame from compare_solvers().
        output_path (str): Path to save HTML file.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Orienteering Problem - Solver Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
            .section { background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #34495e; color: white; }
            tr:hover { background-color: #f9f9f9; }
            .metric { font-weight: bold; color: #2c3e50; }
            .timestamp { color: #7f8c8d; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Orienteering Problem - Solver Performance Report</h1>
            <p class="timestamp">Generated: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Solver</th>
                    <th>Tour Quality</th>
                    <th>Landmarks Visited</th>
                    <th>Execution Time (s)</th>
                    <th>Time Budget Used (%)</th>
                </tr>
    """
    
    for _, row in comparison_df.iterrows():
        html_content += f"""
                <tr>
                    <td class="metric">{row['Solver']}</td>
                    <td>{row['Tour Quality']:.2f}</td>
                    <td>{int(row['Landmarks Visited'])}</td>
                    <td>{row['Execution Time (s)']:.6f}</td>
                    <td>{row['Time Budget Used (%)']:.2f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
    """
    
    best_quality_idx = comparison_df['Tour Quality'].idxmax()
    fastest_idx = comparison_df['Execution Time (s)'].idxmin()
    
    html_content += f"""
            <p><strong>Best Quality:</strong> {comparison_df.loc[best_quality_idx, 'Solver']} ({comparison_df.loc[best_quality_idx, 'Tour Quality']:.2f})</p>
            <p><strong>Fastest:</strong> {comparison_df.loc[fastest_idx, 'Solver']} ({comparison_df.loc[fastest_idx, 'Execution Time (s)']:.6f}s)</p>
        </div>
    </body>
    </html>
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report saved to: {output_path}")
