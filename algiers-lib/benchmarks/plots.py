"""
plots.py -- All plotting functions for the Algiers-only experiment runner (main.py).

Provides clean, simple bar charts for comparing solvers on the single
Algiers problem instance.

Functions:
    compare_solvers          - Bar chart of solver scores
    visualize_tour           - Tour route visualisation (map-like)
    plot_solver_comparison   - 2x2 grid of score, time, landmarks, budget
    plot_solver_distributions - Score distribution boxplots
    generate_performance_report - Text report
    benchmark_solver         - Run a solver N times and collect metrics
    plot_convergence         - Convergence curve for iterative solvers
    save_metrics_json        - Save metrics to JSON
    generate_html_report     - HTML report with embedded plots
    plot_quality_distribution - Quality score distribution
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

warnings.filterwarnings("ignore")

# -- Style constants ----------------------------------------------------------
FIG_DPI = 200
FONT_SIZE = 11
TITLE_SIZE = 13

# Colour palette
PALETTE = "tab20"


def _save(fig: plt.Figure, path: str | Path) -> None:
    fig.tight_layout(pad=1.5)
    fig.savefig(str(path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _solver_colors(n: int) -> list[str]:
    cmap = cm.get_cmap(PALETTE, max(n, 1))
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


# ===================================================================
# Simple Bar Charts
# ===================================================================

def compare_solvers(df: pd.DataFrame, title: str = "Solver Comparison") -> plt.Figure:
    """Simple bar chart comparing solver scores."""
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.5), 6))
    colors = _solver_colors(len(df))

    x = range(len(df))
    scores = df["Tour Quality"] if "Tour Quality" in df.columns else df.get("Score", df.get("score", [0]*len(df)))
    names = df["Solver"] if "Solver" in df.columns else df.index

    bars = ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Tour Quality / Score", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    return fig


def plot_solver_comparison(df: pd.DataFrame, figsize=(14, 10)) -> plt.Figure:
    """2x2 grid: score, time, landmarks visited, budget utilisation."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = _solver_colors(len(df))
    names = df["Solver"] if "Solver" in df.columns else df.index
    x = range(len(names))

    # 1. Score
    score_col = "Tour Quality" if "Tour Quality" in df.columns else "Score"
    axes[0, 0].bar(x, df[score_col], color=colors, edgecolor="white")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Score Comparison", fontweight="bold")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.3)

    # 2. Execution time
    time_col = "Execution Time (s)" if "Execution Time (s)" in df.columns else "Time"
    axes[0, 1].bar(x, df[time_col], color=colors, edgecolor="white")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].set_title("Execution Time", fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.3)

    # 3. Landmarks visited
    lm_col = "Landmarks Visited" if "Landmarks Visited" in df.columns else "Landmarks"
    axes[1, 0].bar(x, df[lm_col], color=colors, edgecolor="white")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Landmarks Visited", fontweight="bold")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.3)

    # 4. Budget utilisation
    bu_col = "Time Budget Used (%)" if "Time Budget Used (%)" in df.columns else None
    if bu_col and bu_col in df.columns:
        axes[1, 1].bar(x, df[bu_col], color=colors, edgecolor="white")
        axes[1, 1].axhline(100, color="red", linestyle="--", alpha=0.5, label="100%")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axes[1, 1].set_ylabel("%")
        axes[1, 1].set_title("Budget Utilisation", fontweight="bold")
        axes[1, 1].legend(loc="best", fontsize=8)
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "No budget data available",
                        ha="center", va="center", fontsize=12)

    fig.suptitle("Solver Performance Overview", fontsize=TITLE_SIZE + 1, fontweight="bold")
    return fig


def plot_solver_distributions(metrics_list, figsize=(16, 10)) -> plt.Figure:
    """Boxplots of score distribution across runs per solver."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = _solver_colors(len(metrics_list))
    names = [getattr(m, "solver_name", f"Solver {i}") for i, m in enumerate(metrics_list)]

    # Collect data from metrics objects
    for ax_idx, (attr, label) in enumerate([
        ("tour_quality", "Tour Quality"),
        ("execution_time", "Execution Time (s)"),
    ]):
        data = []
        valid_names = []
        valid_colors = []
        for m, name, color in zip(metrics_list, names, colors):
            val = getattr(m, attr, None)
            if val is not None:
                data.append([val])
                valid_names.append(name)
                valid_colors.append(color)

        if data:
            bp = axes[ax_idx].boxplot(data, patch_artist=True, vert=True)
            for patch, c in zip(bp["boxes"], valid_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            axes[ax_idx].set_xticklabels(valid_names, rotation=30, ha="right", fontsize=8)
        axes[ax_idx].set_ylabel(label)
        axes[ax_idx].set_title(f"{label} Distribution", fontweight="bold")
        axes[ax_idx].grid(axis="y", linestyle="--", alpha=0.3)

    axes[2].axis("off")
    return fig


def plot_quality_distribution(df: pd.DataFrame, title: str = "Quality Distribution") -> plt.Figure:
    """Histogram of solver quality scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    score_col = "Tour Quality" if "Tour Quality" in df.columns else "Score"
    ax.hist(df[score_col], bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Score", fontsize=FONT_SIZE)
    ax.set_ylabel("Frequency", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig


def plot_convergence(scores: list[float], title: str = "Convergence") -> plt.Figure:
    """Line plot of score improvement over iterations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scores, color="#C44E52", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=FONT_SIZE)
    ax.set_ylabel("Score", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    return fig


# ===================================================================
# Tour Visualisation
# ===================================================================

def visualize_tour(tour, title: str = "Tour Visualisation") -> plt.Figure:
    """Scatter plot of tour stops with route lines."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if tour is None or not hasattr(tour, "visited_landmarks") or not tour.visited_landmarks:
        ax.text(0.5, 0.5, "No tour to display", ha="center", va="center", fontsize=14)
        return fig

    # Collect coordinates
    lats, lons, scores = [], [], []
    for lm in tour.visited_landmarks:
        lats.append(lm.latitude)
        lons.append(lm.longitude)
        scores.append(lm.interest_score)

    # Plot route
    ax.plot(lons, lats, "o-", color="#4C72B0", markersize=6, linewidth=1.5, alpha=0.7)

    # Scatter with size = score
    sc = ax.scatter(lons, lats, c=scores, cmap="YlOrRd", s=[max(s*10, 30) for s in scores],
                    edgecolors="black", linewidths=0.5, zorder=5)
    fig.colorbar(sc, ax=ax, label="Interest Score")

    # Labels
    for lm in tour.visited_landmarks:
        ax.annotate(lm.name[:15], (lm.longitude, lm.latitude),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Longitude", fontsize=FONT_SIZE)
    ax.set_ylabel("Latitude", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    return fig


# ===================================================================
# Metrics & Reports
# ===================================================================

class SolverMetrics:
    """Simple container for solver benchmark metrics."""

    def __init__(self, tour, execution_time: float, solver_name: str = ""):
        self.tour = tour
        self.execution_time = execution_time
        self.solver_name = solver_name
        self.tour_quality = tour.total_score() if tour else 0.0


def benchmark_solver(solver, num_runs: int = 3) -> SolverMetrics:
    """Run a solver multiple times and return average metrics."""
    import time as _time

    times = []
    best_tour = None
    best_score = -1

    for _ in range(num_runs):
        t0 = _time.perf_counter()
        tour = solver.solve()
        elapsed = _time.perf_counter() - t0
        times.append(elapsed)

        if tour is not None:
            score = tour.total_score()
            if score > best_score:
                best_score = score
                best_tour = tour

    avg_time = sum(times) / len(times) if times else 0.0
    return SolverMetrics(best_tour, avg_time, solver.__class__.__name__)


def save_metrics_json(metrics: dict, filepath: str | Path) -> None:
    """Save metrics dictionary to JSON."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def generate_performance_report(df: pd.DataFrame, filepath: str | Path | None = None) -> str:
    """Generate a text performance report from results DataFrame."""
    lines = ["=" * 60, "PERFORMANCE REPORT", "=" * 60, ""]

    score_col = "Tour Quality" if "Tour Quality" in df.columns else "Score"
    time_col = "Execution Time (s)" if "Execution Time (s)" in df.columns else "Time"

    # Sort by score descending
    df_sorted = df.sort_values(score_col, ascending=False)

    for _, row in df_sorted.iterrows():
        name = row.get("Solver", "Unknown")
        score = row[score_col]
        time = row[time_col]
        lines.append(f"{name}:")
        lines.append(f"  Score: {score:.2f}")
        lines.append(f"  Time:  {time:.3f}s")
        lines.append("")

    lines.append("=" * 60)
    report = "\n".join(lines)

    if filepath:
        Path(filepath).write_text(report, encoding="utf-8")

    return report


def generate_html_report(df: pd.DataFrame, filepath: str | Path) -> None:
    """Generate a simple HTML report with an embedded table."""
    score_col = "Tour Quality" if "Tour Quality" in df.columns else "Score"
    time_col = "Execution Time (s)" if "Execution Time (s)" in df.columns else "Time"

    rows_html = ""
    for _, row in df.sort_values(score_col, ascending=False).iterrows():
        rows_html += f"""
        <tr>
            <td>{row.get('Solver', 'Unknown')}</td>
            <td>{row[score_col]:.2f}</td>
            <td>{row[time_col]:.3f}</td>
        </tr>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Solver Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2em; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #4C72B0; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Solver Performance Report</h1>
        <table>
            <tr><th>Solver</th><th>Score</th><th>Time (s)</th></tr>
            {rows_html}
        </table>
    </body>
    </html>"""

    Path(filepath).write_text(html, encoding="utf-8")
