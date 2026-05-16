"""
run_s100.py
============
Run benchmarks on the Solomon-100 dataset.

Workflow:
  - NO CPLEX -- compares against Righini & Salani bestPossible ground truth files
  - All heuristic algorithms run:
      - Stochastic solvers: 5 runs each
      - Deterministic solvers (Greedy): 1 run

Outputs (saved to ./results/):
  - solomon100_scores.png      Bar chart: average scores on Solomon-100
  - solomon100_times.png       Bar chart: average times on Solomon-100
  - results_s100.tex           LaTeX table: mean +- std score, time, gap

Usage:
    python run_s100.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from benchmarks.runner import (
    load_solomon_group,
    load_s100_ground_truth,
    run_group,
    NUM_RUNS,
    NUM_WORKERS,
    MAX_PER_GROUP,
)

from benchmarks.solver_registry import (
    SOLOMON_100_VARIANTS,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT = _HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style constants
# ---------------------------------------------------------------------------
ALGO_ORDER = [
    "Greedy", "GRASP", "SA", "Tabu",
    "Genetic-Tailored", "Genetic-Score",
]

PALETTE = {
    "Greedy":           "#4C72B0",
    "GRASP":            "#DD8452",
    "SA":               "#55A868",
    "Tabu":             "#C44E52",
    "Genetic-Tailored": "#8172B3",
    "Genetic-Score":    "#A78DC4",
}

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-run results by solver name.

    Returns a DataFrame indexed by solver name with columns:
        n, mean_score, std_score, mean_time, std_time, mean_gap, std_gap
    """
    agg = df.groupby("solver").agg(
        n=("run_id", "count"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        mean_time=("time_s", "mean"),
        std_time=("time_s", "std"),
        mean_gap=("optimality_gap_pct", "mean"),
        std_gap=("optimality_gap_pct", "std"),
    )
    # Reindex to canonical order
    ordered = [a for a in ALGO_ORDER if a in agg.index]
    return agg.loc[ordered]


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_bar_scores(agg: pd.DataFrame, title: str, fname: str):
    """Bar chart of mean algorithm scores with std error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(agg))
    colors = [PALETTE.get(a, "#888888") for a in agg.index]
    bars = ax.bar(x, agg["mean_score"], color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)
    errs = agg["std_score"].fillna(0)
    mask = errs > 0
    if mask.any():
        ax.errorbar(x[mask], agg["mean_score"].iloc[mask], yerr=errs[mask],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    for bar, v in zip(bars, agg["mean_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errs.max() * 0.05,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, agg["mean_score"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_bar_times(agg: pd.DataFrame, title: str, fname: str):
    """Bar chart of mean execution times with std error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(agg))
    colors = [PALETTE.get(a, "#888888") for a in agg.index]
    bars = ax.bar(x, agg["mean_time"], color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)
    errs = agg["std_time"].fillna(0)
    mask = errs > 0
    if mask.any():
        ax.errorbar(x[mask], agg["mean_time"].iloc[mask], yerr=errs[mask],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    for bar, v in zip(bars, agg["mean_time"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylabel("Mean Execution Time (s)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, agg["mean_time"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def generate_latex(agg: pd.DataFrame) -> str:
    """Generate a LaTeX table for Solomon-100 results.

    Columns: Algorithm | Score | Time | Gap
    Shows mean +/- std for stochastic solvers (5 runs).
    Ground truth: Righini & Salani bestPossible files.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm performance on Solomon-100 dataset. "
        r"Scores and times shown as mean $\pm$ std over " + str(NUM_RUNS) +
        r" runs. Ground truth: Righini \& Salani bestPossible.}",
        r"\label{tab:results_s100}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l rrr}",
        r"\toprule",
        r"\textbf{Algorithm}",
        r"& \textbf{Score} & \textbf{Time (s)} & \textbf{Gap (\%)} \\",
        r"\midrule",
    ]

    for algo in ALGO_ORDER:
        row = agg.loc[[algo]] if algo in agg.index else None
        if row is None:
            continue
        n = int(row["n"].values[0])

        # Score
        s = row["mean_score"].values[0]
        if n > 1:
            sd = row["std_score"].values[0]
            score_str = rf"{s:.1f} $\pm$ {sd:.1f}" if pd.notna(sd) and sd > 0 else f"{s:.1f}"
        else:
            score_str = f"{s:.1f}"

        # Time
        t = row["mean_time"].values[0]
        if n > 1:
            td = row["std_time"].values[0]
            time_str = rf"{t:.2f} $\pm$ {td:.2f}" if pd.notna(td) and td > 0 else f"{t:.2f}"
        else:
            time_str = f"{t:.2f}"

        # Gap
        g = row["mean_gap"].values[0]
        gap_str = f"{g:.2f}" if pd.notna(g) else "--"

        label = algo.replace("_", r"\_")
        lines.append(f"  {label} & {score_str} & {time_str} & {gap_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total = time.perf_counter()

    print("=" * 60)
    print("  BENCHMARK: Solomon-100")
    print(f"  Multi-run: stochastic x{NUM_RUNS}, deterministic x1")
    print(f"  Workers:   {NUM_WORKERS} CPU cores")
    print(f"  CPLEX:     NOT USED (R&S ground truth)")
    print("=" * 60)

    # Load instances
    from benchmarks.runner import _HERE as _RUNNER_HERE
    s100_dir = _RUNNER_HERE / "datasets" / "c_r_rc_100_100"
    instances_100 = load_solomon_group(s100_dir, MAX_PER_GROUP.get("Solomon-100"))
    print(f"  Loaded {len(instances_100)} instances from {s100_dir.name}")

    # Load R&S ground truth
    gt_100 = load_s100_ground_truth()
    print(f"  Ground truth: {len(gt_100)} entries from Righini & Salani bestPossible")

    # Run benchmarks (NO CPLEX)
    print(f"\n  Running {len(SOLOMON_100_VARIANTS)} algorithms on {len(instances_100)} instances ...")
    df_100 = run_group(
        instances_100,
        SOLOMON_100_VARIANTS,
        ground_truths=gt_100,
        group_label="Solomon-100",
    )

    agg_100 = aggregate(df_100)
    print(f"\n  Solomon-100 results ({len(df_100)} total rows):")
    print(agg_100[["n", "mean_score", "std_score", "mean_time", "mean_gap"]].to_string())

    # ==================================================================
    # Generate charts (PNG)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Generating charts -> {OUT}")
    print(f"{'='*60}")

    print("\n  -- Solomon-100 --")
    plot_bar_scores(agg_100, "Solomon-100 -- Average Score (5 instances)",
                    "solomon100_scores.png")
    plot_bar_times(agg_100, "Solomon-100 -- Average Execution Time (5 instances)",
                   "solomon100_times.png")

    # ==================================================================
    # Generate LaTeX table
    # ==================================================================
    print("\n  -- LaTeX table --")
    latex = generate_latex(agg_100)
    tex_path = OUT / "results_s100.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"  Saved: results_s100.tex")

    # Save raw CSV
    df_100.to_csv(OUT / "results_solomon100.csv", index=False)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  All outputs saved to: {OUT}")
    print(f"  Files: {[f.name for f in sorted(OUT.iterdir())]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
