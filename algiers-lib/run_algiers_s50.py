"""
run_algiers_s50.py
==================
Run benchmarks on the Algiers and Solomon-50 datasets.

Workflow:
  Phase 1 -- CPLEX runs with 30 min time limit on both datasets -> ground truth
  Phase 2 -- All heuristic algorithms run:
             - Stochastic solvers: 5 runs each
             - Deterministic solvers (Greedy): 1 run
             Results compared against CPLEX ground truth

Outputs (saved to ./results/):
  - algiers_scores.png        Bar chart: algorithm scores on Algiers
  - algiers_times.png         Bar chart: execution times on Algiers
  - algiers_optimality.png    Box plot: optimality gap on Algiers
  - solomon50_scores.png      Bar chart: average scores on Solomon-50
  - solomon50_times.png       Bar chart: average times on Solomon-50
  - solomon50_optimality.png  Box plot: optimality gap on Solomon-50
  - results_algiers_s50.tex   LaTeX table: mean +- std score, time, gap

Usage:
    python run_algiers_s50.py
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
    load_algiers_problem,
    load_solomon_group,
    load_s100_ground_truth,
    run_group,
    run_cplex_ground_truth,
    NUM_RUNS,
    NUM_WORKERS,
    MAX_PER_GROUP,
)

from benchmarks.solver_registry import (
    ALGIERS_BEST_VARIANTS,
    SOLOMON_50_VARIANTS,
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
    "CPLEX",
]

PALETTE = {
    "Greedy":           "#4C72B0",
    "GRASP":            "#DD8452",
    "SA":               "#55A868",
    "Tabu":             "#C44E52",
    "Genetic-Tailored": "#8172B3",
    "Genetic-Score":    "#A78DC4",
    "CPLEX":            "#937860",
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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(agg))
    colors = [PALETTE.get(a, "#888888") for a in agg.index]
    bars = ax.bar(x, agg["mean_score"], color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)
    # Error bars from std
    errs = agg["std_score"].fillna(0)
    mask = errs > 0
    if mask.any():
        ax.errorbar(x[mask], agg["mean_score"].iloc[mask], yerr=errs[mask],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    # Value labels
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
    fig, ax = plt.subplots(figsize=(8, 4.5))
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


def plot_optimality_boxplot(df: pd.DataFrame, title: str, fname: str):
    """Box plot of optimality gap distribution per solver (excludes CPLEX)."""
    data = df[df["solver"] != "CPLEX"].copy()
    solvers = [s for s in ALGO_ORDER if s != "CPLEX" and s in data["solver"].values]
    if not solvers:
        print(f"  [SKIP] {fname}: no data (CPLEX only)")
        return
    gaps = [data[data["solver"] == s]["optimality_gap_pct"].dropna().values for s in solvers]
    # Filter out empty
    valid = [(s, g) for s, g in zip(solvers, gaps) if len(g) > 0]
    if not valid:
        print(f"  [SKIP] {fname}: no gap data")
        return
    solvers_v, gaps_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(gaps_v, patch_artist=True, labels=solvers_v, zorder=3,
                    boxprops=dict(linewidth=1.2), medianprops=dict(color="black", linewidth=1.5))
    colors = [PALETTE.get(s, "#888888") for s in solvers_v]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def generate_latex(agg_alg: pd.DataFrame, agg_50: pd.DataFrame) -> str:
    """Generate a LaTeX table for Algiers and Solomon-50 results.

    Columns: Algorithm | Algiers (Score, Time, Gap) | Solomon-50 (Score, Time, Gap)
    Shows mean +/- std for stochastic solvers (5 runs).
    """
    import math

    def _score(row):
        if row.empty:
            return "--"
        s = row["mean_score"].values[0]
        n = int(row["n"].values[0])
        if n > 1:
            sd = row["std_score"].values[0]
            if pd.notna(sd) and sd > 0:
                return rf"{s:.1f} $\pm$ {sd:.1f}"
        return f"{s:.1f}"

    def _time(row):
        if row.empty:
            return "--"
        t = row["mean_time"].values[0]
        n = int(row["n"].values[0])
        if n > 1:
            td = row["std_time"].values[0]
            if pd.notna(td) and td > 0:
                return rf"{t:.2f} $\pm$ {td:.2f}"
        return f"{t:.2f}"

    def _gap(row, is_gt=False):
        if is_gt:
            return r"\textbf{0.00}"
        if row.empty:
            return "--"
        g = row["mean_gap"].values[0]
        if pd.notna(g):
            return f"{g:.2f}"
        return "--"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm performance on Algiers and Solomon-50 datasets. "
        r"Scores and times shown as mean $\pm$ std over " + str(NUM_RUNS) +
        r" runs. Ground truth: CPLEX (30 min time limit).}",
        r"\label{tab:results_algiers_s50}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Algiers}}",
        r"& \multicolumn{3}{c}{\textbf{Solomon-50}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"\textbf{Algorithm}",
        r"& \textbf{Score} & \textbf{Time (s)} & \textbf{Gap (\%)}",
        r"& \textbf{Score} & \textbf{Time (s)} & \textbf{Gap (\%)} \\",
        r"\midrule",
    ]

    for algo in ALGO_ORDER:
        ra = agg_alg.loc[[algo]] if algo in agg_alg.index else pd.DataFrame()
        r50 = agg_50.loc[[algo]] if algo in agg_50.index else pd.DataFrame()
        is_cplex = (algo == "CPLEX")
        label = r"\textit{CPLEX} (GT)" if is_cplex else algo.replace("_", r"\_")

        if ra.empty and r50.empty:
            continue

        lines.append(
            rf"{label}"
            rf" & {_score(ra)} & {_time(ra)} & {_gap(ra, is_cplex)}"
            rf" & {_score(r50)} & {_time(r50)} & {_gap(r50, is_cplex)} \\"
        )

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
    print("  BENCHMARK: Algiers + Solomon-50")
    print(f"  Multi-run: stochastic x{NUM_RUNS}, deterministic x1")
    print(f"  Workers:   {NUM_WORKERS} CPU cores")
    print(f"  CPLEX:     30 min time limit (ground truth)")
    print("=" * 60)

    # ==================================================================
    # 1. ALGIERS
    # ==================================================================
    print("\n--- ALGIERS ---")
    name, problem = load_algiers_problem()
    print(f"  Loaded: {len(problem.landmarks)} landmarks, budget={problem.time_budget} min")

    # Identify CPLEX variant
    cplex_var = next((v for v in ALGIERS_BEST_VARIANTS if "CPLEX" in v["name"]), None)

    if cplex_var:
        # Phase 1: CPLEX ground truth
        live_gt_alg, df_cplex_alg = run_cplex_ground_truth(
            [(name, problem)], cplex_var, group_label="Algiers"
        )
        # Phase 2: Heuristics only
        heuristics = [v for v in ALGIERS_BEST_VARIANTS if "CPLEX" not in v["name"]]
        df_alg = run_group(
            [(name, problem)], heuristics,
            ground_truths=live_gt_alg, group_label="Algiers",
        )
        df_alg = pd.concat([df_cplex_alg, df_alg], ignore_index=True)
    else:
        print("  [WARN] CPLEX not available -- no ground truth for Algiers")
        df_alg = run_group(
            [(name, problem)], ALGIERS_BEST_VARIANTS,
            group_label="Algiers",
        )

    agg_alg = aggregate(df_alg)
    print(f"\n  Algiers results ({len(df_alg)} total rows):")
    print(agg_alg[["n", "mean_score", "std_score", "mean_time", "mean_gap"]].to_string())

    # ==================================================================
    # 2. SOLOMON-50
    # ==================================================================
    print("\n--- SOLOMON-50 ---")
    from benchmarks.runner import _HERE as _RUNNER_HERE
    s50_dir = _RUNNER_HERE / "datasets" / "c_r_rc_100_50"
    instances_50 = load_solomon_group(s50_dir, MAX_PER_GROUP.get("Solomon-50"))
    print(f"  Loaded {len(instances_50)} instances from {s50_dir.name}")

    cplex_var_50 = next((v for v in SOLOMON_50_VARIANTS if "CPLEX" in v["name"]), None)

    if cplex_var_50:
        # Phase 1: CPLEX ground truth
        live_gt_50, df_cplex_50 = run_cplex_ground_truth(
            instances_50, cplex_var_50, group_label="Solomon-50"
        )
        # Phase 2: Heuristics only
        heuristics_50 = [v for v in SOLOMON_50_VARIANTS if "CPLEX" not in v["name"]]
        df_50 = run_group(
            instances_50, heuristics_50,
            ground_truths=live_gt_50, group_label="Solomon-50",
        )
        df_50 = pd.concat([df_cplex_50, df_50], ignore_index=True)
    else:
        print("  [WARN] CPLEX not available -- no ground truth for Solomon-50")
        df_50 = run_group(
            instances_50, SOLOMON_50_VARIANTS,
            group_label="Solomon-50",
        )

    agg_50 = aggregate(df_50)
    print(f"\n  Solomon-50 results ({len(df_50)} total rows):")
    print(agg_50[["n", "mean_score", "std_score", "mean_time", "mean_gap"]].to_string())

    # ==================================================================
    # 3. GENERATE CHARTS (PNG)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Generating charts -> {OUT}")
    print(f"{'='*60}")

    # -- Algiers charts --
    print("\n  -- Algiers --")
    plot_bar_scores(agg_alg, "Algiers -- Algorithm Scores", "algiers_scores.png")
    plot_bar_times(agg_alg, "Algiers -- Execution Times", "algiers_times.png")
    plot_optimality_boxplot(df_alg,
                            "Algiers -- Optimality Gap vs CPLEX Ground Truth",
                            "algiers_optimality.png")

    # -- Solomon-50 charts --
    print("\n  -- Solomon-50 --")
    plot_bar_scores(agg_50, "Solomon-50 -- Average Score (5 instances)",
                    "solomon50_scores.png")
    plot_bar_times(agg_50, "Solomon-50 -- Average Execution Time (5 instances)",
                   "solomon50_times.png")
    plot_optimality_boxplot(df_50,
                            "Solomon-50 -- Optimality Gap vs CPLEX Ground Truth",
                            "solomon50_optimality.png")

    # ==================================================================
    # 4. GENERATE LATEX TABLE
    # ==================================================================
    print("\n  -- LaTeX table --")
    latex = generate_latex(agg_alg, agg_50)
    tex_path = OUT / "results_algiers_s50.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"  Saved: results_algiers_s50.tex")

    # -- Save raw CSVs for inspection --
    df_alg.to_csv(OUT / "results_algiers.csv", index=False)
    df_50.to_csv(OUT / "results_solomon50.csv", index=False)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  All outputs saved to: {OUT}")
    print(f"  Files: {[f.name for f in sorted(OUT.iterdir())]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
