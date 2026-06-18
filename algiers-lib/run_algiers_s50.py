"""
run_algiers_s50.py
==================
Run benchmarks on the Algiers and Solomon-50 datasets.

Algiers workflow:
  - All algorithms run together (including CPLEX as a regular solver)
  - No ground truth for Algiers -- no optimality gap calculation
  - Stochastic solvers: 5 runs each
  - Deterministic solvers (Greedy, CPLEX, Tabu): 1 run
  - Genetic-Score has a 200s time limit (post-execution check).
    If it exceeds 200s, the score is kept but marked INVALID.

Solomon-50 workflow:
  - Phase 1: Pre-screen ALL instances with CPLEX (5 min, parallel)
             to find ones CPLEX can solve within 5 minutes
  - Phase 2: CPLEX ground truth on easy instances (600s time limit)
  - Phase 3: All heuristic algorithms run against CPLEX ground truth
  - Optimality gap computed ONLY on instances where CPLEX terminated

Outputs (saved to ./results/):
  - algiers_scores.png        Bar chart: algorithm scores on Algiers
  - algiers_times.png         Bar chart: execution times on Algiers
  - solomon50_scores.png      Bar chart: average scores on Solomon-50
  - solomon50_times.png       Bar chart: average times on Solomon-50
  - solomon50_optimality.png  Box plot: optimality gap on Solomon-50
  - results_algiers_s50.tex   LaTeX table: mean score, std dev, gap, time

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
    run_group,
    run_cplex_ground_truth,
    screen_easy_instances,
    NUM_RUNS,
    NUM_WORKERS,
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
# Algiers has 10 algorithms
ALGIERS_ALGO_ORDER = [
    "Greedy-Ratio", "Greedy-Score", "Greedy-Random",
    "GRASP",
    "SA-Boltzmann", "SA-Cauchy", "SA",
    "Tabu", "Tabu-Random",
    "Genetic-Tailored", "Genetic-Score",
    "CPLEX",
]

ALGIERS_PALETTE = {
    "Greedy-Ratio":     "#4C72B0",
    "Greedy-Score":     "#2E5BA8",
    "Greedy-Random":    "#1E42A0",
    "GRASP":            "#DD8452",
    "SA-Boltzmann":     "#2ca02c",
    "SA-Cauchy":        "#55A868",
    "SA":               "#98df8a",
    "Tabu":             "#C44E52",
    "Tabu-Random":      "#e377c2",
    "Genetic-Tailored": "#8172B3",
    "Genetic-Score":    "#A78DC4",
    "CPLEX":            "#937860",
}

# S50 has 11 algorithms: Greedy (3 variants), GRASP, SA-Boltzmann, SA-Cauchy, Tabu (2 variants), Genetic (2 variants), CPLEX
S50_ALGO_ORDER = [
    "Greedy-Ratio", "Greedy-Score", "Greedy-Random",
    "GRASP",
    "SA-Boltzmann", "SA-Cauchy",
    "Tabu", "Tabu-Random",
    "Genetic-Tailored", "Genetic-Score",
    "CPLEX",
]

S50_PALETTE = {
    "Greedy-Ratio":     "#4C72B0",
    "Greedy-Score":     "#2E5BA8",
    "Greedy-Random":    "#1E42A0",
    "GRASP":            "#DD8452",
    "SA-Boltzmann":     "#2ca02c",
    "SA-Cauchy":        "#55A868",
    "Tabu":             "#C44E52",
    "Tabu-Random":      "#e377c2",
    "Genetic-Tailored": "#8172B3",
    "Genetic-Score":    "#A78DC4",
    "CPLEX":            "#937860",
}

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})


# ---------------------------------------------------------------------------
# Timeout analysis helpers
# ---------------------------------------------------------------------------
def _compute_timeout_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-solver timeout statistics from raw result DataFrame.

    Returns dict mapping solver name -> {
        "total_runs": int,
        "timed_out_runs": int,
        "timeout_pct": float,
        "mean_time": float,
        "any_timed_out": bool,
    }
    """
    stats = {}
    if df.empty or "timed_out" not in df.columns:
        return stats
    for solver, group in df.groupby("solver"):
        total = len(group)
        timed_out = int(group["timed_out"].sum())
        stats[solver] = {
            "total_runs": total,
            "timed_out_runs": timed_out,
            "timeout_pct": 100.0 * timed_out / total if total > 0 else 0.0,
            "mean_time": group["time_s"].mean(),
            "any_timed_out": timed_out > 0,
        }
    return stats


def _solver_timed_out_pct(df: pd.DataFrame, solver: str) -> float:
    """Return the percentage of runs for a solver that timed out."""
    rows = df[df["solver"] == solver]
    if rows.empty or "timed_out" not in rows.columns:
        return 0.0
    return 100.0 * rows["timed_out"].sum() / len(rows)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, algo_order: list[str]) -> pd.DataFrame:
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
    # Reindex to canonical order (only solvers that appear in data)
    ordered = [a for a in algo_order if a in agg.index]
    return agg.loc[ordered]


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_bar_scores(agg: pd.DataFrame, title: str, fname: str,
                    palette: dict[str, str],
                    timed_out_solvers: set[str] | None = None):
    """Bar chart of mean algorithm scores with std error bars.

    If a solver is in timed_out_solvers, its bar is given a red dashed border
    to indicate the result may be unreliable (INVALID).
    """
    fig, ax = plt.subplots(figsize=(max(8, len(agg) * 1.1), 4.5))
    x = np.arange(len(agg))
    colors = [palette.get(a, "#888888") for a in agg.index]
    bars = ax.bar(x, agg["mean_score"], color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)
    # Error bars from std -- use numpy indexing (avoids iloc/loc issues)
    errs = agg["std_score"].fillna(0)
    mask = errs > 0
    if mask.any():
        ax.errorbar(x[mask.values], agg["mean_score"].values[mask.values],
                    yerr=errs.values[mask.values],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    # Value labels + INVALID marker for timed-out solvers
    has_invalid = False
    for bar, v, name in zip(bars, agg["mean_score"], agg.index):
        suffix = "*" if (timed_out_solvers and name in timed_out_solvers) else ""
        label = f"{v:.1f}{suffix}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + agg["mean_score"].max() * 0.03,
                label, ha="center", va="bottom", fontsize=7.5)
        if timed_out_solvers and name in timed_out_solvers:
            bar.set_edgecolor("red")
            bar.set_linewidth(2.0)
            bar.set_linestyle("--")
            has_invalid = True
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=25, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, agg["mean_score"].max() * 1.3)
    if has_invalid:
        ax.text(0.99, 0.01, "* = exceeded time limit (INVALID)",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="red", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_bar_times(agg: pd.DataFrame, title: str, fname: str,
                   palette: dict[str, str]):
    """Bar chart of mean execution times with std error bars."""
    fig, ax = plt.subplots(figsize=(max(8, len(agg) * 1.1), 4.5))
    x = np.arange(len(agg))
    colors = [palette.get(a, "#888888") for a in agg.index]
    bars = ax.bar(x, agg["mean_time"], color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)
    errs = agg["std_time"].fillna(0)
    mask = errs > 0
    if mask.any():
        ax.errorbar(x[mask.values], agg["mean_time"].values[mask.values],
                    yerr=errs.values[mask.values],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    for bar, v in zip(bars, agg["mean_time"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=25, ha="right")
    ax.set_ylabel("Mean Execution Time (s)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, agg["mean_time"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_optimality_boxplot(df: pd.DataFrame, title: str, fname: str,
                            algo_order: list[str], palette: dict[str, str],
                            exclude_solver: str = "CPLEX"):
    """Box plot of optimality gap distribution per solver.

    Only includes data rows where optimality_gap_pct is not NaN,
    i.e. instances where CPLEX actually terminated (ground truth exists).
    Excludes CPLEX itself from the plot.
    """
    data = df[
        (df["solver"] != exclude_solver) &
        (df["optimality_gap_pct"].notna())
    ].copy()
    solvers = [s for s in algo_order if s != exclude_solver and s in data["solver"].values]
    if not solvers:
        print(f"  [SKIP] {fname}: no gap data (CPLEX did not terminate on any instance)")
        return
    gaps = [data[data["solver"] == s]["optimality_gap_pct"].values for s in solvers]
    valid = [(s, g) for s, g in zip(solvers, gaps) if len(g) > 0]
    if not valid:
        print(f"  [SKIP] {fname}: no gap data")
        return
    solvers_v, gaps_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(7, len(solvers_v) * 1.1), 4.5))
    bp = ax.boxplot(gaps_v, patch_artist=True, labels=solvers_v, zorder=3,
                    boxprops=dict(linewidth=1.2), medianprops=dict(color="black", linewidth=1.5))
    colors = [palette.get(s, "#888888") for s in solvers_v]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(solvers_v, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def generate_latex(
    agg_alg: pd.DataFrame,
    agg_50: pd.DataFrame,
    df_alg_raw: pd.DataFrame,
    df_50_raw: pd.DataFrame,
    alg_order: list[str],
    s50_order: list[str],
) -> str:
    """Generate a LaTeX table for Algiers and Solomon-50 results.

    4 data columns per dataset:
        Mean Score | Std Dev | Avg Optimality Gap (%) | Running Time (s)

    - Algiers: gap column is "--" (no ground truth)
    - Solomon-50: gap = avg over instances where CPLEX terminated
    - If a solver's runs exceeded its time limit (>200s for Genetic-Score),
      the score is marked with * and a footnote indicates INVALID results.

    Stochastic solvers show mean +/- std; deterministic show single value.
    """
    # Compute timeout stats for footnote
    timeout_alg = _compute_timeout_stats(df_alg_raw)
    timeout_50 = _compute_timeout_stats(df_50_raw)

    def _mean_score(row, timed_out_info):
        if row.empty:
            return "--"
        s = row["mean_score"].values[0]
        marker = ""
        if timed_out_info and timed_out_info.get("any_timed_out", False):
            marker = "$^{*}$"
        return f"{s:.1f}{marker}"

    def _std_score(row):
        if row.empty:
            return "--"
        n = int(row["n"].values[0])
        if n <= 1:
            return "--"  # deterministic
        sd = row["std_score"].values[0]
        if pd.notna(sd) and sd > 0:
            return f"{sd:.1f}"
        return "--"

    def _gap_alg(row):
        """Algiers: no ground truth -> always '--'."""
        return "--"

    def _gap_s50(row, timed_out_info):
        """Solomon-50: avg optimality gap. If timed out, mark as '--' (invalid)."""
        if row.empty:
            return "--"
        if timed_out_info and timed_out_info.get("any_timed_out", False):
            return "--"  # gap is meaningless for timed-out results
        g = row["mean_gap"].values[0]
        if pd.notna(g):
            return f"{g:.2f}"
        return "--"

    def _time(row):
        if row.empty:
            return "--"
        t = row["mean_time"].values[0]
        return f"{t:.2f}"

    # Build combined list of all algorithms
    all_algos = list(dict.fromkeys(
        list(agg_alg.index) + [a for a in agg_50.index if a not in agg_alg.index]
    ))

    n_instances_50 = int(agg_50["n"].max()) if not agg_50.empty else 0

    # Check if any solver has timed-out runs (for footnote)
    any_timeout = False
    for algo in all_algos:
        alg_to = timeout_alg.get(algo, {})
        s50_to = timeout_50.get(algo, {})
        if alg_to.get("any_timed_out") or s50_to.get("any_timed_out"):
            any_timeout = True
            break

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm performance on Algiers and Solomon-50 datasets. "
        r"Stochastic solvers: " + str(NUM_RUNS) + r" runs. "
        r"Algiers: no ground truth (CPLEX as regular solver). "
        r"Solomon-50: CPLEX ground truth on "
        + str(n_instances_50) + r" instances where CPLEX terminated "
        r"($\leq$5 min)." + (
            r" $^{*}$\textit{INVALID}: solver exceeded time limit (200\,s)."
            if any_timeout else ""
        ) + r"}",
        r"\label{tab:results_algiers_s50}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l rrrr rrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Algiers}}",
        r"& \multicolumn{4}{c}{\textbf{Solomon-50}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"\textbf{Algorithm}",
        r"& \textbf{Score} & \textbf{Std} & \textbf{Gap (\%)} & \textbf{Time (s)}",
        r"& \textbf{Score} & \textbf{Std} & \textbf{Gap (\%)} & \textbf{Time (s)} \\",
        r"\midrule",
    ]

    for algo in all_algos:
        ra = agg_alg.loc[[algo]] if algo in agg_alg.index else pd.DataFrame()
        r50 = agg_50.loc[[algo]] if algo in agg_50.index else pd.DataFrame()

        if ra.empty and r50.empty:
            continue

        label = algo.replace("_", r"\_")
        to_a = timeout_alg.get(algo, {})
        to_5 = timeout_50.get(algo, {})

        lines.append(
            rf"  {label}"
            rf" & {_mean_score(ra, to_a)} & {_std_score(ra)} & {_gap_alg(ra)} & {_time(ra)}"
            rf" & {_mean_score(r50, to_5)} & {_std_score(r50)} & {_gap_s50(r50, to_5)} & {_time(r50)} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistics and Insights
# ---------------------------------------------------------------------------
def print_statistics(
    df_alg: pd.DataFrame,
    df_50: pd.DataFrame,
    agg_alg: pd.DataFrame,
    agg_50: pd.DataFrame,
    ground_truths_50: dict[str, float],
):
    """Print comprehensive comparison statistics and insights."""

    print(f"\n{'='*60}")
    print(f"  STATISTICS & INSIGHTS")
    print(f"{'='*60}")

    # -----------------------------------------------------------------
    # Helper: check if a DataFrame has timed_out column
    # -----------------------------------------------------------------
    def _has_timed_out(df):
        return "timed_out" in df.columns

    # =================================================================
    # SECTION A: TIMEOUT ANALYSIS (both datasets)
    # =================================================================
    print(f"\n  {'='*56}")
    print(f"  TIMEOUT ANALYSIS (time limit = 200s for Genetic-Score)")
    print(f"  {'='*56}\n")

    any_timeout_found = False

    for label, df in [("Algiers", df_alg), ("Solomon-50", df_50)]:
        if df.empty or not _has_timed_out(df):
            print(f"  {label}: No timeout data available.")
            continue

        timed_out_rows = df[df["timed_out"] == True]
        if timed_out_rows.empty:
            print(f"  {label}: All solvers finished within their time limits.")
            continue

        any_timeout_found = True
        print(f"  {label} -- Timed-out runs ({len(timed_out_rows)} total):")
        for solver, group in timed_out_rows.groupby("solver"):
            total = len(df[df["solver"] == solver])
            pct = 100.0 * len(group) / total
            avg_t = group["time_s"].mean()
            avg_s = group["score"].mean()
            print(f"    {solver:<20}: {len(group)}/{total} runs timed out ({pct:.0f}%)"
                  f"  avg_time={avg_t:.1f}s  avg_score={avg_s:.1f}"
                  f"  --> ALL INVALID")

    if not any_timeout_found:
        print(f"  No solver exceeded its time limit. All results are valid.")

    # =================================================================
    # SECTION B: GENETIC-SCORE INVALIDITY ANALYSIS
    # =================================================================
    print(f"\n  {'='*56}")
    print(f"  GENETIC-SCORE INVALIDITY ANALYSIS")
    print(f"  {'='*56}\n")

    # --- Algiers ---
    gs_alg = df_alg[df_alg["solver"] == "Genetic-Score"]
    if not gs_alg.empty:
        total_gs_alg = len(gs_alg)
        timed_out_alg = int(gs_alg["timed_out"].sum()) if _has_timed_out(df_alg) else 0
        pct_timeout_alg = 100.0 * timed_out_alg / total_gs_alg

        print(f"  Algiers -- Genetic-Score:")
        print(f"    Total runs: {total_gs_alg}")
        print(f"    Timed out (>200s): {timed_out_alg}/{total_gs_alg} = {pct_timeout_alg:.0f}%")

        if timed_out_alg > 0:
            valid_gs = gs_alg[gs_alg["timed_out"] == False]
            invalid_gs = gs_alg[gs_alg["timed_out"] == True]
            print(f"    Valid runs avg score:  {valid_gs['score'].mean():.1f}  (avg time: {valid_gs['time_s'].mean():.1f}s)")
            print(f"    Invalid runs avg score: {invalid_gs['score'].mean():.1f}  (avg time: {invalid_gs['time_s'].mean():.1f}s)")
            print(f"    >>> {pct_timeout_alg:.0f}% of Genetic-Score runs on Algiers are INVALID <<<")
            print(f"    >>> These tours violated the 200s time budget <<<")

        # Also check via score comparison (score > median of other solvers)
        other_scores = df_alg[df_alg["solver"] != "Genetic-Score"]["score"]
        if not other_scores.empty:
            median_other = other_scores.median()
            suspicious = (gs_alg["score"] > median_other * 1.5).sum()
            pct_suspicious = 100.0 * suspicious / total_gs_alg
            print(f"    Score-based check (>{median_other*1.5:.0f} = 1.5x median of others):")
            print(f"      Suspicious runs: {suspicious}/{total_gs_alg} = {pct_suspicious:.0f}%")

    # --- Solomon-50 ---
    gs_50 = df_50[df_50["solver"] == "Genetic-Score"]
    if not gs_50.empty:
        total_gs_50 = len(gs_50)
        timed_out_50 = int(gs_50["timed_out"].sum()) if _has_timed_out(df_50) else 0
        pct_timeout_50 = 100.0 * timed_out_50 / total_gs_50

        print(f"\n  Solomon-50 -- Genetic-Score:")
        print(f"    Total runs: {total_gs_50}")
        print(f"    Timed out (>200s): {timed_out_50}/{total_gs_50} = {pct_timeout_50:.0f}%")

        if timed_out_50 > 0:
            valid_gs = gs_50[gs_50["timed_out"] == False]
            invalid_gs = gs_50[gs_50["timed_out"] == True]
            print(f"    Valid runs avg score:  {valid_gs['score'].mean():.1f}  (avg time: {valid_gs['time_s'].mean():.1f}s)")
            print(f"    Invalid runs avg score: {invalid_gs['score'].mean():.1f}  (avg time: {invalid_gs['time_s'].mean():.1f}s)")
            print(f"    >>> {pct_timeout_50:.0f}% of Genetic-Score runs on Solomon-50 are INVALID <<<")

        # Check against CPLEX ground truth
        if ground_truths_50:
            invalid_vs_cplex = 0
            for _, row in gs_50.iterrows():
                gt = ground_truths_50.get(row["instance"])
                if gt and row["score"] > gt * 1.01:
                    invalid_vs_cplex += 1
            pct_cplex = 100.0 * invalid_vs_cplex / total_gs_50
            avg_gs = gs_50["score"].mean()
            avg_gt = np.mean(list(ground_truths_50.values()))
            print(f"    Score > CPLEX optimal: {invalid_vs_cplex}/{total_gs_50} = {pct_cplex:.0f}%")
            print(f"    Genetic-Score avg: {avg_gs:.1f}  vs  CPLEX avg: {avg_gt:.1f}"
                  f"  (ratio: {avg_gs/avg_gt:.2f}x)")

            if pct_timeout_50 >= 80 or pct_cplex >= 80:
                print(f"\n    >>> CONCLUSION: ScoreFitnessFunction produces tours that are")
                print(f"        unreliable {max(pct_timeout_50, pct_cplex):.0f}% of the time. <<<")
                print(f"        The fitness function does NOT enforce feasibility,")
                print(f"        leading to scores that far exceed the known optimum.")

            # Per-instance breakdown
            print(f"\n    Per-instance breakdown (Solomon-50):")
            for inst in sorted(ground_truths_50.keys()):
                gt = ground_truths_50[inst]
                gs_inst = gs_50[gs_50["instance"] == inst]
                if not gs_inst.empty:
                    gs_mean = gs_inst["score"].mean()
                    gs_time = gs_inst["time_s"].mean()
                    ratio = gs_mean / gt if gt > 0 else float("inf")
                    timed = gs_inst["timed_out"].sum() if _has_timed_out(df_50) else 0
                    status = "INVALID" if ratio > 1.01 else "OK"
                    timeout_flag = f" [TIMEOUT {timed}/{len(gs_inst)}]" if timed > 0 else ""
                    print(f"      {inst:<10} CPLEX={gt:>6.0f}  GS={gs_mean:>8.1f}"
                          f"  ratio={ratio:.2f}x  avg_t={gs_time:.0f}s"
                          f"  [{status}]{timeout_flag}")

    # =================================================================
    # SECTION C: ALGIERS INSIGHTS (no ground truth)
    # =================================================================
    print(f"\n  --- Algiers Solver Comparison (excluding Genetic-Score) ---\n")

    if not agg_alg.empty:
        rank_alg = agg_alg.drop(index="Genetic-Score", errors="ignore")

        # Best solver by mean score
        best_name = rank_alg["mean_score"].idxmax()
        best_score = rank_alg.loc[best_name, "mean_score"]
        print(f"  Best solver (mean score): {best_name} = {best_score:.1f}")

        # Best solver by mean time among top-3 scorers
        top3 = rank_alg.nlargest(3, "mean_score")
        fastest_of_top = top3["mean_time"].idxmin()
        fastest_time = top3.loc[fastest_of_top, "mean_time"]
        print(f"  Fastest among top-3 scorers: {fastest_of_top} = {fastest_time:.2f}s")

        # Worst solver
        worst_name = rank_alg["mean_score"].idxmin()
        worst_score = rank_alg.loc[worst_name, "mean_score"]
        print(f"  Worst solver (mean score): {worst_name} = {worst_score:.1f}")
        print(f"  Score range: {worst_score:.1f} (min) to {best_score:.1f} (max)")

        # Speed ranking
        print(f"\n  Speed ranking (fastest first):")
        speed_rank = rank_alg.sort_values("mean_time")
        for i, (name, row) in enumerate(speed_rank.iterrows(), 1):
            marker = " ***" if name == best_name else ""
            print(f"    {i}. {name:<20} {row['mean_time']:>8.2f}s   (score={row['mean_score']:.1f}){marker}")

        # Deterministic vs Stochastic
        det_solvers = ["Greedy", "Tabu", "CPLEX"]
        det_in_data = [s for s in det_solvers if s in rank_alg.index]
        sto_in_data = [s for s in rank_alg.index if s not in det_solvers]
        if det_in_data and sto_in_data:
            det_best = max(det_in_data, key=lambda s: rank_alg.loc[s, "mean_score"])
            sto_best_name = rank_alg.loc[sto_in_data, "mean_score"].idxmax()
            print(f"\n  Deterministic vs Stochastic:")
            print(f"    Best deterministic: {det_best} = {rank_alg.loc[det_best, 'mean_score']:.1f}")
            print(f"    Best stochastic:    {sto_best_name} = {rank_alg.loc[sto_best_name, 'mean_score']:.1f}")

    # =================================================================
    # SECTION D: SOLOMON-50 INSIGHTS (CPLEX ground truth)
    # =================================================================
    print(f"\n  --- Solomon-50 Solver Comparison ({len(ground_truths_50)} instances "
          f"with ground truth) ---\n")

    if not df_50.empty and ground_truths_50:
        # Exclude Genetic-Score (already analyzed above) and CPLEX (ground truth)
        heuristics = [s for s in agg_50.index
                      if s not in ("CPLEX", "Genetic-Score")]
        if heuristics:
            rank_50 = agg_50.loc[heuristics].sort_values("mean_gap")
            print(f"  Solver ranking by mean optimality gap (lower is better):")
            for i, (name, row) in enumerate(rank_50.iterrows(), 1):
                gap_str = f"{row['mean_gap']:.2f}%" if pd.notna(row["mean_gap"]) else "N/A"
                print(f"    {i}. {name:<20} gap={gap_str:>8}"
                      f"  score={row['mean_score']:.1f}"
                      f"  time={row['mean_time']:.2f}s")

            # Best heuristic
            best_h = rank_50.index[0]
            best_gap = rank_50.iloc[0]["mean_gap"]
            best_score = rank_50.iloc[0]["mean_score"]
            print(f"\n  Best heuristic: {best_h} (gap={best_gap:.2f}%, score={best_score:.1f})")

            # Optimal-finding rate
            print(f"\n  Optimal-finding rate (runs where score == CPLEX optimal):")
            for solver_name in heuristics:
                s_rows = df_50[(df_50["solver"] == solver_name)]
                optimal_count = 0
                total_count = len(s_rows)
                for _, row in s_rows.iterrows():
                    gt = ground_truths_50.get(row["instance"])
                    if gt and abs(row["score"] - gt) < 0.01:
                        optimal_count += 1
                pct = 100 * optimal_count / total_count if total_count > 0 else 0
                print(f"    {solver_name:<20} {optimal_count:>3}/{total_count:>3} = {pct:>5.1f}%")

        # Speed comparison
        print(f"\n  Execution time ranking (fastest first):")
        time_rank = agg_50.sort_values("mean_time")
        for i, (name, row) in enumerate(time_rank.iterrows(), 1):
            to_flag = " [TIMEOUT]" if _solver_timed_out_pct(df_50, name) > 0 else ""
            print(f"    {i}. {name:<20} {row['mean_time']:>8.2f}s{to_flag}")

        # Time-efficiency
        print(f"\n  Score-per-second efficiency (higher is better):")
        eff_data = []
        for name, row in agg_50.iterrows():
            t = row["mean_time"]
            if t and t > 0.001 and pd.notna(t):
                eff = row["mean_score"] / t
                eff_data.append((name, eff, row["mean_score"], t))
        eff_data.sort(key=lambda x: -x[1])
        for i, (name, eff, sc, t) in enumerate(eff_data, 1):
            print(f"    {i}. {name:<20} {eff:>8.1f} pts/s"
                  f"  (score={sc:.1f}, time={t:.2f}s)")

    # =================================================================
    # SECTION E: CROSS-DATASET COMPARISON
    # =================================================================
    print(f"\n  --- Cross-Dataset Comparison ---\n")

    if not agg_alg.empty and not agg_50.empty:
        common = [s for s in S50_ALGO_ORDER
                  if s in agg_alg.index and s in agg_50.index
                  and s not in ("Genetic-Score",)]

        if common:
            print(f"  Solver consistency across datasets (excluding Genetic-Score):")
            print(f"  {'Solver':<20} {'Algiers Rank':>13} {'S50 Rank':>10} {'Consistent?':>12}")
            print(f"  {'-'*20} {'-'*13} {'-'*10} {'-'*12}")

            alg_ranks = agg_alg.loc[common, "mean_score"].rank(ascending=False).astype(int)
            s50_ranks = agg_50.loc[common, "mean_score"].rank(ascending=False).astype(int)

            for solver in common:
                ar = alg_ranks.get(solver, "--")
                sr = s50_ranks.get(solver, "--")
                consistent = "Yes" if isinstance(ar, int) and isinstance(sr, int) and abs(ar - sr) <= 2 else "No"
                print(f"  {solver:<20} {str(ar):>13} {str(sr):>10} {consistent:>12}")

        total_alg_time = df_alg["time_s"].sum() if not df_alg.empty else 0
        total_50_time = df_50["time_s"].sum() if not df_50.empty else 0
        print(f"\n  Total computation time:")
        print(f"    Algiers:    {total_alg_time:>8.1f}s ({total_alg_time/60:.1f} min)")
        print(f"    Solomon-50: {total_50_time:>8.1f}s ({total_50_time/60:.1f} min)")
        print(f"    Combined:   {total_alg_time + total_50_time:>8.1f}s "
              f"({(total_alg_time + total_50_time)/60:.1f} min)")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total = time.perf_counter()

    print("=" * 60)
    print("  BENCHMARK: Algiers + Solomon-50")
    print(f"  Multi-run: stochastic x{NUM_RUNS}, deterministic x1")
    print(f"  Workers:   {NUM_WORKERS} CPU cores")
    print("  Algiers:   CPLEX as regular solver (no ground truth)")
    print("  Solomon-50: CPLEX ground truth (parallel screening, 5 min limit)")
    print("=" * 60)

    # ==================================================================
    # 1. ALGIERS -- all algorithms together, NO ground truth
    # ==================================================================
    print("\n--- ALGIERS ---")
    name, problem = load_algiers_problem()
    print(f"  Loaded: {len(problem.landmarks)} landmarks, budget={problem.time_budget} min")
    print(f"  Running {len(ALGIERS_BEST_VARIANTS)} algorithms (no ground truth) ...")

    df_alg = run_group(
        [(name, problem)], ALGIERS_BEST_VARIANTS,
        ground_truths={},  # empty = no gap calculation
        group_label="Algiers",
    )

    agg_alg = aggregate(df_alg, ALGIERS_ALGO_ORDER)
    print(f"\n  Algiers results ({len(df_alg)} total rows):")

    # Print timeout info alongside the aggregate table
    for solver_name in agg_alg.index:
        n = int(agg_alg.loc[solver_name, "n"])
        ms = agg_alg.loc[solver_name, "mean_score"]
        ss = agg_alg.loc[solver_name, "std_score"]
        mt = agg_alg.loc[solver_name, "mean_time"]
        to_pct = _solver_timed_out_pct(df_alg, solver_name)
        to_flag = f"  TIMEOUT={to_pct:.0f}%" if to_pct > 0 else ""
        ss_str = f"{ss:.1f}" if pd.notna(ss) and ss > 0 and n > 1 else "--"
        print(f"    {solver_name:<20} n={n:<3} score={ms:>7.1f} +/- {ss_str:>5}  time={mt:>8.2f}s{to_flag}")

    # ==================================================================
    # 2. SOLOMON-50 (parallel screening + CPLEX ground truth)
    # ==================================================================
    print("\n--- SOLOMON-50 ---")
    from benchmarks.runner import _HERE as _RUNNER_HERE
    s50_dir = _RUNNER_HERE / "datasets" / "c_r_rc_100_50"

    all_candidates_50 = load_solomon_group(s50_dir, max_instances=None)
    print(f"  Found {len(all_candidates_50)} total S50 instances in {s50_dir.name}")

    instances_50 = screen_easy_instances(
        all_candidates_50,
        screen_timeout=300,   # 5 minutes per instance for screening
        max_keep=5,
        num_workers=NUM_WORKERS,
    )

    ground_truths_50: dict[str, float] = {}

    if not instances_50:
        print("  [ERROR] No solvable S50 instances found! Cannot benchmark Solomon-50.")
        df_50 = pd.DataFrame()
        agg_50 = pd.DataFrame()
    else:
        print(f"  Proceeding with {len(instances_50)} solvable instances: "
              f"{[n for n, _ in instances_50]}")

        cplex_var_50 = next((v for v in SOLOMON_50_VARIANTS if "CPLEX" in v["name"]), None)

        if cplex_var_50:
            ground_truths_50, df_cplex_50 = run_cplex_ground_truth(
                instances_50, cplex_var_50, group_label="Solomon-50"
            )
            heuristics_50 = [v for v in SOLOMON_50_VARIANTS if "CPLEX" not in v["name"]]
            df_50 = run_group(
                instances_50, heuristics_50,
                ground_truths=ground_truths_50, group_label="Solomon-50",
            )
            df_50 = pd.concat([df_cplex_50, df_50], ignore_index=True)
        else:
            print("  [WARN] CPLEX not available -- no ground truth for Solomon-50")
            df_50 = run_group(
                instances_50, SOLOMON_50_VARIANTS,
                ground_truths={}, group_label="Solomon-50",
            )

    # Safely aggregate only if df_50 has data and "solver" column
    if not df_50.empty and "solver" in df_50.columns:
        agg_50 = aggregate(df_50, S50_ALGO_ORDER)
    else:
        agg_50 = pd.DataFrame()
    print(f"\n  Solomon-50 results ({len(df_50)} total rows):")

    for solver_name in agg_50.index:
        n = int(agg_50.loc[solver_name, "n"])
        ms = agg_50.loc[solver_name, "mean_score"]
        ss = agg_50.loc[solver_name, "std_score"]
        mt = agg_50.loc[solver_name, "mean_time"]
        mg = agg_50.loc[solver_name, "mean_gap"]
        to_pct = _solver_timed_out_pct(df_50, solver_name)
        to_flag = f"  TIMEOUT={to_pct:.0f}%" if to_pct > 0 else ""
        ss_str = f"{ss:.1f}" if pd.notna(ss) and ss > 0 and n > 1 else "--"
        gap_str = f"{mg:.2f}%" if pd.notna(mg) else "--"
        print(f"    {solver_name:<20} n={n:<3} score={ms:>7.1f} +/- {ss_str:>5}"
              f"  gap={gap_str:>8}  time={mt:>8.2f}s{to_flag}")

    # ==================================================================
    # 3. GENERATE CHARTS (PNG)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Generating charts -> {OUT}")
    print(f"{'='*60}")

    # Compute which solvers have timed-out runs (for visual markers)
    timed_out_alg = {s for s in ALGIERS_ALGO_ORDER if _solver_timed_out_pct(df_alg, s) > 0}
    timed_out_50 = {s for s in S50_ALGO_ORDER if _solver_timed_out_pct(df_50, s) > 0}

    print("\n  -- Algiers --")
    plot_bar_scores(agg_alg, "Algiers -- Algorithm Scores", "algiers_scores.png",
                    ALGIERS_PALETTE, timed_out_solvers=timed_out_alg)
    plot_bar_times(agg_alg, "Algiers -- Execution Times", "algiers_times.png",
                   ALGIERS_PALETTE)

    if not df_50.empty:
        print("\n  -- Solomon-50 --")
        plot_bar_scores(agg_50, "Solomon-50 -- Average Score", "solomon50_scores.png",
                        S50_PALETTE, timed_out_solvers=timed_out_50)
        plot_bar_times(agg_50, "Solomon-50 -- Average Execution Time", "solomon50_times.png",
                       S50_PALETTE)
        plot_optimality_boxplot(df_50,
                                "Solomon-50 -- Optimality Gap (only CPLEX-terminated instances)",
                                "solomon50_optimality.png",
                                algo_order=S50_ALGO_ORDER,
                                palette=S50_PALETTE,
                                exclude_solver="CPLEX")

    # ==================================================================
    # 4. GENERATE LATEX TABLE
    # ==================================================================
    print("\n  -- LaTeX table --")
    latex = generate_latex(agg_alg, agg_50, df_alg, df_50,
                           ALGIERS_ALGO_ORDER, S50_ALGO_ORDER)
    tex_path = OUT / "results_algiers_s50.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"  Saved: results_algiers_s50.tex")

    # -- Save raw CSVs for inspection --
    df_alg.to_csv(OUT / "results_algiers.csv", index=False)
    if not df_50.empty:
        df_50.to_csv(OUT / "results_solomon50.csv", index=False)

    # ==================================================================
    # 5. STATISTICS AND INSIGHTS
    # ==================================================================
    print_statistics(df_alg, df_50, agg_alg, agg_50, ground_truths_50)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  All outputs saved to: {OUT}")
    print(f"  Files: {[f.name for f in sorted(OUT.iterdir())]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()