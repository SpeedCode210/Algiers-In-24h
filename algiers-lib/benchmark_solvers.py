"""
benchmarks/plots.py -- Clean, simple bar charts for benchmark comparison.

All plotting functions accept a DataFrame (from collect_all_results) and an
output directory, and save PNG files.  Designed for clarity in academic papers
and reports.

Functions:
    plot_mean_score_bar       - Grouped bar chart of mean score per solver per group
    plot_mean_time_bar        - Grouped bar chart of mean runtime per solver per group
    plot_optimality_gap_bar   - Bar chart of mean optimality gap per solver
    plot_optimality_gap_box   - Boxplot of optimality gaps per solver
    plot_algiers_score_bar    - Simple bar chart of scores for Algiers variants
    plot_algiers_time_bar     - Simple bar chart of runtimes for Algiers variants
    plot_solomon_score_bar    - Bar chart comparing solvers on Solomon datasets
    plot_combined_gap_bar     - Combined bar chart of gaps across all groups
    plot_score_vs_time_scatter - Scatter of score vs runtime
    plot_heatmap              - Heatmap of mean scores (solver x group)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")

# -- Style constants ----------------------------------------------------------
FIG_DPI = 200
FONT_SIZE = 11
TITLE_SIZE = 13
TICK_SIZE = 9

# Professional colour palette (solver-family-aware)
SOLVER_COLORS = {
    "Greedy":          "#4C72B0",
    "GRASP":           "#55A868",
    "SA":              "#C44E52",
    "Tabu":            "#8172B2",
    "Genetic":         "#CCB974",
    "CPLEX":           "#64B5CD",
    "CPLEX-120s":      "#64B5CD",
    "CPLEX-60s":       "#64B5CD",
    "CPLEX-Optimal":   "#64B5CD",
    # Algiers detailed variants
    "Greedy-Score":    "#4C72B0",
    "Greedy-Ratio":    "#6E8EBF",
    "GRASP-a0.1-30it": "#55A868",
    "GRASP-a0.3-50it": "#6DBF7D",
    "GRASP-a0.5-50it": "#7FD494",
    "GRASP-a0.3-100it": "#91E9AB",
    "SA-Boltzmann":    "#C44E52",
    "SA-Cauchy":       "#D06B6E",
    "SA-HighTemp":     "#DC888A",
    "SA-Reheat":       "#E8A5A6",
    "Tabu-Default":    "#8172B2",
    "Tabu-LongTenure": "#9B8DC4",
    "Tabu-MoreIter":   "#B5A8D6",
    "Tabu-LargeSlack": "#CFC3E8",
    "Genetic-Feasibility":   "#CCB974",
    "Genetic-Elitism":       "#D8CB8E",
    "Genetic-Penalty":       "#E4DDA8",
    "Genetic-Infeasibility": "#F0EFC2",
    # Solomon benchmark variants
    "Genetic-Tailored":      "#CCB974",
    "Genetic-Infeas":       "#F0EFC2",
}

FALLBACK_CMAP = "tab10"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color(solver: str) -> str:
    """Return a consistent colour for a solver name."""
    if solver in SOLVER_COLORS:
        return SOLVER_COLORS[solver]
    # Try prefix match (e.g. "SA-Reheat" -> "SA")
    for key, val in SOLVER_COLORS.items():
        if solver.startswith(key + "-") or solver.startswith(key + "_"):
            return val
    # Fallback: hash-based colour
    idx = hash(solver) % 10
    cmap = matplotlib.colormaps[FALLBACK_CMAP]
    return matplotlib.colors.to_hex(cmap(idx))


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close."""
    fig.tight_layout(pad=1.5)
    fig.savefig(str(path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] saved {path.name}")


def _mean_by_solver(df: pd.DataFrame, col: str) -> pd.Series:
    """Return mean of *col* grouped by solver, sorted descending."""
    return df.groupby("solver")[col].mean().sort_values(ascending=False)


# ===================================================================
# 1. Mean Score Bar Chart (per group)
# ===================================================================

def plot_mean_score_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: mean score per solver, one subplot per group."""
    groups = [g for g in df["group"].unique() if g != "Algiers"]
    if not groups:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 6),
                              sharey=False)
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        sub = df[df["group"] == grp]
        mean = sub.groupby("solver")["score"].mean().sort_values(ascending=True)
        std  = sub.groupby("solver")["score"].std().reindex(mean.index).fillna(0)
        colors = [_color(s) for s in mean.index]

        bars = ax.barh(range(len(mean)), mean.values, xerr=std.values,
                       color=colors, edgecolor="white", linewidth=0.5,
                       capsize=3, error_kw={"linewidth": 0.8})
        ax.set_yticks(range(len(mean)))
        ax.set_yticklabels(mean.index, fontsize=TICK_SIZE)
        ax.set_xlabel("Mean Score", fontsize=FONT_SIZE)
        ax.set_title(grp, fontsize=TITLE_SIZE, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # Annotate values
        for bar, v in zip(bars, mean.values):
            ax.text(v + std.get(mean.index[list(mean.values).index(v)], 0) + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}", va="center", fontsize=8, fontweight="bold")

    fig.suptitle("Mean Score by Solver", fontsize=TITLE_SIZE + 1, fontweight="bold")
    _save(fig, out_dir / "mean_score_bar.png")


# ===================================================================
# 2. Mean Runtime Bar Chart (per group)
# ===================================================================

def plot_mean_time_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: mean runtime per solver, one subplot per group."""
    groups = [g for g in df["group"].unique() if g != "Algiers"]
    if not groups:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 6),
                              sharey=False)
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        sub = df[df["group"] == grp]
        mean = sub.groupby("solver")["time_s"].mean().sort_values(ascending=True)
        colors = [_color(s) for s in mean.index]

        bars = ax.barh(range(len(mean)), mean.values,
                       color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(mean)))
        ax.set_yticklabels(mean.index, fontsize=TICK_SIZE)
        ax.set_xlabel("Mean Runtime (seconds)", fontsize=FONT_SIZE)
        ax.set_title(grp, fontsize=TITLE_SIZE, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for bar, v in zip(bars, mean.values):
            label = f"{v:.1f}s" if v >= 1 else f"{v*1000:.0f}ms"
            ax.text(v + mean.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=8)

    fig.suptitle("Mean Runtime by Solver", fontsize=TITLE_SIZE + 1, fontweight="bold")
    _save(fig, out_dir / "mean_time_bar.png")


# ===================================================================
# 3. Optimality Gap Bar Chart (mean gap per solver per group)
# ===================================================================

def plot_optimality_gap_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of mean optimality gap (%) per solver, per Solomon group.

    Only considers rows where optimality_gap_pct is not NaN.
    CPLEX reference solvers are excluded from the gap chart.
    """
    ref_solvers = {"CPLEX", "CPLEX-120s", "CPLEX-60s", "CPLEX-Optimal"}
    sub = df[
        (df["optimality_gap_pct"].notna())
        & (~df["solver"].isin(ref_solvers))
    ]
    if sub.empty:
        print("  [skip] No optimality gap data for bar chart")
        return

    groups = sorted(sub["group"].unique())
    solvers = sorted(sub["solver"].unique())

    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 6),
                              sharey=False)
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        g = sub[sub["group"] == grp]
        mean = g.groupby("solver")["optimality_gap_pct"].mean().sort_values(
            ascending=True)
        colors = [_color(s) for s in mean.index]

        bars = ax.barh(range(len(mean)), mean.values,
                       color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(mean)))
        ax.set_yticklabels(mean.index, fontsize=TICK_SIZE)
        ax.set_xlabel("Mean Optimality Gap (%)", fontsize=FONT_SIZE)
        ax.set_title(grp, fontsize=TITLE_SIZE, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for bar, v in zip(bars, mean.values):
            ax.text(v + mean.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8)

    fig.suptitle("Mean Optimality Gap vs Ground Truth",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
    _save(fig, out_dir / "optimality_gap_bar.png")


# ===================================================================
# 4. Optimality Gap Box Plot
# ===================================================================

def plot_optimality_gap_box(df: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot of optimality gaps per solver across all instances.

    Only considers rows where optimality_gap_pct is not NaN.
    CPLEX reference solvers are excluded.
    """
    ref_solvers = {"CPLEX", "CPLEX-120s", "CPLEX-60s", "CPLEX-Optimal"}
    sub = df[
        (df["optimality_gap_pct"].notna())
        & (~df["solver"].isin(ref_solvers))
    ]
    if sub.empty:
        print("  [skip] No optimality gap data for boxplot")
        return

    solvers = sorted(sub["solver"].unique())
    data = [sub[sub["solver"] == s]["optimality_gap_pct"].dropna().values
            for s in solvers]
    colors = [_color(s) for s in solvers]

    fig, ax = plt.subplots(figsize=(max(8, len(solvers) * 1.3), 6))
    bp = ax.boxplot(data, patch_artist=True, vert=True,
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(linewidth=1.0),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(solvers) + 1))
    ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Optimality Gap (%)", fontsize=FONT_SIZE)
    ax.set_title("Optimality Gap Distribution (All Instances)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.axhline(0, color="green", linestyle="--", linewidth=1.5, label="Optimal")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_dir / "optimality_gap_box.png")


# ===================================================================
# 5. Algiers Score Bar Chart (all variants)
# ===================================================================

def plot_algiers_score_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Simple vertical bar chart of scores for all Algiers solver variants."""
    sub = df[df["group"] == "Algiers"]
    if sub.empty:
        return

    mean = sub.groupby("solver")["score"].mean().sort_values(ascending=False)
    colors = [_color(s) for s in mean.index]

    fig, ax = plt.subplots(figsize=(max(10, len(mean) * 1.1), 6))
    bars = ax.bar(range(len(mean)), mean.values,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels(mean.index, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Score", fontsize=FONT_SIZE)
    ax.set_title("Algiers Dataset -- Score by Solver Variant",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, v in zip(bars, mean.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    _save(fig, out_dir / "algiers_score_bar.png")


# ===================================================================
# 6. Algiers Runtime Bar Chart (all variants)
# ===================================================================

def plot_algiers_time_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Simple vertical bar chart of runtimes for all Algiers solver variants."""
    sub = df[df["group"] == "Algiers"]
    if sub.empty:
        return

    mean = sub.groupby("solver")["time_s"].mean().sort_values(ascending=False)
    colors = [_color(s) for s in mean.index]

    fig, ax = plt.subplots(figsize=(max(10, len(mean) * 1.1), 6))
    bars = ax.bar(range(len(mean)), mean.values,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels(mean.index, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Runtime (seconds)", fontsize=FONT_SIZE)
    ax.set_title("Algiers Dataset -- Runtime by Solver Variant",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, v in zip(bars, mean.values):
        label = f"{v:.1f}s" if v >= 1 else f"{v*1000:.0f}ms"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=8)
    _save(fig, out_dir / "algiers_time_bar.png")


# ===================================================================
# 7. Solomon Score Comparison Bar Chart
# ===================================================================

def plot_solomon_score_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart comparing solver scores across Solomon groups.

    One grouped bar per solver, with sub-bars for each Solomon dataset.
    """
    groups = sorted(g for g in df["group"].unique() if g.startswith("Solomon"))
    if not groups:
        return

    solvers = sorted(df[df["group"].isin(groups)]["solver"].unique())
    n_groups = len(groups)
    n_solvers = len(solvers)
    bar_w = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(max(10, n_solvers * 1.5), 7))
    x = np.arange(n_solvers)
    group_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, grp in enumerate(groups):
        sub = df[df["group"] == grp]
        mean = sub.groupby("solver")["score"].mean().reindex(solvers).fillna(0)
        offset = (i - n_groups / 2 + 0.5) * bar_w
        ax.bar(x + offset, mean.values, bar_w * 0.9,
               label=grp, color=group_colors[i % len(group_colors)],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Mean Score", fontsize=FONT_SIZE)
    ax.set_title("Solver Scores Across Solomon Datasets",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _save(fig, out_dir / "solomon_score_bar.png")


# ===================================================================
# 8. Combined Optimality Gap Bar (all groups together)
# ===================================================================

def plot_combined_gap_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Combined bar chart: mean optimality gap per solver, broken down by group.

    This is the main comparison figure for papers -- shows which solver
    performs best relative to the ground truth on each dataset size.
    """
    ref_solvers = {"CPLEX", "CPLEX-120s", "CPLEX-60s", "CPLEX-Optimal"}
    sub = df[
        (df["optimality_gap_pct"].notna())
        & (~df["solver"].isin(ref_solvers))
    ]
    if sub.empty:
        print("  [skip] No optimality gap data for combined bar chart")
        return

    groups = sorted(sub["group"].unique())
    solvers = sorted(sub["solver"].unique())
    n_groups = len(groups)
    n_solvers = len(solvers)
    bar_w = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_solvers * 1.5), 7))
    x = np.arange(n_solvers)
    group_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, grp in enumerate(groups):
        g = sub[sub["group"] == grp]
        mean = g.groupby("solver")["optimality_gap_pct"].mean().reindex(
            solvers).fillna(0)
        offset = (i - n_groups / 2 + 0.5) * bar_w
        ax.bar(x + offset, mean.values, bar_w * 0.9,
               label=grp, color=group_colors[i % len(group_colors)],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Mean Optimality Gap (%)", fontsize=FONT_SIZE)
    ax.set_title("Optimality Gap vs Ground Truth (by Dataset Size)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(title="Dataset", loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _save(fig, out_dir / "combined_gap_bar.png")


# ===================================================================
# 9. Score vs Time Scatter Plot
# ===================================================================

def plot_score_vs_time_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot: score (y) vs runtime (x), coloured by solver."""
    solvers = sorted(df["solver"].unique())
    groups  = sorted(df["group"].unique())
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for gi, grp in enumerate(groups):
        g_sub = df[df["group"] == grp]
        for solver in solvers:
            s_sub = g_sub[g_sub["solver"] == solver]
            if s_sub.empty:
                continue
            ax.scatter(s_sub["time_s"], s_sub["score"],
                       c=_color(solver), marker=markers[gi % len(markers)],
                       s=60, alpha=0.75, edgecolors="black", linewidths=0.4)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    solver_handles = [Patch(color=_color(s), label=s) for s in solvers]
    group_handles = [Line2D([0], [0], marker=markers[i % len(markers)],
                            color="grey", linestyle="None",
                            markersize=8, label=g)
                     for i, g in enumerate(groups)]
    leg1 = ax.legend(handles=solver_handles, title="Solver",
                     loc="upper left", fontsize=8, ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, title="Group", loc="lower right", fontsize=9)

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Runtime (s)", fontsize=FONT_SIZE)
    ax.set_ylabel("Score", fontsize=FONT_SIZE)
    ax.set_title("Score vs Runtime (All Instances)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    _save(fig, out_dir / "score_vs_time_scatter.png")


# ===================================================================
# 10. Heatmap: Mean Score (Solver x Group)
# ===================================================================

def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of mean scores per solver per group."""
    pivot = df.groupby(["solver", "group"])["score"].mean().unstack(fill_value=0)
    if pivot.empty:
        return

    # Normalise per column for colour comparison across different scales
    norm_pivot = pivot.div(pivot.max().replace(0, 1))

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5),
                                    max(5, len(pivot) * 0.55)))
    sns.heatmap(norm_pivot, annot=pivot.round(1), fmt=".1f",
                cmap="YlGnBu", linewidths=0.5, linecolor="grey",
                cbar_kws={"label": "Normalised Score"}, ax=ax)
    ax.set_title("Mean Score Heatmap (Solver x Dataset Group)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Dataset Group", fontsize=FONT_SIZE)
    ax.set_ylabel("Solver", fontsize=FONT_SIZE)
    _save(fig, out_dir / "heatmap_scores.png")


# ===================================================================
# 11. CPLEX Verification Bar (Solomon-50 only)
# ===================================================================

def plot_cplex_verification_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart comparing CPLEX score vs ground truth on Solomon-50.

    This verifies that CPLEX achieves 0% optimality gap on the 50-node
    dataset (i.e. matches the Righini & Salani optimal values).
    """
    sub = df[df["group"] == "Solomon-50"]
    if sub.empty:
        return

    cplex_rows = sub[sub["solver"].str.startswith("CPLEX")]
    if cplex_rows.empty:
        return

    # Build comparison: instance, ground truth (from optimality gap calc),
    # CPLEX score
    instances = sorted(cplex_rows["instance"].unique())
    gt_scores = []
    cplex_scores = []
    for inst in instances:
        row = cplex_rows[cplex_rows["instance"] == inst].iloc[0]
        cplex_scores.append(row["score"])
        # Reverse-compute ground truth from the gap
        gap = row.get("optimality_gap_pct", None)
        if gap is not None and gap == 0.0:
            gt_scores.append(row["score"])
        else:
            # If gap > 0, estimate GT
            if gap is not None and row["score"] > 0:
                gt = row["score"] / (1 - gap / 100.0)
                gt_scores.append(gt)
            else:
                gt_scores.append(None)

    fig, ax = plt.subplots(figsize=(max(8, len(instances) * 1.2), 6))
    x = np.arange(len(instances))
    w = 0.35

    ax.bar(x - w/2, [s if s else 0 for s in gt_scores], w,
           label="Ground Truth", color="#55A868", edgecolor="white")
    ax.bar(x + w/2, cplex_scores, w,
           label="CPLEX", color="#64B5CD", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Score (Prize)", fontsize=FONT_SIZE)
    ax.set_title("CPLEX vs Ground Truth on Solomon-50 (Verification)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate gap
    for i, (gt_val, cp_val) in enumerate(zip(gt_scores, cplex_scores)):
        if gt_val and gt_val > 0:
            gap_pct = abs(gt_val - cp_val) / gt_val * 100
            ax.text(i, max(gt_val, cp_val) + max(gt_scores, default=1) * 0.01,
                    f"{gap_pct:.1f}%", ha="center", fontsize=8,
                    color="green" if gap_pct < 0.1 else "red", fontweight="bold")

    _save(fig, out_dir / "cplex_verification_bar.png")
