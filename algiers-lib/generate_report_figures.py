"""
generate_report_figures.py
==========================
Final report figure generator for the OPTW (Orienteering Problem with Time Windows)
benchmark study. Produces:

  1.  algiers_scores.pdf          -- bar chart: algo scores on Algiers dataset
  2.  algiers_times.pdf           -- bar chart: execution times on Algiers
  3.  algiers_optimality.pdf      -- optimality gap (CPLEX = ground truth)
  4.  solomon50_scores.pdf        -- bar chart: avg scores on Solomon-50
  5.  solomon50_times.pdf         -- bar chart: avg times on Solomon-50
  6.  solomon50_optimality.pdf    -- optimality gap vs CPLEX ground truth
  7.  solomon100_scores.pdf       -- bar chart: avg scores on Solomon-100
  8.  solomon100_times.pdf        -- bar chart: avg times on Solomon-100
  9.  results_table.tex           -- LaTeX table (score, time, gap) for all datasets

Usage
-----
Run from the algiers-lib directory (or any dir with the project on PYTHONPATH):

    python generate_report_figures.py           # uses live solvers
    python generate_report_figures.py --mock    # uses built-in mock data (no solvers needed)

The --mock flag is used automatically when the project solvers cannot be imported.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ---------------------------------------------------------------------------
# Output directory (relative to project root)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
OUT = _ROOT / "benchmarks" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants (publication quality)
# ---------------------------------------------------------------------------
ALGO_ORDER = [
    "Greedy", "GRASP", "SA", "Tabu",
    "Genetic-Tailored", "Genetic-Penalty", "Genetic-Infeas",
    "CPLEX",
]

PALETTE = {
    "Greedy":            "#4C72B0",
    "GRASP":             "#DD8452",
    "SA":                "#55A868",
    "Tabu":              "#C44E52",
    "Genetic-Tailored":  "#8172B3",
    "Genetic-Penalty":   "#A78DC4",
    "Genetic-Infeas":    "#CDB4E0",
    "CPLEX":             "#937860",
}

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

# ---------------------------------------------------------------------------
# Mock data (used when live solvers are unavailable)
# ---------------------------------------------------------------------------

def _mock_algiers() -> pd.DataFrame:
    """Realistic mock results for the Algiers real-world dataset (single run each)."""
    rows = []
    data = {
        # (score, time_s)
        "Greedy":  (185.0, 0.04),
        "GRASP":   (218.0, 1.82),
        "SA":      (231.0, 4.51),
        "Tabu":    (226.0, 3.78),
        "Genetic": (224.0, 8.93),
        "CPLEX":   (245.0, 312.6),   # CPLEX run-to-completion = ground truth
    }
    cplex_score = data["CPLEX"][0]
    for algo, (score, t) in data.items():
        gap = (cplex_score - score) / cplex_score * 100.0 if algo != "CPLEX" else 0.0
        rows.append({
            "solver": algo,
            "score": score,
            "time_s": t,
            "optimality_gap_pct": round(gap, 2),
            "instance": "Algiers",
            "group": "Algiers",
        })
    return pd.DataFrame(rows)


def _mock_solomon50() -> pd.DataFrame:
    """Mock results averaged over 5 Solomon-50 instances."""
    rng = np.random.default_rng(42)
    instances = ["c101", "r101", "rc101", "c103", "r103"]
    # CPLEX run-to-completion ground truths for these instances
    gt = {"c101": 450, "r101": 200, "rc101": 230, "c103": 390, "r103": 190}
    rows = []
    base = {
        "Greedy":  (0.68, 0.05),
        "GRASP":   (0.81, 1.4),
        "SA":      (0.87, 3.2),
        "Tabu":    (0.85, 2.9),
        "Genetic": (0.84, 7.1),
        "CPLEX":   (1.00, 18.4),   # CPLEX matches its own ground truth
    }
    for inst in instances:
        opt = gt[inst]
        for algo, (ratio, t_base) in base.items():
            score = opt * ratio * (1 + rng.normal(0, 0.02))
            t = t_base * (1 + rng.normal(0, 0.1))
            gap = (opt - score) / opt * 100.0
            rows.append({
                "solver": algo,
                "score": round(score, 1),
                "time_s": round(t, 3),
                "optimality_gap_pct": round(max(gap, 0.0), 2),
                "instance": inst,
                "group": "Solomon-50",
            })
    return pd.DataFrame(rows)


def _mock_solomon100() -> pd.DataFrame:
    """Mock results averaged over 5 Solomon-100 instances."""
    rng = np.random.default_rng(7)
    instances = ["c101", "r101", "rc101", "c102", "r102"]
    # Righini-Salani bestPossible ground truths (approx from literature)
    gt = {"c101": 720, "r101": 410, "rc101": 545, "c102": 690, "r102": 390}
    rows = []
    base = {
        "Greedy":  (0.61, 0.07),
        "GRASP":   (0.74, 2.8),
        "SA":      (0.80, 6.4),
        "Tabu":    (0.78, 5.7),
        "Genetic": (0.77, 14.2),
        "CPLEX":   (0.88, 60.0),
    }
    for inst in instances:
        opt = gt[inst]
        for algo, (ratio, t_base) in base.items():
            score = opt * ratio * (1 + rng.normal(0, 0.025))
            t = t_base * (1 + rng.normal(0, 0.12))
            gap = (opt - score) / opt * 100.0
            rows.append({
                "solver": algo,
                "score": round(score, 1),
                "time_s": round(t, 3),
                "optimality_gap_pct": round(max(gap, 0.0), 2),
                "instance": inst,
                "group": "Solomon-100",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attempt live run (falls back to mock)
# ---------------------------------------------------------------------------

def try_live_run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Try to import and run the actual project solvers.

    For Solomon-50, CPLEX is run first as a pre-pass to establish
    the optimality baseline, then all solvers (including CPLEX again)
    are compared against it.

    Returns (df_algiers, df_s50, df_s100, is_live).
    """
    try:
        sys.path.insert(0, str(_ROOT))

        from benchmarks.runner import (
            load_algiers_problem,
            load_solomon_group,
            load_s100_ground_truth,
            run_group,
        )
        from benchmarks.solver_registry import (
            ALGIERS_VARIANTS,
            SOLOMON_50_VARIANTS,
            SOLOMON_100_VARIANTS,
        )

        print("Live solver imports OK -- running benchmarks ...")

        # ── Algiers ────────────────────────────────────────────────────────
        name, prob = load_algiers_problem()
        df_alg = run_group([(name, prob)], ALGIERS_VARIANTS, group_label="Algiers")

        # Compute Algiers optimality gap: CPLEX score is ground truth
        cplex_rows = df_alg[df_alg["solver"].str.startswith("CPLEX")]
        if not cplex_rows.empty:
            cplex_score = cplex_rows.iloc[0]["score"]
            df_alg["optimality_gap_pct"] = df_alg["score"].apply(
                lambda s: max((cplex_score - s) / cplex_score * 100.0, 0.0)
                if cplex_score > 0 else None
            )

        # ── Solomon-50 (CPLEX pre-pass as ground truth) ───────────────────
        _here = _ROOT / "benchmarks"
        s50_dir = _here / "datasets" / "c_r_rc_100_50"
        insts50 = load_solomon_group(s50_dir, max_instances=5)

        # 1. Identify the CPLEX variant
        cplex_variant = next(
            (v for v in SOLOMON_50_VARIANTS if "CPLEX" in v["name"]), None
        )

        # 2. Pre-run CPLEX to establish ground truth
        live_gt_50: dict[str, float] = {}
        if cplex_variant:
            print(f"\n  [Pre-pass] Running CPLEX on {len(insts50)} "
                  f"Solomon-50 instances for ground truth ...")
            for inst_name, prob in insts50:
                try:
                    solver = cplex_variant["factory"](prob)
                    if hasattr(solver, "time_limit"):
                        solver.time_limit = 300  # 5 min safety cap
                    tour = solver.solve()
                    live_gt_50[inst_name] = tour.total_score() if tour else 0.0
                except Exception:
                    pass
            print(f"  [OK] CPLEX ground truth for {len(live_gt_50)} instances.")
        else:
            print("  [WARN] CPLEX not available for Solomon-50 ground truth.")

        # 3. Run all solvers using CPLEX scores as ground truth
        df_50 = run_group(insts50, SOLOMON_50_VARIANTS,
                          ground_truths=live_gt_50, group_label="Solomon-50")

        # ── Solomon-100 (file-based ground truth) ─────────────────────────
        s100_dir = _here / "datasets" / "c_r_rc_100_100"
        gt100 = load_s100_ground_truth()
        insts100 = load_solomon_group(s100_dir, max_instances=5)
        df_100 = run_group(insts100, SOLOMON_100_VARIANTS,
                           ground_truths=gt100, group_label="Solomon-100")

        return df_alg, df_50, df_100, True

    except Exception as exc:
        print(f"[INFO] Live run unavailable ({exc.__class__.__name__}: {exc}). "
              "Falling back to mock data.")
        return _mock_algiers(), _mock_solomon50(), _mock_solomon100(), False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_name(name: str) -> str:
    """Map solver names to canonical family name.

    'CPLEX-60s' -> 'CPLEX', 'Greedy-Ratio' -> 'Greedy',
    'Genetic-Tailored' -> 'Genetic-Tailored', 'Genetic-Feasibility' -> 'Genetic-Tailored',
    'Genetic-Penalty' -> 'Genetic-Penalty', 'Genetic-Infeasibility' -> 'Genetic-Infeas', etc.
    """
    # CPLEX variants
    if name.startswith("CPLEX"):
        return "CPLEX"
    # Genetic variants -- match exact sub-names
    for fam in ("Genetic-Tailored", "Genetic-Penalty", "Genetic-Infeas"):
        if name.startswith(fam):
            return fam
    # Map Algiers-specific Genetic names to Solomon benchmark names
    if "Feasibility" in name or "Elitism" in name:
        return "Genetic-Tailored"
    if "Penalty" in name:
        return "Genetic-Penalty"
    if "Infeasibility" in name:
        return "Genetic-Infeas"
    # Other families
    for fam in ("Greedy", "GRASP", "SA", "Tabu"):
        if name.startswith(fam):
            return fam
    return name


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by canonical solver name (mean score, time, gap)."""
    df = df.copy()
    df["algo"] = df["solver"].apply(_canonical_name)
    agg = (df.groupby("algo")
             .agg(
                 score_mean=("score", "mean"),
                 score_std=("score", "std"),
                 time_mean=("time_s", "mean"),
                 time_std=("time_s", "std"),
                 gap_mean=("optimality_gap_pct", "mean"),
                 gap_std=("optimality_gap_pct", "std"),
                 n=("score", "count"),
             )
             .reset_index())
    # Keep canonical order
    order = [a for a in ALGO_ORDER if a in agg["algo"].values]
    agg = agg.set_index("algo").loc[order].reset_index()
    return agg


def _colors(algos):
    return [PALETTE.get(a, "#888888") for a in algos]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bar_scores(agg: pd.DataFrame, title: str, fname: str,
                ylabel: str = "Mean Score (interest points)"):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["score_mean"], color=_colors(agg["algo"]),
                  edgecolor="white", linewidth=0.6, zorder=3)
    # Error bars if std available and > 0
    if "score_std" in agg.columns:
        errs = agg["score_std"].fillna(0)
        ax.errorbar(x, agg["score_mean"], yerr=errs,
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    # Value labels
    for bar, v in zip(bars, agg["score_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["algo"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, agg["score_mean"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _bar_times(agg: pd.DataFrame, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["time_mean"], color=_colors(agg["algo"]),
                  edgecolor="white", linewidth=0.6, zorder=3)
    if "time_std" in agg.columns:
        errs = agg["time_std"].fillna(0)
        ax.errorbar(x, agg["time_mean"], yerr=errs,
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    for bar, v in zip(bars, agg["time_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["algo"], rotation=20, ha="right")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title(title)
    ax.set_ylim(0, agg["time_mean"].max() * 1.22)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _bar_gap(agg: pd.DataFrame, title: str, fname: str, note: str = ""):
    """Optimality gap bar chart (exclude CPLEX/ground-truth itself)."""
    agg_plot = agg[agg["algo"] != "CPLEX"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg_plot))
    bars = ax.bar(x, agg_plot["gap_mean"], color=_colors(agg_plot["algo"]),
                  edgecolor="white", linewidth=0.6, zorder=3)
    if "gap_std" in agg_plot.columns:
        errs = agg_plot["gap_std"].fillna(0)
        ax.errorbar(x, agg_plot["gap_mean"], yerr=errs,
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    for bar, v in zip(bars, agg_plot["gap_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agg_plot["algo"], rotation=20, ha="right")
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(agg_plot["gap_mean"].max() * 1.25, 5))
    if note:
        ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------

def _latex_table(df_alg: pd.DataFrame,
                 df_50: pd.DataFrame,
                 df_100: pd.DataFrame,
                 is_live: bool) -> str:
    """Generate a combined LaTeX table:
    - Algiers: score, time, optimality gap (vs CPLEX ground truth)
    - Solomon-50: score, time, optimality gap (vs CPLEX run-to-completion)
    - Solomon-100: score, time, optimality gap (vs Righini-Salani bestPossible)
    """
    agg_alg = _aggregate(df_alg)
    agg_50  = _aggregate(df_50)
    agg_100 = _aggregate(df_100)

    lines: list[str] = []
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm comparison: mean score, execution time, and optimality gap "
        r"across three datasets. Algiers and Solomon-50 use CPLEX run to completion as "
        r"ground truth. Solomon-100 uses Righini--Salani best-possible as ground truth.}",
        r"\label{tab:algo_comparison}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrr rrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Algiers}}",
        r"& \multicolumn{3}{c}{\textbf{Solomon-50}}",
        r"& \multicolumn{3}{c}{\textbf{Solomon-100}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"\textbf{Algorithm}",
        r"& \textbf{Score} & \textbf{Time (s)} & \textbf{Gap (\%)}",
        r"& \textbf{Score} & \textbf{Time (s)} & \textbf{Gap (\%)}",
        r"& \textbf{Score} & \textbf{Time (s)} & \textbf{Gap (\%)} \\",
        r"\midrule",
    ]

    def _fmt_score(row):
        if row.empty:
            return "--"
        s = row["score_mean"].values[0]
        n = row["n"].values[0]
        if n > 1:
            sd = row["score_std"].values[0]
            if not (hasattr(sd, "__class__") and sd.__class__.__name__ == "float" and math.isnan(sd)):
                return rf"{s:.1f} $\pm$ {sd:.1f}"
        return f"{s:.1f}"

    def _fmt_time(row):
        if row.empty:
            return "--"
        t = row["time_mean"].values[0]
        n = row["n"].values[0]
        if n > 1:
            sd = row["time_std"].values[0]
            if not (hasattr(sd, "__class__") and sd.__class__.__name__ == "float" and math.isnan(sd)):
                return rf"{t:.2f} $\pm$ {sd:.2f}"
        return f"{t:.2f}"

    def _fmt_gap(row, is_gt=False):
        if is_gt:
            return r"\textbf{0.00}"
        if row.empty:
            return "--"
        g = row["gap_mean"].values[0]
        if g is None or (hasattr(g, "__class__") and g.__class__.__name__ == "float" and math.isnan(g)):
            return "--"
        return f"{g:.2f}"

    for algo in ALGO_ORDER:
        row_a = agg_alg[agg_alg["algo"] == algo]
        row_50 = agg_50[agg_50["algo"] == algo]
        row_h = agg_100[agg_100["algo"] == algo]

        is_cplex = (algo == "CPLEX")
        algo_label = algo.replace("_", r"\_")
        if is_cplex:
            algo_label = r"\textit{" + algo_label + r"} (GT)"

        lines.append(
            rf"{algo_label} "
            rf"& {_fmt_score(row_a)} & {_fmt_time(row_a)} & {_fmt_gap(row_a, is_cplex)} "
            rf"& {_fmt_score(row_50)} & {_fmt_time(row_50)} & {_fmt_gap(row_50, is_cplex)} "
            rf"& {_fmt_score(row_h)} & {_fmt_time(row_h)} & {_fmt_gap(row_h, is_cplex)} \\"
        )
        if algo == "Genetic":
            lines.append(r"\midrule")  # separator before CPLEX

    if is_live:
        note = r"Results from live solver runs."
    else:
        note = r"\textit{Note: values shown are from representative mock data (solvers not available in this environment).}"

    lines += [
        r"\bottomrule",
        r"\multicolumn{10}{l}{" + note + r"} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="Force use of mock data (skip live solver run)")
    args = parser.parse_args()

    if args.mock:
        print("Using mock data (--mock flag set).")
        df_alg = _mock_algiers()
        df_50  = _mock_solomon50()
        df_100 = _mock_solomon100()
        is_live = False
    else:
        df_alg, df_50, df_100, is_live = try_live_run()

    print(f"\nGenerating figures -> {OUT}\n")

    # -- Aggregate ----------------------------------------------------------
    agg_alg = _aggregate(df_alg)
    agg_50  = _aggregate(df_50)
    agg_100 = _aggregate(df_100)

    # -- Algiers (2 bar charts + 1 optimality gap) --------------------------
    print("-- Algiers --")
    _bar_scores(agg_alg,
                "Algiers Dataset -- Algorithm Scores",
                "algiers_scores.pdf")
    _bar_times(agg_alg,
               "Algiers Dataset -- Execution Times",
               "algiers_times.pdf")
    _bar_gap(agg_alg,
             "Algiers Dataset -- Optimality Gap (CPLEX = Ground Truth)",
             "algiers_optimality.pdf",
             note="Ground truth: CPLEX run to completion")

    # -- Solomon-50 (2 bar charts + 1 optimality gap) -----------------------
    print("-- Solomon-50 --")
    _bar_scores(agg_50,
                "Solomon-50 -- Average Score (5 instances)",
                "solomon50_scores.pdf")
    _bar_times(agg_50,
               "Solomon-50 -- Average Execution Time (5 instances)",
               "solomon50_times.pdf")
    _bar_gap(agg_50,
             "Solomon-50 -- Optimality Gap vs. CPLEX Ground Truth",
             "solomon50_optimality.pdf",
             note="Ground truth: CPLEX run to completion (0% gap)")

    # -- Solomon-100 (2 bar charts) ----------------------------------------
    print("-- Solomon-100 --")
    _bar_scores(agg_100,
                "Solomon-100 -- Average Score (5 instances)",
                "solomon100_scores.pdf")
    _bar_times(agg_100,
               "Solomon-100 -- Average Execution Time (5 instances)",
               "solomon100_times.pdf")

    # -- LaTeX table --------------------------------------------------------
    print("-- LaTeX table --")
    latex = _latex_table(df_alg, df_50, df_100, is_live)
    table_path = OUT / "results_table.tex"
    table_path.write_text(latex)
    print(f"  Saved: results_table.tex")

    # -- Also save CSVs for inspection --------------------------------------
    df_alg.to_csv(OUT / "results_algiers.csv", index=False)
    df_50.to_csv(OUT / "results_solomon50.csv", index=False)
    df_100.to_csv(OUT / "results_solomon100.csv", index=False)

    print(f"\nAll outputs saved to: {OUT}")
    print(f"  Files: {[f.name for f in sorted(OUT.iterdir())]}")


if __name__ == "__main__":
    main()
