"""
compare_all.py -- Master benchmark comparison script.

Usage
-----
    python compare_all.py                  # runs 5 instances per group (fast)
    python compare_all.py --full           # runs ALL instances (slow)
    python compare_all.py --no-s200        # skip 200-node group
    python compare_all.py --algiers-only   # only Algiers parameter study

All outputs (PNG plots, LaTeX table, CSV) are saved to:
    benchmarks/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# -- Path setup ---------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from benchmarks.runner import (
    collect_all_results,
    MAX_PER_GROUP,
    RUN_FULL,
)
from benchmarks import plots as P

RESULTS_DIR = _ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -- CLI -----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all solvers on all benchmark datasets and produce plots + LaTeX."
    )
    parser.add_argument("--full",          action="store_true",
                        help="Run ALL instances per group (ignores MAX_PER_GROUP).")
    parser.add_argument("--no-algiers",    action="store_true", help="Skip Algiers dataset.")
    parser.add_argument("--no-s50",        action="store_true", help="Skip Solomon-50.")
    parser.add_argument("--no-s100",       action="store_true", help="Skip Solomon-100.")
    parser.add_argument("--no-s200",       action="store_true", help="Skip Solomon-200.")
    parser.add_argument("--algiers-only",  action="store_true",
                        help="Only run Algiers parameter sensitivity study.")
    return parser.parse_args()


# -- LaTeX table ---------------------------------------------------------------
def generate_latex_table(df: pd.DataFrame, path: Path) -> None:
    """Write a comprehensive LaTeX longtable summarising all results."""
    summary = (
        df.groupby(["group", "solver"])
        .agg(
            instances    = ("instance",             "nunique"),
            mean_score   = ("score",                "mean"),
            mean_time_s  = ("time_s",               "mean"),
            mean_gap_pct = ("optimality_gap_pct",   "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["group", "mean_score"], ascending=[True, False])

    lines: list[str] = [
        r"\documentclass{article}",
        r"\usepackage{booktabs,longtable,array,geometry}",
        r"\geometry{margin=1in}",
        r"\begin{document}",
        r"\begin{center}",
        r"\footnotesize",
        (r"\begin{longtable}{llrrrr}"),
        r"\caption{Performance Benchmark: Solvers vs. Righini \& Salani Ground Truth}\\",
        r"\toprule",
        (r"\textbf{Group} & \textbf{Solver} & \textbf{Inst.} & "
         r"\textbf{Mean Score} & \textbf{Mean Runtime (s)} & \textbf{Optimality Gap (\%)} \\"),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        (r"\textbf{Group} & \textbf{Solver} & \textbf{Inst.} & "
         r"\textbf{Mean Score} & \textbf{Mean Runtime (s)} & \textbf{Optimality Gap (\%)} \\"),
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{6}{r}{Continued on next page\ldots} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    current_group = None
    for _, row in summary.iterrows():
        if row["group"] != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            current_group = row["group"]

        gap_str = (f"{row['mean_gap_pct']:.1f}\\%"
                   if pd.notna(row["mean_gap_pct"]) else "---")

        lines.append(
            f"{row['group']} & {row['solver'].replace('_', r'\_')} & "
            f"{int(row['instances'])} & "
            f"{row['mean_score']:.1f} & "
            f"{row['mean_time_s']:.2f} & "
            f"{gap_str} \\\\"
        )

    lines += [
        r"\end{longtable}",
        r"\end{center}",
        r"\end{document}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] LaTeX table saved -> {path.name}")


# -- Console summary -----------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    summary = (
        df.groupby(["group", "solver"])
        .agg(mean_score=("score", "mean"),
             mean_time=("time_s", "mean"),
             mean_gap=("optimality_gap_pct", "mean"),
             valid_pct=("valid", lambda x: 100 * x.mean()))
        .reset_index()
        .sort_values(["group", "mean_score"], ascending=[True, False])
    )
    print(f"{'Group':<15} {'Solver':<25} {'Score':>8} {'Time(s)':>9} "
          f"{'Gap%':>7} {'Valid%':>7}")
    print("-" * 75)
    current_group = None
    for _, r in summary.iterrows():
        if r["group"] != current_group:
            if current_group:
                print()
            current_group = r["group"]
        gap_str = f"{r['mean_gap']:.1f}" if pd.notna(r["mean_gap"]) else "---"
        print(f"{r['group']:<15} {r['solver']:<25} "
              f"{r['mean_score']:>8.1f} {r['mean_time']:>9.2f} "
              f"{gap_str:>6}% {r['valid_pct']:>6.0f}%")
    print("=" * 90)


# -- Main ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Apply CLI flags to runner module
    import benchmarks.runner as runner_mod
    if args.full:
        runner_mod.RUN_FULL = True
        print("  [INFO] --full: running ALL instances per group.")

    run_algiers = not args.no_algiers
    run_s50     = not args.no_s50     and not args.algiers_only
    run_s100    = not args.no_s100    and not args.algiers_only
    run_s200    = False  # Solomon-200 disabled by default

    if args.algiers_only:
        print("  [INFO] --algiers-only: skipping Solomon groups.")

    # -- Collect all results ---------------------------------------------------
    frames = collect_all_results(
        run_algiers=run_algiers,
        run_s50=run_s50,
        run_s100=run_s100,
        run_s200=run_s200,
    )

    # -- Rebuild combined frame ------------------------------------------------
    non_empty = [df for df in frames.values()
                 if isinstance(df, pd.DataFrame) and not df.empty
                 and df is not frames.get("all")]
    df_all = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    frames["all"] = df_all

    # -- Save CSV --------------------------------------------------------------
    csv_path = RESULTS_DIR / "results_summary.csv"
    if not df_all.empty:
        df_all.to_csv(csv_path, index=False)
        print(f"\n  [OK] CSV saved -> {csv_path}")

    # -- Print console summary -------------------------------------------------
    if not df_all.empty:
        print_summary(df_all)

    # -- Generate plots --------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating plots ...")
    print("=" * 60)

    if not df_all.empty:
        # Generate simple bar charts per dataset group
        P.plot_simple_bar_scores(df_all, RESULTS_DIR)

    # -- LaTeX table -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating LaTeX table ...")
    print("=" * 60)
    if not df_all.empty:
        generate_latex_table(df_all, RESULTS_DIR / "results_table.tex")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
