"""
generate_outputs.py -- Regenerate plots and LaTeX table from saved CSV.

Usage:
    python generate_outputs.py

Reads:  benchmarks/results/results_summary.csv
Writes: benchmarks/results/*.png, benchmarks/results/results_table.tex
"""

import sys
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from benchmarks import plots as P
from compare_all import generate_latex_table, RESULTS_DIR


def main():
    csv_path = RESULTS_DIR / "results_summary.csv"
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}")
        print("Run compare_all.py first to generate the results.")
        return

    df_all = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df_all)} rows from {csv_path.name}")

    if df_all.empty:
        print("DataFrame is empty, nothing to plot.")
        return

    # -- Generate all plots ----------------------------------------------------
    print("\nGenerating plots ...")

    # Score & runtime bars (Solomon groups)
    P.plot_mean_score_bar(df_all, RESULTS_DIR)
    P.plot_mean_time_bar(df_all, RESULTS_DIR)

    # Optimality gap charts
    P.plot_optimality_gap_bar(df_all, RESULTS_DIR)
    P.plot_optimality_gap_box(df_all, RESULTS_DIR)
    P.plot_combined_gap_bar(df_all, RESULTS_DIR)

    # Solomon comparison
    P.plot_solomon_score_bar(df_all, RESULTS_DIR)

    # CPLEX verification (Solomon-50 only)
    P.plot_cplex_verification_bar(df_all, RESULTS_DIR)

    # Algiers detailed bars
    P.plot_algiers_score_bar(df_all, RESULTS_DIR)
    P.plot_algiers_time_bar(df_all, RESULTS_DIR)

    # Score vs time scatter
    P.plot_score_vs_time_scatter(df_all, RESULTS_DIR)

    # Heatmap
    P.plot_heatmap(df_all, RESULTS_DIR)

    # -- LaTeX table -----------------------------------------------------------
    print("\nGenerating LaTeX table ...")
    generate_latex_table(df_all, RESULTS_DIR / "results_table.tex")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
