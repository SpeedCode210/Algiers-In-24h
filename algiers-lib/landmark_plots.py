from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Final

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

__all__ = [
    "load_landmark_data",
    "get_unique_landmarks",
    "count_landmarks_per_category",
    "plot_landmark_category_counts",
    "plot_interest_score_distribution",
    "plot_visit_duration_by_category",
    "plot_availability_heatmap",
]

CATEGORY_ORDER: Final[list[str]] = [
    "historical",
    "attraction",
    "tradition_art",
    "religious",
    "shopping",
]

CATEGORY_LABELS: Final[dict[str, str]] = {
    "historical": "Historical",
    "attraction": "Attraction",
    "tradition_art": "Tradition & Art",
    "religious": "Religious",
    "shopping": "Shopping",
}

CATEGORY_COLORS: Final[dict[str, str]] = {
    "historical": "#4C72B0",
    "attraction": "#55A868",
    "tradition_art": "#C44E52",
    "religious": "#DD8452",
    "shopping": "#8172B2",
    "unknown": "#999999",
}

DAY_ORDER: Final[list[str]] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

DAY_LABELS: Final[list[str]] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_DPI: Final[int] = 300
_TITLE_FONTSIZE: Final[int] = 16
_AXIS_LABEL_FONTSIZE: Final[int] = 13
_TICK_FONTSIZE: Final[int] = 11
_ANNOTATION_FONTSIZE: Final[int] = 11



def load_landmark_data(csv_path: str | Path) -> pd.DataFrame:
    """Load landmark data from *csv_path* and normalise column names.

    Args:
        csv_path: Path to the landmark CSV file.

    Returns:
        Normalised landmark data.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_unique_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated DataFrame with one row per landmark ID.

    Args:
        df: Landmark data that may contain duplicate rows.

    Returns:
        Deduplicated landmark data indexed by landmark ID.
    """
    return df.drop_duplicates(subset="id")


def count_landmarks_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Count unique landmarks per category, preserving *CATEGORY_ORDER*.

    Args:
        df: Landmark data to count by category.

    Returns:
        Category counts for unique landmarks, ordered by *CATEGORY_ORDER*.
    """
    unique = get_unique_landmarks(df)
    counts = (
        unique.groupby("category")
        .size()
        .reindex(CATEGORY_ORDER, fill_value=0)
        .reset_index(name="count")
    )
    return counts




def _numeric_column(series: pd.Series) -> np.ndarray:
    """Coerce *series* to float, drop NaN values, and return a NumPy array.

    Args:
        series: Series of values that should be converted to numeric.

    Returns:
        Numeric values from the series as a float array.
    """
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def _kernel_density_estimate(values: np.ndarray, x: np.ndarray, bandwidth: float = 0.3) -> np.ndarray:
    """Estimate a Gaussian kernel density on the specified grid.

    Args:
        values: Data values used to build the kernel density.
        x: Points at which to evaluate the density.
        bandwidth: Smoothing bandwidth for the Gaussian kernel.

    Returns:
        Density values evaluated at *x*.
    """
    if values.size == 0:
        return np.zeros_like(x)

    diff = x[:, None] - values[None, :]
    kernels = np.exp(-0.5 * (diff / bandwidth) ** 2)
    return np.sum(kernels, axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))


def _save_or_show(
    fig: Figure,
    output_path: str | Path | None,
    show: bool,
) -> None:
    """Save *fig* to *output_path* and/or display it, then close.

    Args:
        fig: Figure object to save or display.
        output_path: Destination path for the saved figure. If None, the figure is not saved.
        show: Whether to display the figure after generating it.
    """
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=_DPI)
        print(f"  ✓  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)




def plot_landmark_category_counts(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot a bar chart of unique landmark counts per category.

    Args:
        df: Landmark data used to compute category counts.
        output_path: Path to save the generated figure. If None, the figure is not saved.
        show: Whether to display the figure after creation.
    """
    counts = count_landmarks_per_category(df)
    labels: list[str] = [CATEGORY_LABELS.get(c, str(c)) for c in counts["category"]]
    values = counts["count"].tolist()
    colors = [CATEGORY_COLORS.get(c, CATEGORY_COLORS["unknown"]) for c in counts["category"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.6,
    )

    ax.set_title(
        f"Landmarks per category  (n = {sum(values)})",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Number of landmarks", fontsize=_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(value),
            ha="center",
            va="bottom",
            fontsize=_ANNOTATION_FONTSIZE,
            fontweight="bold",
        )

    fig.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_interest_score_distribution(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot a histogram with KDE overlay of landmark interest scores.

    Args:
        df: Landmark data from which unique interest scores are extracted.
        output_path: Path to save the generated figure. If None, the figure is not saved.
        show: Whether to display the figure after creation.
    """
    scores = _numeric_column(get_unique_landmarks(df)["interest_score"])
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))

    x = np.linspace(scores.min() - 0.5, scores.max() + 0.5, 300)
    density = _kernel_density_estimate(scores, x, bandwidth=0.3)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(
        scores,
        bins=10,
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
        label="_nolegend_",
    )
    ax.set_xlabel("Interest Score (1–10)", fontsize=_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of landmarks", fontsize=_AXIS_LABEL_FONTSIZE, color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0", labelsize=_TICK_FONTSIZE)
    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    (line_density,) = ax2.plot(
        x, density, color="#C44E52", linewidth=2.5, label="Density (KDE)"
    )
    ax2.set_ylabel("Density", fontsize=_AXIS_LABEL_FONTSIZE, color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52", labelsize=_TICK_FONTSIZE)
    ax2.set_ylim(0, density.max() * 1.2)

    mean_line = ax.axvline(mean_score, color="#A50F15", linestyle="--", linewidth=2)
    median_line = ax.axvline(median_score, color="#DAA520", linestyle=":", linewidth=2)

    ax.set_title(
        f"Distribution of Interest Scores  (n = {len(scores)})",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        pad=14,
    )
    ax.legend(
        [mean_line, median_line, line_density],
        [
            f"Mean = {mean_score:.2f}",
            f"Median = {median_score:.2f}",
            "Density (KDE)",
        ],
        frameon=False,
        fontsize=_ANNOTATION_FONTSIZE,
    )

    fig.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_visit_duration_by_category(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot boxplots of recommended visit durations grouped by category.

    Args:
        df: Landmark data used to compute visit duration distributions.
        output_path: Path to save the generated figure. If None, the figure is not saved.
        show: Whether to display the figure after creation.
    """
    unique = get_unique_landmarks(df)
    data = [
        _numeric_column(unique.loc[unique["category"] == cat, "visit_duration_minutes"]).tolist()
        for cat in CATEGORY_ORDER
    ]
    labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=5, linestyle="none", alpha=0.6),
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=_TICK_FONTSIZE)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(
        "Visit duration by category",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Recommended duration (minutes)", fontsize=_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_availability_heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot a heatmap of landmark availability by day of week and category.

    Args:
        df: Landmark data with day and category availability information.
        output_path: Path to save the generated figure. If None, the figure is not saved.
        show: Whether to display the figure after creation.
    """
    avail = df[["id", "day", "category"]].drop_duplicates()
    pivot = (
        avail.groupby(["day", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DAY_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    )

    matrix = pivot.to_numpy()
    threshold = matrix.max() * 0.65

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")

    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=_TICK_FONTSIZE
    )
    ax.set_yticks(range(len(DAY_ORDER)))
    ax.set_yticklabels(DAY_LABELS, fontsize=_TICK_FONTSIZE)

    for row_idx, row in enumerate(matrix):
        for col_idx, val in enumerate(row):
            text_color = "white" if val > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                str(val),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Number of landmarks open", fontsize=10)

    ax.set_title(
        "Landmark availability (day × category)",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    _save_or_show(fig, output_path, show)



_FIGURES: Final[list[tuple[str, Callable[..., None]]]] = [
    ("landmarks_per_category.png", plot_landmark_category_counts),
    ("interest_score_distribution.png", plot_interest_score_distribution),
    ("visit_duration_by_category.png", plot_visit_duration_by_category),
    ("availability_heatmap.png", plot_availability_heatmap),
]


def main() -> None:
    """Parse CLI arguments, load data, and generate all figures."""
    parser = argparse.ArgumentParser(
        description="Analyse and visualise Algiers landmark data."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to data.csv (default: <repo_root>/algiers-lib/data/data.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder for figures (default: <repo_root>/algiers-lib/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv) if args.csv else repo_root / "algiers-lib" / "data" / "data.csv"
    out_dir = Path(args.out) if args.out else repo_root / "algiers-lib"

    print(f"Loading: {csv_path}")
    df = load_landmark_data(csv_path)
    print(f"  -> {len(get_unique_landmarks(df))} unique landmarks loaded")

    print(f"\nGenerating {len(_FIGURES)} figures -> {out_dir}/")
    for filename, plot_fn in _FIGURES:
        plot_fn(df, output_path=out_dir / filename, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()