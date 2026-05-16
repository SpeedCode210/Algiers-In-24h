"""
Ground truth parser for Solomon OPTW benchmark instances.

Parses *.dssr.conservative / *.dssr files from Righini & Salani benchmark
directories to extract the known optimal (or best-known) prize for each instance.

Supported directories:
    ground_truth/Solomon_OPT/bestPossible/   (100-node instances)
    ground_truth/Solomon_OPT/optimal50/       (50-node instances)
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Generic parsers
# ---------------------------------------------------------------------------

def _parse_prize_from_text(text: str) -> list[float]:
    """Extract all Prize:N values from a ground-truth text blob."""
    return [float(m) for m in re.findall(r"Prize\s*:\s*(\d+)", text)]


def parse_best_possible_file(filepath: str | Path) -> float | None:
    """Parse a single *.dssr.conservative file and return the best prize.

    Args:
        filepath: Path to a single .dssr.conservative file.

    Returns:
        Best prize as float, or None if not found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    text = filepath.read_text(errors="ignore")
    prizes = _parse_prize_from_text(text)
    return max(prizes) if prizes else None


def parse_best_possible_dir(directory: str | Path) -> dict[str, float]:
    """Parse all *.dssr.conservative files in a directory.

    Each file contains lines like::

        Prize:320  Customers:10

    We extract the Prize value for the "Best Feasible Path" section.
    If multiple such lines exist (conservative vs full), we take the max.

    Args:
        directory: Path to the bestPossible/ directory.

    Returns:
        Dict mapping instance name (e.g. "c101") to best-known prize (float).
        Returns empty dict if directory doesn't exist.
    """
    directory = Path(directory)
    if not directory.exists():
        return {}

    results: dict[str, float] = {}

    for fpath in sorted(directory.glob("*.dssr.conservative")):
        # Stem example: "c101_100.dssr.conservative" -> instance = "c101"
        stem = fpath.stem  # e.g. "c101_100.dssr"
        instance_name = stem.split("_")[0]  # "c101"

        text = fpath.read_text(errors="ignore")
        prizes = _parse_prize_from_text(text)
        if prizes:
            results[instance_name] = max(prizes)

    return results


# ---------------------------------------------------------------------------
# Solomon-50 ground truth (Righini & Salani optimal)
# ---------------------------------------------------------------------------

def parse_solomon_50_ground_truth(directory: str | Path) -> dict[str, float]:
    """Parse ground-truth files for 50-node Solomon OPTW instances.

    Looks for ``*.dssr`` or ``*.dssr.conservative`` files inside *directory*
    (the Righini & Salani optimal solution archive for n=50).

    Expected file naming convention::

        c101_50.dssr            -> instance "c101"
        r101_50.dssr            -> instance "r101"
        rc101_50.dssr           -> instance "rc101"
        c101_50.dssr.conservative  -> instance "c101"

    Args:
        directory: Path to the directory containing 50-node ground-truth files.

    Returns:
        Dict mapping instance name (e.g. "c101") to optimal prize (float).
    """
    directory = Path(directory)
    if not directory.exists():
        return {}

    results: dict[str, float] = {}

    # Accept both .dssr and .dssr.conservative extensions
    for fpath in sorted(directory.glob("*.dssr*")):
        # Skip if it is a directory or if the name doesn't match expected pattern
        if fpath.is_dir():
            continue

        stem = fpath.name  # e.g. "c101_50.dssr.conservative"

        # Remove the .dssr.conservative / .dssr suffix
        base = stem.replace(".dssr.conservative", "").replace(".dssr", "")
        # Remove the "_50" size suffix if present
        instance_name = re.sub(r"_\d+$", "", base)  # "c101"

        text = fpath.read_text(errors="ignore")
        prizes = _parse_prize_from_text(text)
        if prizes:
            results[instance_name] = max(prizes)

    return results


# ---------------------------------------------------------------------------
# Convenience loader that searches standard paths
# ---------------------------------------------------------------------------

def load_all_ground_truth(base_dir: str | Path) -> dict[str, dict[str, float]]:
    """Load ground truth from all standard sub-directories.

    Args:
        base_dir: The ``benchmarks/`` directory (or any parent that contains
                  a ``ground_truth/`` folder).

    Returns:
        Dict with keys ``"solomon_50"`` and ``"solomon_100"``, each mapping
        instance names to their optimal / best-known prize.
    """
    base_dir = Path(base_dir)
    gt_dir = base_dir / "ground_truth" / "Solomon_OPT"

    all_gt: dict[str, dict[str, float]] = {}

    # Solomon-50 optimal (Righini & Salani)
    s50_path = gt_dir / "optimal50"
    if s50_path.exists():
        all_gt["solomon_50"] = parse_solomon_50_ground_truth(s50_path)
    else:
        all_gt["solomon_50"] = {}

    # Solomon-100 bestPossible
    s100_path = gt_dir / "bestPossible"
    if s100_path.exists():
        all_gt["solomon_100"] = parse_best_possible_dir(s100_path)
    else:
        all_gt["solomon_100"] = {}

    return all_gt
