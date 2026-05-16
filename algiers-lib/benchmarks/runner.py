"""
runner.py -- Core benchmark execution engine.

Provides:
  - run_single_solver()  : time + execute one solver on one problem
  - run_group()          : run all variants on a list of (name, problem) pairs
                           (stochastic solvers run NUM_RUNS times)
  - load_algiers_problem(): load the Algiers CSV dataset
  - load_solomon_group() : load all .txt files from a benchmark directory
  - collect_all_results(): master function that runs everything and returns DataFrames

Stochastic solvers (GRASP, SA, Tabu, Genetic) are run NUM_RUNS times to account
for randomness.  Results are stored as individual rows (one per run) so that
aggregation functions can compute mean, std, best, worst.
Deterministic solvers (Greedy, CPLEX) run exactly once.

Parallel execution:
  When NUM_WORKERS > 1, tasks are distributed across CPU cores:
    - Linux/macOS: uses 'fork' -- child processes inherit parent memory,
      so solver factories (lambdas) need no pickling.
    - Windows:     uses 'spawn' + cloudpickle/dill to serialize solver
      factories and problem instances to worker processes.
  Install cloudpickle or dill for Windows parallel support:
      pip install cloudpickle
  Set NUM_WORKERS = 1 to force sequential execution.

Workflow for Solomon-50:
  Phase 1: Run CPLEX to completion on all instances -> print ground truth table
  Phase 2: Run all OTHER algorithms (stochastic ones x NUM_RUNS) -> compare vs CPLEX
Workflow for Solomon-100:
  Run ALL algorithms (including CPLEX-60s once, stochastic x NUM_RUNS) -> compare vs R&S
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Cross-platform serialization for multiprocessing
# ---------------------------------------------------------------------------
# On Linux/macOS, 'fork' copies memory directly (no pickling needed).
# On Windows, only 'spawn' is available, which requires pickling everything.
# Standard pickle CANNOT serialize lambda functions.  We detect cloudpickle
# or dill (both are drop-in pickle replacements that handle lambdas).

try:
    import cloudpickle as _pickle_ext          # preferred: pip install cloudpickle
    _HAS_PICKLE_EXT = True
    _PICKLE_EXT_NAME = "cloudpickle"
except ImportError:
    try:
        import dill as _pickle_ext             # fallback:  pip install dill
        _HAS_PICKLE_EXT = True
        _PICKLE_EXT_NAME = "dill"
    except ImportError:
        _pickle_ext = None
        _HAS_PICKLE_EXT = False
        _PICKLE_EXT_NAME = None

_IS_WINDOWS = platform.system() == "Windows"

# Ensure project root is on path
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from models.landmark import Day, loadLandmarks, loadHotel
from models.problem import Problem
from models.tour import Tour
from benchmarks.benchmark_problem import BenchmarkProblem
from benchmarks.ground_truth_parser import (
    parse_best_possible_dir,
    parse_solomon_50_ground_truth,
)


# ---------------------------------------------------------------------------
# Configuration constants (edit here to change scope)
# ---------------------------------------------------------------------------

# How many instances to sample per Solomon group (None = all)
MAX_PER_GROUP: dict[str, int | None] = {
    "Solomon-50":  5,   # run 5 representative instances
    "Solomon-100": 5,
    "Solomon-200": 5,
}

# Number of independent runs for each STOCHASTIC solver per instance.
# Deterministic solvers (Greedy, CPLEX) always run exactly once.
NUM_RUNS: int = 5

# Number of CPU cores for parallel execution (1 = sequential mode).
# Defaults to the number of physical cores (os.cpu_count() on most systems).
# On your 8-core/16-thread machine, this will be 8.
NUM_WORKERS: int = os.cpu_count() or 4

# Set to True to override MAX_PER_GROUP and run everything
RUN_FULL: bool = False


# ---------------------------------------------------------------------------
# Single solver runner
# ---------------------------------------------------------------------------

def run_single_solver(
    problem: Problem,
    entry: dict[str, Any],
    ground_truth: float | None = None,
) -> dict[str, Any]:
    """Run one solver variant on one problem instance and return a result dict.

    Args:
        problem       : The problem instance to solve.
        entry         : Solver registry entry {"name": str, "factory": callable, ...}.
        ground_truth  : Known optimal score for optimality-gap computation (optional).

    Returns:
        Dict with keys: solver, score, time_s, valid, landmarks_visited,
        total_duration, optimality_gap_pct, run_id.
    """
    name = entry["name"]
    result: dict[str, Any] = {
        "solver":             name,
        "score":              0.0,
        "time_s":             0.0,
        "valid":              False,
        "landmarks_visited":  0,
        "total_duration":     0.0,
        "optimality_gap_pct": None,
        "error":              None,
    }

    t0 = time.perf_counter()
    try:
        solver = entry["factory"](problem)
        tour: Tour = solver.solve()
        result["time_s"] = time.perf_counter() - t0

        if tour is not None:
            sim = tour.simulation_cache()
            result["score"]             = tour.total_score()
            result["valid"]             = sim.is_valid
            result["landmarks_visited"] = len(tour.visited_landmarks)
            result["total_duration"]    = sim.total_duration

        if ground_truth and ground_truth > 0 and result["score"] is not None:
            gap = (ground_truth - result["score"]) / ground_truth * 100.0
            result["optimality_gap_pct"] = round(max(gap, 0.0), 2)

    except Exception as exc:
        result["error"] = str(exc)
        result["time_s"] = time.perf_counter() - t0
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Multiprocessing parallel execution support (cross-platform)
# ---------------------------------------------------------------------------

_mp_state: dict[str, Any] = {}


def _mp_init_worker(serialized_state: bytes) -> None:
    """Initialize a spawned worker process with the shared benchmark state.

    Called once per worker on Windows (spawn) or macOS.  The state dict
    (containing problem instances, solver factories with lambdas, etc.)
    was serialized with cloudpickle/dill in the main process and sent
    as raw bytes.  We deserialize it here to reconstruct the objects.
    """
    global _mp_state
    if _pickle_ext is not None:
        _mp_state = _pickle_ext.loads(serialized_state)
    else:
        import pickle as _std_pickle
        _mp_state = _std_pickle.loads(serialized_state)


def _mp_run_task(
    task_key: tuple[int, int, int],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    """Run a single (instance, solver, run_id) in a worker process.

    On Linux/macOS (fork): _mp_state is inherited from the parent --
    no deserialization needed.

    On Windows (spawn): _mp_state was set up by _mp_init_worker() using
    cloudpickle/dill deserialization.

    Returns:
        (result_dict, task_key) tuple so the main process can track progress.
    """
    inst_idx, var_idx, run_id = task_key
    st = _mp_state
    inst_name, problem = st["instances"][inst_idx]
    entry = st["variants"][var_idx]
    gt = st["ground_truths"].get(inst_name)

    rec = run_single_solver(problem, entry, ground_truth=gt)
    rec["instance"]    = inst_name
    rec["group"]       = st["group_label"]
    rec["n_landmarks"] = len(problem.landmarks)
    rec["time_budget"] = problem.time_budget
    rec["run_id"]      = run_id
    return rec, task_key


# ---------------------------------------------------------------------------
# Group runner (with multi-run for stochastic solvers)
# ---------------------------------------------------------------------------

def run_group(
    instances: list[tuple[str, Problem]],
    variants: list[dict[str, Any]],
    ground_truths: dict[str, float] | None = None,
    group_label: str = "",
    num_runs: int = NUM_RUNS,
    num_workers: int = NUM_WORKERS,
) -> pd.DataFrame:
    """Run all variants on all instances in a group.

    Stochastic solvers (entry["stochastic"] == True) are run *num_runs* times
    per instance.  Deterministic solvers run exactly once.

    When num_workers > 1, tasks are distributed across CPU cores:
      - Linux/macOS: 'fork' context (fast, no pickling)
      - Windows:     'spawn' context + cloudpickle/dill serialization

    Each individual run is stored as a separate row in the DataFrame.
    A "run_id" column distinguishes repeated runs (0-based).

    Args:
        instances     : List of (instance_name, Problem) tuples.
        variants      : List of solver registry entries.
        ground_truths : Optional dict mapping instance_name -> optimal score.
        group_label   : Label string added to every row (e.g. "Solomon-50").
        num_runs      : Number of runs for stochastic solvers (default: NUM_RUNS).
        num_workers   : Number of CPU cores for parallel execution (default: NUM_WORKERS).
                        Set to 1 to force sequential execution.

    Returns:
        pd.DataFrame with one row per (instance, solver, run) combination.
    """
    ground_truths = ground_truths or {}
    rows: list[dict[str, Any]] = []

    # -- Build flat task list: (inst_idx, var_idx, run_id) -----------------
    task_keys: list[tuple[int, int, int]] = []
    for i, (_inst_name, _prob) in enumerate(instances):
        for j, entry in enumerate(variants):
            n = num_runs if entry.get("stochastic", False) else 1
            for run_id in range(n):
                task_keys.append((i, j, run_id))
    total_tasks = len(task_keys)

    # -- Try parallel execution -----------------------------------------------
    if num_workers > 1 and total_tasks > 1:
        global _mp_state
        _mp_state = {
            "instances": instances,
            "variants": variants,
            "ground_truths": ground_truths,
            "group_label": group_label,
        }

        # Determine best multiprocessing start method
        use_spawn = False
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            # fork not available (Windows) -- use spawn
            use_spawn = True
            ctx = multiprocessing.get_context("spawn")

        if use_spawn and not _HAS_PICKLE_EXT:
            # On Windows without cloudpickle/dill, we cannot serialize lambdas.
            print(f"  [WARN] Windows detected but neither cloudpickle nor dill is installed.")
            print(f"         Solver factories (lambdas) cannot be serialized.")
            print(f"         Install one of them for parallel execution:")
            print(f"           pip install cloudpickle")
            print(f"         Falling back to sequential execution.\n")
        else:
            method_label = "spawn" if use_spawn else "fork"
            print(f"  Parallel mode ({method_label}): {total_tasks} tasks on "
                  f"{num_workers} CPU cores")
            try:
                if use_spawn:
                    # Serialize state with cloudpickle/dill, pass as bytes to workers.
                    # bytes are always picklable by standard pickle, so this round-trip works:
                    #   main: cloudpickle.dumps(state) -> bytes
                    #   multiprocessing sends bytes to worker via standard pickle
                    #   worker: cloudpickle.loads(bytes) -> state with live lambdas
                    state_bytes = _pickle_ext.dumps(_mp_state)
                    pool_kwargs = {
                        "processes": num_workers,
                        "initializer": _mp_init_worker,
                        "initargs": (state_bytes,),
                    }
                else:
                    # fork: child inherits parent memory, no serialization needed
                    pool_kwargs = {"processes": num_workers}

                with ctx.Pool(**pool_kwargs) as pool:
                    for done, (rec, (ii, jj, rid)) in enumerate(
                        pool.imap_unordered(_mp_run_task, task_keys), 1
                    ):
                        inst_name = instances[ii][0]
                        entry = variants[jj]
                        is_stoch = entry.get("stochastic", False)
                        n_runs = num_runs if is_stoch else 1
                        rl = f" (run {rid+1}/{n_runs})" if is_stoch else ""
                        status = f"score={rec['score']:.1f}  t={rec['time_s']:.2f}s"
                        if rec["optimality_gap_pct"] is not None:
                            status += f"  gap={rec['optimality_gap_pct']:.2f}%"
                        elif rec.get("error"):
                            status = f"ERROR: {rec['error'][:60]}"
                        print(f"  [{done}/{total_tasks}] {group_label} | "
                              f"{inst_name} | {entry['name']}{rl} ... {status}",
                              flush=True)
                        rows.append(rec)
                return pd.DataFrame(rows)
            except Exception as exc:
                print(f"  [WARN] Multiprocessing error: {exc}")
                print(f"  Falling back to sequential execution.\n")

    # -- Sequential execution (original behaviour) --------------------------
    done = 0
    for inst_name, problem in instances:
        gt = ground_truths.get(inst_name)
        for entry in variants:
            is_stochastic = entry.get("stochastic", False)
            runs_to_do = num_runs if is_stochastic else 1

            for run_id in range(runs_to_do):
                done += 1
                run_label = f" (run {run_id+1}/{runs_to_do})" if is_stochastic else ""
                print(
                    f"  [{done}/{total_tasks}] {group_label} | {inst_name} "
                    f"| {entry['name']}{run_label} ...",
                    end=" ", flush=True,
                )
                rec = run_single_solver(problem, entry, ground_truth=gt)
                rec["instance"]    = inst_name
                rec["group"]       = group_label
                rec["n_landmarks"] = len(problem.landmarks)
                rec["time_budget"] = problem.time_budget
                rec["run_id"]      = run_id
                rows.append(rec)

                status = f"score={rec['score']:.1f}  t={rec['time_s']:.2f}s"
                if rec["optimality_gap_pct"] is not None:
                    status += f"  gap={rec['optimality_gap_pct']:.2f}%"
                if rec["error"]:
                    status = f"ERROR: {rec['error'][:60]}"
                print(status)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-group summary printer
# ---------------------------------------------------------------------------

def print_group_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table for a DataFrame of benchmark results.

    For stochastic solvers (multiple runs), shows mean +/- std.
    For deterministic solvers (single run), shows the single value.
    """
    if df.empty:
        return

    # Aggregate: for each (instance, solver), compute stats
    agg = df.groupby(["instance", "solver"]).agg(
        n_runs=("run_id", "count"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        best_score=("score", "max"),
        worst_score=("score", "min"),
        mean_time=("time_s", "mean"),
        mean_gap=("optimality_gap_pct", "mean"),
        std_gap=("optimality_gap_pct", "std"),
        valid_pct=("valid", lambda x: 100 * x.mean()),
    ).reset_index()

    print(f"\n  {'Instance':<12} {'Solver':<22} {'Runs':>4} "
          f"{'Score':>14} {'Best':>8} {'Time(s)':>8} {'Gap%':>8} {'Valid':>6}")
    print(f"  {'-'*12} {'-'*22} {'-'*4} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    prev_inst = None
    for _, row in agg.iterrows():
        if row["instance"] != prev_inst:
            if prev_inst is not None:
                print()
            prev_inst = row["instance"]

        # Score column: mean +/- std if multiple runs, otherwise single value
        if row["n_runs"] > 1 and pd.notna(row["std_score"]):
            score_str = f"{row['mean_score']:.1f} +/- {row['std_score']:.1f}"
        else:
            score_str = f"{row['mean_score']:.1f}"

        # Gap column
        if pd.notna(row["mean_gap"]):
            if row["n_runs"] > 1 and pd.notna(row["std_gap"]):
                gap_str = f"{row['mean_gap']:.1f} +/- {row['std_gap']:.1f}"
            else:
                gap_str = f"{row['mean_gap']:.1f}"
        else:
            gap_str = "---"

        print(f"  {row['instance']:<12} {row['solver']:<22} "
              f"{int(row['n_runs']):>4} {score_str:>14} "
              f"{row['best_score']:>8.1f} {row['mean_time']:>8.2f} "
              f"{gap_str:>8} {row['valid_pct']:>5.0f}%")

    print()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_algiers_problem() -> tuple[str, Problem]:
    """Load the Algiers CSV dataset.

    Returns:
        ("Algiers", Problem) tuple.
    """
    data_dir = _ROOT / "data"
    hotel     = loadHotel(str(data_dir / "hotel.csv"))
    landmarks = loadLandmarks(str(data_dir / "data.csv"))
    problem   = Problem(
        hotel=hotel,
        landmarks=landmarks,
        time_budget=480,      # 8-hour day
        tour_day=Day.SATURDAY,
        start_time=540,       # 9:00 AM
    )
    return "Algiers", problem


def _pick_instances(paths: list[Path], max_n: int | None) -> list[Path]:
    """Select a representative subset: prefer one from each type (c, r, rc)."""
    if max_n is None or RUN_FULL:
        return sorted(paths)

    buckets: dict[str, list[Path]] = {"c": [], "r": [], "rc": []}
    for p in sorted(paths):
        stem = p.stem.lstrip("0123456789_")   # strip leading digits / underscores
        if stem.startswith("rc"):
            buckets["rc"].append(p)
        elif stem.startswith("r"):
            buckets["r"].append(p)
        elif stem.startswith("c"):
            buckets["c"].append(p)

    selected: list[Path] = []
    # Round-robin across buckets until we reach max_n
    iters = [iter(v) for v in buckets.values() if v]
    while len(selected) < max_n:
        advanced = False
        for it in iters:
            if len(selected) >= max_n:
                break
            try:
                selected.append(next(it))
                advanced = True
            except StopIteration:
                pass
        if not advanced:
            break

    return selected


def load_solomon_group(
    directory: str | Path,
    max_instances: int | None = None,
) -> list[tuple[str, BenchmarkProblem]]:
    """Load Solomon OPTW .txt files from a directory.

    Args:
        directory     : Path to the dataset directory.
        max_instances : Maximum number of instances to load (None = all).

    Returns:
        List of (instance_name, BenchmarkProblem) tuples.
    """
    directory = Path(directory)
    paths = list(directory.glob("*.txt"))
    if not paths:
        print(f"  [WARN] No .txt files found in {directory}")
        return []

    chosen = _pick_instances(paths, max_instances)
    result = []
    for p in chosen:
        # Normalize name: strip leading "50_" / "100_" prefix if present
        stem = p.stem
        for prefix in ("50_", "100_", "200_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        try:
            prob = BenchmarkProblem.parse_solomon_file(p)
            result.append((stem, prob))
        except Exception as exc:
            print(f"  [WARN] Failed to parse {p.name}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Ground truth loaders
# ---------------------------------------------------------------------------

def load_s50_ground_truth() -> dict[str, float]:
    """Load Righini & Salani optimal ground truth for Solomon-50 instances.

    Searches for *.dssr / *.dssr.conservative files in:
        benchmarks/ground_truth/Solomon_OPT/optimal50/
    """
    gt_dir = _HERE / "ground_truth" / "Solomon_OPT" / "optimal50"
    if gt_dir.exists():
        gt = parse_solomon_50_ground_truth(gt_dir)
        print(f"  Loaded {len(gt)} Solomon-50 ground truth entries from {gt_dir.name}/")
        return gt
    # Fallback: try bestPossible/ (may contain 50-node files too)
    gt_dir2 = _HERE / "ground_truth" / "Solomon_OPT" / "bestPossible"
    if gt_dir2.exists():
        gt = parse_best_possible_dir(gt_dir2)
        # Filter to only 50-node instances if possible
        print(f"  Loaded {len(gt)} ground truth entries from {gt_dir2.name}/ (fallback)")
        return gt
    print("  [WARN] No Solomon-50 ground truth directory found.")
    return {}


def load_s100_ground_truth() -> dict[str, float]:
    """Load best-known ground truth for Solomon-100 instances.

    Searches for *.dssr.conservative files in:
        benchmarks/ground_truth/Solomon_OPT/bestPossible/
    """
    gt_dir = _HERE / "ground_truth" / "Solomon_OPT" / "bestPossible"
    if gt_dir.exists():
        gt = parse_best_possible_dir(gt_dir)
        print(f"  Loaded {len(gt)} Solomon-100 ground truth entries from bestPossible/")
        return gt
    print("  [WARN] No Solomon-100 ground truth directory found.")
    return {}


# ---------------------------------------------------------------------------
# CPLEX pre-pass helper (Solomon-50 only)
# ---------------------------------------------------------------------------

def run_cplex_ground_truth(
    instances: list[tuple[str, Any]],
    cplex_variant: dict[str, Any],
    group_label: str = "Solomon-50",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run CPLEX to completion on all instances to establish ground truth.

    Prints a formatted table of CPLEX results to the terminal.
    Does NOT run CPLEX again during the solver comparison phase.

    Args:
        instances     : List of (instance_name, Problem) tuples.
        cplex_variant : Solver registry entry for CPLEX.
        group_label   : Label for terminal output.

    Returns:
        (ground_truths, df_cplex) where:
          - ground_truths: dict mapping instance_name -> optimal score
          - df_cplex: DataFrame of CPLEX results (for inclusion in plots)
    """
    live_gt: dict[str, float] = {}
    cplex_rows: list[dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  PHASE 1: {group_label} -- CPLEX Ground Truth (run to completion)")
    print(f"{'='*60}")
    print(f"  Running CPLEX on {len(instances)} instances (1 run each, deterministic) ...")
    print()

    for idx, (inst_name, prob) in enumerate(instances, 1):
        print(f"  [{idx}/{len(instances)}] CPLEX | {inst_name} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            solver = cplex_variant["factory"](prob)
            # Run CPLEX to completion (no time limit on 50-node)
            tour = solver.solve()
            elapsed = time.perf_counter() - t0

            if tour is not None:
                sim = tour.simulation_cache()
                score = tour.total_score()
                live_gt[inst_name] = score
                cplex_rows.append({
                    "solver":             cplex_variant["name"],
                    "score":              score,
                    "time_s":             elapsed,
                    "valid":              sim.is_valid,
                    "landmarks_visited":  len(tour.visited_landmarks),
                    "total_duration":     sim.total_duration,
                    "optimality_gap_pct": 0.0,  # CPLEX is the ground truth
                    "error":              None,
                    "instance":           inst_name,
                    "group":              group_label,
                    "n_landmarks":        len(prob.landmarks),
                    "time_budget":        prob.time_budget,
                    "run_id":             0,
                })
                print(f"optimal = {score:.0f}  |  landmarks = {len(tour.visited_landmarks)}"
                      f"  |  time = {elapsed:.2f}s")
            else:
                print("FAILED: no tour returned")
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()

    # Print summary table
    print()
    print(f"  {'Instance':<15} {'CPLEX Optimal':>15} {'Landmarks':>10} {'Time (s)':>10}")
    print(f"  {'-'*15} {'-'*15} {'-'*10} {'-'*10}")
    for inst_name, prob in instances:
        score = live_gt.get(inst_name)
        if score is not None:
            cplex_row = next((r for r in cplex_rows if r["instance"] == inst_name), None)
            lm = cplex_row["landmarks_visited"] if cplex_row else "?"
            t = cplex_row["time_s"] if cplex_row else "?"
            print(f"  {inst_name:<15} {score:>15.0f} {lm:>10} {t:>10.2f}")
        else:
            print(f"  {inst_name:<15} {'FAILED':>15}")

    print(f"\n  Ground truth established for {len(live_gt)}/{len(instances)} instances.")
    print(f"{'='*60}\n")

    return live_gt, pd.DataFrame(cplex_rows)


# ---------------------------------------------------------------------------
# Master collection function
# ---------------------------------------------------------------------------

def collect_all_results(
    run_algiers:  bool = True,
    run_s50:      bool = True,
    run_s100:     bool = True,
    run_s200:     bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full benchmark suite and return results as DataFrames.

    Solomon-50 workflow:
      Phase 1 -- CPLEX runs to completion (1 run, deterministic) -> ground truth
      Phase 2 -- All OTHER algorithms run:
                 - Stochastic solvers: NUM_RUNS times each
                 - Deterministic solvers (Greedy): 1 run
                 Results compared against CPLEX ground truth

    Solomon-100 workflow:
      All algorithms run against Righini & Salani bestPossible ground truth:
      - CPLEX-60s: 1 run (deterministic)
      - Stochastic solvers: NUM_RUNS times each
      - Greedy: 1 run (deterministic)

    Algiers workflow:
      Same as Solomon-100 (CPLEX-60s as ground truth, stochastic x NUM_RUNS)

    Returns a dict with keys:
        "algiers"    -> DataFrame of Algiers results
        "solomon_50" -> DataFrame of Solomon-50 results
        "solomon_100"-> DataFrame of Solomon-100 results
        "solomon_200"-> DataFrame of Solomon-200 results
        "all"        -> Concatenated DataFrame of everything
    """
    # Lazy import to avoid circular imports at module level
    from benchmarks.solver_registry import (
        ALGIERS_VARIANTS,
        SOLOMON_50_VARIANTS,
        SOLOMON_100_VARIANTS,
        SOLOMON_200_VARIANTS,
    )

    # Count stochastic vs deterministic for info
    def _count(variants, label=""):
        det = sum(1 for v in variants if not v.get("stochastic", False))
        sto = sum(1 for v in variants if v.get("stochastic", False))
        print(f"  [{label}] {len(variants)} variants: "
              f"{det} deterministic, {sto} stochastic (x{NUM_RUNS} runs each)")

    frames: dict[str, pd.DataFrame] = {}

    # -- Algiers ---------------------------------------------------------------
    if run_algiers:
        print("\n" + "=" * 60)
        print("GROUP: Algiers dataset (parameter sensitivity)")
        print(f"  Multi-run: stochastic solvers x{NUM_RUNS}, deterministic x1")
        print(f"  Parallel:  {NUM_WORKERS} CPU cores")
        print("=" * 60)
        _count(ALGIERS_VARIANTS, "Algiers")
        name, problem = load_algiers_problem()
        print(f"  Loaded: {len(problem.landmarks)} landmarks, budget={problem.time_budget} min")
        df_alg = run_group(
            [(name, problem)],
            ALGIERS_VARIANTS,
            group_label="Algiers",
        )
        frames["algiers"] = df_alg
        print_group_summary(df_alg)

    # -- Solomon-50 (CPLEX as ground truth, CPLEX excluded from Phase 2) -----
    if run_s50:
        s50_dir = _HERE / "datasets" / "c_r_rc_100_50"
        instances_50 = load_solomon_group(s50_dir, MAX_PER_GROUP.get("Solomon-50"))
        print(f"  Loaded {len(instances_50)} instances from {s50_dir.name}")

        # Identify the CPLEX variant from the registry
        cplex_variant = next(
            (v for v in SOLOMON_50_VARIANTS if "CPLEX" in v["name"]), None
        )

        if cplex_variant:
            # PHASE 1: Run CPLEX to completion -> ground truth
            live_gt_50, df_cplex_50 = run_cplex_ground_truth(
                instances_50, cplex_variant, group_label="Solomon-50"
            )

            if live_gt_50:
                # PHASE 2: Run all solvers EXCEPT CPLEX, using CPLEX as ground truth
                s50_solvers = [v for v in SOLOMON_50_VARIANTS
                               if "CPLEX" not in v["name"]]

                print(f"{'='*60}")
                print(f"  PHASE 2: Solomon-50 -- Heuristic Algorithms vs CPLEX")
                print(f"  Multi-run: stochastic solvers x{NUM_RUNS}, deterministic x1")
                print(f"  Parallel:  {NUM_WORKERS} CPU cores")
                print(f"{'='*60}")
                _count(s50_solvers, "S50 Phase 2")
                print(f"  (CPLEX already completed in Phase 1 -- excluded from this phase)")
                print()

                df_50 = run_group(
                    instances_50,
                    s50_solvers,
                    ground_truths=live_gt_50,
                    group_label="Solomon-50",
                )

                # Append CPLEX rows so it appears in plots/tables as ground truth
                df_50 = pd.concat([df_cplex_50, df_50], ignore_index=True)
                frames["solomon_50"] = df_50
                print_group_summary(df_50)
            else:
                # CPLEX failed on all instances -- fall back to file-based GT
                print("  [WARN] CPLEX failed on all instances. "
                      "Falling back to file-based ground truth.")
                live_gt_50 = load_s50_ground_truth()
                df_50 = run_group(
                    instances_50,
                    SOLOMON_50_VARIANTS,
                    ground_truths=live_gt_50,
                    group_label="Solomon-50",
                )
                frames["solomon_50"] = df_50
                print_group_summary(df_50)
        else:
            # No CPLEX available -- fall back to file-based ground truth
            print("\n  [WARN] CPLEX not available. Falling back to file-based ground truth.")
            live_gt_50 = load_s50_ground_truth()
            df_50 = run_group(
                instances_50,
                SOLOMON_50_VARIANTS,
                ground_truths=live_gt_50,
                group_label="Solomon-50",
            )
            frames["solomon_50"] = df_50
            print_group_summary(df_50)

    # -- Solomon-100 (ALL solvers including CPLEX-60s, R&S ground truth) -----
    if run_s100:
        print(f"\n{'='*60}")
        print(f"GROUP: Solomon-100 (Righini & Salani bestPossible ground truth)")
        print(f"  Multi-run: stochastic solvers x{NUM_RUNS}, deterministic x1")
        print(f"  Parallel:  {NUM_WORKERS} CPU cores")
        print(f"{'='*60}")
        s100_dir = _HERE / "datasets" / "c_r_rc_100_100"
        gt_100 = load_s100_ground_truth()

        instances_100 = load_solomon_group(s100_dir, MAX_PER_GROUP.get("Solomon-100"))
        print(f"  Loaded {len(instances_100)} instances from {s100_dir.name}")
        _count(SOLOMON_100_VARIANTS, "S100")
        print(f"  Running ALL solvers (including CPLEX-60s x1) against "
              f"{len(gt_100)} ground truth entries")

        df_100 = run_group(
            instances_100,
            SOLOMON_100_VARIANTS,   # includes CPLEX-60s (deterministic)
            ground_truths=gt_100,
            group_label="Solomon-100",
        )
        frames["solomon_100"] = df_100
        print_group_summary(df_100)

    # -- Solomon-200 ----------------------------------------------------------
    if run_s200:
        print(f"\n{'='*60}")
        print(f"GROUP: Solomon-200 (single best variant per solver)")
        print(f"  Multi-run: stochastic solvers x{NUM_RUNS}, deterministic x1")
        print(f"{'='*60}")
        s200_dir = _HERE / "datasets" / "c_r_rc_200_100"
        instances_200 = load_solomon_group(s200_dir, MAX_PER_GROUP.get("Solomon-200"))
        print(f"  Loaded {len(instances_200)} instances from {s200_dir.name}")
        _count(SOLOMON_200_VARIANTS, "S200")

        df_200 = run_group(
            instances_200,
            SOLOMON_200_VARIANTS,
            group_label="Solomon-200",
        )
        frames["solomon_200"] = df_200
        print_group_summary(df_200)

    # -- Concatenate all ------------------------------------------------------
    non_empty = [df for df in frames.values() if not df.empty]
    if non_empty:
        frames["all"] = pd.concat(non_empty, ignore_index=True)
    else:
        frames["all"] = pd.DataFrame()

    return frames
