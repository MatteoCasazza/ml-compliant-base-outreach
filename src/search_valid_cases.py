"""
search_valid_cases.py
=====================

Exploratory search for physically feasible extra-reach cases using the true
dynamic simulator.

Purpose
-------
This script is not part of the ML training pipeline. It is used before or during
dataset design to understand which outreach targets are physically realistic.

The script:
1. samples the parameter space using Latin Hypercube Sampling;
2. runs the true dynamic simulator;
3. identifies feasible extra-reach cases;
4. saves the full search results and the best feasible candidates.

Main objective
--------------
Find parameter combinations that increase peak_y while keeping the robot
relative displacement within its admissible limit.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

from dynamics import simulate_system


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "search_valid_cases"
FULL_RESULTS_PATH = RESULTS_DIR / "targeted_search_results.csv"
BEST_RESULTS_PATH = RESULTS_DIR / "best_feasible_extra_reach_cases.csv"
SUMMARY_PATH = RESULTS_DIR / "targeted_search_summary.csv"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_N_SAMPLES = 2000
DEFAULT_T_SIM = 60.0
DEFAULT_DT = 0.001
DEFAULT_X_R_MAX = 0.500
DEFAULT_TOLERANCE = 0.002
DEFAULT_SEED = 123
DEFAULT_N_JOBS = -1
DEFAULT_TOP_K = 10

PARAM_NAMES = [
    "Kb",
    "Kr",
    "Mb",
    "hb",
    "hr",
    "f0",
    "f1",
    "A",
    "x_r_start",
]

PARAM_BOUNDS = {
    "Kb": (100.0, 1500.0),
    "Kr": (500.0, 5000.0),
    "Mb": (10.0, 80.0),
    "hb": (0.05, 0.25),
    "hr": (0.10, 0.80),
    "f0": (0.05, 0.50),
    "f1": (1.00, 8.00),
    "A": (0.03, 0.12),
    "x_r_start": (0.35, 0.45),
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds in PARAM_NAMES order."""
    lower_bounds = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=float)
    upper_bounds = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=float)

    return lower_bounds, upper_bounds


def sample_parameters(
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate parameter samples using Latin Hypercube Sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed.

    Returns
    -------
    ndarray
        Sampled parameter matrix with shape (n_samples, n_parameters).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    lower_bounds, upper_bounds = get_bounds()

    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    samples_unit = sampler.random(n=n_samples)
    samples_physical = qmc.scale(samples_unit, lower_bounds, upper_bounds)

    return samples_physical


def failed_row(params_array: np.ndarray) -> dict[str, Any]:
    """Return a result row for a failed simulation."""
    row = dict(zip(PARAM_NAMES, params_array))

    row.update(
        {
            "peak_y": np.nan,
            "max_xr": np.nan,
            "min_xr": np.nan,
            "max_abs_xr": np.nan,
            "max_xb": np.nan,
            "max_abs_xb": np.nan,
            "extra_reach": np.nan,
            "constraint_violation": np.nan,
            "constraint_violation_abs": np.nan,
            "feasible": False,
            "feasible_abs": False,
            "feasible_with_tolerance": False,
            "feasible_abs_with_tolerance": False,
        }
    )

    return row


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_one(
    params_array: np.ndarray,
    idx: int,
    x_r_max: float = DEFAULT_X_R_MAX,
    tolerance: float = DEFAULT_TOLERANCE,
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
) -> dict[str, Any]:
    """
    Simulate one sample and return feasibility information.

    Both feasibility definitions are stored:
    - feasible_with_tolerance: max_xr <= x_r_max + tolerance;
    - feasible_abs_with_tolerance: max_abs_xr <= x_r_max + tolerance.

    The absolute constraint is the preferred physical constraint in the final
    pipeline.
    """
    try:
        _, metrics = simulate_system(
            params_array,
            T_sim=T_sim,
            dt=dt,
            return_metrics=True,
            x_r_max=x_r_max,
        )

        peak_y = float(metrics["peak_y"])
        max_xr = float(metrics.get("max_xr", np.nan))
        min_xr = float(metrics.get("min_xr", np.nan))
        max_abs_xr = float(metrics.get("max_abs_xr", abs(max_xr)))
        max_xb = float(metrics.get("max_xb", np.nan))
        max_abs_xb = float(metrics.get("max_abs_xb", abs(max_xb)))

        constraint_violation = float(metrics.get("constraint_violation", max(0.0, max_xr - x_r_max)))
        constraint_violation_abs = float(
            metrics.get("constraint_violation_abs", max(0.0, max_abs_xr - x_r_max))
        )

        feasible_with_tolerance = constraint_violation <= tolerance
        feasible_abs_with_tolerance = constraint_violation_abs <= tolerance

        row = dict(zip(PARAM_NAMES, params_array))

        row.update(
            {
                "peak_y": peak_y,
                "max_xr": max_xr,
                "min_xr": min_xr,
                "max_abs_xr": max_abs_xr,
                "max_xb": max_xb,
                "max_abs_xb": max_abs_xb,
                "extra_reach": peak_y - x_r_max,
                "constraint_violation": constraint_violation,
                "constraint_violation_abs": constraint_violation_abs,
                "feasible": bool(metrics.get("feasible", constraint_violation <= 0.0)),
                "feasible_abs": bool(metrics.get("feasible_abs", constraint_violation_abs <= 0.0)),
                "feasible_with_tolerance": bool(feasible_with_tolerance),
                "feasible_abs_with_tolerance": bool(feasible_abs_with_tolerance),
            }
        )

        return row

    except Exception as exc:
        print(f"Simulation {idx} failed: {exc}")
        return failed_row(params_array)


def run_search(
    n_samples: int = DEFAULT_N_SAMPLES,
    x_r_max: float = DEFAULT_X_R_MAX,
    tolerance: float = DEFAULT_TOLERANCE,
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """Run the complete targeted search."""
    if x_r_max <= 0:
        raise ValueError("x_r_max must be positive.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if T_sim <= 0:
        raise ValueError("T_sim must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    print("=" * 70)
    print("TARGETED SEARCH FOR FEASIBLE EXTRA-REACH CASES")
    print("=" * 70)
    print(f"Samples:        {n_samples}")
    print(f"x_r_max:        {x_r_max:.3f} m")
    print(f"Tolerance:      {tolerance:.4f} m")
    print(f"Simulation time:{T_sim:.1f} s")
    print(f"Time step:      {dt:.4f} s")
    print(f"Seed:           {seed}")
    print(f"Parallel jobs:  {n_jobs}")
    print("=" * 70)

    samples = sample_parameters(n_samples=n_samples, seed=seed)

    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(simulate_one)(
            samples[i],
            i,
            x_r_max,
            tolerance,
            T_sim,
            dt,
        )
        for i in range(n_samples)
    )

    elapsed_time = time.time() - start_time

    df = pd.DataFrame(results)
    df = df.dropna(subset=["peak_y"]).reset_index(drop=True)

    print(f"\nSearch completed in {elapsed_time:.1f} s")
    print(f"Valid simulations: {len(df)}/{n_samples}")

    return df


# =============================================================================
# SUMMARY AND SAVE
# =============================================================================

def summarize_search(
    df: pd.DataFrame,
    x_r_max: float = DEFAULT_X_R_MAX,
    tolerance: float = DEFAULT_TOLERANCE,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute summary statistics and return the best feasible extra-reach cases.

    The preferred feasibility definition is feasible_abs_with_tolerance because
    it corresponds to |x_r(t)| <= x_r_max + tolerance.
    """
    feasible_abs = df[df["feasible_abs_with_tolerance"].astype(bool)].copy()
    good_abs = feasible_abs[feasible_abs["extra_reach"] > 0.0].copy()

    feasible_one_sided = df[df["feasible_with_tolerance"].astype(bool)].copy()
    good_one_sided = feasible_one_sided[feasible_one_sided["extra_reach"] > 0.0].copy()

    if len(good_abs) > 0:
        best_cases = good_abs.sort_values("peak_y", ascending=False).head(top_k).copy()
    else:
        best_cases = pd.DataFrame(columns=df.columns)

    summary = {
        "samples_valid": int(len(df)),
        "x_r_max_m": float(x_r_max),
        "tolerance_m": float(tolerance),
        "feasible_abs_samples": int(len(feasible_abs)),
        "feasible_abs_rate_percent": float(100.0 * len(feasible_abs) / len(df)) if len(df) else np.nan,
        "good_feasible_abs_extra_reach_samples": int(len(good_abs)),
        "best_feasible_abs_peak_y_m": float(good_abs["peak_y"].max()) if len(good_abs) else np.nan,
        "best_feasible_abs_extra_reach_m": float(good_abs["extra_reach"].max()) if len(good_abs) else np.nan,
        "feasible_one_sided_samples": int(len(feasible_one_sided)),
        "good_one_sided_extra_reach_samples": int(len(good_one_sided)),
        "best_one_sided_peak_y_m": float(good_one_sided["peak_y"].max()) if len(good_one_sided) else np.nan,
    }

    summary_df = pd.DataFrame([summary])

    print("\nResults:")
    print(f"Total valid simulations:          {len(df)}")
    print(f"Feasible_abs samples:             {len(feasible_abs)}")
    print(f"Good feasible_abs extra-reach:    {len(good_abs)}")

    if len(good_abs) > 0:
        print(f"Best feasible_abs peak_y:         {good_abs['peak_y'].max():.6f} m")
        print(f"Best feasible_abs extra_reach:    {good_abs['extra_reach'].max():.6f} m")

        display_cols = [
            "peak_y",
            "extra_reach",
            "max_abs_xr",
            "max_xr",
            "max_xb",
            "constraint_violation_abs",
            "Kb",
            "Kr",
            "Mb",
            "hb",
            "hr",
            "f0",
            "f1",
            "A",
            "x_r_start",
        ]

        display_cols = [col for col in display_cols if col in best_cases.columns]

        print(f"\nTop {top_k} feasible_abs extra-reach cases:")
        print(best_cases[display_cols].to_string(index=False))
    else:
        print("No feasible_abs extra-reach cases found.")

    return summary_df, best_cases


def save_results(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_cases: pd.DataFrame,
) -> None:
    """Save full results, summary, and best feasible cases."""
    ensure_dirs()

    df.to_csv(FULL_RESULTS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    best_cases.to_csv(BEST_RESULTS_PATH, index=False)

    print("\nSaved files:")
    print(f"  - {FULL_RESULTS_PATH}")
    print(f"  - {SUMMARY_PATH}")
    print(f"  - {BEST_RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search for feasible extra-reach cases using the true simulator."
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of LHS samples. Default: {DEFAULT_N_SAMPLES}.",
    )

    parser.add_argument(
        "--x_r_max",
        type=float,
        default=DEFAULT_X_R_MAX,
        help=f"Robot relative-displacement limit. Default: {DEFAULT_X_R_MAX}.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Feasibility tolerance in meters. Default: {DEFAULT_TOLERANCE}.",
    )

    parser.add_argument(
        "--T_sim",
        type=float,
        default=DEFAULT_T_SIM,
        help=f"Simulation duration in seconds. Default: {DEFAULT_T_SIM}.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"Simulation time step. Default: {DEFAULT_DT}.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel jobs. Use -1 for all cores. Default: {DEFAULT_N_JOBS}.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of best feasible cases to save. Default: {DEFAULT_TOP_K}.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the targeted feasible extra-reach search."""
    args = parse_args()
    ensure_dirs()

    df = run_search(
        n_samples=args.n_samples,
        x_r_max=args.x_r_max,
        tolerance=args.tolerance,
        T_sim=args.T_sim,
        dt=args.dt,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    summary_df, best_cases = summarize_search(
        df=df,
        x_r_max=args.x_r_max,
        tolerance=args.tolerance,
        top_k=args.top_k,
    )

    save_results(
        df=df,
        summary_df=summary_df,
        best_cases=best_cases,
    )

    print("\nTargeted search completed successfully.")


if __name__ == "__main__":
    main()