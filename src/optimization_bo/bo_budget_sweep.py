"""
bo_budget_sweep.py
==================

Bayesian Optimization online-budget sweep.

Purpose
-------
This script evaluates how the selected true-simulator Bayesian Optimization
configuration improves as the number of online simulator calls increases.

It is Step D of the BO configuration-selection workflow:

    Step A: acquisition-function benchmark
    Step B: internal GP kernel benchmark
    Step C: internal GP alpha benchmark
    Step D: online budget sweep

Selected BO configuration from previous benchmarks:

    acquisition = PI
    kernel      = RBF
    gp_alpha    = 1e-8

Method
------
For each representative target and random seed, the script runs one long BO
trajectory and extracts the best feasible solution at multiple intermediate
budgets.

This avoids running independent BO experiments for each budget and provides a
clean best-so-far analysis.

Representative targets
----------------------
0.65 m:
    High but reachable target.

0.75 m:
    Saturated / difficult target.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Existing true-simulator BO implementation. This file must be placed in src/.
import optimization_bo as obo


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_budget_sweep"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_budget_sweep"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_TARGETS = (0.65, 0.75)
DEFAULT_SEEDS = (2026, 2027)

DEFAULT_BUDGETS = (50, 100, 150, 200)
DEFAULT_LONG_TOTAL_EVALUATIONS = 200
DEFAULT_INITIAL_POINTS = 20

DEFAULT_KERNEL_KIND = "rbf"
DEFAULT_KERNEL_LABEL = "RBF"

DEFAULT_ACQUISITION_NAME = "pi"
DEFAULT_ACQUISITION_LABEL = "PI"
DEFAULT_PI_XI = 0.01

DEFAULT_GP_ALPHA = 1e-8
DEFAULT_GP_RESTARTS = 1

DEFAULT_CANDIDATE_POOL_GLOBAL = 4000
DEFAULT_CANDIDATE_POOL_LOCAL = 1000
DEFAULT_LOCAL_SIGMA_UNIT = 0.07

LONG_HISTORY_FILENAME = "bo_budget_sweep_long_history.csv"
TIMING_FILENAME = "bo_budget_sweep_timing.csv"
RESULTS_FILENAME = "bo_budget_sweep_results.csv"
SUMMARY_FILENAME = "bo_budget_sweep_summary.csv"
OVERALL_SUMMARY_FILENAME = "bo_budget_sweep_overall_summary.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class BudgetSweepConfig:
    """Configuration for the BO budget sweep."""

    targets: tuple[float, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    long_total_evaluations: int = DEFAULT_LONG_TOTAL_EVALUATIONS
    initial_points: int = DEFAULT_INITIAL_POINTS

    kernel_kind: str = DEFAULT_KERNEL_KIND
    kernel_label: str = DEFAULT_KERNEL_LABEL

    acquisition_name: str = DEFAULT_ACQUISITION_NAME
    acquisition_label: str = DEFAULT_ACQUISITION_LABEL
    pi_xi: float = DEFAULT_PI_XI

    gp_alpha: float = DEFAULT_GP_ALPHA
    gp_restarts: int = DEFAULT_GP_RESTARTS

    candidate_pool_global: int = DEFAULT_CANDIDATE_POOL_GLOBAL
    candidate_pool_local: int = DEFAULT_CANDIDATE_POOL_LOCAL
    local_sigma_unit: float = DEFAULT_LOCAL_SIGMA_UNIT

    resume_existing: bool = True
    skip_plots: bool = False

    @property
    def alpha_label(self) -> str:
        """Return alpha label in scientific notation."""
        return f"{self.gp_alpha:.0e}"


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""
    values = [item.strip() for item in raw.split(",") if item.strip()]

    if not values:
        raise ValueError("At least one value must be provided.")

    return tuple(float(value) for value in values)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    """Parse comma-separated integers."""
    values = [item.strip() for item in raw.split(",") if item.strip()]

    if not values:
        raise ValueError("At least one value must be provided.")

    return tuple(int(value) for value in values)


def safe_scientific_label(value: float) -> str:
    """Return file-safe scientific-notation label."""
    return f"{value:.0e}".replace("+", "p").replace("-", "m")


def mean_or_nan(df: pd.DataFrame, column: str) -> float:
    """Return mean of a column if it exists, otherwise NaN."""
    if column not in df.columns or df.empty:
        return np.nan

    return float(df[column].mean())


def sum_or_nan(df: pd.DataFrame, column: str) -> float:
    """Return sum of a column if it exists, otherwise NaN."""
    if column not in df.columns or df.empty:
        return np.nan

    return float(df[column].sum())


def save_all_tables(
    long_history_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save all budget-sweep tables."""
    paths = {
        "long_history": RESULTS_DIR / LONG_HISTORY_FILENAME,
        "timing": RESULTS_DIR / TIMING_FILENAME,
        "results": RESULTS_DIR / RESULTS_FILENAME,
        "summary": RESULTS_DIR / SUMMARY_FILENAME,
        "overall": RESULTS_DIR / OVERALL_SUMMARY_FILENAME,
    }

    long_history_df.to_csv(paths["long_history"], index=False)
    timing_df.to_csv(paths["timing"], index=False)
    results_df.to_csv(paths["results"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    overall_df.to_csv(paths["overall"], index=False)

    return paths


# =============================================================================
# MONKEY PATCH: PI ACQUISITION
# =============================================================================

def probability_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float = DEFAULT_PI_XI,
) -> np.ndarray:
    """
    Probability of Improvement for minimization.

    PI(x) = P(J(x) < J_best - xi)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma_safe = np.maximum(sigma, 1e-12)
    z = (best_y - mu - xi) / sigma_safe

    pi = norm.cdf(z)
    pi[sigma <= 1e-12] = 0.0

    return np.maximum(pi, 0.0)


def patched_acquisition_scores(
    gp: Any,
    X_cand: np.ndarray,
    best_y: float,
    acquisition: str,
    config: Any,
) -> np.ndarray:
    """
    Acquisition scores with EI, PI and LCB support.

    Higher score is better for all acquisition functions.
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    acquisition = acquisition.lower()

    if acquisition == "ei":
        return obo.expected_improvement(
            mu,
            sigma,
            best_y,
            xi=config.ei_xi,
        )

    if acquisition == "pi":
        xi = getattr(config, "pi_xi", config.ei_xi)
        return probability_improvement(mu, sigma, best_y, xi=xi)

    if acquisition == "lcb":
        lcb = mu - config.lcb_kappa * sigma
        return -lcb

    raise ValueError(f"Unsupported acquisition: {acquisition!r}")


# Apply patch so optimization_bo.run_bo_single_target can use acquisition="pi".
obo.acquisition_scores = patched_acquisition_scores


# =============================================================================
# BO CONFIGURATION AND RUNNERS
# =============================================================================

def make_bo_config(config: BudgetSweepConfig) -> Any:
    """Create BOConfig for the selected BO budget-sweep configuration."""
    bo_config = obo.BOConfig()

    # Disable workflows controlled by optimization_bo.py main().
    bo_config.run_pilot = False
    bo_config.stop_after_pilot = False
    bo_config.run_main_bo = False
    bo_config.run_random_search = False
    bo_config.run_kernel_sensitivity = False
    bo_config.run_acquisition_sensitivity = False
    bo_config.run_budget_sweep = False
    bo_config.make_time_response_plots = False

    # Selected BO settings.
    bo_config.main_kernel = config.kernel_kind
    bo_config.main_acquisition = config.acquisition_name
    bo_config.gp_alpha = float(config.gp_alpha)
    bo_config.gp_restarts = int(config.gp_restarts)

    bo_config.candidate_pool_global = int(config.candidate_pool_global)
    bo_config.candidate_pool_local = int(config.candidate_pool_local)
    bo_config.local_sigma_unit = float(config.local_sigma_unit)

    # Dynamically used by patched acquisition.
    bo_config.pi_xi = float(config.pi_xi)

    return bo_config


def run_label_for(
    target: float,
    seed: int,
    config: BudgetSweepConfig,
) -> str:
    """Return stable run label for a long BO trajectory."""
    target_mm = int(round(target * 1000.0))

    label = (
        f"budget_long_{config.acquisition_label}_{config.kernel_label}_"
        f"alpha{safe_scientific_label(config.gp_alpha)}_"
        f"target{target_mm:04d}_seed{seed}"
    )

    return (
        label.replace("+", "p")
        .replace("-", "m")
        .replace("/", "")
        .replace(" ", "")
    )


def has_complete_existing_history(
    existing_history: pd.DataFrame,
    run_label: str,
    config: BudgetSweepConfig,
) -> bool:
    """Check whether a complete long run already exists in the saved history."""
    if existing_history.empty:
        return False

    required_cols = {"run_label", "iteration"}

    if not required_cols.issubset(existing_history.columns):
        return False

    mask = existing_history["run_label"].astype(str) == run_label

    if not mask.any():
        return False

    max_iteration = int(existing_history.loc[mask, "iteration"].max())

    return max_iteration >= int(config.long_total_evaluations)


def timing_from_loaded_history(
    history: pd.DataFrame,
    run_label: str,
    target: float,
    seed: int,
    config: BudgetSweepConfig,
) -> dict[str, Any]:
    """Create timing row when a long run is loaded from existing history."""
    return {
        "run_label": run_label,
        "target": float(target),
        "seed": int(seed),
        "loaded_from_existing": True,
        "kernel": config.kernel_kind,
        "kernel_label": config.kernel_label,
        "acquisition": config.acquisition_name,
        "acquisition_label": config.acquisition_label,
        "gp_alpha": float(config.gp_alpha),
        "alpha_label": config.alpha_label,
        "long_total_evaluations": int(config.long_total_evaluations),
        "total_wall_time_s": sum_or_nan(history, "iteration_time_s"),
        "mean_iteration_time_s": mean_or_nan(history, "iteration_time_s"),
        "mean_simulation_time_s": mean_or_nan(history, "simulation_time_s"),
        "mean_gp_fit_time_s": mean_or_nan(history, "gp_fit_time_s"),
        "mean_acquisition_time_s": mean_or_nan(history, "acquisition_time_s"),
    }


def run_or_load_long_history(
    target: float,
    seed: int,
    config: BudgetSweepConfig,
    existing_history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run or load one long BO trajectory for one target and seed."""
    bo_config = make_bo_config(config)
    run_label = run_label_for(target, seed, config)

    if config.resume_existing and has_complete_existing_history(
        existing_history=existing_history,
        run_label=run_label,
        config=config,
    ):
        history = existing_history[
            existing_history["run_label"].astype(str) == run_label
        ].copy()

        history = history.sort_values("iteration").reset_index(drop=True)

        timing = timing_from_loaded_history(
            history=history,
            run_label=run_label,
            target=target,
            seed=seed,
            config=config,
        )

        print(f"    SKIP existing long run | target={target:.2f}, seed={seed}")

        return history, timing

    _, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=bo_config,
        run_label=run_label,
        total_evaluations=config.long_total_evaluations,
        n_initial_points=config.initial_points,
        kernel_kind=config.kernel_kind,
        acquisition=config.acquisition_name,
        seed=seed,
    )

    history = history.copy()
    history["benchmark"] = "budget_sweep"
    history["target"] = float(target)
    history["seed"] = int(seed)
    history["kernel"] = config.kernel_kind
    history["kernel_label"] = config.kernel_label
    history["acquisition"] = config.acquisition_name
    history["acquisition_label"] = config.acquisition_label
    history["gp_alpha"] = float(config.gp_alpha)
    history["alpha_label"] = config.alpha_label
    history["long_total_evaluations"] = int(config.long_total_evaluations)

    timing = dict(timing)
    timing.update(
        {
            "run_label": run_label,
            "target": float(target),
            "seed": int(seed),
            "loaded_from_existing": False,
            "kernel": config.kernel_kind,
            "kernel_label": config.kernel_label,
            "acquisition": config.acquisition_name,
            "acquisition_label": config.acquisition_label,
            "gp_alpha": float(config.gp_alpha),
            "alpha_label": config.alpha_label,
            "long_total_evaluations": int(config.long_total_evaluations),
        }
    )

    return history, timing


# =============================================================================
# BUDGET EXTRACTION
# =============================================================================

def extract_budget_rows(
    long_history: pd.DataFrame,
    config: BudgetSweepConfig,
) -> pd.DataFrame:
    """Extract the best feasible solution at each budget from long BO histories."""
    if long_history.empty:
        return pd.DataFrame()

    bo_config = make_bo_config(config)
    rows: list[dict[str, Any]] = []

    required_cols = {"target", "seed", "run_label", "iteration"}
    missing = required_cols - set(long_history.columns)

    if missing:
        raise KeyError(f"Long history is missing required columns: {sorted(missing)}")

    for (target, seed, run_label), group in long_history.groupby(
        ["target", "seed", "run_label"],
        dropna=False,
    ):
        group = group.sort_values("iteration").copy()
        max_iteration = int(group["iteration"].max())

        for budget in config.budgets:
            if int(budget) > max_iteration:
                continue

            subset = group[group["iteration"] <= int(budget)].copy()

            if subset.empty:
                continue

            best_row = obo.select_best_from_history(subset)
            result = obo.build_result_row(best_row, bo_config)

            result.update(
                {
                    "benchmark": "budget_sweep",
                    "target": float(target),
                    "seed": int(seed),
                    "run_label": str(run_label),
                    "budget": int(budget),
                    "long_total_evaluations": int(config.long_total_evaluations),
                    "initial_points": int(config.initial_points),
                    "kernel": config.kernel_kind,
                    "kernel_label": config.kernel_label,
                    "acquisition": config.acquisition_name,
                    "acquisition_label": config.acquisition_label,
                    "gp_alpha": float(config.gp_alpha),
                    "alpha_label": config.alpha_label,
                    "n_true_simulator_evaluations": int(budget),
                }
            )

            rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def summarize_budget_results(
    results_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize budget-sweep results by target/budget and overall by budget."""
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        results_df.groupby(["target", "budget"], dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            feasibility_rate_percent=(
                "feasible_abs",
                lambda series: 100.0 * series.astype(bool).mean(),
            ),
            mean_target_error_mm=("target_error_mm", "mean"),
            std_target_error_mm=("target_error_mm", "std"),
            min_target_error_mm=("target_error_mm", "min"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_peak_y_true=("peak_y_true", "mean"),
            mean_max_abs_xr_true=("max_abs_xr_true", "mean"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .reset_index()
        .sort_values(["target", "budget"])
        .reset_index(drop=True)
    )

    overall = (
        results_df.groupby("budget", dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            n_targets=("target", "nunique"),
            feasibility_rate_percent=(
                "feasible_abs",
                lambda series: 100.0 * series.astype(bool).mean(),
            ),
            mean_target_error_mm=("target_error_mm", "mean"),
            median_target_error_mm=("target_error_mm", "median"),
            std_target_error_mm=("target_error_mm", "std"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .reset_index()
        .sort_values("budget")
        .reset_index(drop=True)
    )

    return summary, overall


# =============================================================================
# PLOTS
# =============================================================================

def plot_target_error_vs_budget(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Plot mean and range of target error versus budget for each target."""
    if results_df.empty or summary_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = summary_df[np.isclose(summary_df["target"], target)].copy()

        if df_target.empty:
            continue

        df_target = df_target.sort_values("budget")

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        ax.plot(
            df_target["budget"],
            df_target["mean_target_error_mm"],
            marker="o",
            linewidth=2.2,
            label="Mean across seeds",
        )

        ax.fill_between(
            df_target["budget"].astype(float),
            df_target["min_target_error_mm"].astype(float),
            df_target["max_target_error_mm"].astype(float),
            alpha=0.18,
            label="Seed min-max range",
        )

        ax.set_xlabel("True simulator calls")
        ax.set_ylabel("Best feasible target error [mm]")
        ax.set_title(f"BO budget sweep: target error, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_budget_sweep_error_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_target_margin_vs_budget(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Plot residual margin versus budget for each target."""
    if results_df.empty or summary_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = summary_df[np.isclose(summary_df["target"], target)].copy()

        if df_target.empty:
            continue

        df_target = df_target.sort_values("budget")

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        ax.plot(
            df_target["budget"],
            df_target["mean_residual_margin_mm"],
            marker="o",
            linewidth=2.2,
            label="Mean residual margin",
        )

        ax.plot(
            df_target["budget"],
            df_target["min_residual_margin_mm"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="Minimum residual margin",
        )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.8,
            label="Constraint boundary",
        )

        ax.set_xlabel("True simulator calls")
        ax.set_ylabel("Residual margin [mm]")
        ax.set_title(f"BO budget sweep: residual margin, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_budget_sweep_margin_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_target_convergence(
    long_history: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    """Plot long BO convergence curves by seed for each target."""
    if long_history.empty:
        return

    if "best_feasible_error_so_far_mm" not in long_history.columns:
        print("Convergence plot skipped: missing best_feasible_error_so_far_mm.")
        return

    for target in sorted(long_history["target"].unique()):
        df_target = long_history[np.isclose(long_history["target"], target)].copy()

        if df_target.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for seed, group in df_target.groupby("seed"):
            group = group.sort_values("iteration")

            ax.plot(
                group["iteration"],
                group["best_feasible_error_so_far_mm"],
                linewidth=1.8,
                label=f"seed {int(seed)}",
            )

        if not results_df.empty:
            for budget in sorted(results_df["budget"].unique()):
                ax.axvline(int(budget), linestyle=":", linewidth=0.9)

        ax.set_xlabel("True simulator calls")
        ax.set_ylabel("Best feasible target error so far [mm]")
        ax.set_title(f"BO convergence, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_budget_sweep_convergence_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_overall_budget_error(results_df: pd.DataFrame) -> None:
    """Plot overall mean and worst-case target error versus budget."""
    if results_df.empty:
        return

    overall = (
        results_df.groupby("budget")
        .agg(
            mean_target_error_mm=("target_error_mm", "mean"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
        )
        .reset_index()
        .sort_values("budget")
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        overall["budget"],
        overall["mean_target_error_mm"],
        marker="o",
        linewidth=2.2,
        label="Mean target error",
    )

    ax.plot(
        overall["budget"],
        overall["max_target_error_mm"],
        marker="s",
        linewidth=2.0,
        label="Worst-case target error",
    )

    ax.set_xlabel("True simulator calls")
    ax.set_ylabel("Target error [mm]")
    ax.set_title("BO budget sweep: overall target error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "bo_budget_sweep_overall_error.png"

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def make_plots(
    results_df: pd.DataFrame,
    long_history: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Generate all budget-sweep plots."""
    print("\nGenerating budget-sweep plots...")

    plot_target_error_vs_budget(results_df, summary_df)
    plot_target_margin_vs_budget(results_df, summary_df)
    plot_target_convergence(long_history, results_df)
    plot_overall_budget_error(results_df)


# =============================================================================
# PRINTING
# =============================================================================

def print_settings(config: BudgetSweepConfig) -> None:
    """Print budget-sweep settings."""
    print("=" * 80)
    print("BO BUDGET SWEEP")
    print("=" * 80)
    print(f"Targets:                 {config.targets}")
    print(f"Seeds:                   {config.seeds}")
    print(f"Acquisition:             {config.acquisition_label}")
    print(f"Kernel:                  {config.kernel_label}")
    print(f"GP alpha:                {config.alpha_label}")
    print(f"Budgets extracted:       {config.budgets}")
    print(f"Long BO calls/run:       {config.long_total_evaluations}")
    print(f"Initial LHS points:      {config.initial_points}")
    print(f"GP restarts:             {config.gp_restarts}")
    print(f"Global candidate pool:   {config.candidate_pool_global}")
    print(f"Local candidate pool:    {config.candidate_pool_local}")
    print(f"Resume existing:         {config.resume_existing}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)


def print_summary(
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    """Print budget-sweep summaries."""
    print("\n" + "=" * 80)
    print("BUDGET SWEEP SUMMARY")
    print("=" * 80)

    if not summary_df.empty:
        cols = [
            "target",
            "budget",
            "n_runs",
            "feasibility_rate_percent",
            "mean_target_error_mm",
            "std_target_error_mm",
            "min_target_error_mm",
            "max_target_error_mm",
            "mean_residual_margin_mm",
            "min_residual_margin_mm",
            "mean_best_iteration",
        ]

        available_cols = [col for col in cols if col in summary_df.columns]
        print(summary_df[available_cols].to_string(index=False))
    else:
        print("No summary available.")

    print("\n" + "=" * 80)
    print("OVERALL BUDGET SUMMARY ACROSS REPRESENTATIVE TARGETS")
    print("=" * 80)

    if not overall_df.empty:
        cols = [
            "budget",
            "n_runs",
            "n_targets",
            "feasibility_rate_percent",
            "mean_target_error_mm",
            "median_target_error_mm",
            "std_target_error_mm",
            "max_target_error_mm",
            "mean_residual_margin_mm",
            "min_residual_margin_mm",
            "mean_best_iteration",
        ]

        available_cols = [col for col in cols if col in overall_df.columns]
        print(overall_df[available_cols].to_string(index=False))
    else:
        print("No overall summary available.")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BO online-budget sweep."
    )

    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(str(target) for target in DEFAULT_TARGETS),
        help="Comma-separated representative outreach targets.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--budgets",
        type=str,
        default=",".join(str(budget) for budget in DEFAULT_BUDGETS),
        help="Comma-separated budgets to extract from the long BO histories.",
    )

    parser.add_argument(
        "--long_budget",
        type=int,
        default=DEFAULT_LONG_TOTAL_EVALUATIONS,
        help=f"Long BO trajectory length. Default: {DEFAULT_LONG_TOTAL_EVALUATIONS}.",
    )

    parser.add_argument(
        "--initial_points",
        type=int,
        default=DEFAULT_INITIAL_POINTS,
        help=f"Initial LHS points. Default: {DEFAULT_INITIAL_POINTS}.",
    )

    parser.add_argument(
        "--gp_alpha",
        type=float,
        default=DEFAULT_GP_ALPHA,
        help=f"Internal BO GP alpha. Default: {DEFAULT_GP_ALPHA}.",
    )

    parser.add_argument(
        "--gp_restarts",
        type=int,
        default=DEFAULT_GP_RESTARTS,
        help=f"Internal BO GP optimizer restarts. Default: {DEFAULT_GP_RESTARTS}.",
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing long-history file.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BudgetSweepConfig:
    """Build budget-sweep configuration from command-line arguments."""
    budgets = parse_int_tuple(args.budgets)

    if any(budget <= 0 for budget in budgets):
        raise ValueError("All budgets must be positive.")

    if args.long_budget <= 0:
        raise ValueError("--long_budget must be positive.")

    if max(budgets) > int(args.long_budget):
        raise ValueError("The largest extracted budget cannot exceed --long_budget.")

    if args.initial_points <= 0:
        raise ValueError("--initial_points must be positive.")

    if args.initial_points >= min(budgets):
        raise ValueError("--initial_points must be smaller than the smallest extracted budget.")

    if args.initial_points >= args.long_budget:
        raise ValueError("--initial_points must be smaller than --long_budget.")

    if args.gp_restarts < 0:
        raise ValueError("--gp_restarts must be non-negative.")

    return BudgetSweepConfig(
        targets=parse_float_tuple(args.targets),
        seeds=parse_int_tuple(args.seeds),
        budgets=budgets,
        long_total_evaluations=int(args.long_budget),
        initial_points=int(args.initial_points),
        gp_alpha=float(args.gp_alpha),
        gp_restarts=int(args.gp_restarts),
        resume_existing=not bool(args.no_resume),
        skip_plots=bool(args.skip_plots),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete BO budget-sweep analysis."""
    args = parse_args()
    config = build_config(args)

    ensure_dirs()
    print_settings(config)

    long_history_path = RESULTS_DIR / LONG_HISTORY_FILENAME

    existing_history = pd.DataFrame()

    if config.resume_existing and long_history_path.exists():
        existing_history = pd.read_csv(long_history_path)
        print(f"Resuming from existing long history with {len(existing_history)} rows.")

    all_histories: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []

    total_runs = len(config.targets) * len(config.seeds)
    run_counter = 0
    global_start = time.perf_counter()

    for target in config.targets:
        for seed in config.seeds:
            run_counter += 1

            print(f"\n[{run_counter}/{total_runs}] target={target:.2f} | seed={seed}")

            history, timing = run_or_load_long_history(
                target=float(target),
                seed=int(seed),
                config=config,
                existing_history=existing_history,
            )

            all_histories.append(history)
            timing_rows.append(timing)

            max_iteration = int(history["iteration"].max())

            preview_budgets = [
                budget for budget in (100, config.long_total_evaluations)
                if budget <= max_iteration
            ]

            for budget in preview_budgets:
                best = obo.select_best_from_history(
                    history[history["iteration"] <= int(budget)]
                )

                print(
                    f"    budget{budget}: "
                    f"y={float(best['peak_y_true']):.4f} m | "
                    f"err={float(best['target_error_mm']):.2f} mm | "
                    f"margin={float(best['residual_margin_mm']):.2f} mm"
                )

            # Incremental save for resume safety.
            partial_long_history = pd.concat(all_histories, ignore_index=True)
            partial_long_history = obo.add_cumulative_best_columns(partial_long_history)

            partial_timing_df = pd.DataFrame(timing_rows)
            partial_results_df = extract_budget_rows(partial_long_history, config)
            partial_summary_df, partial_overall_df = summarize_budget_results(
                partial_results_df
            )

            save_all_tables(
                long_history_df=partial_long_history,
                timing_df=partial_timing_df,
                results_df=partial_results_df,
                summary_df=partial_summary_df,
                overall_df=partial_overall_df,
            )

    total_time_s = time.perf_counter() - global_start

    if all_histories:
        long_history = pd.concat(all_histories, ignore_index=True)
    else:
        long_history = pd.DataFrame()

    long_history = obo.add_cumulative_best_columns(long_history)

    timing_df = pd.DataFrame(timing_rows)
    results_df = extract_budget_rows(long_history, config)
    summary_df, overall_df = summarize_budget_results(results_df)

    saved_paths = save_all_tables(
        long_history_df=long_history,
        timing_df=timing_df,
        results_df=results_df,
        summary_df=summary_df,
        overall_df=overall_df,
    )

    print_summary(summary_df, overall_df)

    print("\nSaved files:")
    for path in saved_paths.values():
        print(f"  - {path}")

    print(f"Total budget-sweep wall time: {total_time_s / 60.0:.1f} min")

    if not config.skip_plots:
        make_plots(
            results_df=results_df,
            long_history=long_history,
            summary_df=summary_df,
        )

    print("\nBudget sweep completed.")


if __name__ == "__main__":
    main()