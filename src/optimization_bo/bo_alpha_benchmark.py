"""
bo_alpha_benchmark.py
=====================

Bayesian Optimization internal-GP alpha benchmark.

Purpose
-------
This script benchmarks the numerical regularization parameter alpha used by the
internal Gaussian Process of the true-simulator Bayesian Optimization strategy.

It is Step C of the BO configuration-selection workflow:

    Step A: acquisition-function benchmark
    Step B: internal GP kernel benchmark
    Step C: internal GP alpha benchmark

The previous benchmarks selected the following candidate configuration:

    acquisition = PI
    kernel      = RBF

This script compares several gp_alpha values on two representative targets:

    0.65 m: high but reachable target
    0.75 m: saturated / difficult target
    
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

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_alpha_benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_alpha_benchmark"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_TARGETS = (0.65, 0.75)
DEFAULT_SEEDS = (2026, 2027)

DEFAULT_TOTAL_EVALUATIONS = 100
DEFAULT_INITIAL_POINTS = 20

DEFAULT_KERNEL_KIND = "rbf"
DEFAULT_KERNEL_LABEL = "RBF"

DEFAULT_ACQUISITION_NAME = "pi"
DEFAULT_ACQUISITION_LABEL = "PI"
DEFAULT_PI_XI = 0.01

DEFAULT_ALPHA_VALUES = (1e-8, 1e-6, 1e-4, 1e-3)

DEFAULT_GP_RESTARTS = 1
DEFAULT_CANDIDATE_POOL_GLOBAL = 4000
DEFAULT_CANDIDATE_POOL_LOCAL = 1000
DEFAULT_LOCAL_SIGMA_UNIT = 0.07

RESULTS_FILENAME = "bo_alpha_benchmark.csv"
HISTORY_FILENAME = "bo_alpha_benchmark_history.csv"
TIMING_FILENAME = "bo_alpha_benchmark_timing.csv"
SUMMARY_FILENAME = "bo_alpha_benchmark_summary.csv"
OVERALL_SUMMARY_FILENAME = "bo_alpha_benchmark_overall_summary.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AlphaBenchmarkConfig:
    """Configuration for the BO alpha benchmark."""

    targets: tuple[float, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    total_evaluations: int = DEFAULT_TOTAL_EVALUATIONS
    initial_points: int = DEFAULT_INITIAL_POINTS

    kernel_kind: str = DEFAULT_KERNEL_KIND
    kernel_label: str = DEFAULT_KERNEL_LABEL

    acquisition_name: str = DEFAULT_ACQUISITION_NAME
    acquisition_label: str = DEFAULT_ACQUISITION_LABEL
    pi_xi: float = DEFAULT_PI_XI

    alpha_values: tuple[float, ...] = DEFAULT_ALPHA_VALUES

    gp_restarts: int = DEFAULT_GP_RESTARTS
    candidate_pool_global: int = DEFAULT_CANDIDATE_POOL_GLOBAL
    candidate_pool_local: int = DEFAULT_CANDIDATE_POOL_LOCAL
    local_sigma_unit: float = DEFAULT_LOCAL_SIGMA_UNIT

    resume_existing: bool = True
    skip_plots: bool = False


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


def alpha_label(alpha: float) -> str:
    """Return compact scientific notation label for alpha."""
    return f"{float(alpha):.0e}"


def safe_alpha_label(alpha: float) -> str:
    """Return file-safe alpha label."""
    return f"alpha{float(alpha):.0e}".replace("-", "m").replace("+", "p")


def alpha_order(config: AlphaBenchmarkConfig) -> list[float]:
    """Return alpha plotting order."""
    return list(config.alpha_values)


def save_all_tables(
    results_df: pd.DataFrame,
    history_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save all benchmark tables."""
    paths = {
        "results": RESULTS_DIR / RESULTS_FILENAME,
        "history": RESULTS_DIR / HISTORY_FILENAME,
        "timing": RESULTS_DIR / TIMING_FILENAME,
        "summary": RESULTS_DIR / SUMMARY_FILENAME,
        "overall": RESULTS_DIR / OVERALL_SUMMARY_FILENAME,
    }

    results_df.to_csv(paths["results"], index=False)
    history_df.to_csv(paths["history"], index=False)
    timing_df.to_csv(paths["timing"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    overall_df.to_csv(paths["overall"], index=False)

    return paths


def get_valid_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Remove failed benchmark rows before summary and plotting."""
    if results_df.empty:
        return pd.DataFrame()

    if "target_error_mm" not in results_df.columns:
        return pd.DataFrame()

    return results_df.dropna(subset=["target_error_mm"]).copy()


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

def make_bo_config(
    benchmark_config: AlphaBenchmarkConfig,
    alpha: float,
) -> Any:
    """Create BOConfig for one alpha benchmark run."""
    config = obo.BOConfig()

    # Disable workflows controlled by optimization_bo.py main().
    config.run_pilot = False
    config.stop_after_pilot = False
    config.run_main_bo = False
    config.run_random_search = False
    config.run_kernel_sensitivity = False
    config.run_acquisition_sensitivity = False
    config.run_budget_sweep = False
    config.make_time_response_plots = False

    # Benchmark settings.
    config.main_kernel = benchmark_config.kernel_kind
    config.main_acquisition = benchmark_config.acquisition_name
    config.gp_restarts = benchmark_config.gp_restarts
    config.gp_alpha = float(alpha)

    config.candidate_pool_global = benchmark_config.candidate_pool_global
    config.candidate_pool_local = benchmark_config.candidate_pool_local
    config.local_sigma_unit = benchmark_config.local_sigma_unit

    # Dynamically used by patched acquisition.
    config.pi_xi = benchmark_config.pi_xi

    return config


def run_single_benchmark(
    target: float,
    alpha: float,
    seed: int,
    benchmark_config: AlphaBenchmarkConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run one BO alpha benchmark case."""
    config = make_bo_config(benchmark_config, alpha)

    run_label = (
        f"alpha_{safe_alpha_label(alpha)}_"
        f"target{int(round(target * 1000)):04d}_"
        f"seed{seed}"
    )

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=config,
        run_label=run_label,
        total_evaluations=benchmark_config.total_evaluations,
        n_initial_points=benchmark_config.initial_points,
        kernel_kind=benchmark_config.kernel_kind,
        acquisition=benchmark_config.acquisition_name,
        seed=seed,
    )

    result = obo.build_result_row(best, config)

    result.update(
        {
            "benchmark": "alpha",
            "target": float(target),
            "kernel": benchmark_config.kernel_kind,
            "kernel_label": benchmark_config.kernel_label,
            "acquisition": benchmark_config.acquisition_name,
            "acquisition_label": benchmark_config.acquisition_label,
            "seed": int(seed),
            "budget": int(benchmark_config.total_evaluations),
            "initial_points": int(benchmark_config.initial_points),
            "gp_alpha": float(alpha),
            "alpha_label": alpha_label(alpha),
            "optimization_time_s": float(timing["total_wall_time_s"]),
            "mean_iteration_time_s": float(timing["mean_iteration_time_s"]),
            "mean_simulation_time_s": float(timing["mean_simulation_time_s"]),
            "mean_gp_fit_time_s": float(timing["mean_gp_fit_time_s"]),
            "mean_acquisition_time_s": float(timing["mean_acquisition_time_s"]),
        }
    )

    history = history.copy()
    history["benchmark"] = "alpha"
    history["target"] = float(target)
    history["kernel"] = benchmark_config.kernel_kind
    history["kernel_label"] = benchmark_config.kernel_label
    history["acquisition"] = benchmark_config.acquisition_name
    history["acquisition_label"] = benchmark_config.acquisition_label
    history["seed"] = int(seed)
    history["budget"] = int(benchmark_config.total_evaluations)
    history["initial_points"] = int(benchmark_config.initial_points)
    history["gp_alpha"] = float(alpha)
    history["alpha_label"] = alpha_label(alpha)

    timing = dict(timing)
    timing.update(
        {
            "benchmark": "alpha",
            "target": float(target),
            "kernel": benchmark_config.kernel_kind,
            "kernel_label": benchmark_config.kernel_label,
            "acquisition": benchmark_config.acquisition_name,
            "acquisition_label": benchmark_config.acquisition_label,
            "seed": int(seed),
            "budget": int(benchmark_config.total_evaluations),
            "initial_points": int(benchmark_config.initial_points),
            "gp_alpha": float(alpha),
            "alpha_label": alpha_label(alpha),
        }
    )

    return result, history, timing


# =============================================================================
# RESUME SUPPORT
# =============================================================================

def load_existing_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load partial benchmark files if they exist."""
    results_path = RESULTS_DIR / RESULTS_FILENAME
    history_path = RESULTS_DIR / HISTORY_FILENAME
    timing_path = RESULTS_DIR / TIMING_FILENAME

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    timing = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()

    return results, history, timing


def already_done(
    existing_results: pd.DataFrame,
    target: float,
    alpha: float,
    seed: int,
    budget: int,
    kernel_label: str,
    acquisition_label: str,
) -> bool:
    """Check whether a benchmark run is already present."""
    if existing_results.empty:
        return False

    required_cols = {
        "target",
        "gp_alpha",
        "seed",
        "budget",
        "kernel_label",
        "acquisition_label",
    }

    if not required_cols.issubset(existing_results.columns):
        return False

    mask = (
        np.isclose(existing_results["target"].astype(float), float(target))
        & np.isclose(existing_results["gp_alpha"].astype(float), float(alpha))
        & (existing_results["seed"].astype(int) == int(seed))
        & (existing_results["budget"].astype(int) == int(budget))
        & (existing_results["kernel_label"].astype(str) == kernel_label)
        & (existing_results["acquisition_label"].astype(str) == acquisition_label)
    )

    return bool(mask.any())


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate alpha benchmark results by target and alpha."""
    if results_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for (target, alpha), group in results_df.groupby(
        ["target", "gp_alpha"],
        sort=False,
    ):
        group = group.copy()

        feasible_rate = 100.0 * float(group["feasible_abs"].astype(bool).mean())

        rows.append(
            {
                "target": float(target),
                "gp_alpha": float(alpha),
                "alpha_label": alpha_label(alpha),
                "n_runs": int(len(group)),
                "feasibility_rate_percent": feasible_rate,
                "mean_target_error_mm": float(group["target_error_mm"].mean()),
                "std_target_error_mm": float(group["target_error_mm"].std(ddof=0)),
                "min_target_error_mm": float(group["target_error_mm"].min()),
                "max_target_error_mm": float(group["target_error_mm"].max()),
                "mean_peak_y_true": float(group["peak_y_true"].mean()),
                "mean_max_abs_xr_true": float(group["max_abs_xr_true"].mean()),
                "mean_residual_margin_mm": float(group["residual_margin_mm"].mean()),
                "min_residual_margin_mm": float(group["residual_margin_mm"].min()),
                "mean_best_iteration": float(group["best_iteration"].mean()),
                "mean_optimization_time_s": float(group["optimization_time_s"].mean()),
                "mean_gp_fit_time_s": float(group["mean_gp_fit_time_s"].mean()),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["target", "mean_target_error_mm", "mean_residual_margin_mm"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return summary


def build_overall_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate alpha performance across representative targets."""
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    df["rank_by_error_within_target_seed"] = df.groupby(
        ["target", "seed"]
    )["target_error_mm"].rank(method="min", ascending=True)

    rows: list[dict[str, Any]] = []

    for alpha, group in df.groupby("gp_alpha", sort=False):
        feasible_rate = 100.0 * float(group["feasible_abs"].astype(bool).mean())

        rows.append(
            {
                "gp_alpha": float(alpha),
                "alpha_label": alpha_label(alpha),
                "n_runs": int(len(group)),
                "n_targets": int(group["target"].nunique()),
                "feasibility_rate_percent": feasible_rate,
                "mean_target_error_mm": float(group["target_error_mm"].mean()),
                "median_target_error_mm": float(group["target_error_mm"].median()),
                "std_target_error_mm": float(group["target_error_mm"].std(ddof=0)),
                "max_target_error_mm": float(group["target_error_mm"].max()),
                "mean_residual_margin_mm": float(group["residual_margin_mm"].mean()),
                "min_residual_margin_mm": float(group["residual_margin_mm"].min()),
                "mean_rank_by_error": float(
                    group["rank_by_error_within_target_seed"].mean()
                ),
                "mean_best_iteration": float(group["best_iteration"].mean()),
                "mean_optimization_time_s": float(group["optimization_time_s"].mean()),
                "mean_gp_fit_time_s": float(group["mean_gp_fit_time_s"].mean()),
            }
        )

    overall = pd.DataFrame(rows)

    overall = overall.sort_values(
        [
            "feasibility_rate_percent",
            "mean_rank_by_error",
            "mean_target_error_mm",
            "max_target_error_mm",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return overall


# =============================================================================
# PLOTS
# =============================================================================

def reindex_by_alpha(
    df: pd.DataFrame,
    config: AlphaBenchmarkConfig,
) -> pd.DataFrame:
    """Return DataFrame ordered by configured alpha values."""
    return (
        df.set_index("gp_alpha")
        .reindex(alpha_order(config))
        .reset_index()
    )


def plot_alpha_error(
    summary_df: pd.DataFrame,
    config: AlphaBenchmarkConfig,
) -> None:
    """Plot final target error as a function of alpha."""
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_target = summary_df[np.isclose(summary_df["target"], target)].copy()
        df_target = reindex_by_alpha(df_target, config)

        x = df_target["gp_alpha"].astype(float).to_numpy()
        y = df_target["mean_target_error_mm"].astype(float).to_numpy()
        yerr = df_target["std_target_error_mm"].astype(float).to_numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.2,
            capsize=4,
        )

        ax.set_xscale("log")
        ax.set_xlabel("GP alpha")
        ax.set_ylabel("Mean final target error [mm]")
        ax.set_title(f"BO alpha benchmark: target error, target = {target:.2f} m")
        ax.grid(True, alpha=0.3, which="both")

        path = FIGURES_DIR / (
            f"bo_alpha_error_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_alpha_margin(
    summary_df: pd.DataFrame,
    config: AlphaBenchmarkConfig,
) -> None:
    """Plot residual margin as a function of alpha."""
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_target = summary_df[np.isclose(summary_df["target"], target)].copy()
        df_target = reindex_by_alpha(df_target, config)

        x = df_target["gp_alpha"].astype(float).to_numpy()
        y_mean = df_target["mean_residual_margin_mm"].astype(float).to_numpy()
        y_min = df_target["min_residual_margin_mm"].astype(float).to_numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            x,
            y_mean,
            marker="o",
            linewidth=2.2,
            label="Mean residual margin",
        )
        ax.plot(
            x,
            y_min,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label="Minimum residual margin",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1.8, label="Constraint boundary")
        ax.set_xscale("log")
        ax.set_xlabel("GP alpha")
        ax.set_ylabel("Residual margin to true limit [mm]")
        ax.set_title(f"BO alpha benchmark: residual margin, target = {target:.2f} m")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_alpha_margin_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_alpha_convergence(
    history_df: pd.DataFrame,
    config: AlphaBenchmarkConfig,
) -> None:
    """Plot BO convergence curves for each alpha."""
    if history_df.empty:
        return

    if "best_feasible_error_so_far_mm" not in history_df.columns:
        print("Convergence plot skipped: missing best_feasible_error_so_far_mm.")
        return

    for target in sorted(history_df["target"].unique()):
        df_target = history_df[np.isclose(history_df["target"], target)].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        for alpha in alpha_order(config):
            df_alpha = df_target[
                np.isclose(df_target["gp_alpha"].astype(float), float(alpha))
            ].copy()

            if df_alpha.empty:
                continue

            grouped = (
                df_alpha.groupby("iteration")["best_feasible_error_so_far_mm"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .sort_values("iteration")
            )

            label = f"alpha={alpha:.0e}"

            ax.plot(
                grouped["iteration"],
                grouped["mean"],
                linewidth=2.2,
                label=label,
            )

            ax.fill_between(
                grouped["iteration"],
                grouped["min"],
                grouped["max"],
                alpha=0.12,
            )

        ax.set_xlabel("True simulator calls")
        ax.set_ylabel("Best feasible target error so far [mm]")
        ax.set_title(f"BO alpha convergence, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_alpha_convergence_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_overall_alpha_bars(
    overall_df: pd.DataFrame,
    config: AlphaBenchmarkConfig,
) -> None:
    """Plot overall mean target error by alpha."""
    if overall_df.empty:
        return

    df = reindex_by_alpha(overall_df, config)

    labels = [f"{alpha:.0e}" for alpha in df["gp_alpha"].astype(float).to_numpy()]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x,
        df["mean_target_error_mm"],
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("GP alpha")
    ax.set_ylabel("Mean target error across benchmark targets [mm]")
    ax.set_title("BO alpha benchmark: overall mean target error")
    ax.grid(True, axis="y", alpha=0.3)

    path = FIGURES_DIR / "bo_alpha_overall_mean_error.png"

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def make_plots(
    results_df: pd.DataFrame,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    config: AlphaBenchmarkConfig,
) -> None:
    """Generate all alpha benchmark plots."""
    print("\nGenerating alpha benchmark plots...")

    _ = results_df  # kept for a consistent plot API with other benchmark scripts

    plot_alpha_error(summary_df, config)
    plot_alpha_margin(summary_df, config)
    plot_alpha_convergence(history_df, config)
    plot_overall_alpha_bars(overall_df, config)


# =============================================================================
# PRINTING
# =============================================================================

def print_settings(config: AlphaBenchmarkConfig) -> None:
    """Print benchmark settings."""
    print("=" * 80)
    print("BO ALPHA BENCHMARK")
    print("=" * 80)
    print(f"Targets:                 {config.targets}")
    print(f"Seeds:                   {config.seeds}")
    print(f"Kernel:                  {config.kernel_label}")
    print(f"Acquisition:             {config.acquisition_label}")
    print(f"PI xi:                   {config.pi_xi}")
    print(f"Budget:                  {config.total_evaluations} calls")
    print(f"Initial LHS points:      {config.initial_points}")
    print(f"Alpha values:            {[f'{alpha:.0e}' for alpha in config.alpha_values]}")
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
    """Print final benchmark summaries."""
    print("\n" + "=" * 80)
    print("ALPHA BENCHMARK SUMMARY")
    print("=" * 80)

    if not summary_df.empty:
        cols = [
            "target",
            "alpha_label",
            "n_runs",
            "feasibility_rate_percent",
            "mean_target_error_mm",
            "std_target_error_mm",
            "min_target_error_mm",
            "max_target_error_mm",
            "mean_residual_margin_mm",
            "min_residual_margin_mm",
            "mean_best_iteration",
            "mean_gp_fit_time_s",
        ]

        available_cols = [col for col in cols if col in summary_df.columns]
        print(summary_df[available_cols].to_string(index=False))
    else:
        print("No valid results to summarize.")

    print("\n" + "=" * 80)
    print("OVERALL ALPHA SUMMARY ACROSS REPRESENTATIVE TARGETS")
    print("=" * 80)

    if not overall_df.empty:
        cols = [
            "alpha_label",
            "n_runs",
            "n_targets",
            "feasibility_rate_percent",
            "mean_target_error_mm",
            "median_target_error_mm",
            "std_target_error_mm",
            "max_target_error_mm",
            "mean_residual_margin_mm",
            "min_residual_margin_mm",
            "mean_rank_by_error",
            "mean_gp_fit_time_s",
        ]

        available_cols = [col for col in cols if col in overall_df.columns]
        print(overall_df[available_cols].to_string(index=False))
    else:
        print("No valid results to summarize.")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BO internal-GP alpha benchmark."
    )

    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(str(target) for target in DEFAULT_TARGETS),
        help="Comma-separated outreach targets.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_TOTAL_EVALUATIONS,
        help=f"Total true simulator calls per run. Default: {DEFAULT_TOTAL_EVALUATIONS}.",
    )

    parser.add_argument(
        "--initial_points",
        type=int,
        default=DEFAULT_INITIAL_POINTS,
        help=f"Initial LHS points. Default: {DEFAULT_INITIAL_POINTS}.",
    )

    parser.add_argument(
        "--alphas",
        type=str,
        default=",".join(f"{alpha:.0e}" for alpha in DEFAULT_ALPHA_VALUES),
        help="Comma-separated alpha values.",
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
        help="Do not resume from existing benchmark result files.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AlphaBenchmarkConfig:
    """Build benchmark configuration from command-line arguments."""
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")

    if args.initial_points <= 0:
        raise ValueError("--initial_points must be positive.")

    if args.initial_points >= args.budget:
        raise ValueError("--initial_points must be smaller than --budget.")

    if args.gp_restarts < 0:
        raise ValueError("--gp_restarts must be non-negative.")

    return AlphaBenchmarkConfig(
        targets=parse_float_tuple(args.targets),
        seeds=parse_int_tuple(args.seeds),
        total_evaluations=int(args.budget),
        initial_points=int(args.initial_points),
        alpha_values=parse_float_tuple(args.alphas),
        gp_restarts=int(args.gp_restarts),
        resume_existing=not bool(args.no_resume),
        skip_plots=bool(args.skip_plots),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete BO alpha benchmark."""
    args = parse_args()
    config = build_config(args)

    ensure_dirs()
    print_settings(config)

    existing_results, existing_history, existing_timing = load_existing_results()

    result_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []

    if config.resume_existing and not existing_results.empty:
        print(f"Resuming from existing file with {len(existing_results)} completed runs.")

        result_rows.extend(existing_results.to_dict(orient="records"))

        if not existing_history.empty:
            history_frames.append(existing_history)

        if not existing_timing.empty:
            timing_rows.extend(existing_timing.to_dict(orient="records"))

    total_runs = len(config.targets) * len(config.alpha_values) * len(config.seeds)
    run_counter = 0
    t_global = time.perf_counter()

    for target in config.targets:
        for alpha in config.alpha_values:
            for seed in config.seeds:
                run_counter += 1

                if config.resume_existing and already_done(
                    existing_results=existing_results,
                    target=target,
                    alpha=alpha,
                    seed=seed,
                    budget=config.total_evaluations,
                    kernel_label=config.kernel_label,
                    acquisition_label=config.acquisition_label,
                ):
                    print(
                        f"[{run_counter}/{total_runs}] SKIP existing | "
                        f"target={target:.2f}, alpha={alpha:.0e}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{run_counter}/{total_runs}] "
                    f"target={target:.2f} | alpha={alpha:.0e} | "
                    f"kernel={config.kernel_label} | "
                    f"acquisition={config.acquisition_label} | seed={seed}"
                )

                try:
                    result, history, timing = run_single_benchmark(
                        target=target,
                        alpha=alpha,
                        seed=seed,
                        benchmark_config=config,
                    )

                except Exception as exc:
                    print(f"    ERROR: {exc}")

                    error_row = {
                        "benchmark": "alpha",
                        "target": float(target),
                        "kernel": config.kernel_kind,
                        "kernel_label": config.kernel_label,
                        "acquisition": config.acquisition_name,
                        "acquisition_label": config.acquisition_label,
                        "seed": int(seed),
                        "budget": int(config.total_evaluations),
                        "initial_points": int(config.initial_points),
                        "gp_alpha": float(alpha),
                        "alpha_label": alpha_label(alpha),
                        "error": repr(exc),
                    }

                    result_rows.append(error_row)

                    results_df = pd.DataFrame(result_rows)
                    valid_results_df = get_valid_results(results_df)
                    history_df = (
                        pd.concat(history_frames, ignore_index=True)
                        if history_frames
                        else pd.DataFrame()
                    )
                    timing_df = pd.DataFrame(timing_rows)
                    summary_df = build_summary(valid_results_df)
                    overall_df = build_overall_summary(valid_results_df)

                    save_all_tables(
                        results_df=results_df,
                        history_df=history_df,
                        timing_df=timing_df,
                        summary_df=summary_df,
                        overall_df=overall_df,
                    )

                    continue

                result_rows.append(result)
                history_frames.append(history)
                timing_rows.append(timing)

                results_df = pd.DataFrame(result_rows)
                valid_results_df = get_valid_results(results_df)
                history_df = (
                    pd.concat(history_frames, ignore_index=True)
                    if history_frames
                    else pd.DataFrame()
                )
                timing_df = pd.DataFrame(timing_rows)
                summary_df = build_summary(valid_results_df)
                overall_df = build_overall_summary(valid_results_df)

                save_all_tables(
                    results_df=results_df,
                    history_df=history_df,
                    timing_df=timing_df,
                    summary_df=summary_df,
                    overall_df=overall_df,
                )

                print(
                    f"    best_y={float(result['peak_y_true']):.4f} m | "
                    f"err={float(result['target_error_mm']):.2f} mm | "
                    f"xr={float(result['max_abs_xr_true']):.4f} m | "
                    f"margin={float(result['residual_margin_mm']):.2f} mm | "
                    f"feasible={bool(result['feasible_abs'])} | "
                    f"time={float(result['optimization_time_s']):.1f} s"
                )

    total_time_s = time.perf_counter() - t_global

    results_df = pd.DataFrame(result_rows)
    valid_results_df = get_valid_results(results_df)

    history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )

    timing_df = pd.DataFrame(timing_rows)

    summary_df = build_summary(valid_results_df)
    overall_df = build_overall_summary(valid_results_df)

    saved_paths = save_all_tables(
        results_df=results_df,
        history_df=history_df,
        timing_df=timing_df,
        summary_df=summary_df,
        overall_df=overall_df,
    )

    print_summary(summary_df, overall_df)

    print("\nSaved files:")
    for path in saved_paths.values():
        print(f"  - {path}")

    print(f"Total benchmark wall time: {total_time_s / 60.0:.1f} min")

    if not config.skip_plots:
        make_plots(
            results_df=valid_results_df,
            history_df=history_df,
            summary_df=summary_df,
            overall_df=overall_df,
            config=config,
        )

    print("\nAlpha benchmark completed.")


if __name__ == "__main__":
    main()