"""
bo_kernel_benchmark.py
======================

Bayesian Optimization internal-kernel benchmark.

Purpose
-------
This script benchmarks the internal Gaussian Process kernel used by the
true-simulator Bayesian Optimization strategy.

It is Step B of the BO configuration-selection workflow:

    Step A: acquisition-function benchmark
    Step B: internal GP kernel benchmark

After Step A selects PI as the primary acquisition candidate, this script
compares:

    - Matern 3/2
    - Matern 5/2
    - RBF

using:
    - acquisition = PI
    - official targets: 0.55, 0.60, 0.65, 0.70, 0.75 m
    - multiple random seeds
    - fixed true-simulator budget
    - fixed number of Latin Hypercube initial points

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
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

# Existing true-simulator BO implementation. This file must be placed in src/.
import optimization_bo as obo


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_kernel_benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_kernel_benchmark"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_TARGETS = (0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_SEEDS = (2026, 2027)

DEFAULT_TOTAL_EVALUATIONS = 100
DEFAULT_INITIAL_POINTS = 20

DEFAULT_ACQUISITION_NAME = "pi"
DEFAULT_ACQUISITION_LABEL = "PI"
DEFAULT_PI_XI = 0.01

DEFAULT_KERNELS = (
    ("matern32", "Matern 3/2"),
    ("matern", "Matern 5/2"),
    ("rbf", "RBF"),
)

DEFAULT_GP_RESTARTS = 1
DEFAULT_GP_ALPHA = 1e-8

DEFAULT_CANDIDATE_POOL_GLOBAL = 4000
DEFAULT_CANDIDATE_POOL_LOCAL = 1000
DEFAULT_LOCAL_SIGMA_UNIT = 0.07

RESULTS_FILENAME = "bo_kernel_benchmark.csv"
HISTORY_FILENAME = "bo_kernel_benchmark_history.csv"
TIMING_FILENAME = "bo_kernel_benchmark_timing.csv"
SUMMARY_FILENAME = "bo_kernel_benchmark_summary.csv"
OVERALL_SUMMARY_FILENAME = "bo_kernel_benchmark_overall_summary.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class KernelBenchmarkConfig:
    """Configuration for the BO internal-kernel benchmark."""

    targets: tuple[float, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    total_evaluations: int = DEFAULT_TOTAL_EVALUATIONS
    initial_points: int = DEFAULT_INITIAL_POINTS

    acquisition_name: str = DEFAULT_ACQUISITION_NAME
    acquisition_label: str = DEFAULT_ACQUISITION_LABEL
    pi_xi: float = DEFAULT_PI_XI

    kernels: tuple[tuple[str, str], ...] = DEFAULT_KERNELS

    gp_restarts: int = DEFAULT_GP_RESTARTS
    gp_alpha: float = DEFAULT_GP_ALPHA

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


def safe_label(text: str) -> str:
    """Convert a label to a compact file/run-safe string."""
    return (
        text.replace(" ", "")
        .replace("/", "")
        .replace("=", "")
        .replace(".", "p")
        .replace("_", "")
    )


def kernel_order() -> list[str]:
    """Return preferred plotting order."""
    return ["Matern 3/2", "Matern 5/2", "RBF"]


def ordered_kernel_labels(available_labels: set[str]) -> list[str]:
    """Return available kernel labels in preferred order."""
    return [label for label in kernel_order() if label in available_labels]


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

    valid = results_df.dropna(subset=["target_error_mm"]).copy()

    required_cols = [
        "target",
        "kernel_label",
        "seed",
        "target_error_mm",
        "feasible_abs",
        "residual_margin_mm",
    ]

    missing = [col for col in required_cols if col not in valid.columns]

    if missing:
        print(f"Warning: valid result table is missing columns: {missing}")

    return valid


# =============================================================================
# MONKEY PATCHES: PI ACQUISITION AND MATERN 3/2 KERNEL
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


def patched_build_kernel(kernel_kind: str, config: Any) -> Any:
    """Build BO internal GP kernel with Matern 3/2, Matern 5/2 and RBF support."""
    dim = len(config.optimized_columns())

    length_scale_bounds = (
        config.gp_length_scale_bounds_low,
        config.gp_length_scale_bounds_high,
    )

    noise_bounds = (
        config.white_noise_bounds_low,
        config.white_noise_bounds_high,
    )

    kind = kernel_kind.lower()

    if kind in {"matern32", "matern_32", "matern3/2", "matern_3_2"}:
        base_kernel = Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=length_scale_bounds,
            nu=1.5,
        )

    elif kind in {"matern", "matern52", "matern_52", "matern5/2", "matern_5_2"}:
        base_kernel = Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=length_scale_bounds,
            nu=2.5,
        )

    elif kind == "rbf":
        base_kernel = RBF(
            length_scale=np.ones(dim),
            length_scale_bounds=length_scale_bounds,
        )

    else:
        raise ValueError(f"Unsupported kernel_kind: {kernel_kind!r}")

    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * base_kernel
        + WhiteKernel(
            noise_level=config.white_noise_level,
            noise_level_bounds=noise_bounds,
        )
    )


# Apply patches so optimization_bo.run_bo_single_target can use PI and Matern 3/2.
obo.acquisition_scores = patched_acquisition_scores
obo.build_kernel = patched_build_kernel


# =============================================================================
# BO CONFIGURATION AND RUNNERS
# =============================================================================

def make_bo_config(benchmark_config: KernelBenchmarkConfig) -> Any:
    """Create BOConfig for a kernel benchmark run."""
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
    config.main_acquisition = benchmark_config.acquisition_name
    config.gp_restarts = benchmark_config.gp_restarts
    config.gp_alpha = benchmark_config.gp_alpha

    config.candidate_pool_global = benchmark_config.candidate_pool_global
    config.candidate_pool_local = benchmark_config.candidate_pool_local
    config.local_sigma_unit = benchmark_config.local_sigma_unit

    # Dynamically used by patched acquisition.
    config.pi_xi = benchmark_config.pi_xi

    return config


def run_single_benchmark(
    target: float,
    kernel_kind: str,
    kernel_label: str,
    seed: int,
    benchmark_config: KernelBenchmarkConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run one BO kernel benchmark case."""
    config = make_bo_config(benchmark_config)

    run_label = (
        f"kernel_{safe_label(kernel_label)}_"
        f"target{int(round(target * 1000)):04d}_"
        f"seed{seed}"
    )

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=config,
        run_label=run_label,
        total_evaluations=benchmark_config.total_evaluations,
        n_initial_points=benchmark_config.initial_points,
        kernel_kind=kernel_kind,
        acquisition=benchmark_config.acquisition_name,
        seed=seed,
    )

    result = obo.build_result_row(best, config)

    result.update(
        {
            "benchmark": "kernel",
            "target": float(target),
            "kernel": kernel_kind,
            "kernel_label": kernel_label,
            "acquisition": benchmark_config.acquisition_name,
            "acquisition_label": benchmark_config.acquisition_label,
            "seed": int(seed),
            "budget": int(benchmark_config.total_evaluations),
            "initial_points": int(benchmark_config.initial_points),
            "gp_alpha": float(benchmark_config.gp_alpha),
            "optimization_time_s": float(timing["total_wall_time_s"]),
            "mean_iteration_time_s": float(timing["mean_iteration_time_s"]),
            "mean_simulation_time_s": float(timing["mean_simulation_time_s"]),
            "mean_gp_fit_time_s": float(timing["mean_gp_fit_time_s"]),
            "mean_acquisition_time_s": float(timing["mean_acquisition_time_s"]),
        }
    )

    history = history.copy()
    history["benchmark"] = "kernel"
    history["kernel"] = kernel_kind
    history["kernel_label"] = kernel_label
    history["acquisition"] = benchmark_config.acquisition_name
    history["acquisition_label"] = benchmark_config.acquisition_label
    history["seed"] = int(seed)
    history["budget"] = int(benchmark_config.total_evaluations)
    history["initial_points"] = int(benchmark_config.initial_points)
    history["gp_alpha"] = float(benchmark_config.gp_alpha)

    timing = dict(timing)
    timing.update(
        {
            "benchmark": "kernel",
            "kernel": kernel_kind,
            "kernel_label": kernel_label,
            "acquisition": benchmark_config.acquisition_name,
            "acquisition_label": benchmark_config.acquisition_label,
            "target": float(target),
            "seed": int(seed),
            "budget": int(benchmark_config.total_evaluations),
            "initial_points": int(benchmark_config.initial_points),
            "gp_alpha": float(benchmark_config.gp_alpha),
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
    kernel_label: str,
    seed: int,
    budget: int,
    acquisition_label: str,
) -> bool:
    """Check whether a benchmark run is already present."""
    if existing_results.empty:
        return False

    required_cols = {
        "target",
        "kernel_label",
        "seed",
        "budget",
        "acquisition_label",
    }

    if not required_cols.issubset(existing_results.columns):
        return False

    mask = (
        np.isclose(existing_results["target"].astype(float), float(target))
        & (existing_results["kernel_label"].astype(str) == kernel_label)
        & (existing_results["seed"].astype(int) == int(seed))
        & (existing_results["budget"].astype(int) == int(budget))
        & (existing_results["acquisition_label"].astype(str) == acquisition_label)
    )

    return bool(mask.any())


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate kernel benchmark results by target and kernel."""
    if results_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for (target, kernel_label), group in results_df.groupby(
        ["target", "kernel_label"],
        sort=False,
    ):
        group = group.copy()

        feasible_rate = 100.0 * float(group["feasible_abs"].astype(bool).mean())

        rows.append(
            {
                "target": float(target),
                "kernel_label": kernel_label,
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
    """Aggregate kernel performance across all targets."""
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    df["rank_by_error_within_target_seed"] = df.groupby(
        ["target", "seed"]
    )["target_error_mm"].rank(method="min", ascending=True)

    rows: list[dict[str, Any]] = []

    for kernel_label, group in df.groupby("kernel_label", sort=False):
        feasible_rate = 100.0 * float(group["feasible_abs"].astype(bool).mean())

        rows.append(
            {
                "kernel_label": kernel_label,
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

def plot_final_error_boxplots(results_df: pd.DataFrame) -> None:
    """Plot final target-error boxplots by kernel for each target."""
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)].copy()
        labels = ordered_kernel_labels(set(df_target["kernel_label"]))

        data = [
            df_target[df_target["kernel_label"] == label]["target_error_mm"].to_numpy()
            for label in labels
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_ylabel("Final target error [mm]")
        ax.set_xlabel("Internal GP kernel")
        ax.set_title(f"BO kernel benchmark: final error, target = {target:.2f} m")
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / (
            f"bo_kernel_final_error_boxplot_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_margin_boxplots(results_df: pd.DataFrame) -> None:
    """Plot residual-margin boxplots by kernel for each target."""
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)].copy()
        labels = ordered_kernel_labels(set(df_target["kernel_label"]))

        data = [
            df_target[df_target["kernel_label"] == label]["residual_margin_mm"].to_numpy()
            for label in labels
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.boxplot(data, labels=labels, showfliers=True)
        ax.axhline(0.0, linestyle="--", linewidth=1.8)

        ax.set_ylabel("Residual margin to true limit [mm]")
        ax.set_xlabel("Internal GP kernel")
        ax.set_title(f"BO kernel benchmark: residual margin, target = {target:.2f} m")
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / (
            f"bo_kernel_margin_boxplot_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_mean_error_bars(summary_df: pd.DataFrame) -> None:
    """Plot mean target error by kernel for each target."""
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_target = summary_df[np.isclose(summary_df["target"], target)].copy()
        labels = ordered_kernel_labels(set(df_target["kernel_label"]))

        df_target = (
            df_target.set_index("kernel_label")
            .loc[labels]
            .reset_index()
        )

        x = np.arange(len(df_target))

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.bar(
            x,
            df_target["mean_target_error_mm"],
            yerr=df_target["std_target_error_mm"],
            capsize=4,
            edgecolor="black",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df_target["kernel_label"].to_numpy())
        ax.set_ylabel("Mean final target error [mm]")
        ax.set_xlabel("Internal GP kernel")
        ax.set_title(f"BO kernel benchmark: mean final error, target = {target:.2f} m")
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / (
            f"bo_kernel_mean_error_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_convergence(history_df: pd.DataFrame) -> None:
    """Plot BO convergence by kernel for each target."""
    if history_df.empty:
        return

    if "best_feasible_error_so_far_mm" not in history_df.columns:
        print("Convergence plot skipped: missing best_feasible_error_so_far_mm.")
        return

    for target in sorted(history_df["target"].unique()):
        df_target = history_df[np.isclose(history_df["target"], target)].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        for label in ordered_kernel_labels(set(df_target["kernel_label"])):
            df_kernel = df_target[df_target["kernel_label"] == label].copy()

            if df_kernel.empty:
                continue

            grouped = (
                df_kernel.groupby("iteration")["best_feasible_error_so_far_mm"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .sort_values("iteration")
            )

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
        ax.set_title(f"BO kernel convergence, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_kernel_convergence_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_error_margin_tradeoff(results_df: pd.DataFrame) -> None:
    """Plot final target-error vs residual-margin trade-off."""
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        for label in ordered_kernel_labels(set(df_target["kernel_label"])):
            df_kernel = df_target[df_target["kernel_label"] == label]

            if df_kernel.empty:
                continue

            ax.scatter(
                df_kernel["target_error_mm"],
                df_kernel["residual_margin_mm"],
                s=70,
                edgecolor="black",
                label=label,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.8)
        ax.set_xlabel("Final target error [mm]")
        ax.set_ylabel("Residual margin to true limit [mm]")
        ax.set_title(f"BO kernel trade-off, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_kernel_error_margin_tradeoff_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_mean_error_heatmap(summary_df: pd.DataFrame) -> None:
    """Plot target-vs-kernel heatmap of mean final error."""
    if summary_df.empty:
        return

    targets = sorted(summary_df["target"].unique())
    labels = ordered_kernel_labels(set(summary_df["kernel_label"]))

    matrix = np.full((len(targets), len(labels)), np.nan)

    for i, target in enumerate(targets):
        for j, label in enumerate(labels):
            row = summary_df[
                np.isclose(summary_df["target"], target)
                & (summary_df["kernel_label"] == label)
            ]

            if not row.empty:
                matrix[i, j] = float(row["mean_target_error_mm"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax, label="Mean final target error [mm]")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels([f"{target:.2f}" for target in targets])

    ax.set_xlabel("Internal GP kernel")
    ax.set_ylabel("Target [m]")
    ax.set_title("BO kernel benchmark: mean error heatmap")

    for i in range(len(targets)):
        for j in range(len(labels)):
            value = matrix[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    path = FIGURES_DIR / "bo_kernel_mean_error_heatmap.png"

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_overall_mean_error(overall_df: pd.DataFrame) -> None:
    """Plot overall mean error across all targets by kernel."""
    if overall_df.empty:
        return

    labels = ordered_kernel_labels(set(overall_df["kernel_label"]))

    df = (
        overall_df.set_index("kernel_label")
        .loc[labels]
        .reset_index()
    )

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x,
        df["mean_target_error_mm"],
        yerr=df["std_target_error_mm"],
        capsize=4,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["kernel_label"])
    ax.set_ylabel("Mean target error across all targets [mm]")
    ax.set_xlabel("Internal GP kernel")
    ax.set_title("BO kernel benchmark: overall mean error")
    ax.grid(True, axis="y", alpha=0.3)

    path = FIGURES_DIR / "bo_kernel_overall_mean_error.png"

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def make_plots(
    results_df: pd.DataFrame,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    """Generate all benchmark plots."""
    print("\nGenerating benchmark plots...")

    plot_final_error_boxplots(results_df)
    plot_margin_boxplots(results_df)
    plot_mean_error_bars(summary_df)
    plot_convergence(history_df)
    plot_error_margin_tradeoff(results_df)
    plot_mean_error_heatmap(summary_df)
    plot_overall_mean_error(overall_df)


# =============================================================================
# PRINTING
# =============================================================================

def print_settings(config: KernelBenchmarkConfig) -> None:
    """Print benchmark settings."""
    print("=" * 80)
    print("BO KERNEL BENCHMARK")
    print("=" * 80)
    print(f"Targets:                 {config.targets}")
    print(f"Seeds:                   {config.seeds}")
    print(f"Acquisition:             {config.acquisition_label}")
    print(f"PI xi:                   {config.pi_xi}")
    print(f"Budget:                  {config.total_evaluations} calls")
    print(f"Initial LHS points:      {config.initial_points}")
    print(f"Kernels:                 {[kernel[1] for kernel in config.kernels]}")
    print(f"GP alpha:                {config.gp_alpha:.0e}")
    print(f"GP restarts:             {config.gp_restarts}")
    print(f"Global candidate pool:   {config.candidate_pool_global}")
    print(f"Local candidate pool:    {config.candidate_pool_local}")
    print(f"Resume existing:         {config.resume_existing}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)


def print_summary(summary_df: pd.DataFrame, overall_df: pd.DataFrame) -> None:
    """Print final benchmark summaries."""
    print("\n" + "=" * 80)
    print("KERNEL BENCHMARK SUMMARY")
    print("=" * 80)

    if not summary_df.empty:
        cols = [
            "target",
            "kernel_label",
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
    print("OVERALL KERNEL SUMMARY ACROSS ALL TARGETS")
    print("=" * 80)

    if not overall_df.empty:
        cols = [
            "kernel_label",
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
        description="Run BO internal-kernel benchmark."
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
        help="Do not resume from existing benchmark result files.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> KernelBenchmarkConfig:
    """Build benchmark configuration from command-line arguments."""
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")

    if args.initial_points <= 0:
        raise ValueError("--initial_points must be positive.")

    if args.initial_points >= args.budget:
        raise ValueError("--initial_points must be smaller than --budget.")

    if args.gp_restarts < 0:
        raise ValueError("--gp_restarts must be non-negative.")

    return KernelBenchmarkConfig(
        targets=parse_float_tuple(args.targets),
        seeds=parse_int_tuple(args.seeds),
        total_evaluations=int(args.budget),
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
    """Run the complete BO internal-kernel benchmark."""
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

    total_runs = len(config.targets) * len(config.kernels) * len(config.seeds)
    run_index = 0
    t_start = time.perf_counter()

    for target in config.targets:
        for kernel_kind, kernel_label in config.kernels:
            for seed in config.seeds:
                run_index += 1

                if config.resume_existing and already_done(
                    existing_results=existing_results,
                    target=target,
                    kernel_label=kernel_label,
                    seed=seed,
                    budget=config.total_evaluations,
                    acquisition_label=config.acquisition_label,
                ):
                    print(
                        f"[{run_index}/{total_runs}] SKIP existing | "
                        f"target={target:.2f}, kernel={kernel_label}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{run_index}/{total_runs}] "
                    f"target={target:.2f} | kernel={kernel_label} | "
                    f"acquisition={config.acquisition_label} | seed={seed}"
                )

                try:
                    result, history, timing = run_single_benchmark(
                        target=target,
                        kernel_kind=kernel_kind,
                        kernel_label=kernel_label,
                        seed=seed,
                        benchmark_config=config,
                    )

                except Exception as exc:
                    print(f"    ERROR: {exc}")

                    error_row = {
                        "benchmark": "kernel",
                        "target": float(target),
                        "kernel": kernel_kind,
                        "kernel_label": kernel_label,
                        "acquisition": config.acquisition_name,
                        "acquisition_label": config.acquisition_label,
                        "seed": int(seed),
                        "budget": int(config.total_evaluations),
                        "initial_points": int(config.initial_points),
                        "gp_alpha": float(config.gp_alpha),
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

    total_time_s = time.perf_counter() - t_start

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
        make_plots(valid_results_df, history_df, summary_df, overall_df)

    print("\nKernel benchmark completed.")


if __name__ == "__main__":
    main()