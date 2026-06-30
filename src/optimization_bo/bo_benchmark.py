"""
bo_benchmark.py
===============

Bayesian Optimization acquisition-function benchmark.

Purpose
-------
This script compares acquisition functions for the true-simulator Bayesian
Optimization strategy used in the extra-reach inverse problem.

Compared acquisition functions:
    - Expected Improvement (EI)
    - Probability of Improvement (PI)
    - Lower Confidence Bound (LCB) with several kappa values

The benchmark is run over the official outreach targets:

    0.55, 0.60, 0.65, 0.70, 0.75 m

and multiple random seeds. Results are saved incrementally so the benchmark can
be resumed if interrupted.

Important
---------
This script is an analysis/configuration-selection script. It does not replace
the final Bayesian Optimization script.

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

# Existing BO implementation. This file must be placed in src/.
import optimization_bo as obo


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_benchmark"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_TARGETS = (0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_SEEDS = (2026, 2027)

DEFAULT_TOTAL_EVALUATIONS = 100
DEFAULT_INITIAL_POINTS = 20

DEFAULT_KERNEL_KIND = "matern"

DEFAULT_GP_RESTARTS = 1
DEFAULT_CANDIDATE_POOL_GLOBAL = 4000
DEFAULT_CANDIDATE_POOL_LOCAL = 1000
DEFAULT_LOCAL_SIGMA_UNIT = 0.07

DEFAULT_EI_XI = 0.01
DEFAULT_PI_XI = 0.01

DEFAULT_ACQUISITIONS = (
    ("ei", "EI", None),
    ("pi", "PI", None),
    ("lcb", "LCB k=1", 1.0),
    ("lcb", "LCB k=2", 2.0),
    ("lcb", "LCB k=3", 3.0),
)

RESULTS_FILENAME = "bo_acquisition_benchmark.csv"
HISTORY_FILENAME = "bo_acquisition_benchmark_history.csv"
TIMING_FILENAME = "bo_acquisition_benchmark_timing.csv"
SUMMARY_FILENAME = "bo_acquisition_benchmark_summary.csv"
OVERALL_SUMMARY_FILENAME = "bo_acquisition_benchmark_overall_summary.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AcquisitionBenchmarkConfig:
    """Configuration for the BO acquisition-function benchmark."""

    targets: tuple[float, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    total_evaluations: int = DEFAULT_TOTAL_EVALUATIONS
    initial_points: int = DEFAULT_INITIAL_POINTS

    kernel_kind: str = DEFAULT_KERNEL_KIND
    acquisitions: tuple[tuple[str, str, float | None], ...] = DEFAULT_ACQUISITIONS

    gp_restarts: int = DEFAULT_GP_RESTARTS
    candidate_pool_global: int = DEFAULT_CANDIDATE_POOL_GLOBAL
    candidate_pool_local: int = DEFAULT_CANDIDATE_POOL_LOCAL
    local_sigma_unit: float = DEFAULT_LOCAL_SIGMA_UNIT

    ei_xi: float = DEFAULT_EI_XI
    pi_xi: float = DEFAULT_PI_XI

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


def sanitize_label(label: str) -> str:
    """Convert an acquisition label to a compact file/run-safe string."""
    return (
        label.replace(" ", "")
        .replace("=", "")
        .replace(".", "p")
        .replace("/", "_")
    )


def acquisition_order() -> list[str]:
    """Return preferred plotting order."""
    return ["EI", "PI", "LCB k=1", "LCB k=2", "LCB k=3"]


def ordered_labels(available_labels: set[str]) -> list[str]:
    """Return acquisition labels in preferred order."""
    return [label for label in acquisition_order() if label in available_labels]


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


# =============================================================================
# MONKEY PATCH: ADD PI ACQUISITION TO optimization_bo
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

    PI is usually greedier than EI because it rewards the probability of
    improvement rather than the magnitude of improvement.
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
    benchmark_config: AcquisitionBenchmarkConfig,
    kappa: float | None,
) -> Any:
    """Create a BOConfig configured for benchmark runs."""
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
    config.gp_restarts = benchmark_config.gp_restarts
    config.candidate_pool_global = benchmark_config.candidate_pool_global
    config.candidate_pool_local = benchmark_config.candidate_pool_local
    config.local_sigma_unit = benchmark_config.local_sigma_unit

    config.ei_xi = benchmark_config.ei_xi
    config.pi_xi = benchmark_config.pi_xi  # dynamically used by patched acquisition

    if kappa is not None:
        config.lcb_kappa = float(kappa)

    return config


def run_single_benchmark(
    target: float,
    acquisition_name: str,
    acquisition_label: str,
    kappa: float | None,
    seed: int,
    benchmark_config: AcquisitionBenchmarkConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run one BO acquisition benchmark case."""
    config = make_bo_config(benchmark_config, kappa)

    run_label = (
        f"acq_{sanitize_label(acquisition_label)}_"
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
        acquisition=acquisition_name,
        seed=seed,
    )

    result = obo.build_result_row(best, config)

    result.update(
        {
            "benchmark": "acquisition",
            "target": float(target),
            "kernel": benchmark_config.kernel_kind,
            "acquisition": acquisition_name,
            "acquisition_label": acquisition_label,
            "kappa": np.nan if kappa is None else float(kappa),
            "seed": int(seed),
            "budget": int(benchmark_config.total_evaluations),
            "initial_points": int(benchmark_config.initial_points),
            "optimization_time_s": float(timing["total_wall_time_s"]),
            "mean_iteration_time_s": float(timing["mean_iteration_time_s"]),
            "mean_simulation_time_s": float(timing["mean_simulation_time_s"]),
            "mean_gp_fit_time_s": float(timing["mean_gp_fit_time_s"]),
            "mean_acquisition_time_s": float(timing["mean_acquisition_time_s"]),
        }
    )

    history = history.copy()
    history["benchmark"] = "acquisition"
    history["acquisition"] = acquisition_name
    history["acquisition_label"] = acquisition_label
    history["kappa"] = np.nan if kappa is None else float(kappa)
    history["seed"] = int(seed)
    history["budget"] = int(benchmark_config.total_evaluations)
    history["initial_points"] = int(benchmark_config.initial_points)

    timing = dict(timing)
    timing.update(
        {
            "benchmark": "acquisition",
            "acquisition": acquisition_name,
            "acquisition_label": acquisition_label,
            "kappa": np.nan if kappa is None else float(kappa),
            "seed": int(seed),
            "target": float(target),
            "budget": int(benchmark_config.total_evaluations),
            "initial_points": int(benchmark_config.initial_points),
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
    acquisition_label: str,
    seed: int,
    budget: int,
) -> bool:
    """Check whether a benchmark run is already present."""
    if existing_results.empty:
        return False

    required_cols = {"target", "acquisition_label", "seed", "budget"}

    if not required_cols.issubset(existing_results.columns):
        return False

    mask = (
        np.isclose(existing_results["target"].astype(float), float(target))
        & (existing_results["acquisition_label"].astype(str) == acquisition_label)
        & (existing_results["seed"].astype(int) == int(seed))
        & (existing_results["budget"].astype(int) == int(budget))
    )

    return bool(mask.any())


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark results by target and acquisition."""
    if results_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for (target, acquisition_label), group in results_df.groupby(
        ["target", "acquisition_label"],
        sort=False,
    ):
        group = group.copy()

        feasible_rate = 100.0 * float(group["feasible_abs"].astype(bool).mean())

        rows.append(
            {
                "target": float(target),
                "acquisition_label": acquisition_label,
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
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["target", "mean_target_error_mm", "mean_residual_margin_mm"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return summary


def build_overall_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate acquisition performance across all targets.

    This table is used to select a single global acquisition strategy. It
    includes average error, worst-case error, residual margin and mean rank.
    """
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    df["rank_by_error_within_target_seed"] = df.groupby(
        ["target", "seed"]
    )["target_error_mm"].rank(method="min", ascending=True)

    rows: list[dict[str, Any]] = []

    for acquisition_label, group in df.groupby("acquisition_label", sort=False):
        feasible_rate = 100.0 * float(group["feasible_abs"].astype(bool).mean())

        rows.append(
            {
                "acquisition_label": acquisition_label,
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
    """Plot final target-error boxplots by acquisition for each target."""
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)].copy()
        labels = ordered_labels(set(df_target["acquisition_label"]))

        data = [
            df_target[df_target["acquisition_label"] == label]["target_error_mm"].to_numpy()
            for label in labels
        ]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_ylabel("Final target error [mm]")
        ax.set_xlabel("Acquisition function")
        ax.set_title(f"BO acquisition benchmark: final error, target = {target:.2f} m")
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / (
            f"bo_acquisition_final_error_boxplot_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_margin_boxplots(results_df: pd.DataFrame) -> None:
    """Plot residual-margin boxplots by acquisition for each target."""
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)].copy()
        labels = ordered_labels(set(df_target["acquisition_label"]))

        data = [
            df_target[df_target["acquisition_label"] == label]["residual_margin_mm"].to_numpy()
            for label in labels
        ]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.boxplot(data, labels=labels, showfliers=True)
        ax.axhline(0.0, linestyle="--", linewidth=1.8)

        ax.set_ylabel("Residual margin to true limit [mm]")
        ax.set_xlabel("Acquisition function")
        ax.set_title(f"BO acquisition benchmark: residual margin, target = {target:.2f} m")
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / (
            f"bo_acquisition_margin_boxplot_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_mean_error_bars(summary_df: pd.DataFrame) -> None:
    """Plot mean target error by acquisition for each target."""
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_target = summary_df[np.isclose(summary_df["target"], target)].copy()
        labels = ordered_labels(set(df_target["acquisition_label"]))

        df_target = (
            df_target.set_index("acquisition_label")
            .loc[labels]
            .reset_index()
        )

        x = np.arange(len(df_target))

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.bar(
            x,
            df_target["mean_target_error_mm"],
            yerr=df_target["std_target_error_mm"],
            capsize=4,
            edgecolor="black",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df_target["acquisition_label"].to_numpy())
        ax.set_ylabel("Mean final target error [mm]")
        ax.set_xlabel("Acquisition function")
        ax.set_title(f"BO acquisition benchmark: mean final error, target = {target:.2f} m")
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / (
            f"bo_acquisition_mean_error_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_convergence(history_df: pd.DataFrame) -> None:
    """Plot BO convergence by acquisition for each target."""
    if history_df.empty:
        return

    if "best_feasible_error_so_far_mm" not in history_df.columns:
        print("Convergence plot skipped: missing best_feasible_error_so_far_mm.")
        return

    for target in sorted(history_df["target"].unique()):
        df_target = history_df[np.isclose(history_df["target"], target)].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        for label in ordered_labels(set(df_target["acquisition_label"])):
            df_acq = df_target[df_target["acquisition_label"] == label].copy()

            if df_acq.empty:
                continue

            grouped = (
                df_acq.groupby("iteration")["best_feasible_error_so_far_mm"]
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
        ax.set_title(f"BO acquisition convergence, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_acquisition_convergence_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_error_margin_tradeoff(results_df: pd.DataFrame) -> None:
    """Plot target-error vs residual-margin trade-off."""
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        for label in ordered_labels(set(df_target["acquisition_label"])):
            df_acq = df_target[df_target["acquisition_label"] == label]

            if df_acq.empty:
                continue

            ax.scatter(
                df_acq["target_error_mm"],
                df_acq["residual_margin_mm"],
                s=70,
                edgecolor="black",
                label=label,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.8)
        ax.set_xlabel("Final target error [mm]")
        ax.set_ylabel("Residual margin to true limit [mm]")
        ax.set_title(f"BO acquisition trade-off, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / (
            f"bo_acquisition_error_margin_tradeoff_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_mean_error_heatmap(summary_df: pd.DataFrame) -> None:
    """Plot target-vs-acquisition heatmap of mean final error."""
    if summary_df.empty:
        return

    targets = sorted(summary_df["target"].unique())
    labels = ordered_labels(set(summary_df["acquisition_label"]))

    matrix = np.full((len(targets), len(labels)), np.nan)

    for i, target in enumerate(targets):
        for j, label in enumerate(labels):
            row = summary_df[
                np.isclose(summary_df["target"], target)
                & (summary_df["acquisition_label"] == label)
            ]

            if not row.empty:
                matrix[i, j] = float(row["mean_target_error_mm"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax, label="Mean final target error [mm]")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels([f"{target:.2f}" for target in targets])

    ax.set_xlabel("Acquisition function")
    ax.set_ylabel("Target [m]")
    ax.set_title("BO acquisition benchmark: mean error heatmap")

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

    path = FIGURES_DIR / "bo_acquisition_mean_error_heatmap.png"

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_overall_mean_error(overall_df: pd.DataFrame) -> None:
    """Plot overall mean error across all targets by acquisition."""
    if overall_df.empty:
        return

    labels = ordered_labels(set(overall_df["acquisition_label"]))

    df = (
        overall_df.set_index("acquisition_label")
        .loc[labels]
        .reset_index()
    )

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x,
        df["mean_target_error_mm"],
        yerr=df["std_target_error_mm"],
        capsize=4,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["acquisition_label"])
    ax.set_ylabel("Mean target error across all targets [mm]")
    ax.set_xlabel("Acquisition function")
    ax.set_title("BO acquisition benchmark: overall mean error")
    ax.grid(True, axis="y", alpha=0.3)

    path = FIGURES_DIR / "bo_acquisition_overall_mean_error.png"

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

def print_settings(config: AcquisitionBenchmarkConfig) -> None:
    """Print benchmark settings."""
    print("=" * 80)
    print("BO ACQUISITION BENCHMARK")
    print("=" * 80)
    print(f"Targets:                 {config.targets}")
    print(f"Seeds:                   {config.seeds}")
    print(f"Kernel:                  {config.kernel_kind}")
    print(f"Budget:                  {config.total_evaluations} calls")
    print(f"Initial LHS points:      {config.initial_points}")
    print(f"Acquisitions:            {[item[1] for item in config.acquisitions]}")
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
    print("ACQUISITION BENCHMARK SUMMARY")
    print("=" * 80)

    if not summary_df.empty:
        cols = [
            "target",
            "acquisition_label",
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

    print("\n" + "=" * 80)
    print("OVERALL ACQUISITION SUMMARY ACROSS ALL TARGETS")
    print("=" * 80)

    if not overall_df.empty:
        cols = [
            "acquisition_label",
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
        ]

        available_cols = [col for col in cols if col in overall_df.columns]
        print(overall_df[available_cols].to_string(index=False))


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BO acquisition-function benchmark."
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
        "--kernel",
        type=str,
        default=DEFAULT_KERNEL_KIND,
        choices=["matern", "rbf"],
        help=f"Internal BO GP kernel. Default: {DEFAULT_KERNEL_KIND}.",
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


def build_config(args: argparse.Namespace) -> AcquisitionBenchmarkConfig:
    """Build benchmark configuration from command-line arguments."""
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")

    if args.initial_points <= 0:
        raise ValueError("--initial_points must be positive.")

    if args.initial_points >= args.budget:
        raise ValueError("--initial_points must be smaller than --budget.")

    return AcquisitionBenchmarkConfig(
        targets=parse_float_tuple(args.targets),
        seeds=parse_int_tuple(args.seeds),
        total_evaluations=int(args.budget),
        initial_points=int(args.initial_points),
        kernel_kind=str(args.kernel),
        gp_restarts=int(args.gp_restarts),
        resume_existing=not bool(args.no_resume),
        skip_plots=bool(args.skip_plots),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete BO acquisition benchmark."""
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

    total_runs = len(config.targets) * len(config.acquisitions) * len(config.seeds)
    run_counter = 0

    t0_all = time.perf_counter()

    for target in config.targets:
        for acquisition_name, acquisition_label, kappa in config.acquisitions:
            for seed in config.seeds:
                run_counter += 1

                if config.resume_existing and already_done(
                    existing_results=existing_results,
                    target=target,
                    acquisition_label=acquisition_label,
                    seed=seed,
                    budget=config.total_evaluations,
                ):
                    print(
                        f"[{run_counter}/{total_runs}] SKIP existing | "
                        f"target={target:.2f}, acquisition={acquisition_label}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{run_counter}/{total_runs}] "
                    f"target={target:.2f} | acquisition={acquisition_label} | seed={seed}"
                )

                result, history, timing = run_single_benchmark(
                    target=target,
                    acquisition_name=acquisition_name,
                    acquisition_label=acquisition_label,
                    kappa=kappa,
                    seed=seed,
                    benchmark_config=config,
                )

                result_rows.append(result)
                history_frames.append(history)
                timing_rows.append(timing)

                results_df = pd.DataFrame(result_rows)
                history_df = (
                    pd.concat(history_frames, ignore_index=True)
                    if history_frames
                    else pd.DataFrame()
                )
                timing_df = pd.DataFrame(timing_rows)
                summary_df = build_summary(results_df)
                overall_df = build_overall_summary(results_df)

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

    total_wall_time_s = time.perf_counter() - t0_all

    results_df = pd.DataFrame(result_rows)
    history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )
    timing_df = pd.DataFrame(timing_rows)
    summary_df = build_summary(results_df)
    overall_df = build_overall_summary(results_df)

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

    print(f"Total benchmark wall time: {total_wall_time_s / 60.0:.1f} min")

    if not config.skip_plots:
        make_plots(results_df, history_df, summary_df, overall_df)

    print("\nDone.")


if __name__ == "__main__":
    main()