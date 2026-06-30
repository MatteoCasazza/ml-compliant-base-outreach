"""
bo_final_run.py
===============

Final Bayesian Optimization run after the BO benchmark stages.

Selected configuration
----------------------
acquisition = PI
kernel      = RBF
gp_alpha    = 1e-8
budget      = 200 true simulator calls per target
initial LHS = 20 points

Purpose
-------
Run the final true-simulator Bayesian Optimization on all official targets:

    0.55, 0.60, 0.65, 0.70, 0.75 m

By default, two random seeds are used so that stochastic variability can be
reported. A same-budget Random Search baseline can also be run for comparison.

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

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_final"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_final"

# Make helper plots/functions inside optimization_bo save in the final folders if used.
obo.RESULTS_DIR = RESULTS_DIR
obo.FIGURES_DIR = FIGURES_DIR


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_TARGETS = (0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_SEEDS = (2026, 2027)

DEFAULT_TOTAL_EVALUATIONS = 200
DEFAULT_INITIAL_POINTS = 20

DEFAULT_ACQUISITION_NAME = "pi"
DEFAULT_ACQUISITION_LABEL = "PI"

DEFAULT_KERNEL_KIND = "rbf"
DEFAULT_KERNEL_LABEL = "RBF"

DEFAULT_GP_ALPHA = 1e-8
DEFAULT_PI_XI = 0.01
DEFAULT_GP_RESTARTS = 1

DEFAULT_CANDIDATE_POOL_GLOBAL = 4000
DEFAULT_CANDIDATE_POOL_LOCAL = 1000
DEFAULT_LOCAL_SIGMA_UNIT = 0.07


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class FinalBOConfig:
    """Final BO configuration selected after acquisition/kernel/alpha/budget tests."""

    targets: tuple[float, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    acquisition_name: str = DEFAULT_ACQUISITION_NAME
    acquisition_label: str = DEFAULT_ACQUISITION_LABEL

    kernel_kind: str = DEFAULT_KERNEL_KIND
    kernel_label: str = DEFAULT_KERNEL_LABEL

    gp_alpha: float = DEFAULT_GP_ALPHA
    pi_xi: float = DEFAULT_PI_XI
    gp_restarts: int = DEFAULT_GP_RESTARTS

    total_evaluations: int = DEFAULT_TOTAL_EVALUATIONS
    initial_points: int = DEFAULT_INITIAL_POINTS

    candidate_pool_global: int = DEFAULT_CANDIDATE_POOL_GLOBAL
    candidate_pool_local: int = DEFAULT_CANDIDATE_POOL_LOCAL
    local_sigma_unit: float = DEFAULT_LOCAL_SIGMA_UNIT

    run_random_search_baseline: bool = True
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
    """Return mean of a column if available."""
    if df.empty or column not in df.columns:
        return np.nan

    return float(df[column].mean())


def sum_or_nan(df: pd.DataFrame, column: str) -> float:
    """Return sum of a column if available."""
    if df.empty or column not in df.columns:
        return np.nan

    return float(df[column].sum())


def save_table_markdown(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as markdown without requiring tabulate."""
    columns = list(df.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# PI ACQUISITION PATCH
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
# BO CONFIGURATION
# =============================================================================

def make_bo_config(config: FinalBOConfig) -> Any:
    """Create the optimization_bo.BOConfig used by the final run."""
    bo_config = obo.BOConfig()

    # This script controls the workflow directly.
    bo_config.run_pilot = False
    bo_config.stop_after_pilot = False
    bo_config.run_main_bo = False
    bo_config.run_random_search = False
    bo_config.run_kernel_sensitivity = False
    bo_config.run_acquisition_sensitivity = False
    bo_config.run_budget_sweep = False
    bo_config.make_time_response_plots = False

    bo_config.targets = tuple(config.targets)
    bo_config.main_kernel = config.kernel_kind
    bo_config.main_acquisition = config.acquisition_name
    bo_config.main_total_evaluations = int(config.total_evaluations)
    bo_config.main_initial_points = int(config.initial_points)

    bo_config.gp_alpha = float(config.gp_alpha)
    bo_config.gp_restarts = int(config.gp_restarts)

    bo_config.candidate_pool_global = int(config.candidate_pool_global)
    bo_config.candidate_pool_local = int(config.candidate_pool_local)
    bo_config.local_sigma_unit = float(config.local_sigma_unit)

    # Dynamically used by patched acquisition.
    bo_config.pi_xi = float(config.pi_xi)

    return bo_config


def bo_run_label(
    target: float,
    seed: int,
    config: FinalBOConfig,
) -> str:
    """Return stable run label for a final BO trajectory."""
    target_mm = int(round(target * 1000.0))

    label = (
        f"final_BO_{config.acquisition_label}_{config.kernel_label}_"
        f"alpha{safe_scientific_label(config.gp_alpha)}_"
        f"budget{config.total_evaluations}_"
        f"target{target_mm:04d}_seed{seed}"
    )

    return (
        label.replace("+", "p")
        .replace("-", "m")
        .replace("/", "")
        .replace(" ", "")
    )


def random_search_run_label(
    target: float,
    seed: int,
    config: FinalBOConfig,
) -> str:
    """Return stable run label for a Random Search baseline trajectory."""
    target_mm = int(round(target * 1000.0))

    return f"final_RandomSearch_budget{config.total_evaluations}_target{target_mm:04d}_seed{seed}"


# =============================================================================
# RESUME HELPERS
# =============================================================================

def has_complete_existing_history(
    existing_history: pd.DataFrame,
    run_label: str,
    required_iterations: int,
) -> bool:
    """Check whether a complete run already exists in the saved history."""
    if existing_history.empty:
        return False

    required_cols = {"run_label", "iteration"}

    if not required_cols.issubset(existing_history.columns):
        return False

    mask = existing_history["run_label"].astype(str) == run_label

    if not mask.any():
        return False

    max_iteration = int(existing_history.loc[mask, "iteration"].max())

    return max_iteration >= int(required_iterations)


def prepare_loaded_history(
    history: pd.DataFrame,
    method: str,
    target: float,
    seed: int,
    config: FinalBOConfig,
) -> pd.DataFrame:
    """Attach metadata to loaded history."""
    history = history.sort_values("iteration").copy()
    history = obo.add_cumulative_best_columns(history)

    history["method"] = method
    history["target"] = float(target)
    history["seed"] = int(seed)

    if method == "BO":
        history["kernel"] = config.kernel_kind
        history["kernel_label"] = config.kernel_label
        history["acquisition"] = config.acquisition_name
        history["acquisition_label"] = config.acquisition_label
        history["gp_alpha"] = float(config.gp_alpha)
        history["alpha_label"] = config.alpha_label
    else:
        history["kernel"] = "none"
        history["kernel_label"] = "none"
        history["acquisition"] = "none"
        history["acquisition_label"] = "none"
        history["gp_alpha"] = np.nan
        history["alpha_label"] = "none"

    history["budget"] = int(config.total_evaluations)

    return history


def timing_from_loaded_history(
    history: pd.DataFrame,
    method: str,
    run_label: str,
    target: float,
    seed: int,
    config: FinalBOConfig,
) -> dict[str, Any]:
    """Create timing row when a run is loaded from saved history."""
    return {
        "method": method,
        "run_label": run_label,
        "target": float(target),
        "seed": int(seed),
        "loaded_from_existing": True,
        "budget": int(config.total_evaluations),
        "total_wall_time_s": sum_or_nan(history, "iteration_time_s"),
        "mean_iteration_time_s": mean_or_nan(history, "iteration_time_s"),
        "mean_simulation_time_s": mean_or_nan(history, "simulation_time_s"),
        "mean_gp_fit_time_s": (
            mean_or_nan(history, "gp_fit_time_s") if method == "BO" else 0.0
        ),
        "mean_acquisition_time_s": (
            mean_or_nan(history, "acquisition_time_s") if method == "BO" else 0.0
        ),
    }


# =============================================================================
# RUNNERS
# =============================================================================

def run_or_load_bo(
    target: float,
    seed: int,
    config: FinalBOConfig,
    existing_history: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Run or load one final BO trajectory."""
    bo_config = make_bo_config(config)
    run_label = bo_run_label(target, seed, config)

    if config.resume_existing and has_complete_existing_history(
        existing_history=existing_history,
        run_label=run_label,
        required_iterations=config.total_evaluations,
    ):
        history = existing_history[
            existing_history["run_label"].astype(str) == run_label
        ].copy()

        history = prepare_loaded_history(
            history=history,
            method="BO",
            target=target,
            seed=seed,
            config=config,
        )

        best = obo.select_best_from_history(history)

        timing = timing_from_loaded_history(
            history=history,
            method="BO",
            run_label=run_label,
            target=target,
            seed=seed,
            config=config,
        )

        print(f"    SKIP existing BO | target={target:.2f}, seed={seed}")

        return best, history, timing

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=bo_config,
        run_label=run_label,
        total_evaluations=config.total_evaluations,
        n_initial_points=config.initial_points,
        kernel_kind=config.kernel_kind,
        acquisition=config.acquisition_name,
        seed=seed,
    )

    history = history.copy()
    history["method"] = "BO"
    history["target"] = float(target)
    history["seed"] = int(seed)
    history["kernel"] = config.kernel_kind
    history["kernel_label"] = config.kernel_label
    history["acquisition"] = config.acquisition_name
    history["acquisition_label"] = config.acquisition_label
    history["gp_alpha"] = float(config.gp_alpha)
    history["alpha_label"] = config.alpha_label
    history["budget"] = int(config.total_evaluations)

    timing = dict(timing)
    timing.update(
        {
            "method": "BO",
            "run_label": run_label,
            "target": float(target),
            "seed": int(seed),
            "loaded_from_existing": False,
            "budget": int(config.total_evaluations),
            "kernel": config.kernel_kind,
            "kernel_label": config.kernel_label,
            "acquisition": config.acquisition_name,
            "acquisition_label": config.acquisition_label,
            "gp_alpha": float(config.gp_alpha),
            "alpha_label": config.alpha_label,
        }
    )

    return best, history, timing


def run_or_load_random_search(
    target: float,
    seed: int,
    config: FinalBOConfig,
    existing_history: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Run or load one same-budget Random Search trajectory."""
    bo_config = make_bo_config(config)
    run_label = random_search_run_label(target, seed, config)

    if config.resume_existing and has_complete_existing_history(
        existing_history=existing_history,
        run_label=run_label,
        required_iterations=config.total_evaluations,
    ):
        history = existing_history[
            existing_history["run_label"].astype(str) == run_label
        ].copy()

        history = prepare_loaded_history(
            history=history,
            method="RandomSearch",
            target=target,
            seed=seed,
            config=config,
        )

        best = obo.select_best_from_history(history)

        timing = timing_from_loaded_history(
            history=history,
            method="RandomSearch",
            run_label=run_label,
            target=target,
            seed=seed,
            config=config,
        )

        print(f"    SKIP existing Random Search | target={target:.2f}, seed={seed}")

        return best, history, timing

    best, history, timing = obo.run_random_search_single_target(
        y_target=target,
        config=bo_config,
        run_label=run_label,
        total_evaluations=config.total_evaluations,
        seed=seed,
    )

    history = history.copy()
    history["method"] = "RandomSearch"
    history["target"] = float(target)
    history["seed"] = int(seed)
    history["kernel"] = "none"
    history["kernel_label"] = "none"
    history["acquisition"] = "none"
    history["acquisition_label"] = "none"
    history["gp_alpha"] = np.nan
    history["alpha_label"] = "none"
    history["budget"] = int(config.total_evaluations)

    timing = dict(timing)
    timing.update(
        {
            "method": "RandomSearch",
            "run_label": run_label,
            "target": float(target),
            "seed": int(seed),
            "loaded_from_existing": False,
            "budget": int(config.total_evaluations),
            "kernel": "none",
            "kernel_label": "none",
            "acquisition": "none",
            "acquisition_label": "none",
            "gp_alpha": np.nan,
            "alpha_label": "none",
        }
    )

    return best, history, timing


# =============================================================================
# RESULTS AND SUMMARIES
# =============================================================================

def build_result_record(
    best_row: pd.Series,
    method: str,
    timing: dict[str, Any],
    config: FinalBOConfig,
) -> dict[str, Any]:
    """Build final result row with consistent metadata."""
    bo_config = make_bo_config(config)

    result = obo.build_result_row(best_row, bo_config)
    result = dict(result)

    result["method"] = method
    result["target"] = float(timing["target"])
    result["seed"] = int(timing["seed"])
    result["run_label"] = str(timing["run_label"])
    result["budget"] = int(config.total_evaluations)
    result["n_true_simulator_evaluations"] = int(config.total_evaluations)
    result["optimization_time_s"] = float(timing["total_wall_time_s"])
    result["loaded_from_existing"] = bool(timing.get("loaded_from_existing", False))

    if method == "BO":
        result.update(
            {
                "kernel": config.kernel_kind,
                "kernel_label": config.kernel_label,
                "acquisition": config.acquisition_name,
                "acquisition_label": config.acquisition_label,
                "gp_alpha": float(config.gp_alpha),
                "alpha_label": config.alpha_label,
                "initial_points": int(config.initial_points),
                "candidate_pool_global": int(config.candidate_pool_global),
                "candidate_pool_local": int(config.candidate_pool_local),
                "pi_xi": float(config.pi_xi),
                "gp_restarts": int(config.gp_restarts),
            }
        )
    else:
        result.update(
            {
                "kernel": "none",
                "kernel_label": "none",
                "acquisition": "none",
                "acquisition_label": "none",
                "gp_alpha": np.nan,
                "alpha_label": "none",
                "initial_points": np.nan,
                "candidate_pool_global": np.nan,
                "candidate_pool_local": np.nan,
                "pi_xi": np.nan,
                "gp_restarts": np.nan,
            }
        )

    return result


def summarize_by_target(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final results by method and target."""
    if results_df.empty:
        return pd.DataFrame()

    return (
        results_df.groupby(["method", "target"], dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            feasibility_rate_percent=(
                "feasible_abs",
                lambda series: 100.0 * series.astype(bool).mean(),
            ),
            mean_peak_y_true=("peak_y_true", "mean"),
            std_peak_y_true=("peak_y_true", "std"),
            mean_target_error_mm=("target_error_mm", "mean"),
            std_target_error_mm=("target_error_mm", "std"),
            min_target_error_mm=("target_error_mm", "min"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_max_abs_xr_true=("max_abs_xr_true", "mean"),
            max_max_abs_xr_true=("max_abs_xr_true", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_optimization_time_s=("optimization_time_s", "mean"),
            total_true_simulator_calls=("n_true_simulator_evaluations", "sum"),
        )
        .reset_index()
        .sort_values(["method", "target"])
        .reset_index(drop=True)
    )


def summarize_overall(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final results overall by method."""
    if results_df.empty:
        return pd.DataFrame()

    return (
        results_df.groupby("method", dropna=False)
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
            mean_peak_y_true=("peak_y_true", "mean"),
            mean_max_abs_xr_true=("max_abs_xr_true", "mean"),
            max_max_abs_xr_true=("max_abs_xr_true", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_optimization_time_s=("optimization_time_s", "mean"),
            total_optimization_time_s=("optimization_time_s", "sum"),
            total_true_simulator_calls=("n_true_simulator_evaluations", "sum"),
        )
        .reset_index()
        .sort_values("method")
        .reset_index(drop=True)
    )


def make_report_table(by_target_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact report table."""
    if by_target_df.empty:
        return pd.DataFrame()

    report = by_target_df[
        [
            "method",
            "target",
            "n_runs",
            "feasibility_rate_percent",
            "mean_peak_y_true",
            "mean_target_error_mm",
            "std_target_error_mm",
            "mean_max_abs_xr_true",
            "mean_residual_margin_mm",
            "min_residual_margin_mm",
            "mean_best_iteration",
        ]
    ].copy()

    report = report.rename(
        columns={
            "method": "Method",
            "target": "Target [m]",
            "n_runs": "Runs",
            "feasibility_rate_percent": "Feasible [%]",
            "mean_peak_y_true": "Mean peak_y [m]",
            "mean_target_error_mm": "Mean error [mm]",
            "std_target_error_mm": "Std error [mm]",
            "mean_max_abs_xr_true": "Mean max_abs_xr [m]",
            "mean_residual_margin_mm": "Mean margin [mm]",
            "min_residual_margin_mm": "Min margin [mm]",
            "mean_best_iteration": "Mean best call",
        }
    )

    numeric_cols = report.select_dtypes(include=[np.number]).columns
    report[numeric_cols] = report[numeric_cols].round(3)

    return report


def save_outputs(
    bo_history: pd.DataFrame,
    bo_results: pd.DataFrame,
    timing_df: pd.DataFrame,
    all_history: pd.DataFrame,
    all_results: pd.DataFrame,
    by_target_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    report_df: pd.DataFrame,
    random_history: pd.DataFrame | None = None,
    random_results: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Save all final BO outputs."""
    paths = {
        "bo_history": RESULTS_DIR / "bo_final_history.csv",
        "bo_results": RESULTS_DIR / "bo_final_results.csv",
        "bo_results_by_seed": RESULTS_DIR / "bo_final_results_by_seed.csv",
        "timing": RESULTS_DIR / "bo_final_timing.csv",
        "all_history": RESULTS_DIR / "bo_final_all_history.csv",
        "all_results": RESULTS_DIR / "bo_final_all_results.csv",
        "all_results_by_seed": RESULTS_DIR / "bo_final_all_results_by_seed.csv",
        "summary_by_target": RESULTS_DIR / "bo_final_summary_by_target.csv",
        "summary_overall": RESULTS_DIR / "bo_final_summary_overall.csv",
        "report_table": RESULTS_DIR / "bo_final_report_table.csv",
        "report_table_md": RESULTS_DIR / "bo_final_report_table.md",
    }

    bo_history.to_csv(paths["bo_history"], index=False)
    bo_results.to_csv(paths["bo_results"], index=False)
    bo_results.to_csv(paths["bo_results_by_seed"], index=False)

    timing_df.to_csv(paths["timing"], index=False)

    all_history.to_csv(paths["all_history"], index=False)
    all_results.to_csv(paths["all_results"], index=False)
    all_results.to_csv(paths["all_results_by_seed"], index=False)

    by_target_df.to_csv(paths["summary_by_target"], index=False)
    overall_df.to_csv(paths["summary_overall"], index=False)

    report_df.to_csv(paths["report_table"], index=False)
    save_table_markdown(report_df, paths["report_table_md"])

    if random_history is not None and not random_history.empty:
        path = RESULTS_DIR / "random_search_final_history.csv"
        random_history.to_csv(path, index=False)
        paths["random_history"] = path

    if random_results is not None and not random_results.empty:
        path = RESULTS_DIR / "random_search_results.csv"
        random_results.to_csv(path, index=False)
        paths["random_results"] = path

        path_by_seed = RESULTS_DIR / "random_search_final_results_by_seed.csv"
        random_results.to_csv(path_by_seed, index=False)
        paths["random_results_by_seed"] = path_by_seed

    return paths


# =============================================================================
# PLOTS
# =============================================================================

def plot_bo_target_tracking(by_target_df: pd.DataFrame, config: FinalBOConfig) -> None:
    """Plot final BO target tracking averaged across seeds."""
    mean_df = by_target_df[by_target_df["method"] == "BO"].sort_values("target").copy()

    if mean_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.plot(
        mean_df["target"],
        mean_df["target"],
        linestyle="--",
        linewidth=2.0,
        label="Ideal tracking",
    )

    ax.errorbar(
        mean_df["target"],
        mean_df["mean_peak_y_true"],
        yerr=mean_df["std_peak_y_true"].fillna(0.0),
        marker="o",
        linewidth=2.0,
        capsize=4,
        label="BO mean ± std",
    )

    ax.axhline(
        0.500,
        linestyle=":",
        linewidth=1.8,
        label="Nominal robot reach",
    )

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("True achieved peak_y [m]")
    ax.set_title("Final BO target tracking")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "bo_final_target_tracking_mean.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_bo_target_error(by_target_df: pd.DataFrame) -> None:
    """Plot final BO target error averaged across seeds."""
    mean_df = by_target_df[by_target_df["method"] == "BO"].sort_values("target").copy()

    if mean_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.bar(
        [f"{target:.2f}" for target in mean_df["target"]],
        mean_df["mean_target_error_mm"],
        yerr=mean_df["std_target_error_mm"].fillna(0.0),
        capsize=4,
        edgecolor="black",
    )

    ax.axhline(10.0, linestyle="--", linewidth=1.8, label="10 mm reference")

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("Target error [mm]")
    ax.set_title("Final BO target error, mean ± std across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "bo_final_target_error_mean.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_bo_residual_margin(by_target_df: pd.DataFrame) -> None:
    """Plot final BO residual margin averaged across seeds."""
    mean_df = by_target_df[by_target_df["method"] == "BO"].sort_values("target").copy()

    if mean_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.bar(
        [f"{target:.2f}" for target in mean_df["target"]],
        mean_df["mean_residual_margin_mm"],
        edgecolor="black",
        label="Mean residual margin",
    )

    ax.scatter(
        [f"{target:.2f}" for target in mean_df["target"]],
        mean_df["min_residual_margin_mm"],
        marker="v",
        s=75,
        label="Minimum margin across seeds",
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.8, label="Constraint boundary")

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("Residual margin [mm]")
    ax.set_title("Final BO residual safety margin")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "bo_final_residual_margin_mean.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_representative_convergence(
    history_df: pd.DataFrame,
    config: FinalBOConfig,
) -> None:
    """Plot final BO convergence for representative targets."""
    if history_df.empty or "best_feasible_error_so_far_mm" not in history_df.columns:
        return

    for target in (0.65, 0.75):
        sub = history_df[
            (history_df["method"].astype(str) == "BO")
            & np.isclose(history_df["target"].astype(float), target)
        ].copy()

        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.0, 5.5))

        for seed, group in sub.groupby("seed"):
            group = group.sort_values("iteration")

            ax.plot(
                group["iteration"],
                group["best_feasible_error_so_far_mm"],
                linewidth=1.9,
                label=f"seed {int(seed)}",
            )

        ax.axvline(100, linestyle=":", linewidth=1.2, label="100 calls")
        ax.axvline(
            config.total_evaluations,
            linestyle="--",
            linewidth=1.2,
            label=f"{config.total_evaluations} calls",
        )

        ax.set_xlabel("True simulator calls")
        ax.set_ylabel("Best feasible target error so far [mm]")
        ax.set_title(f"Final BO convergence, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = FIGURES_DIR / f"bo_final_convergence_target{int(round(target * 1000)):04d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def plot_bo_vs_random_error(
    results_df: pd.DataFrame,
    by_target_df: pd.DataFrame,
    config: FinalBOConfig,
) -> None:
    """Plot BO vs Random Search mean target error."""
    if "RandomSearch" not in set(results_df["method"].astype(str)):
        return

    comp = by_target_df.pivot(
        index="target",
        columns="method",
        values="mean_target_error_mm",
    ).reset_index()

    if not {"BO", "RandomSearch"}.issubset(comp.columns):
        return

    x = np.arange(len(comp))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    ax.bar(x - width / 2, comp["BO"], width, label="BO", edgecolor="black")
    ax.bar(
        x + width / 2,
        comp["RandomSearch"],
        width,
        label="Random Search",
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{target:.2f}" for target in comp["target"]])

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("Mean target error [mm]")
    ax.set_title(f"BO vs Random Search at {config.total_evaluations} simulator calls")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "bo_final_vs_random_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_bo_vs_random_convergence(
    history_df: pd.DataFrame,
) -> None:
    """Plot BO vs Random Search convergence on representative targets."""
    if history_df.empty or "best_feasible_error_so_far_mm" not in history_df.columns:
        return

    if "RandomSearch" not in set(history_df["method"].astype(str)):
        return

    for target in (0.65, 0.75):
        sub = history_df[np.isclose(history_df["target"].astype(float), target)].copy()

        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        for (method, seed), group in sub.groupby(["method", "seed"]):
            group = group.sort_values("iteration")

            ax.plot(
                group["iteration"],
                group["best_feasible_error_so_far_mm"],
                linewidth=1.3,
                label=f"{method}, seed {int(seed)}",
            )

        ax.set_xlabel("True simulator calls")
        ax.set_ylabel("Best feasible target error so far [mm]")
        ax.set_title(f"BO vs Random Search convergence, target = {target:.2f} m")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        path = FIGURES_DIR / (
            f"bo_final_vs_random_convergence_"
            f"target{int(round(target * 1000)):04d}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def generate_final_plots(
    all_results_df: pd.DataFrame,
    all_history_df: pd.DataFrame,
    by_target_df: pd.DataFrame,
    config: FinalBOConfig,
) -> None:
    """Generate all final BO figures."""
    print("\nGenerating final BO plots...")

    plot_bo_target_tracking(by_target_df, config)
    plot_bo_target_error(by_target_df)
    plot_bo_residual_margin(by_target_df)
    plot_representative_convergence(all_history_df, config)
    plot_bo_vs_random_error(all_results_df, by_target_df, config)
    plot_bo_vs_random_convergence(all_history_df)


# =============================================================================
# PRINTING
# =============================================================================

def print_settings(config: FinalBOConfig) -> None:
    """Print final BO settings."""
    print("=" * 80)
    print("FINAL BAYESIAN OPTIMIZATION RUN")
    print("=" * 80)
    print(f"Targets:                 {config.targets}")
    print(f"Seeds:                   {config.seeds}")
    print(f"Acquisition:             {config.acquisition_label}")
    print(f"Kernel:                  {config.kernel_label}")
    print(f"GP alpha:                {config.alpha_label}")
    print(f"Budget:                  {config.total_evaluations} true simulator calls/target/seed")
    print(f"Initial LHS points:      {config.initial_points}")
    print(f"GP restarts:             {config.gp_restarts}")
    print(f"Global candidate pool:   {config.candidate_pool_global}")
    print(f"Local candidate pool:    {config.candidate_pool_local}")
    print(f"Random Search baseline:  {config.run_random_search_baseline}")
    print(f"Resume existing:         {config.resume_existing}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)


def print_summary(
    by_target_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    """Print final summary tables."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY BY TARGET")
    print("=" * 80)

    if by_target_df.empty:
        print("No by-target summary available.")
    else:
        print(by_target_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FINAL OVERALL SUMMARY")
    print("=" * 80)

    if overall_df.empty:
        print("No overall summary available.")
    else:
        print(overall_df.to_string(index=False))


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run final Bayesian Optimization and optional Random Search baseline."
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
        help=f"True simulator calls per target and seed. Default: {DEFAULT_TOTAL_EVALUATIONS}.",
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
        "--no_random_search",
        action="store_true",
        help="Skip same-budget Random Search baseline.",
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing histories.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip final figure generation.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FinalBOConfig:
    """Build final BO configuration from command-line arguments."""
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")

    if args.initial_points <= 0:
        raise ValueError("--initial_points must be positive.")

    if args.initial_points >= args.budget:
        raise ValueError("--initial_points must be smaller than --budget.")

    if args.gp_restarts < 0:
        raise ValueError("--gp_restarts must be non-negative.")

    return FinalBOConfig(
        targets=parse_float_tuple(args.targets),
        seeds=parse_int_tuple(args.seeds),
        total_evaluations=int(args.budget),
        initial_points=int(args.initial_points),
        gp_alpha=float(args.gp_alpha),
        gp_restarts=int(args.gp_restarts),
        run_random_search_baseline=not bool(args.no_random_search),
        resume_existing=not bool(args.no_resume),
        skip_plots=bool(args.skip_plots),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the final BO pipeline."""
    args = parse_args()
    config = build_config(args)

    ensure_dirs()
    print_settings(config)

    bo_history_path = RESULTS_DIR / "bo_final_history.csv"
    random_history_path = RESULTS_DIR / "random_search_final_history.csv"

    existing_bo_history = pd.DataFrame()
    existing_random_history = pd.DataFrame()

    if config.resume_existing and bo_history_path.exists():
        existing_bo_history = pd.read_csv(bo_history_path)
        print(f"Resuming BO from existing history with {len(existing_bo_history)} rows.")

    if (
        config.resume_existing
        and config.run_random_search_baseline
        and random_history_path.exists()
    ):
        existing_random_history = pd.read_csv(random_history_path)
        print(
            f"Resuming Random Search from existing history with "
            f"{len(existing_random_history)} rows."
        )

    bo_records: list[dict[str, Any]] = []
    bo_histories: list[pd.DataFrame] = []
    random_records: list[dict[str, Any]] = []
    random_histories: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Final BO runs
    # -------------------------------------------------------------------------
    total_bo_runs = len(config.targets) * len(config.seeds)
    run_counter = 0

    for target in config.targets:
        for seed in config.seeds:
            run_counter += 1

            print(f"\n[BO {run_counter}/{total_bo_runs}] target={target:.2f} | seed={seed}")

            best, history, timing = run_or_load_bo(
                target=float(target),
                seed=int(seed),
                config=config,
                existing_history=existing_bo_history,
            )

            result = build_result_record(
                best_row=best,
                method="BO",
                timing=timing,
                config=config,
            )

            bo_records.append(result)
            bo_histories.append(history)
            timing_rows.append(timing)

            print(
                f"    best_y={float(best['peak_y_true']):.4f} m | "
                f"err={float(best['target_error_mm']):.2f} mm | "
                f"xr={float(best['max_abs_xr_true']):.4f} m | "
                f"margin={float(best['residual_margin_mm']):.2f} mm | "
                f"feasible={bool(best['feasible_abs'])} | "
                f"time={float(timing['total_wall_time_s']):.1f} s"
            )

    bo_history_df = (
        pd.concat(bo_histories, ignore_index=True)
        if bo_histories
        else pd.DataFrame()
    )

    if not bo_history_df.empty:
        bo_history_df = obo.add_cumulative_best_columns(bo_history_df)

    bo_results_df = pd.DataFrame(bo_records)

    # -------------------------------------------------------------------------
    # Same-budget Random Search baseline
    # -------------------------------------------------------------------------
    if config.run_random_search_baseline:
        total_random_runs = len(config.targets) * len(config.seeds)
        run_counter = 0

        for target in config.targets:
            for seed in config.seeds:
                run_counter += 1

                print(
                    f"\n[Random Search {run_counter}/{total_random_runs}] "
                    f"target={target:.2f} | seed={seed}"
                )

                best, history, timing = run_or_load_random_search(
                    target=float(target),
                    seed=int(seed),
                    config=config,
                    existing_history=existing_random_history,
                )

                result = build_result_record(
                    best_row=best,
                    method="RandomSearch",
                    timing=timing,
                    config=config,
                )

                random_records.append(result)
                random_histories.append(history)
                timing_rows.append(timing)

                print(
                    f"    best_y={float(best['peak_y_true']):.4f} m | "
                    f"err={float(best['target_error_mm']):.2f} mm | "
                    f"xr={float(best['max_abs_xr_true']):.4f} m | "
                    f"margin={float(best['residual_margin_mm']):.2f} mm | "
                    f"feasible={bool(best['feasible_abs'])} | "
                    f"time={float(timing['total_wall_time_s']):.1f} s"
                )

    random_history_df = (
        pd.concat(random_histories, ignore_index=True)
        if random_histories
        else pd.DataFrame()
    )

    if not random_history_df.empty:
        random_history_df = obo.add_cumulative_best_columns(random_history_df)

    random_results_df = pd.DataFrame(random_records)

    # -------------------------------------------------------------------------
    # Save final tables
    # -------------------------------------------------------------------------
    history_frames = [df for df in [bo_history_df, random_history_df] if not df.empty]
    result_frames = [df for df in [bo_results_df, random_results_df] if not df.empty]

    all_history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )

    if not all_history_df.empty:
        all_history_df = obo.add_cumulative_best_columns(all_history_df)

    all_results_df = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )

    timing_df = pd.DataFrame(timing_rows)

    by_target_df = summarize_by_target(all_results_df)
    overall_df = summarize_overall(all_results_df)
    report_df = make_report_table(by_target_df)

    saved_paths = save_outputs(
        bo_history=bo_history_df,
        bo_results=bo_results_df,
        timing_df=timing_df,
        all_history=all_history_df,
        all_results=all_results_df,
        by_target_df=by_target_df,
        overall_df=overall_df,
        report_df=report_df,
        random_history=random_history_df,
        random_results=random_results_df,
    )

    total_time_s = time.perf_counter() - total_start

    print_summary(by_target_df, overall_df)

    print("\nSaved files:")
    for path in saved_paths.values():
        print(f"  - {path}")

    print(f"Total final BO script wall time: {total_time_s / 60.0:.1f} min")

    if not config.skip_plots:
        generate_final_plots(
            all_results_df=all_results_df,
            all_history_df=all_history_df,
            by_target_df=by_target_df,
            config=config,
        )

    print("\nFinal BO run completed.")


if __name__ == "__main__":
    main()