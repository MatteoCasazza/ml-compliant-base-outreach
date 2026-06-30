"""
cross_validation_gp_parametric.py
=================================

Parametric cross-validation study for Gaussian Process kernel and alpha
selection on the ML-based extra-reach planning dataset.

The script can evaluate either:
1. GP_peak_y:
   [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> peak_y

2. GP_max_abs_xr:
   [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> max_abs_xr

Purpose
-------
This script is used to justify the final GP surrogate configurations used in
the report and in the inverse optimization pipeline.

Run from project root:
    python src/cross_validation_gp_parametric.py --target peak_y
    python src/cross_validation_gp_parametric.py --target max_abs_xr
    python src/cross_validation_gp_parametric.py --target both

Quick debug:
    python src/cross_validation_gp_parametric.py --target peak_y --quick_debug

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "cross_validation"
BASE_FIGURES_DIR = PROJECT_ROOT / "figures" / "cross_validation"

DEFAULT_DATASET_PATH = DATA_DIR / "dataset_augmented.csv"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

PARAM_COLUMNS = [
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

DEFAULT_N_SPLITS = 5
DEFAULT_N_RESTARTS = 1
DEFAULT_ALPHAS = [1e-10, 1e-6]
DEFAULT_KERNEL_NAMES = ["RBF", "Matern_3/2", "Matern_5/2", "RationalQuadratic"]
DEFAULT_RANDOM_STATE = 42

DEBUG_N_SPLITS = 3
DEBUG_N_RESTARTS = 0
DEBUG_ALPHAS = [1e-6]
DEBUG_KERNEL_NAMES = ["Matern_3/2", "Matern_5/2"]

HIGH_OUTREACH_THRESHOLD = 0.600

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495

NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520

LENGTH_SCALE_BOUNDS = (1e-2, 1e3)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TargetConfig:
    """Configuration for one GP target."""

    target_column: str
    target_slug: str
    target_label: str
    target_unit: str
    special_mode: str  # "outreach" or "constraint"

    @property
    def results_dir(self) -> Path:
        """Return output directory for result tables."""
        return BASE_RESULTS_DIR / self.target_slug

    @property
    def figures_dir(self) -> Path:
        """Return output directory for figures."""
        return BASE_FIGURES_DIR / self.target_slug


@dataclass(frozen=True)
class CVConfig:
    """Configuration for the cross-validation experiment."""

    dataset_path: Path
    n_splits: int
    n_restarts: int
    alphas: list[float]
    kernel_names: list[str]
    random_state: int
    skip_plots: bool
    quick_debug: bool


def get_target_config(target: str) -> TargetConfig:
    """Return target-specific configuration."""
    if target == "peak_y":
        return TargetConfig(
            target_column="peak_y",
            target_slug="gp_peak_y",
            target_label="peak_y",
            target_unit="m",
            special_mode="outreach",
        )

    if target == "max_abs_xr":
        return TargetConfig(
            target_column="max_abs_xr",
            target_slug="gp_max_abs_xr",
            target_label="max_abs_xr",
            target_unit="m",
            special_mode="constraint",
        )

    raise ValueError(f"Unknown target: {target}")


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs(target_config: TargetConfig) -> None:
    """Create output directories if they do not exist."""
    target_config.results_dir.mkdir(parents=True, exist_ok=True)
    target_config.figures_dir.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the previous pipeline step first."
        )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(value) for value in values]


def parse_string_list(raw: str) -> list[str]:
    """Parse comma-separated strings."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def save_table_markdown(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as markdown without external dependencies."""
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
# DATA
# =============================================================================

def load_dataset(
    target_config: TargetConfig,
    cv_config: CVConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the final augmented dataset for the selected target."""
    require_file(cv_config.dataset_path)

    df = pd.read_csv(cv_config.dataset_path, comment="#")

    required_cols = PARAM_COLUMNS + [target_config.target_column]
    require_columns(df, required_cols, "Dataset")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaNs in required columns.")

    X = df[PARAM_COLUMNS].to_numpy(dtype=np.float64)
    y = df[target_config.target_column].to_numpy(dtype=np.float64)

    return X, y, df


def print_dataset_summary(
    df: pd.DataFrame,
    target_config: TargetConfig,
    cv_config: CVConfig,
) -> None:
    """Print compact dataset statistics for the selected target."""
    y = df[target_config.target_column]

    print("\n" + "=" * 90)
    print("DATASET SUMMARY")
    print("=" * 90)
    print(f"Dataset path:              {cv_config.dataset_path}")
    print(f"Samples:                   {len(df)}")
    print(f"Input dimensions:          {len(PARAM_COLUMNS)}")
    print(f"Target:                    {target_config.target_column}")
    print(f"Mean target:               {y.mean():.6f} {target_config.target_unit}")
    print(f"Std target:                {y.std():.6f} {target_config.target_unit}")
    print(f"Min target:                {y.min():.6f} {target_config.target_unit}")
    print(f"Max target:                {y.max():.6f} {target_config.target_unit}")

    if target_config.special_mode == "outreach":
        high_count = int((df[target_config.target_column] > HIGH_OUTREACH_THRESHOLD).sum())
        print(
            f"High-outreach samples:     {high_count} "
            f"({100.0 * high_count / len(df):.1f}%)"
        )

    if target_config.special_mode == "constraint":
        true_feasible = df[target_config.target_column] <= ROBOT_LIMIT_TRUE
        opt_feasible = df[target_config.target_column] <= ROBOT_LIMIT_OPT
        near_boundary = (
            (df[target_config.target_column] >= NEAR_BOUNDARY_LOW)
            & (df[target_config.target_column] <= NEAR_BOUNDARY_HIGH)
        )

        print(f"True robot limit:          {ROBOT_LIMIT_TRUE:.3f} m")
        print(f"Optimization safety limit: {ROBOT_LIMIT_OPT:.3f} m")
        print(
            f"Feasible, true limit:      {int(true_feasible.sum())} "
            f"({100.0 * true_feasible.mean():.1f}%)"
        )
        print(
            f"Feasible, safety limit:    {int(opt_feasible.sum())} "
            f"({100.0 * opt_feasible.mean():.1f}%)"
        )
        print(
            f"Near-boundary samples:     {int(near_boundary.sum())} "
            f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
        )

    if "constraint_violation_abs" in df.columns:
        feasible_abs_count = int((df["constraint_violation_abs"] <= 0.002).sum())
        print(
            f"Feasible_abs samples:      {feasible_abs_count} "
            f"({100.0 * feasible_abs_count / len(df):.1f}%)"
        )

    if "dataset_type" in df.columns:
        print("Dataset types:")
        for name, count in df["dataset_type"].value_counts().items():
            print(f"  {str(name):22s}: {count}")

    print("=" * 90)


# =============================================================================
# KERNELS
# =============================================================================

def create_kernel(kernel_name: str, n_dims: int):
    """Create a GP kernel by name."""
    length_scale_init = np.ones(n_dims)

    if kernel_name == "RBF":
        base_kernel = RBF(
            length_scale=length_scale_init,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
        )

    elif kernel_name == "Matern_3/2":
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
            nu=1.5,
        )

    elif kernel_name == "Matern_5/2":
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
            nu=2.5,
        )

    elif kernel_name == "RationalQuadratic":
        # Rational Quadratic is included as a non-ARD baseline kernel family.
        base_kernel = RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
            alpha_bounds=(1e-3, 1e3),
        )

    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}")

    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * base_kernel
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )


# =============================================================================
# METRICS
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute scalar regression metrics in meters."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
    }


def high_outreach_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> dict[str, float]:
    """Compute peak_y metrics only on high-outreach validation samples."""
    mask = y_true > threshold

    if not np.any(mask):
        return {
            "high_n": 0,
            "high_rmse": np.nan,
            "high_mae": np.nan,
            "high_r2": np.nan,
            "high_median_abs_error": np.nan,
            "high_max_abs_error": np.nan,
        }

    reg = regression_metrics(y_true[mask], y_pred[mask])

    return {
        "high_n": int(mask.sum()),
        "high_rmse": reg["rmse"],
        "high_mae": reg["mae"],
        "high_r2": reg["r2"],
        "high_median_abs_error": reg["median_abs_error"],
        "high_max_abs_error": reg["max_abs_error"],
    }


def constraint_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    limit: float,
    prefix: str,
) -> dict[str, float]:
    """
    Compute feasibility classification metrics for max_abs_xr.

    False feasible is the most critical error:
    the surrogate predicts feasible while the simulator says infeasible.
    """
    true_feasible = y_true <= limit
    pred_feasible = y_pred <= limit

    false_feasible = pred_feasible & ~true_feasible
    false_infeasible = ~pred_feasible & true_feasible

    return {
        f"{prefix}_limit": float(limit),
        f"{prefix}_accuracy": float(np.mean(true_feasible == pred_feasible)),
        f"{prefix}_false_feasible_rate": float(np.mean(false_feasible)),
        f"{prefix}_false_infeasible_rate": float(np.mean(false_infeasible)),
        f"{prefix}_n_false_feasible": int(false_feasible.sum()),
        f"{prefix}_n_false_infeasible": int(false_infeasible.sum()),
    }


def safety_margin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_limit: float = ROBOT_LIMIT_TRUE,
    opt_limit: float = ROBOT_LIMIT_OPT,
) -> dict[str, float]:
    """
    Compute safety-margin false-feasible metric.

    Unsafe accepted means:
    - predicted safe under optimization margin: y_pred <= 0.495;
    - truly infeasible under physical limit: y_true > 0.500.
    """
    pred_safe_margin = y_pred <= opt_limit
    true_infeasible = y_true > true_limit

    unsafe_accepted = pred_safe_margin & true_infeasible
    n_pred_safe = int(pred_safe_margin.sum())

    return {
        "safety_margin_pred_safe_count": n_pred_safe,
        "safety_margin_unsafe_accepted_count": int(unsafe_accepted.sum()),
        "safety_margin_false_feasible_rate": float(np.mean(unsafe_accepted)),
        "safety_margin_unsafe_given_pred_safe_rate": (
            float(unsafe_accepted.sum() / n_pred_safe) if n_pred_safe > 0 else np.nan
        ),
    }


def near_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low: float = NEAR_BOUNDARY_LOW,
    high: float = NEAR_BOUNDARY_HIGH,
) -> dict[str, float]:
    """Compute max_abs_xr metrics near the robot workspace boundary."""
    mask = (y_true >= low) & (y_true <= high)

    if not np.any(mask):
        return {
            "near_boundary_low": low,
            "near_boundary_high": high,
            "near_boundary_n": 0,
            "near_boundary_rmse": np.nan,
            "near_boundary_mae": np.nan,
            "near_boundary_accuracy": np.nan,
            "near_boundary_false_feasible_rate": np.nan,
            "near_boundary_false_infeasible_rate": np.nan,
            "near_boundary_safety_margin_false_feasible_rate": np.nan,
        }

    y_true_nb = y_true[mask]
    y_pred_nb = y_pred[mask]

    reg = regression_metrics(y_true_nb, y_pred_nb)

    clf = constraint_classification_metrics(
        y_true_nb,
        y_pred_nb,
        limit=ROBOT_LIMIT_TRUE,
        prefix="near_boundary_true_limit",
    )

    safety = safety_margin_metrics(y_true_nb, y_pred_nb)

    return {
        "near_boundary_low": low,
        "near_boundary_high": high,
        "near_boundary_n": int(mask.sum()),
        "near_boundary_rmse": reg["rmse"],
        "near_boundary_mae": reg["mae"],
        "near_boundary_accuracy": clf["near_boundary_true_limit_accuracy"],
        "near_boundary_false_feasible_rate": (
            clf["near_boundary_true_limit_false_feasible_rate"]
        ),
        "near_boundary_false_infeasible_rate": (
            clf["near_boundary_true_limit_false_infeasible_rate"]
        ),
        "near_boundary_safety_margin_false_feasible_rate": (
            safety["safety_margin_false_feasible_rate"]
        ),
    }


def special_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_config: TargetConfig,
) -> dict[str, float]:
    """Compute target-specific metrics."""
    if target_config.special_mode == "outreach":
        return high_outreach_metrics(y_true, y_pred)

    if target_config.special_mode == "constraint":
        metrics: dict[str, float] = {}
        metrics.update(
            constraint_classification_metrics(
                y_true,
                y_pred,
                limit=ROBOT_LIMIT_TRUE,
                prefix="true_limit",
            )
        )
        metrics.update(
            constraint_classification_metrics(
                y_true,
                y_pred,
                limit=ROBOT_LIMIT_OPT,
                prefix="opt_limit",
            )
        )
        metrics.update(safety_margin_metrics(y_true, y_pred))
        metrics.update(near_boundary_metrics(y_true, y_pred))

        return metrics

    raise ValueError(f"Unknown special mode: {target_config.special_mode}")


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    target_config: TargetConfig,
    cv_config: CVConfig,
) -> pd.DataFrame:
    """
    Run the full cross-validation study.

    Scaling is fitted only on the training fold to avoid data leakage.
    """
    n_samples, n_dims = X.shape

    kfold = KFold(
        n_splits=cv_config.n_splits,
        shuffle=True,
        random_state=cv_config.random_state,
    )

    total_fits = (
        len(cv_config.kernel_names)
        * len(cv_config.alphas)
        * cv_config.n_splits
    )

    fit_counter = 0
    global_start_time = time.time()

    all_rows = []

    print("\n" + "=" * 90)
    print(f"GAUSSIAN PROCESS CROSS-VALIDATION: {target_config.target_column}")
    print("=" * 90)
    print(f"Samples:                   {n_samples}")
    print(f"Input dimensions:          {n_dims}")
    print(f"CV folds:                  {cv_config.n_splits}")
    print(f"Optimizer restarts:        {cv_config.n_restarts}")
    print(f"Alpha values:              {cv_config.alphas}")
    print(f"Kernels:                   {cv_config.kernel_names}")
    print(f"Total GP fits:             {total_fits}")

    if target_config.special_mode == "outreach":
        print(f"High-outreach threshold:   {HIGH_OUTREACH_THRESHOLD:.3f} m")

    if target_config.special_mode == "constraint":
        print(f"Robot true limit:          {ROBOT_LIMIT_TRUE:.3f} m")
        print(f"Optimization safety limit: {ROBOT_LIMIT_OPT:.3f} m")
        print(
            f"Near-boundary range:       "
            f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
        )

    print("=" * 90)

    for kernel_name in cv_config.kernel_names:
        for alpha in cv_config.alphas:
            print("\n" + "#" * 90)
            print(f"KERNEL: {kernel_name} | ALPHA: {alpha:.0e}")
            print("#" * 90)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
                fit_counter += 1

                X_train = X[train_idx]
                X_val = X[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]

                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_train_scaled = scaler_X.fit_transform(X_train)
                X_val_scaled = scaler_X.transform(X_val)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

                kernel = create_kernel(kernel_name, n_dims=n_dims)

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=False,
                    n_restarts_optimizer=cv_config.n_restarts,
                    random_state=cv_config.random_state,
                )

                print(
                    f"\nFit {fit_counter}/{total_fits} | "
                    f"target={target_config.target_column} | "
                    f"{kernel_name} | alpha={alpha:.0e} | "
                    f"fold {fold_idx}/{cv_config.n_splits}"
                )

                fit_start_time = time.time()

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(X_train_scaled, y_train_scaled)

                train_time_s = time.time() - fit_start_time

                n_convergence_warnings = sum(
                    issubclass(warning.category, ConvergenceWarning)
                    for warning in caught_warnings
                )

                predict_start_time = time.time()

                y_pred_scaled, y_std_scaled = gp.predict(
                    X_val_scaled,
                    return_std=True,
                )

                prediction_time_s = time.time() - predict_start_time

                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).ravel()

                y_std = y_std_scaled * scaler_y.scale_[0]

                reg = regression_metrics(y_val, y_pred)
                spec = special_metrics(y_val, y_pred, target_config)

                row = {
                    "target": target_config.target_column,
                    "kernel": kernel_name,
                    "alpha": alpha,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "train_time_s": float(train_time_s),
                    "prediction_time_s": float(prediction_time_s),
                    "rmse": reg["rmse"],
                    "mae": reg["mae"],
                    "r2": reg["r2"],
                    "mean_error": reg["mean_error"],
                    "std_error": reg["std_error"],
                    "median_abs_error": reg["median_abs_error"],
                    "max_abs_error": reg["max_abs_error"],
                    "mean_pred_std": float(np.mean(y_std)),
                    "max_pred_std": float(np.max(y_std)),
                    "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
                    "n_convergence_warnings": int(n_convergence_warnings),
                    "optimized_kernel": str(gp.kernel_),
                    **spec,
                }

                all_rows.append(row)

                elapsed_total_s = time.time() - global_start_time
                avg_fit_time_s = elapsed_total_s / fit_counter
                remaining_s = avg_fit_time_s * (total_fits - fit_counter)

                message = (
                    f"  RMSE: {reg['rmse'] * 1000.0:.3f} mm | "
                    f"MAE: {reg['mae'] * 1000.0:.3f} mm | "
                    f"R2: {reg['r2']:.5f} | "
                    f"LML: {gp.log_marginal_likelihood_value_:.2f} | "
                    f"Warnings: {n_convergence_warnings} | "
                    f"Time: {train_time_s:.1f} s"
                )

                if target_config.special_mode == "outreach":
                    high_rmse = spec.get("high_rmse", np.nan)
                    message += f" | High RMSE: {high_rmse * 1000.0:.3f} mm"

                if target_config.special_mode == "constraint":
                    false_feasible = spec.get("true_limit_false_feasible_rate", np.nan)
                    nb_rmse = spec.get("near_boundary_rmse", np.nan)
                    message += (
                        f" | False feasible: {false_feasible * 100.0:.2f}%"
                        f" | Near-boundary RMSE: {nb_rmse * 1000.0:.3f} mm"
                    )

                print(message)
                print(
                    f"  Elapsed: {elapsed_total_s / 60.0:.1f} min | "
                    f"Estimated remaining: {remaining_s / 60.0:.1f} min"
                )

                checkpoint_df = pd.DataFrame(all_rows)
                checkpoint_path = (
                    target_config.results_dir
                    / "cross_validation_raw_results_checkpoint.csv"
                )
                checkpoint_df.to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_rows)


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_results(
    raw_df: pd.DataFrame,
    target_config: TargetConfig,
) -> pd.DataFrame:
    """Summarize cross-validation results by kernel and alpha."""
    agg_dict: dict[str, tuple[str, str]] = {
        "rmse_mean": ("rmse", "mean"),
        "rmse_std": ("rmse", "std"),
        "mae_mean": ("mae", "mean"),
        "mae_std": ("mae", "std"),
        "r2_mean": ("r2", "mean"),
        "r2_std": ("r2", "std"),
        "mean_error_mean": ("mean_error", "mean"),
        "std_error_mean": ("std_error", "mean"),
        "median_abs_error_mean": ("median_abs_error", "mean"),
        "max_abs_error_mean": ("max_abs_error", "mean"),
        "mean_pred_std_mean": ("mean_pred_std", "mean"),
        "mean_pred_std_std": ("mean_pred_std", "std"),
        "max_pred_std_mean": ("max_pred_std", "mean"),
        "lml_mean": ("log_marginal_likelihood", "mean"),
        "lml_std": ("log_marginal_likelihood", "std"),
        "train_time_s_mean": ("train_time_s", "mean"),
        "train_time_s_std": ("train_time_s", "std"),
        "train_time_s_total": ("train_time_s", "sum"),
        "prediction_time_s_mean": ("prediction_time_s", "mean"),
        "convergence_warnings_total": ("n_convergence_warnings", "sum"),
    }

    if target_config.special_mode == "outreach":
        agg_dict.update(
            {
                "high_n_mean": ("high_n", "mean"),
                "high_rmse_mean": ("high_rmse", "mean"),
                "high_rmse_std": ("high_rmse", "std"),
                "high_mae_mean": ("high_mae", "mean"),
                "high_mae_std": ("high_mae", "std"),
                "high_r2_mean": ("high_r2", "mean"),
                "high_r2_std": ("high_r2", "std"),
                "high_max_abs_error_mean": ("high_max_abs_error", "mean"),
            }
        )

    if target_config.special_mode == "constraint":
        agg_dict.update(
            {
                "true_limit_accuracy_mean": ("true_limit_accuracy", "mean"),
                "true_limit_false_feasible_mean": (
                    "true_limit_false_feasible_rate",
                    "mean",
                ),
                "true_limit_false_infeasible_mean": (
                    "true_limit_false_infeasible_rate",
                    "mean",
                ),
                "opt_limit_accuracy_mean": ("opt_limit_accuracy", "mean"),
                "opt_limit_false_feasible_mean": (
                    "opt_limit_false_feasible_rate",
                    "mean",
                ),
                "safety_margin_false_feasible_mean": (
                    "safety_margin_false_feasible_rate",
                    "mean",
                ),
                "safety_margin_unsafe_given_pred_safe_mean": (
                    "safety_margin_unsafe_given_pred_safe_rate",
                    "mean",
                ),
                "near_boundary_n_mean": ("near_boundary_n", "mean"),
                "near_boundary_rmse_mean": ("near_boundary_rmse", "mean"),
                "near_boundary_rmse_std": ("near_boundary_rmse", "std"),
                "near_boundary_mae_mean": ("near_boundary_mae", "mean"),
                "near_boundary_accuracy_mean": ("near_boundary_accuracy", "mean"),
                "near_boundary_false_feasible_mean": (
                    "near_boundary_false_feasible_rate",
                    "mean",
                ),
                "near_boundary_false_infeasible_mean": (
                    "near_boundary_false_infeasible_rate",
                    "mean",
                ),
                "near_boundary_safety_margin_false_feasible_mean": (
                    "near_boundary_safety_margin_false_feasible_rate",
                    "mean",
                ),
            }
        )

    summary_df = (
        raw_df.groupby(["target", "kernel", "alpha"])
        .agg(**agg_dict)
        .reset_index()
    )

    metric_cols_m = [
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "mean_error_mean",
        "std_error_mean",
        "median_abs_error_mean",
        "max_abs_error_mean",
        "mean_pred_std_mean",
        "mean_pred_std_std",
        "max_pred_std_mean",
    ]

    if target_config.special_mode == "outreach":
        metric_cols_m += [
            "high_rmse_mean",
            "high_rmse_std",
            "high_mae_mean",
            "high_mae_std",
            "high_max_abs_error_mean",
        ]

    if target_config.special_mode == "constraint":
        metric_cols_m += [
            "near_boundary_rmse_mean",
            "near_boundary_rmse_std",
            "near_boundary_mae_mean",
        ]

    for col in metric_cols_m:
        if col in summary_df.columns:
            summary_df[f"{col}_mm"] = summary_df[col] * 1000.0

    summary_df = rank_summary(summary_df, target_config)

    return summary_df


def selection_columns(target_config: TargetConfig) -> tuple[list[str], list[bool]]:
    """Return target-specific selection criteria."""
    if target_config.special_mode == "outreach":
        return (
            [
                "rmse_mean",
                "high_rmse_mean",
                "convergence_warnings_total",
                "lml_mean",
            ],
            [True, True, True, False],
        )

    if target_config.special_mode == "constraint":
        return (
            [
                "true_limit_false_feasible_mean",
                "near_boundary_false_feasible_mean",
                "near_boundary_rmse_mean",
                "rmse_mean",
                "convergence_warnings_total",
                "lml_mean",
            ],
            [True, True, True, True, True, False],
        )

    raise ValueError(f"Unknown special mode: {target_config.special_mode}")


def rank_summary(
    summary_df: pd.DataFrame,
    target_config: TargetConfig,
) -> pd.DataFrame:
    """Sort summary according to target-specific selection criteria."""
    sort_cols, ascending = selection_columns(target_config)

    ranked = summary_df.copy()

    for col, is_ascending in zip(sort_cols, ascending):
        if col not in ranked.columns:
            continue

        ranked[col] = ranked[col].fillna(np.inf if is_ascending else -np.inf)

    ranked = ranked.sort_values(
        by=sort_cols,
        ascending=ascending,
    ).reset_index(drop=True)

    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)

    return ranked


def select_best_configuration(summary_df: pd.DataFrame) -> pd.Series:
    """Select the first row after target-specific ranking."""
    return summary_df.sort_values("selection_rank").iloc[0]


def make_report_table(
    summary_df: pd.DataFrame,
    target_config: TargetConfig,
) -> pd.DataFrame:
    """Create compact report-friendly table."""
    base = pd.DataFrame(
        {
            "Target": summary_df["target"],
            "Kernel": summary_df["kernel"],
            "Alpha": summary_df["alpha"].map(lambda x: f"{x:.0e}"),
            "RMSE [mm]": summary_df["rmse_mean_mm"].round(3),
            "RMSE std [mm]": summary_df["rmse_std_mm"].round(3),
            "MAE [mm]": summary_df["mae_mean_mm"].round(3),
            "R2": summary_df["r2_mean"].round(5),
            "Mean pred std [mm]": summary_df["mean_pred_std_mean_mm"].round(3),
            "LML": summary_df["lml_mean"].round(3),
            "Warnings": summary_df["convergence_warnings_total"].astype(int),
            "Total train time [min]": (
                summary_df["train_time_s_total"] / 60.0
            ).round(2),
            "Rank": summary_df["selection_rank"].astype(int),
        }
    )

    if target_config.special_mode == "outreach":
        base.insert(
            7,
            "High RMSE [mm]",
            summary_df["high_rmse_mean_mm"].round(3),
        )
        base.insert(
            8,
            "High RMSE std [mm]",
            summary_df["high_rmse_std_mm"].round(3),
        )

    if target_config.special_mode == "constraint":
        base.insert(
            7,
            "False feasible [%]",
            (summary_df["true_limit_false_feasible_mean"] * 100.0).round(3),
        )
        base.insert(
            8,
            "Constraint accuracy [%]",
            (summary_df["true_limit_accuracy_mean"] * 100.0).round(3),
        )
        base.insert(
            9,
            "Near-boundary RMSE [mm]",
            summary_df["near_boundary_rmse_mean_mm"].round(3),
        )
        base.insert(
            10,
            "Near-boundary false feasible [%]",
            (summary_df["near_boundary_false_feasible_mean"] * 100.0).round(3),
        )
        base.insert(
            11,
            "Safety-margin false feasible [%]",
            (summary_df["safety_margin_false_feasible_mean"] * 100.0).round(3),
        )

    return base


def save_best_kernel_summary(
    summary_df: pd.DataFrame,
    report_df: pd.DataFrame,
    target_config: TargetConfig,
    cv_config: CVConfig,
) -> Path:
    """Save text summary with selected kernel."""
    best = select_best_configuration(summary_df)

    lines = [
        f"CROSS-VALIDATION SUMMARY: {target_config.target_column}",
        "=" * 90,
        "",
        "Setup:",
        f"  Dataset: {cv_config.dataset_path}",
        f"  Target: {target_config.target_column}",
        f"  Output folder: {target_config.results_dir}",
        f"  Folds: {cv_config.n_splits}",
        f"  Optimizer restarts: {cv_config.n_restarts}",
        f"  Kernels: {', '.join(cv_config.kernel_names)}",
        f"  Alpha values: {cv_config.alphas}",
    ]

    if target_config.special_mode == "outreach":
        lines.append(f"  High-outreach threshold: {HIGH_OUTREACH_THRESHOLD:.3f} m")

    if target_config.special_mode == "constraint":
        lines.append(f"  Robot true limit: {ROBOT_LIMIT_TRUE:.3f} m")
        lines.append(f"  Optimization safety limit: {ROBOT_LIMIT_OPT:.3f} m")
        lines.append(
            f"  Near-boundary range: "
            f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
        )

    lines.extend(
        [
            "",
            "Recommended final configuration:",
            f"  Kernel: {best['kernel']}",
            f"  Alpha: {best['alpha']:.0e}",
            f"  RMSE: {best['rmse_mean_mm']:.3f} ± {best['rmse_std_mm']:.3f} mm",
            f"  MAE: {best['mae_mean_mm']:.3f} ± {best['mae_std_mm']:.3f} mm",
            f"  R2: {best['r2_mean']:.5f} ± {best['r2_std']:.5f}",
            f"  Mean predictive std: {best['mean_pred_std_mean_mm']:.3f} mm",
            f"  Mean LML: {best['lml_mean']:.3f}",
            f"  Convergence warnings: {int(best['convergence_warnings_total'])}",
        ]
    )

    if target_config.special_mode == "outreach":
        lines.extend(
            [
                (
                    f"  High-outreach RMSE: "
                    f"{best['high_rmse_mean_mm']:.3f} ± "
                    f"{best['high_rmse_std_mm']:.3f} mm"
                ),
                "",
                "Selection criterion:",
                (
                    "  The selected configuration minimizes global RMSE, then "
                    "high-outreach RMSE, then convergence warnings, and finally "
                    "maximizes LML."
                ),
            ]
        )

    if target_config.special_mode == "constraint":
        lines.extend(
            [
                (
                    f"  False feasible rate, true limit: "
                    f"{best['true_limit_false_feasible_mean'] * 100.0:.3f}%"
                ),
                (
                    f"  Constraint accuracy, true limit: "
                    f"{best['true_limit_accuracy_mean'] * 100.0:.3f}%"
                ),
                (
                    f"  Near-boundary RMSE: "
                    f"{best['near_boundary_rmse_mean_mm']:.3f} ± "
                    f"{best['near_boundary_rmse_std_mm']:.3f} mm"
                ),
                (
                    f"  Near-boundary false feasible rate: "
                    f"{best['near_boundary_false_feasible_mean'] * 100.0:.3f}%"
                ),
                "",
                "Selection criterion:",
                (
                    "  The selected configuration is chosen with a safety-first "
                    "criterion: lowest false-feasible rate, then lowest "
                    "near-boundary false-feasible rate, then near-boundary RMSE, "
                    "global RMSE, convergence warnings, and LML."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Compact report table:",
            report_df.to_string(index=False),
        ]
    )

    output_path = target_config.results_dir / "best_kernel_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return output_path


def print_summary(
    summary_df: pd.DataFrame,
    report_df: pd.DataFrame,
    target_config: TargetConfig,
) -> None:
    """Print CV summary to terminal."""
    print("\n" + "=" * 90)
    print(f"CROSS-VALIDATION SUMMARY: {target_config.target_column}")
    print("=" * 90)
    print(report_df.to_string(index=False))

    best = select_best_configuration(summary_df)

    print("\nRecommended configuration:")
    print(f"  Kernel:             {best['kernel']}")
    print(f"  Alpha:              {best['alpha']:.0e}")
    print(f"  RMSE:               {best['rmse_mean_mm']:.3f} mm")
    print(f"  R2:                 {best['r2_mean']:.5f}")
    print(f"  LML:                {best['lml_mean']:.3f}")
    print(f"  Warnings:           {int(best['convergence_warnings_total'])}")

    if target_config.special_mode == "outreach":
        print(f"  High RMSE:          {best['high_rmse_mean_mm']:.3f} mm")

    if target_config.special_mode == "constraint":
        print(
            f"  False feasible:     "
            f"{best['true_limit_false_feasible_mean'] * 100.0:.3f}%"
        )
        print(f"  Near-boundary RMSE: {best['near_boundary_rmse_mean_mm']:.3f} mm")
        print(
            f"  NB false feasible:  "
            f"{best['near_boundary_false_feasible_mean'] * 100.0:.3f}%"
        )

    print("=" * 90)


# =============================================================================
# PLOTS
# =============================================================================

def make_labels(summary_df: pd.DataFrame) -> list[str]:
    """Create compact labels for kernel-alpha combinations."""
    return [
        f"{row.kernel}\nalpha={row.alpha:.0e}"
        for row in summary_df.itertuples()
    ]


def get_bar_colors(summary_df: pd.DataFrame) -> list[str]:
    """Return bar colors by kernel type."""
    color_map = {
        "RBF": "#4C78A8",
        "Matern_3/2": "#F58518",
        "Matern_5/2": "#54A24B",
        "RationalQuadratic": "#B279A2",
    }

    return [color_map.get(kernel, "gray") for kernel in summary_df["kernel"]]


def plot_bar_metric(
    summary_df: pd.DataFrame,
    target_config: TargetConfig,
    value_col: str,
    error_col: str | None,
    ylabel: str,
    title: str,
    filename: str,
    convert_to_mm: bool = False,
    convert_to_percent: bool = False,
    lower_is_better: bool = True,
) -> None:
    """Generic bar plot for CV summary metrics."""
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    values = summary_df[value_col].to_numpy(dtype=float).copy()
    errors = summary_df[error_col].to_numpy(dtype=float).copy() if error_col else None

    if convert_to_mm:
        values *= 1000.0
        if errors is not None:
            errors *= 1000.0

    if convert_to_percent:
        values *= 100.0
        if errors is not None:
            errors *= 100.0

    fig, ax = plt.subplots(figsize=(13, 6))

    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=5 if errors is not None else 0,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.88,
    )

    finite_values = values[np.isfinite(values)]

    if len(finite_values) > 0:
        best_idx = int(
            np.nanargmin(values) if lower_is_better else np.nanargmax(values)
        )

        bars[best_idx].set_linewidth(2.6)
        bars[best_idx].set_edgecolor("black")

        y_span = np.nanmax(finite_values) - np.nanmin(finite_values)
        offset = 0.015 * (abs(y_span) + 1e-9)

        for idx, (bar, value) in enumerate(zip(bars, values)):
            if not np.isfinite(value):
                continue

            if "LML" in ylabel:
                label = f"{value:.1f}"
            elif convert_to_percent:
                label = f"{value:.2f}"
            elif convert_to_mm:
                label = f"{value:.2f}"
            else:
                label = f"{value:.4f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if idx == best_idx else "normal",
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    output_path = target_config.figures_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_warnings(
    summary_df: pd.DataFrame,
    target_config: TargetConfig,
) -> None:
    """Plot total convergence warnings by configuration."""
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    values = summary_df["convergence_warnings_total"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 6))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.88,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.05,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Total convergence warnings")
    ax.set_title(f"Cross-validation convergence warnings: {target_config.target_column}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    output_path = target_config.figures_dir / "cv_warnings_by_kernel_alpha.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_summary_figure(
    summary_df: pd.DataFrame,
    target_config: TargetConfig,
) -> None:
    """Plot compact 2x2 summary figure."""
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.bar(
        labels,
        summary_df["rmse_mean_mm"],
        yerr=summary_df["rmse_std_mm"],
        capsize=4,
        color=colors,
        edgecolor="black",
    )
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Global prediction error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    if target_config.special_mode == "outreach":
        ax = axes[0, 1]
        ax.bar(
            labels,
            summary_df["high_rmse_mean_mm"],
            yerr=summary_df["high_rmse_std_mm"],
            capsize=4,
            color=colors,
            edgecolor="black",
        )
        ax.set_ylabel("High-outreach RMSE [mm]")
        ax.set_title("High-outreach prediction error")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

        ax = axes[1, 0]
        ax.bar(
            labels,
            summary_df["r2_mean"],
            yerr=summary_df["r2_std"],
            capsize=4,
            color=colors,
            edgecolor="black",
        )
        ax.set_ylabel("R2")
        ax.set_title("Explained variance")
        ax.set_ylim(max(0.90, summary_df["r2_mean"].min() - 0.02), 1.01)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

        ax = axes[1, 1]
        ax.bar(
            labels,
            summary_df["lml_mean"],
            yerr=summary_df["lml_std"],
            capsize=4,
            color=colors,
            edgecolor="black",
        )
        ax.set_ylabel("Mean LML")
        ax.set_title("Log-marginal likelihood")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

    if target_config.special_mode == "constraint":
        ax = axes[0, 1]
        ax.bar(
            labels,
            summary_df["true_limit_false_feasible_mean"] * 100.0,
            color=colors,
            edgecolor="black",
        )
        ax.set_ylabel("False feasible [%]")
        ax.set_title("Constraint safety error")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

        ax = axes[1, 0]
        ax.bar(
            labels,
            summary_df["near_boundary_rmse_mean_mm"],
            yerr=summary_df["near_boundary_rmse_std_mm"],
            capsize=4,
            color=colors,
            edgecolor="black",
        )
        ax.set_ylabel("Near-boundary RMSE [mm]")
        ax.set_title("Near-boundary prediction error")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

        ax = axes[1, 1]
        ax.bar(
            labels,
            summary_df["near_boundary_false_feasible_mean"] * 100.0,
            color=colors,
            edgecolor="black",
        )
        ax.set_ylabel("Near-boundary false feasible [%]")
        ax.set_title("Near-boundary safety error")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

    fig.suptitle(
        f"Gaussian Process cross-validation summary: {target_config.target_column}",
        fontsize=16,
        fontweight="bold",
    )

    output_path = target_config.figures_dir / "cv_summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def generate_plots(
    summary_df: pd.DataFrame,
    target_config: TargetConfig,
) -> None:
    """Generate all CV plots for the selected target."""
    print("\n" + "=" * 90)
    print(f"GENERATING CROSS-VALIDATION FIGURES: {target_config.target_column}")
    print("=" * 90)

    plot_bar_metric(
        summary_df,
        target_config,
        value_col="rmse_mean",
        error_col="rmse_std",
        ylabel="RMSE [mm]",
        title=f"Cross-validation RMSE by kernel and alpha: {target_config.target_column}",
        filename="cv_rmse_by_kernel_alpha.png",
        convert_to_mm=True,
        lower_is_better=True,
    )

    plot_bar_metric(
        summary_df,
        target_config,
        value_col="r2_mean",
        error_col="r2_std",
        ylabel="R2",
        title=f"Cross-validation R2 by kernel and alpha: {target_config.target_column}",
        filename="cv_r2_by_kernel_alpha.png",
        lower_is_better=False,
    )

    plot_bar_metric(
        summary_df,
        target_config,
        value_col="lml_mean",
        error_col="lml_std",
        ylabel="Mean LML",
        title=f"Cross-validation log-marginal likelihood: {target_config.target_column}",
        filename="cv_lml_by_kernel_alpha.png",
        lower_is_better=False,
    )

    plot_bar_metric(
        summary_df,
        target_config,
        value_col="train_time_s_total",
        error_col=None,
        ylabel="Total training time [s]",
        title=f"Cross-validation training time: {target_config.target_column}",
        filename="cv_train_time_by_kernel_alpha.png",
        lower_is_better=True,
    )

    if target_config.special_mode == "outreach":
        plot_bar_metric(
            summary_df,
            target_config,
            value_col="high_rmse_mean",
            error_col="high_rmse_std",
            ylabel="High-outreach RMSE [mm]",
            title="Cross-validation high-outreach RMSE by kernel and alpha",
            filename="cv_high_rmse_by_kernel_alpha.png",
            convert_to_mm=True,
            lower_is_better=True,
        )

    if target_config.special_mode == "constraint":
        plot_bar_metric(
            summary_df,
            target_config,
            value_col="true_limit_false_feasible_mean",
            error_col=None,
            ylabel="False feasible [%]",
            title="Cross-validation false-feasible rate at true robot limit",
            filename="cv_false_feasible_by_kernel_alpha.png",
            convert_to_percent=True,
            lower_is_better=True,
        )

        plot_bar_metric(
            summary_df,
            target_config,
            value_col="near_boundary_rmse_mean",
            error_col="near_boundary_rmse_std",
            ylabel="Near-boundary RMSE [mm]",
            title="Cross-validation near-boundary RMSE by kernel and alpha",
            filename="cv_near_boundary_rmse_by_kernel_alpha.png",
            convert_to_mm=True,
            lower_is_better=True,
        )

        plot_bar_metric(
            summary_df,
            target_config,
            value_col="near_boundary_false_feasible_mean",
            error_col=None,
            ylabel="Near-boundary false feasible [%]",
            title="Cross-validation near-boundary false-feasible rate",
            filename="cv_near_boundary_false_feasible_by_kernel_alpha.png",
            convert_to_percent=True,
            lower_is_better=True,
        )

    plot_warnings(summary_df, target_config)
    plot_summary_figure(summary_df, target_config)


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    target: str,
    cv_config: CVConfig,
) -> None:
    """Run cross-validation for one target."""
    target_config = get_target_config(target)
    ensure_dirs(target_config)

    print("\n" + "=" * 90)
    print(f"RUNNING GP CROSS-VALIDATION FOR TARGET: {target_config.target_column}")
    print("=" * 90)
    print(f"Results directory: {target_config.results_dir}")
    print(f"Figures directory: {target_config.figures_dir}")

    X, y, df = load_dataset(target_config, cv_config)
    print_dataset_summary(df, target_config, cv_config)

    raw_df = run_cross_validation(
        X=X,
        y=y,
        target_config=target_config,
        cv_config=cv_config,
    )

    raw_path = target_config.results_dir / "cross_validation_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_results(raw_df, target_config)
    summary_path = target_config.results_dir / "cross_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report_df = make_report_table(summary_df, target_config)
    report_path = target_config.results_dir / "cross_validation_report_table.csv"
    report_df.to_csv(report_path, index=False)

    report_md_path = target_config.results_dir / "cross_validation_report_table.md"
    save_table_markdown(report_df, report_md_path)

    best_summary_path = save_best_kernel_summary(
        summary_df=summary_df,
        report_df=report_df,
        target_config=target_config,
        cv_config=cv_config,
    )

    print_summary(summary_df, report_df, target_config)

    if not cv_config.skip_plots:
        generate_plots(summary_df, target_config)

    print("\n" + "=" * 90)
    print(f"CROSS-VALIDATION COMPLETED: {target_config.target_column}")
    print("=" * 90)
    print("Generated files:")
    print(f"  - {raw_path}")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")
    print(f"  - {report_md_path}")
    print(f"  - {best_summary_path}")

    if not cv_config.skip_plots:
        print(f"  - {target_config.figures_dir / 'cv_summary.png'}")

    print("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parametric GP cross-validation for peak_y and max_abs_xr."
    )

    parser.add_argument(
        "--target",
        choices=["peak_y", "max_abs_xr", "both"],
        default="peak_y",
        help="Target to evaluate. Use 'both' to run both targets sequentially.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset path. Default: {DEFAULT_DATASET_PATH}.",
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of K-fold splits. Default: {DEFAULT_N_SPLITS}.",
    )

    parser.add_argument(
        "--n_restarts",
        type=int,
        default=DEFAULT_N_RESTARTS,
        help=f"Number of GP optimizer restarts. Default: {DEFAULT_N_RESTARTS}.",
    )

    parser.add_argument(
        "--alphas",
        type=str,
        default=",".join(f"{alpha:.0e}" for alpha in DEFAULT_ALPHAS),
        help="Comma-separated alpha values.",
    )

    parser.add_argument(
        "--kernels",
        type=str,
        default=",".join(DEFAULT_KERNEL_NAMES),
        help="Comma-separated kernel names.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    parser.add_argument(
        "--quick_debug",
        action="store_true",
        help="Run a reduced configuration for quick testing.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CVConfig:
    """Build cross-validation configuration from command-line arguments."""
    if args.quick_debug:
        return CVConfig(
            dataset_path=args.dataset,
            n_splits=DEBUG_N_SPLITS,
            n_restarts=DEBUG_N_RESTARTS,
            alphas=DEBUG_ALPHAS,
            kernel_names=DEBUG_KERNEL_NAMES,
            random_state=DEFAULT_RANDOM_STATE,
            skip_plots=args.skip_plots,
            quick_debug=True,
        )

    return CVConfig(
        dataset_path=args.dataset,
        n_splits=args.n_splits,
        n_restarts=args.n_restarts,
        alphas=parse_float_list(args.alphas),
        kernel_names=parse_string_list(args.kernels),
        random_state=DEFAULT_RANDOM_STATE,
        skip_plots=args.skip_plots,
        quick_debug=False,
    )


def main() -> None:
    """Run the selected GP cross-validation pipeline."""
    args = parse_args()
    cv_config = build_config(args)

    print("\n" + "=" * 90)
    print("PARAMETRIC GP CROSS-VALIDATION")
    print("=" * 90)

    if cv_config.quick_debug:
        print("Running in quick-debug mode.")

    if args.target == "both":
        for target in ["peak_y", "max_abs_xr"]:
            run_pipeline(target, cv_config)
    else:
        run_pipeline(args.target, cv_config)

    print("\nAll requested cross-validation runs completed.")


if __name__ == "__main__":
    main()