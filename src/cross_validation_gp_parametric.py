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

Usage
-----
Run peak_y only:
    python src/cross_validation_gp_parametric.py --target peak_y

Run max_abs_xr only:
    python src/cross_validation_gp_parametric.py --target max_abs_xr

Run both targets:
    python src/cross_validation_gp_parametric.py --target both

Outputs
-------
For peak_y:
    results/cross_validation/gp_peak_y/
    figures/cross_validation/gp_peak_y/

For max_abs_xr:
    results/cross_validation/gp_max_abs_xr/
    figures/cross_validation/gp_max_abs_xr/

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
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
DATA_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"

BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "cross_validation"
BASE_FIGURES_DIR = PROJECT_ROOT / "figures" / "cross_validation"


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

PARAM_COLS = [
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

N_SPLITS = 5
N_RESTARTS = 1
ALPHAS = [1e-10, 1e-6]
KERNEL_NAMES = ["RBF", "Matern_3/2", "Matern_5/2", "RationalQuadratic"]
RANDOM_STATE = 42

HIGH_OUTREACH_THRESHOLD = 0.600
ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495
NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TargetConfig:
    target_col: str
    target_slug: str
    target_label: str
    target_unit: str
    special_mode: str  # "outreach" or "constraint"

    @property
    def results_dir(self) -> Path:
        return BASE_RESULTS_DIR / self.target_slug

    @property
    def figures_dir(self) -> Path:
        return BASE_FIGURES_DIR / self.target_slug


def get_target_config(target: str) -> TargetConfig:
    if target == "peak_y":
        return TargetConfig(
            target_col="peak_y",
            target_slug="gp_peak_y",
            target_label="peak_y",
            target_unit="m",
            special_mode="outreach",
        )

    if target == "max_abs_xr":
        return TargetConfig(
            target_col="max_abs_xr",
            target_slug="gp_max_abs_xr",
            target_label="max_abs_xr",
            target_unit="m",
            special_mode="constraint",
        )

    raise ValueError(f"Unknown target: {target}")


# =============================================================================
# DATA
# =============================================================================

def load_dataset(config: TargetConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the final augmented dataset for the selected target."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = PARAM_COLS + [config.target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaNs in required columns.")

    X = df[PARAM_COLS].to_numpy(dtype=np.float64)
    y = df[config.target_col].to_numpy(dtype=np.float64)

    return X, y, df


def print_dataset_summary(df: pd.DataFrame, config: TargetConfig) -> None:
    """Print compact dataset statistics for the selected target."""
    y = df[config.target_col]

    print("\n" + "=" * 90)
    print("DATASET SUMMARY")
    print("=" * 90)
    print(f"Dataset path:              {DATA_PATH}")
    print(f"Samples:                   {len(df)}")
    print(f"Input dimensions:           {len(PARAM_COLS)}")
    print(f"Target:                     {config.target_col}")
    print(f"Mean target:                {y.mean():.6f} {config.target_unit}")
    print(f"Std target:                 {y.std():.6f} {config.target_unit}")
    print(f"Min target:                 {y.min():.6f} {config.target_unit}")
    print(f"Max target:                 {y.max():.6f} {config.target_unit}")

    if config.special_mode == "outreach":
        high_count = int((df[config.target_col] > HIGH_OUTREACH_THRESHOLD).sum())
        print(
            f"High-outreach samples:      {high_count} "
            f"({100 * high_count / len(df):.1f}%)"
        )

    if config.special_mode == "constraint":
        true_feasible = df[config.target_col] <= ROBOT_LIMIT_TRUE
        opt_feasible = df[config.target_col] <= ROBOT_LIMIT_OPT
        near_boundary = (
            (df[config.target_col] >= NEAR_BOUNDARY_LOW)
            & (df[config.target_col] <= NEAR_BOUNDARY_HIGH)
        )
        print(f"True robot limit:           {ROBOT_LIMIT_TRUE:.3f} m")
        print(f"Optimization safety limit:  {ROBOT_LIMIT_OPT:.3f} m")
        print(
            f"Feasible, true limit:       {int(true_feasible.sum())} "
            f"({100 * true_feasible.mean():.1f}%)"
        )
        print(
            f"Feasible, safety limit:     {int(opt_feasible.sum())} "
            f"({100 * opt_feasible.mean():.1f}%)"
        )
        print(
            f"Near-boundary samples:      {int(near_boundary.sum())} "
            f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
        )

    if "constraint_violation_abs" in df.columns:
        feasible_abs_count = int((df["constraint_violation_abs"] <= 0.002).sum())
        print(
            f"Feasible_abs samples:       {feasible_abs_count} "
            f"({100 * feasible_abs_count / len(df):.1f}%)"
        )

    if "dataset_type" in df.columns:
        print("Dataset types:")
        for name, count in df["dataset_type"].value_counts().items():
            print(f"  {name:22s}: {count}")

    print("=" * 90)


# =============================================================================
# KERNELS
# =============================================================================

def create_kernel(kernel_name: str, n_dims: int):
    """Create a GP kernel by name."""
    length_scale_init = np.ones(n_dims)
    length_scale_bounds = (1e-2, 1e3)

    if kernel_name == "RBF":
        base_kernel = RBF(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
        )
    elif kernel_name == "Matern_3/2":
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=1.5,
        )
    elif kernel_name == "Matern_5/2":
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=2.5,
        )
    elif kernel_name == "RationalQuadratic":
        # Rational Quadratic is included as a non-ARD baseline kernel family.
        base_kernel = RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e3),
            alpha_bounds=(1e-3, 1e3),
        )
    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}")

    return (
        C(1.0, (1e-3, 1e3))
        * base_kernel
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )


# =============================================================================
# METRICS
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
) -> Dict[str, float]:
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
) -> Dict[str, float]:
    """
    Compute feasibility classification metrics for max_abs_xr.

    False feasible is the most critical error:
        predicted feasible, but actually infeasible.
    """
    true_feasible = y_true <= limit
    pred_feasible = y_pred <= limit

    false_feasible = pred_feasible & (~true_feasible)
    false_infeasible = (~pred_feasible) & true_feasible

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
) -> Dict[str, float]:
    """
    Compute safety-margin false feasible metric.

    Unsafe accepted means:
        predicted safe under optimization margin: y_pred <= 0.495
        but truly infeasible under physical limit: y_true > 0.500
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
) -> Dict[str, float]:
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
        "near_boundary_false_feasible_rate": clf["near_boundary_true_limit_false_feasible_rate"],
        "near_boundary_false_infeasible_rate": clf["near_boundary_true_limit_false_infeasible_rate"],
        "near_boundary_safety_margin_false_feasible_rate": safety[
            "safety_margin_false_feasible_rate"
        ],
    }


def special_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: TargetConfig,
) -> Dict[str, float]:
    """Compute target-specific metrics."""
    if config.special_mode == "outreach":
        return high_outreach_metrics(y_true, y_pred)

    if config.special_mode == "constraint":
        metrics: Dict[str, float] = {}
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

    raise ValueError(f"Unknown special mode: {config.special_mode}")


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: TargetConfig,
) -> pd.DataFrame:
    """
    Run the full cross-validation study.

    Scaling is fitted only on the training fold to avoid data leakage.
    """
    n_samples, n_dims = X.shape

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    total_fits = len(KERNEL_NAMES) * len(ALPHAS) * N_SPLITS
    fit_counter = 0
    global_start_time = time.time()

    all_rows = []

    print("\n" + "=" * 90)
    print(f"GAUSSIAN PROCESS CROSS-VALIDATION: {config.target_col}")
    print("=" * 90)
    print(f"Samples:                   {n_samples}")
    print(f"Input dimensions:          {n_dims}")
    print(f"CV folds:                  {N_SPLITS}")
    print(f"Optimizer restarts:        {N_RESTARTS}")
    print(f"Alpha values:              {ALPHAS}")
    print(f"Kernels:                   {KERNEL_NAMES}")
    print(f"Total GP fits:             {total_fits}")
    if config.special_mode == "outreach":
        print(f"High-outreach threshold:   {HIGH_OUTREACH_THRESHOLD:.3f} m")
    if config.special_mode == "constraint":
        print(f"Robot true limit:          {ROBOT_LIMIT_TRUE:.3f} m")
        print(f"Optimization safety limit: {ROBOT_LIMIT_OPT:.3f} m")
        print(
            f"Near-boundary range:       "
            f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
        )
    print("=" * 90)

    for kernel_name in KERNEL_NAMES:
        for alpha in ALPHAS:
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
                    n_restarts_optimizer=N_RESTARTS,
                    random_state=RANDOM_STATE,
                )

                print(
                    f"\nFit {fit_counter}/{total_fits} | "
                    f"target={config.target_col} | "
                    f"{kernel_name} | alpha={alpha:.0e} | "
                    f"fold {fold_idx}/{N_SPLITS}"
                )

                fit_start_time = time.time()

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(X_train_scaled, y_train_scaled)

                train_time_s = time.time() - fit_start_time

                n_convergence_warnings = sum(
                    issubclass(w.category, ConvergenceWarning) for w in caught_warnings
                )

                predict_start_time = time.time()
                y_pred_scaled, y_std_scaled = gp.predict(X_val_scaled, return_std=True)
                prediction_time_s = time.time() - predict_start_time

                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_std = y_std_scaled * scaler_y.scale_[0]

                reg = regression_metrics(y_val, y_pred)
                spec = special_metrics(y_val, y_pred, config)

                row = {
                    "target": config.target_col,
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

                msg = (
                    f"  RMSE: {reg['rmse'] * 1000:.3f} mm | "
                    f"MAE: {reg['mae'] * 1000:.3f} mm | "
                    f"R2: {reg['r2']:.5f} | "
                    f"LML: {gp.log_marginal_likelihood_value_:.2f} | "
                    f"Warnings: {n_convergence_warnings} | "
                    f"Time: {train_time_s:.1f} s"
                )

                if config.special_mode == "outreach":
                    msg += f" | High RMSE: {spec['high_rmse'] * 1000:.3f} mm"
                elif config.special_mode == "constraint":
                    msg += (
                        f" | False feasible: {spec['true_limit_false_feasible_rate'] * 100:.2f}%"
                        f" | Near-boundary RMSE: {spec['near_boundary_rmse'] * 1000:.3f} mm"
                    )

                print(msg)
                print(
                    f"  Elapsed: {elapsed_total_s / 60:.1f} min | "
                    f"Estimated remaining: {remaining_s / 60:.1f} min"
                )

                checkpoint_df = pd.DataFrame(all_rows)
                checkpoint_path = config.results_dir / "cross_validation_raw_results_checkpoint.csv"
                checkpoint_df.to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_rows)


# =============================================================================
# SUMMARY
# =============================================================================

def _mean_std_agg(metric_name: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    return (
        (f"{metric_name}_mean", (metric_name, "mean")),
        (f"{metric_name}_std", (metric_name, "std")),
    )


def summarize_results(raw_df: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
    """Summarize cross-validation results by kernel and alpha."""
    agg_dict = {
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

    if config.special_mode == "outreach":
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

    if config.special_mode == "constraint":
        agg_dict.update(
            {
                "true_limit_accuracy_mean": ("true_limit_accuracy", "mean"),
                "true_limit_false_feasible_mean": ("true_limit_false_feasible_rate", "mean"),
                "true_limit_false_infeasible_mean": ("true_limit_false_infeasible_rate", "mean"),
                "opt_limit_accuracy_mean": ("opt_limit_accuracy", "mean"),
                "opt_limit_false_feasible_mean": ("opt_limit_false_feasible_rate", "mean"),
                "safety_margin_false_feasible_mean": ("safety_margin_false_feasible_rate", "mean"),
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

    # Metric columns in meters to duplicate as millimetres.
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

    if config.special_mode == "outreach":
        metric_cols_m += [
            "high_rmse_mean",
            "high_rmse_std",
            "high_mae_mean",
            "high_mae_std",
            "high_max_abs_error_mean",
        ]

    if config.special_mode == "constraint":
        metric_cols_m += [
            "near_boundary_rmse_mean",
            "near_boundary_rmse_std",
            "near_boundary_mae_mean",
        ]

    for col in metric_cols_m:
        if col in summary_df.columns:
            summary_df[f"{col}_mm"] = summary_df[col] * 1000.0

    summary_df = rank_summary(summary_df, config)
    return summary_df


def rank_summary(summary_df: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
    """Sort summary according to target-specific selection criteria."""
    sort_cols, ascending = selection_columns(config)

    ranked = summary_df.copy()
    for col in sort_cols:
        if col not in ranked.columns:
            continue
        # Missing values should not be selected as best.
        if ascending[sort_cols.index(col)]:
            ranked[col] = ranked[col].fillna(np.inf)
        else:
            ranked[col] = ranked[col].fillna(-np.inf)

    ranked = ranked.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def selection_columns(config: TargetConfig) -> Tuple[List[str], List[bool]]:
    """Return target-specific selection criteria."""
    if config.special_mode == "outreach":
        return (
            ["rmse_mean", "high_rmse_mean", "convergence_warnings_total", "lml_mean"],
            [True, True, True, False],
        )

    if config.special_mode == "constraint":
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

    raise ValueError(f"Unknown special mode: {config.special_mode}")


def select_best_configuration(summary_df: pd.DataFrame) -> pd.Series:
    """Select the first row after target-specific ranking."""
    return summary_df.sort_values("selection_rank").iloc[0]


def make_report_table(summary_df: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
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
            "Total train time [min]": (summary_df["train_time_s_total"] / 60.0).round(2),
            "Rank": summary_df["selection_rank"].astype(int),
        }
    )

    if config.special_mode == "outreach":
        base.insert(7, "High RMSE [mm]", summary_df["high_rmse_mean_mm"].round(3))
        base.insert(8, "High RMSE std [mm]", summary_df["high_rmse_std_mm"].round(3))

    if config.special_mode == "constraint":
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
    config: TargetConfig,
) -> Path:
    """Save text summary with selected kernel."""
    best = select_best_configuration(summary_df)

    lines = []
    lines.append(f"CROSS-VALIDATION SUMMARY: {config.target_col}")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Setup:")
    lines.append(f"  Dataset: {DATA_PATH}")
    lines.append(f"  Target: {config.target_col}")
    lines.append(f"  Output folder: {config.results_dir}")
    lines.append(f"  Folds: {N_SPLITS}")
    lines.append(f"  Optimizer restarts: {N_RESTARTS}")
    lines.append(f"  Kernels: {', '.join(KERNEL_NAMES)}")
    lines.append(f"  Alpha values: {ALPHAS}")
    if config.special_mode == "outreach":
        lines.append(f"  High-outreach threshold: {HIGH_OUTREACH_THRESHOLD:.3f} m")
    if config.special_mode == "constraint":
        lines.append(f"  Robot true limit: {ROBOT_LIMIT_TRUE:.3f} m")
        lines.append(f"  Optimization safety limit: {ROBOT_LIMIT_OPT:.3f} m")
        lines.append(
            f"  Near-boundary range: "
            f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
        )
    lines.append("")

    lines.append("Recommended final configuration:")
    lines.append(f"  Kernel: {best['kernel']}")
    lines.append(f"  Alpha: {best['alpha']:.0e}")
    lines.append(f"  RMSE: {best['rmse_mean_mm']:.3f} ± {best['rmse_std_mm']:.3f} mm")
    lines.append(f"  MAE: {best['mae_mean_mm']:.3f} ± {best['mae_std_mm']:.3f} mm")
    lines.append(f"  R2: {best['r2_mean']:.5f} ± {best['r2_std']:.5f}")
    lines.append(f"  Mean predictive std: {best['mean_pred_std_mean_mm']:.3f} mm")
    lines.append(f"  Mean LML: {best['lml_mean']:.3f}")
    lines.append(f"  Convergence warnings: {int(best['convergence_warnings_total'])}")

    if config.special_mode == "outreach":
        lines.append(
            f"  High-outreach RMSE: "
            f"{best['high_rmse_mean_mm']:.3f} ± {best['high_rmse_std_mm']:.3f} mm"
        )
        lines.append("")
        lines.append("Selection criterion:")
        lines.append(
            "  The selected configuration minimizes global RMSE, then high-outreach RMSE, "
            "then convergence warnings, and finally maximizes LML."
        )

    if config.special_mode == "constraint":
        lines.append(
            f"  False feasible rate, true limit: "
            f"{best['true_limit_false_feasible_mean'] * 100:.3f}%"
        )
        lines.append(
            f"  Constraint accuracy, true limit: "
            f"{best['true_limit_accuracy_mean'] * 100:.3f}%"
        )
        lines.append(
            f"  Near-boundary RMSE: "
            f"{best['near_boundary_rmse_mean_mm']:.3f} ± "
            f"{best['near_boundary_rmse_std_mm']:.3f} mm"
        )
        lines.append(
            f"  Near-boundary false feasible rate: "
            f"{best['near_boundary_false_feasible_mean'] * 100:.3f}%"
        )
        lines.append("")
        lines.append("Selection criterion:")
        lines.append(
            "  The selected configuration is chosen with a safety-first criterion: "
            "lowest false-feasible rate, then lowest near-boundary false-feasible rate, "
            "then near-boundary RMSE, global RMSE, convergence warnings, and LML."
        )

    lines.append("")
    lines.append("Compact report table:")
    lines.append(report_df.to_string(index=False))

    output_path = config.results_dir / "best_kernel_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def print_summary(summary_df: pd.DataFrame, report_df: pd.DataFrame, config: TargetConfig) -> None:
    """Print CV summary to terminal."""
    print("\n" + "=" * 90)
    print(f"CROSS-VALIDATION SUMMARY: {config.target_col}")
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

    if config.special_mode == "outreach":
        print(f"  High RMSE:          {best['high_rmse_mean_mm']:.3f} mm")

    if config.special_mode == "constraint":
        print(
            f"  False feasible:     "
            f"{best['true_limit_false_feasible_mean'] * 100:.3f}%"
        )
        print(
            f"  Near-boundary RMSE: {best['near_boundary_rmse_mean_mm']:.3f} mm"
        )
        print(
            f"  NB false feasible:  "
            f"{best['near_boundary_false_feasible_mean'] * 100:.3f}%"
        )

    print("=" * 90)


# =============================================================================
# PLOTS
# =============================================================================

def make_labels(summary_df: pd.DataFrame) -> List[str]:
    return [f"{row.kernel}\nα={row.alpha:.0e}" for row in summary_df.itertuples()]


def get_bar_colors(summary_df: pd.DataFrame):
    color_map = {
        "RBF": "#4C78A8",
        "Matern_3/2": "#F58518",
        "Matern_5/2": "#54A24B",
        "RationalQuadratic": "#B279A2",
    }
    return [color_map.get(kernel, "gray") for kernel in summary_df["kernel"]]


def plot_bar_metric(
    summary_df: pd.DataFrame,
    config: TargetConfig,
    value_col: str,
    error_col: Optional[str],
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
        values = values * 1000.0
        if errors is not None:
            errors = errors * 1000.0

    if convert_to_percent:
        values = values * 100.0
        if errors is not None:
            errors = errors * 100.0

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
        best_idx = int(np.nanargmin(values) if lower_is_better else np.nanargmax(values))
        bars[best_idx].set_linewidth(2.6)
        bars[best_idx].set_edgecolor("black")

        y_span = np.nanmax(finite_values) - np.nanmin(finite_values)
        offset = 0.015 * (y_span + 1e-9)

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
                bar.get_x() + bar.get_width() / 2,
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

    plt.tight_layout()

    output_path = config.figures_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_warnings(summary_df: pd.DataFrame, config: TargetConfig) -> None:
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)
    values = summary_df["convergence_warnings_total"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0, alpha=0.88)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Total convergence warnings")
    ax.set_title(f"Cross-validation convergence warnings: {config.target_col}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)
    plt.tight_layout()

    output_path = config.figures_dir / "cv_warnings_by_kernel_alpha.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_figure(summary_df: pd.DataFrame, config: TargetConfig) -> None:
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # RMSE
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

    if config.special_mode == "outreach":
        # High-outreach RMSE
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

        # R2
        ax = axes[1, 0]
        ax.bar(labels, summary_df["r2_mean"], yerr=summary_df["r2_std"], capsize=4, color=colors, edgecolor="black")
        ax.set_ylabel("R2")
        ax.set_title("Explained variance")
        ax.set_ylim(max(0.90, summary_df["r2_mean"].min() - 0.02), 1.01)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

        # LML
        ax = axes[1, 1]
        ax.bar(labels, summary_df["lml_mean"], yerr=summary_df["lml_std"], capsize=4, color=colors, edgecolor="black")
        ax.set_ylabel("Mean LML")
        ax.set_title("Log-marginal likelihood")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)

    else:
        # False feasible
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

        # Near-boundary RMSE
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

        # Near-boundary false feasible
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

    plt.suptitle(
        f"Gaussian Process cross-validation summary: {config.target_col}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = config.figures_dir / "cv_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_plots(summary_df: pd.DataFrame, config: TargetConfig) -> None:
    """Generate all CV plots for the selected target."""
    print("\n" + "=" * 90)
    print(f"GENERATING CROSS-VALIDATION FIGURES: {config.target_col}")
    print("=" * 90)

    plot_bar_metric(
        summary_df,
        config,
        value_col="rmse_mean",
        error_col="rmse_std",
        ylabel="RMSE [mm]",
        title=f"Cross-validation RMSE by kernel and alpha: {config.target_col}",
        filename="cv_rmse_by_kernel_alpha.png",
        convert_to_mm=True,
        lower_is_better=True,
    )

    plot_bar_metric(
        summary_df,
        config,
        value_col="r2_mean",
        error_col="r2_std",
        ylabel="R2",
        title=f"Cross-validation R2 by kernel and alpha: {config.target_col}",
        filename="cv_r2_by_kernel_alpha.png",
        lower_is_better=False,
    )

    plot_bar_metric(
        summary_df,
        config,
        value_col="lml_mean",
        error_col="lml_std",
        ylabel="Mean LML",
        title=f"Cross-validation log-marginal likelihood: {config.target_col}",
        filename="cv_lml_by_kernel_alpha.png",
        lower_is_better=False,
    )

    plot_bar_metric(
        summary_df,
        config,
        value_col="train_time_s_total",
        error_col=None,
        ylabel="Total training time [s]",
        title=f"Cross-validation training time: {config.target_col}",
        filename="cv_train_time_by_kernel_alpha.png",
        lower_is_better=True,
    )

    if config.special_mode == "outreach":
        plot_bar_metric(
            summary_df,
            config,
            value_col="high_rmse_mean",
            error_col="high_rmse_std",
            ylabel="High-outreach RMSE [mm]",
            title="Cross-validation high-outreach RMSE by kernel and alpha",
            filename="cv_high_rmse_by_kernel_alpha.png",
            convert_to_mm=True,
            lower_is_better=True,
        )

    if config.special_mode == "constraint":
        plot_bar_metric(
            summary_df,
            config,
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
            config,
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
            config,
            value_col="near_boundary_false_feasible_mean",
            error_col=None,
            ylabel="Near-boundary false feasible [%]",
            title="Cross-validation near-boundary false-feasible rate",
            filename="cv_near_boundary_false_feasible_by_kernel_alpha.png",
            convert_to_percent=True,
            lower_is_better=True,
        )

    plot_warnings(summary_df, config)
    plot_summary_figure(summary_df, config)


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(target: str) -> None:
    config = get_target_config(target)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"RUNNING GP CROSS-VALIDATION FOR TARGET: {config.target_col}")
    print("=" * 90)
    print(f"Results directory: {config.results_dir}")
    print(f"Figures directory: {config.figures_dir}")

    X, y, df = load_dataset(config)
    print_dataset_summary(df, config)

    raw_df = run_cross_validation(X, y, config)
    raw_path = config.results_dir / "cross_validation_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_results(raw_df, config)
    summary_path = config.results_dir / "cross_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report_df = make_report_table(summary_df, config)
    report_path = config.results_dir / "cross_validation_report_table.csv"
    report_df.to_csv(report_path, index=False)

    best_summary_path = save_best_kernel_summary(summary_df, report_df, config)

    print_summary(summary_df, report_df, config)
    generate_plots(summary_df, config)

    print("\n" + "=" * 90)
    print(f"CROSS-VALIDATION COMPLETED: {config.target_col}")
    print("=" * 90)
    print("Generated files:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")
    print(f"  {best_summary_path}")
    print(f"  {config.figures_dir / 'cv_summary.png'}")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parametric GP cross-validation for peak_y and max_abs_xr."
    )
    parser.add_argument(
        "--target",
        choices=["peak_y", "max_abs_xr", "both"],
        default="peak_y",
        help="Target to evaluate. Use 'both' to run peak_y and max_abs_xr sequentially.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target == "both":
        for target in ["peak_y", "max_abs_xr"]:
            run_pipeline(target)
    else:
        run_pipeline(args.target)


if __name__ == "__main__":
    main()
