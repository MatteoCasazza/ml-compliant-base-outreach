"""
model_max_xr.py
===============

Gaussian Process surrogate for predicting the absolute robot-displacement
constraint quantity max_abs_xr.

Workflow
--------
1. Load the augmented simulation dataset.
2. Split the dataset into training and test sets.
3. Standardize input and output variables.
4. Train a Gaussian Process Regressor.
5. Evaluate regression accuracy.
6. Evaluate constraint-classification accuracy.
7. Evaluate near-boundary performance around the robot limit.
8. Analyze parameter relevance using ARD length-scales.
9. Generate diagnostic plots.
10. Save model, scalers, metrics, ARD results and model information.

Supervised-learning task
------------------------
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> max_abs_xr

The GP_max_abs_xr model is used as the constraint surrogate inside the
constraint-aware inverse optimization pipeline. It complements the GP_peak_y
surrogate and does not replace it.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEFAULT_DATASET_PATH = DATA_DIR / "dataset_augmented.csv"

GP_CONSTRAINT_RESULTS_DIR = RESULTS_DIR / "gp_constraints"
GP_CONSTRAINT_FIGURES_DIR = FIGURES_DIR / "gp_constraints"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

INPUT_COLUMNS = [
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

TARGET_COLUMN = "max_abs_xr"

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495

NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

DEFAULT_KERNEL = "matern32"
DEFAULT_ALPHA = 1e-6
DEFAULT_N_RESTARTS = 3

SUPPORTED_KERNELS = ("matern32", "matern52", "rbf")


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    GP_CONSTRAINT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GP_CONSTRAINT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_constraint_dataset(
    filepath: Path | str = DEFAULT_DATASET_PATH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the dataset for the max_abs_xr constraint surrogate."""
    filepath = Path(filepath)
    require_file(filepath)

    df = pd.read_csv(filepath, comment="#")

    require_columns(df, INPUT_COLUMNS, "Dataset")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Dataset is missing target column '{TARGET_COLUMN}'. "
            "Regenerate dataset.py and augment_high_outreach.py so that "
            "dataset_augmented.csv contains max_abs_xr."
        )

    df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    X = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    return X, y, df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print dataset and absolute-constraint statistics."""
    require_columns(df, [TARGET_COLUMN], "Dataset")

    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    true_feasible = y <= ROBOT_LIMIT_TRUE
    opt_feasible = y <= ROBOT_LIMIT_OPT

    near_boundary = df[
        (df[TARGET_COLUMN] >= NEAR_BOUNDARY_LOW)
        & (df[TARGET_COLUMN] <= NEAR_BOUNDARY_HIGH)
    ]

    print("\n" + "=" * 70)
    print("DATASET ABSOLUTE CONSTRAINT SUMMARY")
    print("=" * 70)
    print(f"Dataset samples:               {len(df)}")
    print(f"Target column:                 {TARGET_COLUMN}")
    print(f"Mean max_abs_xr:               {np.mean(y):.6f} m")
    print(f"Std max_abs_xr:                {np.std(y):.6f} m")
    print(f"Min max_abs_xr:                {np.min(y):.6f} m")
    print(f"Max max_abs_xr:                {np.max(y):.6f} m")
    print(f"True robot limit:              {ROBOT_LIMIT_TRUE:.3f} m")
    print(f"Optimization safety limit:     {ROBOT_LIMIT_OPT:.3f} m")
    print(f"Violation rate, true limit:    {np.mean(~true_feasible) * 100.0:.2f}%")
    print(f"Violation rate, opt limit:     {np.mean(~opt_feasible) * 100.0:.2f}%")
    print(
        f"Near-boundary samples "
        f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m: "
        f"{len(near_boundary)}"
    )

    if "constraint_violation_abs" in df.columns:
        print(
            f"Mean abs constraint violation: "
            f"{df['constraint_violation_abs'].mean() * 1000.0:.3f} mm"
        )
        print(
            f"Max abs constraint violation:  "
            f"{df['constraint_violation_abs'].max() * 1000.0:.3f} mm"
        )

    if "feasible_abs" in df.columns:
        feasible_abs = df["feasible_abs"].astype(bool)
        print(f"feasible_abs samples:          {int(feasible_abs.sum())}")

    print("=" * 70)


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
    np.ndarray,
    np.ndarray,
]:
    """Split and standardize data for GP training."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    print("\n" + "=" * 70)
    print("PREPROCESSING DATASET")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    print(f"Train set: {len(y_train)} samples ({100.0 * (1.0 - test_size):.0f}%)")
    print(f"Test set:  {len(y_test)} samples ({100.0 * test_size:.0f}%)")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print("\nInput standardization check:")
    print(f"  Mean of standardized X_train, first 3 features: {X_train_scaled.mean(axis=0)[:3]}")
    print(f"  Std  of standardized X_train, first 3 features: {X_train_scaled.std(axis=0)[:3]}")

    print("\nOutput standardization check:")
    print(f"  Mean of standardized y_train: {y_train_scaled.mean():.6f}")
    print(f"  Std  of standardized y_train: {y_train_scaled.std():.6f}")

    if not np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-10):
        raise RuntimeError("X_train is not centered feature-wise.")

    if not np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-10):
        raise RuntimeError("X_train is not normalized feature-wise.")

    if not np.allclose(y_train_scaled.mean(), 0.0, atol=1e-10):
        raise RuntimeError("y_train is not centered.")

    if not np.allclose(y_train_scaled.std(), 1.0, atol=1e-10):
        raise RuntimeError("y_train is not normalized.")

    print("\nPreprocessing completed.")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        y_train,
        y_test,
    )


# =============================================================================
# GAUSSIAN PROCESS TRAINING
# =============================================================================

def create_gp_model(
    kernel_type: str = DEFAULT_KERNEL,
    n_dims: int = len(INPUT_COLUMNS),
    length_scale_init: np.ndarray | None = None,
    length_scale_bounds: tuple[float, float] = (1e-2, 1e3),
    noise_level: float = 1e-5,
    alpha: float = DEFAULT_ALPHA,
    n_restarts: int = DEFAULT_N_RESTARTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> GaussianProcessRegressor:
    """Create a Gaussian Process Regressor with ARD length-scales."""
    if kernel_type not in SUPPORTED_KERNELS:
        raise ValueError(
            f"Unsupported kernel type: {kernel_type}. "
            f"Supported kernels are: {SUPPORTED_KERNELS}."
        )

    if length_scale_init is None:
        length_scale_init = np.ones(n_dims, dtype=float)

    if kernel_type == "matern32":
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=1.5,
        )
    elif kernel_type == "matern52":
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=2.5,
        )
    else:
        base_kernel = RBF(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
        )

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * base_kernel
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
        random_state=random_state,
        alpha=alpha,
    )

    print(f"\nCreated GP with kernel: {kernel_type}")
    print(f"Kernel formula: {kernel}")
    print(f"Optimizer restarts: {n_restarts}")
    print(f"Alpha regularization: {alpha}")

    return gp


def train_gp(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    kernel_type: str = DEFAULT_KERNEL,
    n_restarts: int = DEFAULT_N_RESTARTS,
    alpha: float = DEFAULT_ALPHA,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> tuple[GaussianProcessRegressor, float]:
    """Train the GP_max_abs_xr surrogate."""
    print("\n" + "=" * 70)
    print("TRAINING GP_MAX_ABS_XR")
    print("=" * 70)

    gp = create_gp_model(
        kernel_type=kernel_type,
        n_dims=X_train_scaled.shape[1],
        n_restarts=n_restarts,
        alpha=alpha,
        random_state=random_state,
    )

    print("\nFitting GP_max_abs_xr. This may take some time...")
    start_time = time.time()

    gp.fit(X_train_scaled, y_train_scaled)

    training_time_s = time.time() - start_time
    log_marginal_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)

    print(f"Training completed in {training_time_s:.1f} s")
    print(f"Log-marginal likelihood: {log_marginal_likelihood:.3f}")

    if verbose:
        print("\nOptimized kernel:")
        print(f"  {gp.kernel_}")

    return gp, training_time_s


# =============================================================================
# EVALUATION
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse_m": rmse,
        "rmse_mm": rmse * 1000.0,
        "mae_m": mae,
        "mae_mm": mae * 1000.0,
        "r2": r2,
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    limit: float,
    label: str,
) -> dict[str, float]:
    """
    Evaluate feasibility classification using max_abs_xr <= limit.

    False feasible is the most critical error:
    the surrogate predicts a sample as feasible while the simulator says it is
    infeasible.
    """
    true_feasible = y_true <= limit
    pred_feasible = y_pred <= limit

    accuracy = float(np.mean(true_feasible == pred_feasible))
    false_feasible = float(np.mean(pred_feasible & ~true_feasible))
    false_infeasible = float(np.mean(~pred_feasible & true_feasible))

    tn, fp, fn, tp = confusion_matrix(
        true_feasible,
        pred_feasible,
        labels=[False, True],
    ).ravel()

    return {
        f"{label}_limit_m": float(limit),
        f"{label}_classification_accuracy": accuracy,
        f"{label}_false_feasible_rate": false_feasible,
        f"{label}_false_infeasible_rate": false_infeasible,
        f"{label}_true_infeasible_pred_infeasible": int(tn),
        f"{label}_true_infeasible_pred_feasible": int(fp),
        f"{label}_true_feasible_pred_infeasible": int(fn),
        f"{label}_true_feasible_pred_feasible": int(tp),
    }


def evaluate_near_boundary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    low: float = NEAR_BOUNDARY_LOW,
    high: float = NEAR_BOUNDARY_HIGH,
) -> dict[str, float]:
    """Evaluate prediction quality near the absolute robot constraint boundary."""
    mask = (y_true >= low) & (y_true <= high)

    if not np.any(mask):
        return {
            "near_boundary_low_m": float(low),
            "near_boundary_high_m": float(high),
            "near_boundary_n_samples": 0,
            "near_boundary_rmse_mm": np.nan,
            "near_boundary_mae_mm": np.nan,
            "near_boundary_mean_std_mm": np.nan,
            "near_boundary_classification_accuracy_true_limit": np.nan,
            "near_boundary_false_feasible_rate_true_limit": np.nan,
        }

    y_true_nb = y_true[mask]
    y_pred_nb = y_pred[mask]
    y_std_nb = y_std[mask]

    reg = regression_metrics(y_true_nb, y_pred_nb)
    clf = evaluate_classification(
        y_true_nb,
        y_pred_nb,
        limit=ROBOT_LIMIT_TRUE,
        label="near_boundary_true_limit",
    )

    return {
        "near_boundary_low_m": float(low),
        "near_boundary_high_m": float(high),
        "near_boundary_n_samples": int(len(y_true_nb)),
        "near_boundary_rmse_mm": float(reg["rmse_mm"]),
        "near_boundary_mae_mm": float(reg["mae_mm"]),
        "near_boundary_mean_std_mm": float(np.mean(y_std_nb) * 1000.0),
        "near_boundary_classification_accuracy_true_limit": float(
            clf["near_boundary_true_limit_classification_accuracy"]
        ),
        "near_boundary_false_feasible_rate_true_limit": float(
            clf["near_boundary_true_limit_false_feasible_rate"]
        ),
    }


def evaluate_model(
    gp: GaussianProcessRegressor,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> dict[str, Any]:
    """Evaluate the trained GP on training and test sets."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    y_train_pred_scaled, y_train_std_scaled = gp.predict(
        X_train_scaled,
        return_std=True,
    )
    y_test_pred_scaled, y_test_std_scaled = gp.predict(
        X_test_scaled,
        return_std=True,
    )

    y_train_pred = scaler_y.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_train_true = scaler_y.inverse_transform(
        y_train_scaled.reshape(-1, 1)
    ).ravel()
    y_test_true = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).ravel()

    y_train_std = y_train_std_scaled * scaler_y.scale_[0]
    y_test_std = y_test_std_scaled * scaler_y.scale_[0]

    train_metrics = regression_metrics(y_train_true, y_train_pred)
    test_metrics = regression_metrics(y_test_true, y_test_pred)

    clf_true_limit = evaluate_classification(
        y_test_true,
        y_test_pred,
        limit=ROBOT_LIMIT_TRUE,
        label="true_limit",
    )

    clf_opt_limit = evaluate_classification(
        y_test_true,
        y_test_pred,
        limit=ROBOT_LIMIT_OPT,
        label="opt_limit",
    )

    near_boundary_metrics = evaluate_near_boundary(
        y_test_true,
        y_test_pred,
        y_test_std,
    )

    print("\nTraining set:")
    print(f"  RMSE: {train_metrics['rmse_m']:.6f} m ({train_metrics['rmse_mm']:.3f} mm)")
    print(f"  MAE:  {train_metrics['mae_m']:.6f} m ({train_metrics['mae_mm']:.3f} mm)")
    print(f"  R²:   {train_metrics['r2']:.6f}")

    print("\nTest set:")
    print(f"  RMSE: {test_metrics['rmse_m']:.6f} m ({test_metrics['rmse_mm']:.3f} mm)")
    print(f"  MAE:  {test_metrics['mae_m']:.6f} m ({test_metrics['mae_mm']:.3f} mm)")
    print(f"  R²:   {test_metrics['r2']:.6f}")

    print("\nPredictive uncertainty on test set:")
    print(f"  Mean std: {y_test_std.mean():.6f} m ({y_test_std.mean() * 1000.0:.3f} mm)")
    print(f"  Max std:  {y_test_std.max():.6f} m ({y_test_std.max() * 1000.0:.3f} mm)")

    print("\nAbsolute constraint classification, true limit:")
    print(f"  Limit:             {ROBOT_LIMIT_TRUE:.3f} m")
    print(f"  Accuracy:          {clf_true_limit['true_limit_classification_accuracy'] * 100.0:.2f}%")
    print(f"  False feasible:    {clf_true_limit['true_limit_false_feasible_rate'] * 100.0:.2f}%")
    print(f"  False infeasible:  {clf_true_limit['true_limit_false_infeasible_rate'] * 100.0:.2f}%")

    print("\nAbsolute constraint classification, optimization safety limit:")
    print(f"  Limit:             {ROBOT_LIMIT_OPT:.3f} m")
    print(f"  Accuracy:          {clf_opt_limit['opt_limit_classification_accuracy'] * 100.0:.2f}%")
    print(f"  False feasible:    {clf_opt_limit['opt_limit_false_feasible_rate'] * 100.0:.2f}%")
    print(f"  False infeasible:  {clf_opt_limit['opt_limit_false_infeasible_rate'] * 100.0:.2f}%")

    print("\nNear-boundary region:")
    print(
        f"  Range:             "
        f"[{near_boundary_metrics['near_boundary_low_m']:.3f}, "
        f"{near_boundary_metrics['near_boundary_high_m']:.3f}] m"
    )
    print(f"  Samples:           {near_boundary_metrics['near_boundary_n_samples']}")
    print(f"  RMSE:              {near_boundary_metrics['near_boundary_rmse_mm']:.3f} mm")
    print(f"  MAE:               {near_boundary_metrics['near_boundary_mae_mm']:.3f} mm")
    print(f"  Mean std:          {near_boundary_metrics['near_boundary_mean_std_mm']:.3f} mm")
    print(
        f"  Accuracy:          "
        f"{near_boundary_metrics['near_boundary_classification_accuracy_true_limit'] * 100.0:.2f}%"
    )
    print(
        f"  False feasible:    "
        f"{near_boundary_metrics['near_boundary_false_feasible_rate_true_limit'] * 100.0:.2f}%"
    )

    if test_metrics["r2"] > 0.95:
        print("\nExcellent constraint surrogate accuracy: R² > 0.95")
    elif test_metrics["r2"] > 0.85:
        print("\nGood constraint surrogate accuracy: R² > 0.85")
    else:
        print("\nConstraint surrogate accuracy may need improvement.")

    return {
        "train_rmse_m": train_metrics["rmse_m"],
        "train_rmse_mm": train_metrics["rmse_mm"],
        "train_mae_m": train_metrics["mae_m"],
        "train_mae_mm": train_metrics["mae_mm"],
        "train_r2": train_metrics["r2"],
        "test_rmse_m": test_metrics["rmse_m"],
        "test_rmse_mm": test_metrics["rmse_mm"],
        "test_mae_m": test_metrics["mae_m"],
        "test_mae_mm": test_metrics["mae_mm"],
        "test_r2": test_metrics["r2"],
        "test_mean_std_m": float(y_test_std.mean()),
        "test_mean_std_mm": float(y_test_std.mean() * 1000.0),
        "test_max_std_m": float(y_test_std.max()),
        "test_max_std_mm": float(y_test_std.max() * 1000.0),
        **clf_true_limit,
        **clf_opt_limit,
        **near_boundary_metrics,
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "y_train_std": y_train_std,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "y_test_std": y_test_std,
    }


def analyze_test_outliers(
    X_test_scaled: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    scaler_X: StandardScaler,
    save_dir: Path = GP_CONSTRAINT_RESULTS_DIR,
    top_n: int = 10,
) -> pd.DataFrame:
    """Analyze the largest test errors for GP_max_abs_xr."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_test_original = scaler_X.inverse_transform(X_test_scaled)

    error = y_test_pred - y_test_true
    abs_error = np.abs(error)

    error_df = pd.DataFrame(X_test_original, columns=INPUT_COLUMNS)
    error_df["true_max_abs_xr"] = y_test_true
    error_df["pred_max_abs_xr"] = y_test_pred
    error_df["pred_std_m"] = y_test_std
    error_df["pred_std_mm"] = y_test_std * 1000.0
    error_df["error_m"] = error
    error_df["error_mm"] = error * 1000.0
    error_df["abs_error_m"] = abs_error
    error_df["abs_error_mm"] = abs_error * 1000.0
    error_df["true_feasible_abs"] = y_test_true <= ROBOT_LIMIT_TRUE
    error_df["pred_feasible_abs"] = y_test_pred <= ROBOT_LIMIT_TRUE
    error_df["true_violation_abs_m"] = np.maximum(0.0, y_test_true - ROBOT_LIMIT_TRUE)
    error_df["pred_violation_abs_m"] = np.maximum(0.0, y_test_pred - ROBOT_LIMIT_TRUE)

    error_df = error_df.sort_values("abs_error_m", ascending=False).reset_index(drop=True)

    filepath = save_dir / "test_prediction_errors_max_abs_xr.csv"
    error_df.to_csv(filepath, index=False)

    print("\n" + "=" * 70)
    print("TEST OUTLIER ANALYSIS: GP_MAX_ABS_XR")
    print("=" * 70)
    print(f"Saved full error table to: {filepath}")

    display_cols = [
        "true_max_abs_xr",
        "pred_max_abs_xr",
        "error_mm",
        "abs_error_mm",
        "pred_std_mm",
        "true_feasible_abs",
        "pred_feasible_abs",
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

    print(f"\nTop {top_n} largest absolute test errors:")
    print(error_df[display_cols].head(top_n).to_string(index=False))

    false_feasible_df = error_df[
        error_df["pred_feasible_abs"] & ~error_df["true_feasible_abs"]
    ]

    if len(false_feasible_df) > 0:
        print("\nFalse-feasible_abs test cases:")
        print(false_feasible_df[display_cols].to_string(index=False))
    else:
        print("\nNo false-feasible_abs test cases.")

    return error_df


# =============================================================================
# ARD ANALYSIS
# =============================================================================

def extract_length_scales(gp: GaussianProcessRegressor) -> np.ndarray:
    """
    Extract ARD length-scales from the optimized GP kernel.

    Expected kernel structure:
        (ConstantKernel * Matern/RBF) + WhiteKernel
    """
    kernel = gp.kernel_

    try:
        return np.asarray(kernel.k1.k2.length_scale, dtype=float)
    except AttributeError:
        pass

    try:
        return np.asarray(kernel.k2.length_scale, dtype=float)
    except AttributeError:
        pass

    raise RuntimeError(
        "Could not extract length-scales from the GP kernel. "
        "Check the optimized kernel structure."
    )


def analyze_ard_relevance(
    gp: GaussianProcessRegressor,
    param_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Analyze parameter relevance using ARD length-scales.

    Relevance is computed as:

        relevance_i = (1 / length_scale_i) / sum_j(1 / length_scale_j)
    """
    print("\n" + "=" * 70)
    print("ARD PARAMETER RELEVANCE: GP_MAX_ABS_XR")
    print("=" * 70)

    if param_names is None:
        param_names = INPUT_COLUMNS

    length_scales = extract_length_scales(gp)

    if len(length_scales) != len(param_names):
        raise ValueError(
            f"Length-scale dimension mismatch: got {len(length_scales)}, "
            f"expected {len(param_names)}."
        )

    relevance_raw = 1.0 / length_scales
    relevance_normalized = relevance_raw / relevance_raw.sum()

    ard_df = pd.DataFrame(
        {
            "Parameter": param_names,
            "LengthScale": length_scales,
            "Relevance": relevance_normalized,
        }
    )

    ard_df = ard_df.sort_values("Relevance", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print(f"{'Rank':<6}{'Parameter':<15}{'Length-scale':<18}{'Relevance':<15}")
    print("-" * 70)

    for rank, row in ard_df.iterrows():
        print(
            f"{rank + 1:<6}{row['Parameter']:<15}"
            f"{row['LengthScale']:<18.6f}{row['Relevance']:<15.4f}"
        )

    print("-" * 70)

    print("\nTop 3 most influential parameters for max_abs_xr:")
    for i in range(min(3, len(ard_df))):
        print(
            f"  {i + 1}. {ard_df.iloc[i]['Parameter']:12s} "
            f"(relevance: {ard_df.iloc[i]['Relevance']:.3f})"
        )

    return ard_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_results(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_train_std: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    metrics: dict[str, Any],
    save_dir: Path = GP_CONSTRAINT_FIGURES_DIR,
) -> None:
    """
    Generate GP_max_abs_xr diagnostic plots.

    Generated figures:
    - max_abs_xr_parity_plot.png
    - max_abs_xr_residuals.png
    - max_abs_xr_uncertainty.png
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_parity(
        y_train_true=y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        metrics=metrics,
        save_dir=save_dir,
    )

    plot_residuals(
        y_train_true=y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        metrics=metrics,
        save_dir=save_dir,
    )

    plot_uncertainty(
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        y_test_std=y_test_std,
        save_dir=save_dir,
    )


def plot_parity(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    metrics: dict[str, Any],
    save_dir: Path,
) -> None:
    """Generate train/test parity plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.scatter(
        y_train_true,
        y_train_pred,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
        label="Train",
    )

    min_train = min(float(y_train_true.min()), float(y_train_pred.min()))
    max_train = max(float(y_train_true.max()), float(y_train_pred.max()))

    ax.plot([min_train, max_train], [min_train, max_train], linestyle="--", linewidth=2, label="Identity")
    ax.axhline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2)

    ax.set_xlabel(r"True $x_{r,\max}^{\mathrm{sim}}$ [m]")
    ax.set_ylabel(r"Predicted $x_{r,\max}^{\mathrm{sim}}$ [m]")
    ax.set_title(f"Training set (R² = {metrics['train_r2']:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        y_test_true,
        y_test_pred,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
        label="Test",
    )

    min_test = min(float(y_test_true.min()), float(y_test_pred.min()))
    max_test = max(float(y_test_true.max()), float(y_test_pred.max()))

    ax.plot([min_test, max_test], [min_test, max_test], linestyle="--", linewidth=2, label="Identity")
    ax.axhline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2)

    ax.set_xlabel(r"True $x_{r,\max}^{\mathrm{sim}}$ [m]")
    ax.set_ylabel(r"Predicted $x_{r,\max}^{\mathrm{sim}}$ [m]")
    ax.set_title(f"Test set (R² = {metrics['test_r2']:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Parity plot: GP constraint surrogate", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = save_dir / "max_abs_xr_parity_plot.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_residuals(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    metrics: dict[str, Any],
    save_dir: Path,
) -> None:
    """Generate train/test residual plots."""
    residuals_train_mm = (y_train_pred - y_train_true) * 1000.0
    residuals_test_mm = (y_test_pred - y_test_true) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.scatter(
        y_train_pred,
        residuals_train_mm,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(0.0, linestyle="--", linewidth=2)
    ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax.set_xlabel(r"Predicted $x_{r,\max}^{\mathrm{sim}}$ [m]")
    ax.set_ylabel("Residual [mm]")
    ax.set_title(f"Training residuals (RMSE = {metrics['train_rmse_mm']:.2f} mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        y_test_pred,
        residuals_test_mm,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(0.0, linestyle="--", linewidth=2)
    ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax.set_xlabel(r"Predicted $x_{r,\max}^{\mathrm{sim}}$ [m]")
    ax.set_ylabel("Residual [mm]")
    ax.set_title(f"Test residuals (RMSE = {metrics['test_rmse_mm']:.2f} mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Residual analysis: GP constraint surrogate", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = save_dir / "max_abs_xr_residuals.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_uncertainty(
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    save_dir: Path,
) -> None:
    """Generate predictive uncertainty plot for the test set."""
    idx_sorted = np.argsort(y_test_true)

    y_true_sorted = y_test_true[idx_sorted]
    y_pred_sorted = y_test_pred[idx_sorted]
    y_std_sorted = y_test_std[idx_sorted]

    x = np.arange(len(y_test_true))

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x, y_true_sorted, linewidth=2, label="True max_abs_xr")
    ax.plot(x, y_pred_sorted, linewidth=2, label="Predicted max_abs_xr")

    ax.fill_between(
        x,
        y_pred_sorted - 1.96 * y_std_sorted,
        y_pred_sorted + 1.96 * y_std_sorted,
        alpha=0.3,
        label="95% predictive interval",
    )

    ax.axhline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=2, label="True robot limit")
    ax.axhline(ROBOT_LIMIT_OPT, linestyle=":", linewidth=2, label="Optimization safety limit")

    ax.set_xlabel("Test sample, sorted by true max_abs_xr")
    ax.set_ylabel("max_abs_xr [m]")
    ax.set_title("GP constraint predictions with predictive uncertainty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    filepath = save_dir / "max_abs_xr_uncertainty.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_classification_summary(
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    save_dir: Path = GP_CONSTRAINT_FIGURES_DIR,
) -> None:
    """Plot constraint-classification confusion matrices."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    limits = [
        ("True limit 0.500 m", ROBOT_LIMIT_TRUE),
        ("Safety limit 0.495 m", ROBOT_LIMIT_OPT),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (title, limit) in zip(axes, limits):
        true_feasible = y_test_true <= limit
        pred_feasible = y_test_pred <= limit

        cm = confusion_matrix(
            true_feasible,
            pred_feasible,
            labels=[False, True],
        )

        ax.imshow(cm)

        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred infeasible", "Pred feasible"])
        ax.set_yticklabels(["True infeasible", "True feasible"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    fig.suptitle("Absolute constraint classification confusion matrices")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    filepath = save_dir / "max_abs_xr_constraint_classification.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_ard_relevance(
    ard_df: pd.DataFrame,
    save_dir: Path = GP_CONSTRAINT_FIGURES_DIR,
) -> None:
    """Generate ARD parameter relevance bar plot."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(
        ard_df["Parameter"],
        ard_df["Relevance"],
        edgecolor="black",
        linewidth=1.2,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Normalized relevance")
    ax.set_title("ARD parameter relevance: GP constraint surrogate")
    ax.set_ylim(0.0, ard_df["Relevance"].max() * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

    filepath = save_dir / "max_abs_xr_ard_relevance.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


# =============================================================================
# SAVE AND LOAD MODEL
# =============================================================================

def save_test_predictions(
    X_test_scaled: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    scaler_X: StandardScaler,
    save_dir: Path = GP_CONSTRAINT_RESULTS_DIR,
) -> None:
    """Save detailed test predictions for GP_max_abs_xr."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_test_original = scaler_X.inverse_transform(X_test_scaled)

    predictions_df = pd.DataFrame(X_test_original, columns=INPUT_COLUMNS)

    predictions_df["true_max_abs_xr"] = y_test_true
    predictions_df["pred_max_abs_xr"] = y_test_pred
    predictions_df["pred_std_m"] = y_test_std
    predictions_df["pred_std_mm"] = y_test_std * 1000.0

    predictions_df["error_m"] = y_test_pred - y_test_true
    predictions_df["error_mm"] = (y_test_pred - y_test_true) * 1000.0
    predictions_df["abs_error_m"] = np.abs(y_test_pred - y_test_true)
    predictions_df["abs_error_mm"] = np.abs(y_test_pred - y_test_true) * 1000.0

    predictions_df["true_feasible_abs_true_limit"] = y_test_true <= ROBOT_LIMIT_TRUE
    predictions_df["pred_feasible_abs_true_limit"] = y_test_pred <= ROBOT_LIMIT_TRUE

    predictions_df["true_feasible_abs_opt_limit"] = y_test_true <= ROBOT_LIMIT_OPT
    predictions_df["pred_feasible_abs_opt_limit"] = y_test_pred <= ROBOT_LIMIT_OPT

    predictions_df["true_violation_abs_m"] = np.maximum(0.0, y_test_true - ROBOT_LIMIT_TRUE)
    predictions_df["pred_violation_abs_m"] = np.maximum(0.0, y_test_pred - ROBOT_LIMIT_TRUE)

    predictions_df["near_boundary"] = (
        (y_test_true >= NEAR_BOUNDARY_LOW)
        & (y_test_true <= NEAR_BOUNDARY_HIGH)
    )

    path = save_dir / "test_predictions_max_abs_xr.csv"
    predictions_df.to_csv(path, index=False)

    print(f"Test predictions saved: {path}")


def save_model(
    gp: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    metrics: dict[str, Any],
    ard_df: pd.DataFrame,
    kernel_type: str,
    alpha: float,
    n_restarts: int,
    training_time_s: float,
    save_dir: Path = GP_CONSTRAINT_RESULTS_DIR,
) -> None:
    """Save trained GP_max_abs_xr model, scalers, metrics, ARD and model info."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "gp_max_abs_xr_model.pkl"
    scaler_x_path = save_dir / "scaler_X_max_abs_xr.pkl"
    scaler_y_path = save_dir / "scaler_y_max_abs_xr.pkl"
    metrics_path = save_dir / "metrics_max_abs_xr.csv"
    ard_path = save_dir / "ard_relevance_max_abs_xr.csv"
    info_path = save_dir / "model_info_max_abs_xr.txt"

    joblib.dump(gp, model_path)
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    metrics_clean = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }

    metrics_clean["model"] = "GP_max_abs_xr"
    metrics_clean["target"] = TARGET_COLUMN
    metrics_clean["kernel_type"] = kernel_type
    metrics_clean["alpha"] = alpha
    metrics_clean["n_restarts"] = n_restarts
    metrics_clean["training_time_s"] = training_time_s
    metrics_clean["optimized_kernel"] = str(gp.kernel_)

    pd.DataFrame([metrics_clean]).to_csv(metrics_path, index=False)
    ard_df.to_csv(ard_path, index=False)

    with info_path.open("w", encoding="utf-8") as file:
        file.write("Gaussian Process Absolute Constraint Surrogate Model\n")
        file.write("=" * 60 + "\n")
        file.write("Target: max_abs_xr\n")
        file.write("Constraint interpretation: max_abs_xr <= 0.500 m\n")
        file.write(f"Input parameters: {INPUT_COLUMNS}\n")
        file.write(f"Kernel type: {kernel_type}\n")
        file.write(f"Alpha: {alpha}\n")
        file.write(f"Optimizer restarts: {n_restarts}\n")
        file.write(f"Training time: {training_time_s:.2f} s\n")
        file.write("\nOptimized kernel:\n")
        file.write(str(gp.kernel_) + "\n")
        file.write("\nTest regression metrics:\n")
        file.write(f"R2:   {metrics['test_r2']:.6f}\n")
        file.write(f"RMSE: {metrics['test_rmse_m']:.6f} m ({metrics['test_rmse_mm']:.3f} mm)\n")
        file.write(f"MAE:  {metrics['test_mae_m']:.6f} m ({metrics['test_mae_mm']:.3f} mm)\n")
        file.write("\nConstraint classification, true limit:\n")
        file.write(f"Accuracy: {metrics['true_limit_classification_accuracy'] * 100.0:.2f}%\n")
        file.write(f"False feasible: {metrics['true_limit_false_feasible_rate'] * 100.0:.2f}%\n")
        file.write(f"False infeasible: {metrics['true_limit_false_infeasible_rate'] * 100.0:.2f}%\n")
        file.write("\nNear-boundary metrics:\n")
        file.write(f"Range: [{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m\n")
        file.write(f"Samples: {metrics['near_boundary_n_samples']}\n")
        file.write(f"RMSE: {metrics['near_boundary_rmse_mm']:.3f} mm\n")
        file.write(
            f"False feasible: "
            f"{metrics['near_boundary_false_feasible_rate_true_limit'] * 100.0:.2f}%\n"
        )

    print(f"GP_max_abs_xr model saved: {model_path}")
    print(f"Input scaler saved: {scaler_x_path}")
    print(f"Output scaler saved: {scaler_y_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"ARD relevance saved: {ard_path}")
    print(f"Model info saved: {info_path}")


def load_model_max_abs_xr(
    save_dir: Path | str = GP_CONSTRAINT_RESULTS_DIR,
) -> tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]:
    """Load trained GP_max_abs_xr model and scalers."""
    save_dir = Path(save_dir)

    model_path = save_dir / "gp_max_abs_xr_model.pkl"
    scaler_x_path = save_dir / "scaler_X_max_abs_xr.pkl"
    scaler_y_path = save_dir / "scaler_y_max_abs_xr.pkl"

    require_file(model_path)
    require_file(scaler_x_path)
    require_file(scaler_y_path)

    gp = joblib.load(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    print(f"GP_max_abs_xr loaded from: {save_dir}")

    return gp, scaler_X, scaler_y


def load_model_max_xr(
    save_dir: Path | str = GP_CONSTRAINT_RESULTS_DIR,
) -> tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]:
    """
    Backward-compatible wrapper for older scripts.

    New code should use load_model_max_abs_xr().
    """
    return load_model_max_abs_xr(save_dir=save_dir)


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GP constraint surrogate for max_abs_xr."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset path. Default: {DEFAULT_DATASET_PATH}.",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default=DEFAULT_KERNEL,
        choices=SUPPORTED_KERNELS,
        help=f"Kernel type. Default: {DEFAULT_KERNEL}.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Alpha regularization. Default: {DEFAULT_ALPHA}.",
    )

    parser.add_argument(
        "--n_restarts",
        type=int,
        default=DEFAULT_N_RESTARTS,
        help=f"Number of optimizer restarts. Default: {DEFAULT_N_RESTARTS}.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Test fraction. Default: {DEFAULT_TEST_SIZE}.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random state. Default: {DEFAULT_RANDOM_STATE}.",
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip diagnostic plots.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the GP_max_abs_xr training pipeline."""
    args = parse_args()
    ensure_dirs()

    print("\n" + "=" * 70)
    print("GAUSSIAN PROCESS REGRESSION - GP_MAX_ABS_XR TRAINING PIPELINE")
    print("=" * 70 + "\n")

    X, y, df = load_constraint_dataset(args.dataset)
    print_dataset_summary(df)

    (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        _,
        _,
    ) = prepare_data(
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    gp, training_time_s = train_gp(
        X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled,
        kernel_type=args.kernel,
        n_restarts=args.n_restarts,
        alpha=args.alpha,
        random_state=args.random_state,
        verbose=True,
    )

    metrics = evaluate_model(
        gp=gp,
        X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled,
        X_test_scaled=X_test_scaled,
        y_test_scaled=y_test_scaled,
        scaler_y=scaler_y,
    )

    analyze_test_outliers(
        X_test_scaled=X_test_scaled,
        y_test_true=metrics["y_test_true"],
        y_test_pred=metrics["y_test_pred"],
        y_test_std=metrics["y_test_std"],
        scaler_X=scaler_X,
        save_dir=GP_CONSTRAINT_RESULTS_DIR,
        top_n=10,
    )

    ard_df = analyze_ard_relevance(
        gp=gp,
        param_names=INPUT_COLUMNS,
    )

    if not args.no_plots:
        print("\nGenerating plots...")

        plot_training_results(
            y_train_true=metrics["y_train_true"],
            y_train_pred=metrics["y_train_pred"],
            y_train_std=metrics["y_train_std"],
            y_test_true=metrics["y_test_true"],
            y_test_pred=metrics["y_test_pred"],
            y_test_std=metrics["y_test_std"],
            metrics=metrics,
            save_dir=GP_CONSTRAINT_FIGURES_DIR,
        )

        plot_classification_summary(
            y_test_true=metrics["y_test_true"],
            y_test_pred=metrics["y_test_pred"],
            save_dir=GP_CONSTRAINT_FIGURES_DIR,
        )

        plot_ard_relevance(
            ard_df=ard_df,
            save_dir=GP_CONSTRAINT_FIGURES_DIR,
        )

    save_test_predictions(
        X_test_scaled=X_test_scaled,
        y_test_true=metrics["y_test_true"],
        y_test_pred=metrics["y_test_pred"],
        y_test_std=metrics["y_test_std"],
        scaler_X=scaler_X,
        save_dir=GP_CONSTRAINT_RESULTS_DIR,
    )

    save_model(
        gp=gp,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        metrics=metrics,
        ard_df=ard_df,
        kernel_type=args.kernel,
        alpha=args.alpha,
        n_restarts=args.n_restarts,
        training_time_s=training_time_s,
        save_dir=GP_CONSTRAINT_RESULTS_DIR,
    )

    print("\n" + "=" * 70)
    print("GP_MAX_ABS_XR TRAINING COMPLETED")
    print("=" * 70)

    print("\nFinal metrics:")
    print(f"  Test R²:                  {metrics['test_r2']:.4f}")
    print(f"  Test RMSE:                {metrics['test_rmse_m']:.6f} m ({metrics['test_rmse_mm']:.2f} mm)")
    print(f"  Test MAE:                 {metrics['test_mae_m']:.6f} m ({metrics['test_mae_mm']:.2f} mm)")
    print(f"  Classification accuracy:  {metrics['true_limit_classification_accuracy'] * 100.0:.2f}%")
    print(f"  False feasible rate:      {metrics['true_limit_false_feasible_rate'] * 100.0:.2f}%")
    print(f"  Near-boundary RMSE:       {metrics['near_boundary_rmse_mm']:.2f} mm")

    print("\nTop 3 influential parameters for max_abs_xr:")
    for i in range(min(3, len(ard_df))):
        print(
            f"  {i + 1}. {ard_df.iloc[i]['Parameter']:12s} "
            f"(relevance: {ard_df.iloc[i]['Relevance']:.3f})"
        )

    print("\nOutput:")
    print(f"  Model:             {GP_CONSTRAINT_RESULTS_DIR / 'gp_max_abs_xr_model.pkl'}")
    print(f"  Scalers:           {GP_CONSTRAINT_RESULTS_DIR / 'scaler_*_max_abs_xr.pkl'}")
    print(f"  Metrics:           {GP_CONSTRAINT_RESULTS_DIR / 'metrics_max_abs_xr.csv'}")
    print(f"  ARD:               {GP_CONSTRAINT_RESULTS_DIR / 'ard_relevance_max_abs_xr.csv'}")
    print(f"  Model info:         {GP_CONSTRAINT_RESULTS_DIR / 'model_info_max_abs_xr.txt'}")
    print(f"  Predictions:        {GP_CONSTRAINT_RESULTS_DIR / 'test_predictions_max_abs_xr.csv'}")
    print(f"  Error analysis:     {GP_CONSTRAINT_RESULTS_DIR / 'test_prediction_errors_max_abs_xr.csv'}")

    if not args.no_plots:
        print(f"  Plots:              {GP_CONSTRAINT_FIGURES_DIR / 'max_abs_xr_*.png'}")

    print("\nNext step:")
    print("  Run the GP-based inverse optimization script.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()