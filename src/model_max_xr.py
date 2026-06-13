"""
model_max_xr.py
================

Gaussian Process Regression surrogate model for predicting the absolute robot
relative displacement constraint quantity max_abs_xr.

Workflow
--------
1. Load augmented simulation dataset
2. Train/test split
3. Standardize input and output variables
4. Train a Gaussian Process Regressor
5. Evaluate regression accuracy
6. Evaluate constraint classification accuracy
7. Evaluate near-boundary performance around the robot limit
8. Analyze parameter relevance using ARD length-scales
9. Generate diagnostic plots
10. Save model, scalers, metrics, ARD results and model information

Main supervised-learning task:
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> max_abs_xr

The GP_max_abs_xr model is used as an auxiliary constraint surrogate inside
constraint-aware inverse optimization. It does not replace the main GP_peak_y
surrogate.

Author: MatteoCasazza
Date: 2026
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATASET_PATH = DATA_DIR / "dataset_augmented.csv"

GP_CONSTRAINT_RESULTS_DIR = RESULTS_DIR / "gp_constraints"
GP_CONSTRAINT_FIGURES_DIR = FIGURES_DIR / "gp_constraints"

for directory in [GP_CONSTRAINT_RESULTS_DIR, GP_CONSTRAINT_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

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

# New absolute constraint target.
TARGET_COLUMN = "max_abs_xr"

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495

NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520

TEST_SIZE = 0.20
RANDOM_STATE = 42

KERNEL_TYPE = "matern52"
ALPHA = 1e-10
N_RESTARTS = 3


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_constraint_dataset(
    filepath: str | Path = DATASET_PATH,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load dataset for the max_abs_xr constraint surrogate.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, comment="#")

    missing_inputs = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing_inputs:
        raise ValueError(f"Missing input columns: {missing_inputs}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COLUMN}'. "
            "Regenerate dataset.py and augment_high_outreach.py so that "
            "dataset_augmented.csv contains max_abs_xr."
        )

    df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    X = df[INPUT_COLUMNS].values
    y = df[TARGET_COLUMN].values

    return X, y, df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print dataset and absolute-constraint statistics.
    """
    y = df[TARGET_COLUMN].values

    true_feasible = y <= ROBOT_LIMIT_TRUE
    opt_feasible = y <= ROBOT_LIMIT_OPT

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
    print(f"Violation rate, true limit:    {np.mean(~true_feasible) * 100:.2f}%")
    print(f"Violation rate, opt limit:     {np.mean(~opt_feasible) * 100:.2f}%")

    near_boundary = df[
        (df[TARGET_COLUMN] >= NEAR_BOUNDARY_LOW)
        & (df[TARGET_COLUMN] <= NEAR_BOUNDARY_HIGH)
    ]
    print(
        f"Near-boundary samples "
        f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m: "
        f"{len(near_boundary)}"
    )

    if "constraint_violation_abs" in df.columns:
        print(
            f"Mean abs constraint violation: {df['constraint_violation_abs'].mean() * 1000:.3f} mm"
        )
        print(
            f"Max abs constraint violation:  {df['constraint_violation_abs'].max() * 1000:.3f} mm"
        )

    if "feasible_abs" in df.columns:
        print(f"feasible_abs samples:          {int(df['feasible_abs'].sum())}")

    print("=" * 70)


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
    np.ndarray,
    np.ndarray,
]:
    """
    Prepare data for GP training: train/test split and standardization.
    """
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

    print(f"Train set: {len(y_train)} samples ({100 * (1 - test_size):.0f}%)")
    print(f"Test set:  {len(y_test)} samples ({100 * test_size:.0f}%)")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print("\nInput standardization:")
    print(f"  Mean X_train, first 3 features: {X_train_scaled.mean(axis=0)[:3]}")
    print(f"  Std  X_train, first 3 features: {X_train_scaled.std(axis=0)[:3]}")

    print("\nOutput standardization:")
    print(f"  Mean y_train: {y_train_scaled.mean():.6f}")
    print(f"  Std  y_train: {y_train_scaled.std():.6f}")

    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-10), \
        "X_train is not centered feature-wise."
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-10), \
        "X_train is not normalized feature-wise."
    assert np.allclose(y_train_scaled.mean(), 0.0, atol=1e-10), \
        "y_train is not centered."
    assert np.allclose(y_train_scaled.std(), 1.0, atol=1e-10), \
        "y_train is not normalized."

    print("\n✓ Preprocessing completed")

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


# ============================================================================
# GAUSSIAN PROCESS TRAINING
# ============================================================================

def create_gp_model(
    kernel_type: str = KERNEL_TYPE,
    n_dims: int = 9,
    length_scale_init: Optional[np.ndarray] = None,
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e3),
    noise_level: float = 1e-5,
    alpha: float = ALPHA,
    n_restarts: int = N_RESTARTS,
) -> GaussianProcessRegressor:
    """
    Create a Gaussian Process Regressor with ARD length-scales.
    """
    if length_scale_init is None:
        length_scale_init = np.ones(n_dims)

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
    elif kernel_type == "rbf":
        from sklearn.gaussian_process.kernels import RBF

        base_kernel = RBF(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    kernel = (
        C(1.0, constant_value_bounds=(1e-3, 1e3))
        * base_kernel
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
        random_state=RANDOM_STATE,
        alpha=alpha,
    )

    print(f"\n✓ Created GP with kernel: {kernel_type.upper()}")
    print(f"  Kernel formula: {kernel}")
    print(f"  Optimizer restarts: {n_restarts}")
    print(f"  Alpha regularization: {alpha}")

    return gp


def train_gp(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    kernel_type: str = KERNEL_TYPE,
    n_restarts: int = N_RESTARTS,
    alpha: float = ALPHA,
    verbose: bool = True,
) -> Tuple[GaussianProcessRegressor, float]:
    """
    Train a Gaussian Process Regressor.
    """
    print("\n" + "=" * 70)
    print("TRAINING GP_MAX_ABS_XR")
    print("=" * 70)

    gp = create_gp_model(
        kernel_type=kernel_type,
        n_dims=X_train_scaled.shape[1],
        n_restarts=n_restarts,
        alpha=alpha,
    )

    print("\nFitting GP_max_abs_xr. This may take some time...")
    start_time = time.time()

    gp.fit(X_train_scaled, y_train_scaled)

    elapsed = time.time() - start_time

    print(f"✓ Training completed in {elapsed:.1f} s")

    lml = gp.log_marginal_likelihood(gp.kernel_.theta)
    print(f"\nLog-Marginal-Likelihood: {lml:.3f}")

    if verbose:
        print("\nOptimized kernel:")
        print(f"  {gp.kernel_}")

    return gp, elapsed


# ============================================================================
# EVALUATION
# ============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

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
) -> Dict[str, float]:
    """
    Evaluate feasibility classification using max_abs_xr <= limit.

    False feasible is the most critical error:
    predicted feasible but actually infeasible.
    """
    true_feasible = y_true <= limit
    pred_feasible = y_pred <= limit

    accuracy = np.mean(true_feasible == pred_feasible)
    false_feasible = np.mean((pred_feasible == True) & (true_feasible == False))
    false_infeasible = np.mean((pred_feasible == False) & (true_feasible == True))

    tn, fp, fn, tp = confusion_matrix(
        true_feasible,
        pred_feasible,
        labels=[False, True],
    ).ravel()

    return {
        f"{label}_limit_m": limit,
        f"{label}_classification_accuracy": accuracy,
        f"{label}_false_feasible_rate": false_feasible,
        f"{label}_false_infeasible_rate": false_infeasible,
        f"{label}_true_infeasible_pred_infeasible": tn,
        f"{label}_true_infeasible_pred_feasible": fp,
        f"{label}_true_feasible_pred_infeasible": fn,
        f"{label}_true_feasible_pred_feasible": tp,
    }


def evaluate_near_boundary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    low: float = NEAR_BOUNDARY_LOW,
    high: float = NEAR_BOUNDARY_HIGH,
) -> Dict[str, float]:
    """
    Evaluate prediction quality near the absolute robot constraint boundary.
    """
    mask = (y_true >= low) & (y_true <= high)

    if not np.any(mask):
        return {
            "near_boundary_low_m": low,
            "near_boundary_high_m": high,
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
        "near_boundary_low_m": low,
        "near_boundary_high_m": high,
        "near_boundary_n_samples": len(y_true_nb),
        "near_boundary_rmse_mm": reg["rmse_mm"],
        "near_boundary_mae_mm": reg["mae_mm"],
        "near_boundary_mean_std_mm": np.mean(y_std_nb) * 1000.0,
        "near_boundary_classification_accuracy_true_limit": (
            clf["near_boundary_true_limit_classification_accuracy"]
        ),
        "near_boundary_false_feasible_rate_true_limit": (
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
) -> Dict[str, object]:
    """
    Evaluate the trained GP on training and test sets.
    """
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

    print("\n--- TRAINING SET ---")
    print(f"  RMSE:  {train_metrics['rmse_m']:.6f} m  ({train_metrics['rmse_mm']:.3f} mm)")
    print(f"  MAE:   {train_metrics['mae_m']:.6f} m  ({train_metrics['mae_mm']:.3f} mm)")
    print(f"  R²:    {train_metrics['r2']:.6f}")

    print("\n--- TEST SET ---")
    print(f"  RMSE:  {test_metrics['rmse_m']:.6f} m  ({test_metrics['rmse_mm']:.3f} mm)")
    print(f"  MAE:   {test_metrics['mae_m']:.6f} m  ({test_metrics['mae_mm']:.3f} mm)")
    print(f"  R²:    {test_metrics['r2']:.6f}")

    print("\n--- PREDICTIVE UNCERTAINTY, TEST SET ---")
    print(f"  Mean std: {y_test_std.mean():.6f} m ({y_test_std.mean() * 1000:.3f} mm)")
    print(f"  Max std:  {y_test_std.max():.6f} m ({y_test_std.max() * 1000:.3f} mm)")

    print("\n--- ABSOLUTE CONSTRAINT CLASSIFICATION, TRUE LIMIT ---")
    print(f"  Limit:             {ROBOT_LIMIT_TRUE:.3f} m")
    print(f"  Accuracy:          {clf_true_limit['true_limit_classification_accuracy'] * 100:.2f}%")
    print(f"  False feasible:    {clf_true_limit['true_limit_false_feasible_rate'] * 100:.2f}%")
    print(f"  False infeasible:  {clf_true_limit['true_limit_false_infeasible_rate'] * 100:.2f}%")

    print("\n--- ABSOLUTE CONSTRAINT CLASSIFICATION, OPTIMIZATION SAFETY LIMIT ---")
    print(f"  Limit:             {ROBOT_LIMIT_OPT:.3f} m")
    print(f"  Accuracy:          {clf_opt_limit['opt_limit_classification_accuracy'] * 100:.2f}%")
    print(f"  False feasible:    {clf_opt_limit['opt_limit_false_feasible_rate'] * 100:.2f}%")
    print(f"  False infeasible:  {clf_opt_limit['opt_limit_false_infeasible_rate'] * 100:.2f}%")

    print("\n--- NEAR-BOUNDARY REGION ---")
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
        f"{near_boundary_metrics['near_boundary_classification_accuracy_true_limit'] * 100:.2f}%"
    )
    print(
        f"  False feasible:    "
        f"{near_boundary_metrics['near_boundary_false_feasible_rate_true_limit'] * 100:.2f}%"
    )

    if test_metrics["r2"] > 0.95:
        print("\n✓ Excellent constraint surrogate accuracy: R² > 0.95")
    elif test_metrics["r2"] > 0.85:
        print("\n✓ Good constraint surrogate accuracy: R² > 0.85")
    else:
        print("\n⚠️ Constraint surrogate accuracy may need improvement.")

    metrics = {
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
        "test_mean_std_m": y_test_std.mean(),
        "test_mean_std_mm": y_test_std.mean() * 1000.0,
        "test_max_std_m": y_test_std.max(),
        "test_max_std_mm": y_test_std.max() * 1000.0,
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

    return metrics


def analyze_test_outliers(
    X_test_scaled: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    scaler_X: StandardScaler,
    save_dir: str | Path = GP_CONSTRAINT_RESULTS_DIR,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Analyze largest test errors for GP_max_abs_xr.
    """
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

    error_df = error_df.sort_values(
        "abs_error_m",
        ascending=False,
    ).reset_index(drop=True)

    filepath = save_dir / "test_prediction_errors_max_abs_xr.csv"
    error_df.to_csv(filepath, index=False)

    print("\n" + "=" * 70)
    print("TEST OUTLIER ANALYSIS: GP_MAX_ABS_XR")
    print("=" * 70)
    print(f"Saved full error table to: {filepath}")

    cols = [
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
    print(error_df[cols].head(top_n).to_string(index=False))

    false_feasible_df = error_df[
        (error_df["pred_feasible_abs"] == True)
        & (error_df["true_feasible_abs"] == False)
    ]

    if len(false_feasible_df) > 0:
        print("\nFalse-feasible_abs test cases:")
        print(false_feasible_df[cols].to_string(index=False))
    else:
        print("\nNo false-feasible_abs test cases.")

    return error_df


# ============================================================================
# ARD ANALYSIS
# ============================================================================

def extract_length_scales(gp: GaussianProcessRegressor) -> np.ndarray:
    """
    Robustly extract ARD length-scales from optimized GP kernel.
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
        "Check kernel structure."
    )


def analyze_ard_relevance(
    gp: GaussianProcessRegressor,
    param_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    Analyze parameter relevance using ARD length-scales.
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

    df = pd.DataFrame(
        {
            "Parameter": param_names,
            "LengthScale": length_scales,
            "Relevance": relevance_normalized,
        }
    )

    df = df.sort_values("Relevance", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print(f"{'Rank':<6}{'Parameter':<15}{'Length-scale':<18}{'Relevance':<15}")
    print("-" * 70)

    for i, row in df.iterrows():
        print(
            f"{i + 1:<6}{row['Parameter']:<15}"
            f"{row['LengthScale']:<18.6f}{row['Relevance']:<15.4f}"
        )

    print("-" * 70)

    print("\n✓ Top 3 most influential parameters for max_abs_xr:")
    for i in range(min(3, len(df))):
        print(
            f"  {i + 1}. {df.iloc[i]['Parameter']:12s} "
            f"(relevance: {df.iloc[i]['Relevance']:.3f})"
        )

    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_results(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_train_std: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    metrics: Dict[str, object],
    save_dir: str | Path = GP_CONSTRAINT_FIGURES_DIR,
) -> None:
    """
    Generate GP_max_abs_xr diagnostic plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Plot 1: parity plot
    # ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.scatter(
        y_train_true,
        y_train_pred,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
        label="Train",
    )

    min_train = min(y_train_true.min(), y_train_pred.min())
    max_train = max(y_train_true.max(), y_train_pred.max())

    ax1.plot(
        [min_train, max_train],
        [min_train, max_train],
        "--",
        linewidth=2,
        label="Identity",
    )

    ax1.axhline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax1.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2)

    ax1.set_xlabel("True max_abs_xr [m]")
    ax1.set_ylabel("Predicted max_abs_xr [m]")
    ax1.set_title(f"Training Set (R² = {metrics['train_r2']:.4f})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        y_test_true,
        y_test_pred,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
        label="Test",
    )

    min_test = min(y_test_true.min(), y_test_pred.min())
    max_test = max(y_test_true.max(), y_test_pred.max())

    ax2.plot(
        [min_test, max_test],
        [min_test, max_test],
        "--",
        linewidth=2,
        label="Identity",
    )

    ax2.axhline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax2.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2)

    ax2.set_xlabel("True max_abs_xr [m]")
    ax2.set_ylabel("Predicted max_abs_xr [m]")
    ax2.set_title(f"Test Set (R² = {metrics['test_r2']:.4f})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Parity Plot: GP_max_abs_xr Constraint Surrogate", fontsize=15)
    plt.tight_layout()

    filepath = save_dir / "max_abs_xr_parity_plot.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()

    # ----------------------------------------------------------------------
    # Plot 2: residuals
    # ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    residuals_train = y_train_pred - y_train_true
    residuals_test = y_test_pred - y_test_true

    ax1.scatter(
        y_train_pred,
        residuals_train * 1000.0,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    ax1.axhline(0.0, linestyle="--", linewidth=2)
    ax1.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax1.set_xlabel("Predicted max_abs_xr [m]")
    ax1.set_ylabel("Residual [mm]")
    ax1.set_title(f"Training residuals (RMSE = {metrics['train_rmse_mm']:.2f} mm)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        y_test_pred,
        residuals_test * 1000.0,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    ax2.axhline(0.0, linestyle="--", linewidth=2)
    ax2.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
    ax2.set_xlabel("Predicted max_abs_xr [m]")
    ax2.set_ylabel("Residual [mm]")
    ax2.set_title(f"Test residuals (RMSE = {metrics['test_rmse_mm']:.2f} mm)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Residual Analysis: GP_max_abs_xr", fontsize=15)
    plt.tight_layout()

    filepath = save_dir / "max_abs_xr_residuals.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()

    # ----------------------------------------------------------------------
    # Plot 3: uncertainty
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    idx_sorted = np.argsort(y_test_true)
    y_true_sorted = y_test_true[idx_sorted]
    y_pred_sorted = y_test_pred[idx_sorted]
    y_std_sorted = y_test_std[idx_sorted]

    x = np.arange(len(y_test_true))

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
    ax.set_title("GP_max_abs_xr Predictions with Predictive Uncertainty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = save_dir / "max_abs_xr_uncertainty.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_classification_summary(
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    save_dir: str | Path = GP_CONSTRAINT_FIGURES_DIR,
) -> None:
    """
    Plot constraint classification confusion matrices for true and safety limits.
    """
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

    plt.suptitle("Absolute Constraint Classification Confusion Matrices")
    plt.tight_layout()

    filepath = save_dir / "max_abs_xr_constraint_classification.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_ard_relevance(
    ard_df: pd.DataFrame,
    save_dir: str | Path = GP_CONSTRAINT_FIGURES_DIR,
) -> None:
    """
    Generate ARD parameter relevance bar plot.
    """
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
    ax.set_title("ARD Parameter Relevance: GP_max_abs_xr")
    ax.set_ylim(0, ard_df["Relevance"].max() * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    filepath = save_dir / "max_abs_xr_ard_relevance.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()


# ============================================================================
# SAVE AND LOAD MODEL
# ============================================================================

def save_model(
    gp: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    metrics: Dict[str, object],
    ard_df: pd.DataFrame,
    kernel_type: str,
    alpha: float,
    training_time_s: float,
    save_dir: str | Path = GP_CONSTRAINT_RESULTS_DIR,
) -> None:
    """
    Save trained GP_max_abs_xr model, scalers, metrics, ARD and model info.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(gp, save_dir / "gp_max_abs_xr_model.pkl")
    print(f"✓ GP_max_abs_xr saved: {save_dir / 'gp_max_abs_xr_model.pkl'}")

    joblib.dump(scaler_X, save_dir / "scaler_X_max_abs_xr.pkl")
    joblib.dump(scaler_y, save_dir / "scaler_y_max_abs_xr.pkl")
    print(f"✓ Scalers saved in: {save_dir}")

    metrics_clean = {
        k: v for k, v in metrics.items()
        if not isinstance(v, np.ndarray)
    }

    metrics_clean["model"] = "GP_max_abs_xr"
    metrics_clean["target"] = TARGET_COLUMN
    metrics_clean["kernel_type"] = kernel_type
    metrics_clean["alpha"] = alpha
    metrics_clean["n_restarts"] = N_RESTARTS
    metrics_clean["training_time_s"] = training_time_s
    metrics_clean["optimized_kernel"] = str(gp.kernel_)

    pd.DataFrame([metrics_clean]).to_csv(
        save_dir / "metrics_max_abs_xr.csv",
        index=False,
    )
    print(f"✓ Metrics saved: {save_dir / 'metrics_max_abs_xr.csv'}")

    ard_df.to_csv(save_dir / "ard_relevance_max_abs_xr.csv", index=False)
    print(f"✓ ARD saved: {save_dir / 'ard_relevance_max_abs_xr.csv'}")

    with open(save_dir / "model_info_max_abs_xr.txt", "w", encoding="utf-8") as f:
        f.write("Gaussian Process Absolute Constraint Surrogate Model\n")
        f.write("=" * 60 + "\n")
        f.write("Target: max_abs_xr\n")
        f.write("Constraint interpretation: max_abs_xr <= 0.5 m\n")
        f.write(f"Input parameters: {INPUT_COLUMNS}\n")
        f.write(f"Kernel type: {kernel_type}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Optimizer restarts: {N_RESTARTS}\n")
        f.write(f"Training time: {training_time_s:.2f} s\n")
        f.write("\nOptimized kernel:\n")
        f.write(str(gp.kernel_) + "\n")
        f.write("\nTest regression metrics:\n")
        f.write(f"R2:   {metrics['test_r2']:.6f}\n")
        f.write(f"RMSE: {metrics['test_rmse_m']:.6f} m ({metrics['test_rmse_mm']:.3f} mm)\n")
        f.write(f"MAE:  {metrics['test_mae_m']:.6f} m ({metrics['test_mae_mm']:.3f} mm)\n")
        f.write("\nConstraint classification, true limit:\n")
        f.write(f"Accuracy: {metrics['true_limit_classification_accuracy'] * 100:.2f}%\n")
        f.write(f"False feasible: {metrics['true_limit_false_feasible_rate'] * 100:.2f}%\n")
        f.write(f"False infeasible: {metrics['true_limit_false_infeasible_rate'] * 100:.2f}%\n")
        f.write("\nNear-boundary metrics:\n")
        f.write(f"Range: [{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m\n")
        f.write(f"Samples: {metrics['near_boundary_n_samples']}\n")
        f.write(f"RMSE: {metrics['near_boundary_rmse_mm']:.3f} mm\n")
        f.write(
            f"False feasible: "
            f"{metrics['near_boundary_false_feasible_rate_true_limit'] * 100:.2f}%\n"
        )

    print(f"✓ Model info saved: {save_dir / 'model_info_max_abs_xr.txt'}")


def load_model_max_abs_xr(
    save_dir: str | Path = GP_CONSTRAINT_RESULTS_DIR,
) -> Tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]:
    """
    Load trained GP_max_abs_xr model and scalers.
    """
    save_dir = Path(save_dir)

    gp = joblib.load(save_dir / "gp_max_abs_xr_model.pkl")
    scaler_X = joblib.load(save_dir / "scaler_X_max_abs_xr.pkl")
    scaler_y = joblib.load(save_dir / "scaler_y_max_abs_xr.pkl")

    print(f"✓ GP_max_abs_xr loaded from: {save_dir}")

    return gp, scaler_X, scaler_y


def load_model_max_xr(
    save_dir: str | Path = GP_CONSTRAINT_RESULTS_DIR,
) -> Tuple[GaussianProcessRegressor, StandardScaler, StandardScaler]:
    """
    Backward-compatible wrapper for older scripts.

    New code should use load_model_max_abs_xr().
    """
    return load_model_max_abs_xr(save_dir=save_dir)


def save_test_predictions(
    X_test_scaled: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    scaler_X: StandardScaler,
    save_dir: str | Path = GP_CONSTRAINT_RESULTS_DIR,
) -> None:
    """
    Save detailed test predictions for GP_max_abs_xr.
    """
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

    print(f"✓ Test predictions saved: {path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main() -> None:
    """
    Run the GP_max_abs_xr training pipeline.
    """
    print("\n" + "=" * 70)
    print("GAUSSIAN PROCESS REGRESSION - GP_MAX_ABS_XR TRAINING PIPELINE")
    print("=" * 70 + "\n")

    # 1. Load dataset
    X, y, df = load_constraint_dataset(DATASET_PATH)

    print_dataset_summary(df)

    # 2. Preprocessing
    (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        y_train,
        y_test,
    ) = prepare_data(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # 3. GP training
    gp, training_time_s = train_gp(
        X_train_scaled,
        y_train_scaled,
        kernel_type=KERNEL_TYPE,
        n_restarts=N_RESTARTS,
        alpha=ALPHA,
        verbose=True,
    )

    # 4. Evaluation
    metrics = evaluate_model(
        gp,
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler_y,
    )

    # 5. Test outlier analysis
    error_df = analyze_test_outliers(
        X_test_scaled=X_test_scaled,
        y_test_true=metrics["y_test_true"],
        y_test_pred=metrics["y_test_pred"],
        y_test_std=metrics["y_test_std"],
        scaler_X=scaler_X,
        save_dir=GP_CONSTRAINT_RESULTS_DIR,
        top_n=10,
    )

    # 6. ARD analysis
    ard_df = analyze_ard_relevance(
        gp,
        param_names=INPUT_COLUMNS,
    )

    # 7. Plots
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
        ard_df,
        save_dir=GP_CONSTRAINT_FIGURES_DIR,
    )

    # 8. Save model and results
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
        kernel_type=KERNEL_TYPE,
        alpha=ALPHA,
        training_time_s=training_time_s,
        save_dir=GP_CONSTRAINT_RESULTS_DIR,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("GP_MAX_ABS_XR TRAINING COMPLETED")
    print("=" * 70)

    print("\nFINAL METRICS:")
    print(f"  Test R²:                  {metrics['test_r2']:.4f}")
    print(
        f"  Test RMSE:                {metrics['test_rmse_m']:.6f} m "
        f"({metrics['test_rmse_mm']:.2f} mm)"
    )
    print(
        f"  Test MAE:                 {metrics['test_mae_m']:.6f} m "
        f"({metrics['test_mae_mm']:.2f} mm)"
    )
    print(
        f"  Classification accuracy:  "
        f"{metrics['true_limit_classification_accuracy'] * 100:.2f}%"
    )
    print(
        f"  False feasible rate:      "
        f"{metrics['true_limit_false_feasible_rate'] * 100:.2f}%"
    )
    print(
        f"  Near-boundary RMSE:       "
        f"{metrics['near_boundary_rmse_mm']:.2f} mm"
    )

    print("\nTOP 3 INFLUENTIAL PARAMETERS FOR max_abs_xr:")
    for i in range(min(3, len(ard_df))):
        print(
            f"  {i + 1}. {ard_df.iloc[i]['Parameter']:12s} "
            f"(relevance: {ard_df.iloc[i]['Relevance']:.3f})"
        )

    print("\nOUTPUT:")
    print(f"  Model:             {GP_CONSTRAINT_RESULTS_DIR / 'gp_max_abs_xr_model.pkl'}")
    print(f"  Scalers:           {GP_CONSTRAINT_RESULTS_DIR / 'scaler_*_max_abs_xr.pkl'}")
    print(f"  Metrics:           {GP_CONSTRAINT_RESULTS_DIR / 'metrics_max_abs_xr.csv'}")
    print(f"  ARD:               {GP_CONSTRAINT_RESULTS_DIR / 'ard_relevance_max_abs_xr.csv'}")
    print(f"  Model info:         {GP_CONSTRAINT_RESULTS_DIR / 'model_info_max_abs_xr.txt'}")
    print(f"  Predictions:        {GP_CONSTRAINT_RESULTS_DIR / 'test_predictions_max_abs_xr.csv'}")
    print(f"  Error analysis:     {GP_CONSTRAINT_RESULTS_DIR / 'test_prediction_errors_max_abs_xr.csv'}")
    print(f"  Plots:              {GP_CONSTRAINT_FIGURES_DIR / 'max_abs_xr_*.png'}")

    print("\nNext step: src/optimization_constraint_aware.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
