"""
model_peak_y.py
===============

Single-output Gaussian Process surrogate for predicting peak outreach.

Workflow
--------
1. Load the simulation dataset.
2. Split the dataset into training and test sets.
3. Standardize input and output variables.
4. Train a Gaussian Process Regressor.
5. Evaluate the model using RMSE, MAE and R².
6. Analyze parameter relevance using ARD length-scales.
7. Generate diagnostic plots.
8. Save model, scalers, metrics and ARD results.

Supervised-learning task
------------------------
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> peak_y

This script trains a single-output GP surrogate for peak_y. In the final
constraint-aware pipeline, separate or multi-output surrogates may be used for
both peak_y and max_abs_xr.

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import ParameterRanges, load_dataset, load_dataset_dataframe


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

GP_RESULTS_DIR = RESULTS_DIR / "gp"
GP_FIGURES_DIR = FIGURES_DIR / "gp"

DEFAULT_DATASET_PATH = DATA_DIR / "dataset_augmented.csv"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_KERNEL = "matern52"
DEFAULT_ALPHA = 1e-6
DEFAULT_N_RESTARTS = 3
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_X_R_MAX = 0.500
DEFAULT_CONSTRAINT_TOLERANCE = 0.002
HIGH_OUTREACH_THRESHOLD = 0.600

SUPPORTED_KERNELS = ("matern32", "matern52", "rbf")


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    GP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
# PREPROCESSING
# =============================================================================

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
    """
    Split and standardize data for GP training.

    Standardization is important for Gaussian Processes because it improves
    numerical conditioning and makes ARD length-scales comparable across input
    dimensions.
    """
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


def print_dataset_feasibility_summary(
    filepath: Path = DEFAULT_DATASET_PATH,
    x_r_max: float = DEFAULT_X_R_MAX,
    tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE,
) -> None:
    """
    Print feasibility statistics if the dataset contains physical metrics.

    These statistics are not used for GP training. They are printed only to
    document the composition of the training dataset.
    """
    try:
        df = load_dataset_dataframe(filepath)

        required_cols = {"constraint_violation_abs", "extra_reach", "peak_y"}
        if not required_cols.issubset(df.columns):
            print("\nDataset feasibility metrics not found. Skipping feasibility summary.")
            return

        feasible = df[df["constraint_violation_abs"] <= tolerance]
        extra_reach = feasible[feasible["extra_reach"] > 0.0]
        high_outreach = feasible[feasible["peak_y"] > HIGH_OUTREACH_THRESHOLD]

        print("\n" + "=" * 70)
        print("DATASET FEASIBILITY SUMMARY")
        print("=" * 70)
        print(f"x_r_max:                         {x_r_max:.3f} m")
        print(f"Constraint tolerance:             {tolerance:.4f} m")
        print(f"Total samples:                    {len(df)}")
        print(f"Feasible_abs samples:             {len(feasible)}")
        print(f"Feasible extra-reach cases:       {len(extra_reach)}")
        print(f"Feasible high-outreach cases:     {len(high_outreach)}")
        print(f"Max feasible peak_y:              {feasible['peak_y'].max():.6f} m")

        if len(extra_reach) > 0:
            print(f"Max feasible extra_reach:         {extra_reach['extra_reach'].max():.6f} m")
        else:
            print("Max feasible extra_reach:         n/a")

        print("=" * 70)

    except Exception as exc:
        print(f"\nCould not compute feasibility summary: {exc}")


# =============================================================================
# GAUSSIAN PROCESS TRAINING
# =============================================================================

def create_gp_model(
    kernel_type: str = DEFAULT_KERNEL,
    n_dims: int = 9,
    length_scale_init: np.ndarray | None = None,
    length_scale_bounds: tuple[float, float] = (1e-2, 1e3),
    noise_level: float = 1e-5,
    alpha: float = DEFAULT_ALPHA,
    n_restarts: int = DEFAULT_N_RESTARTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> GaussianProcessRegressor:
    """
    Create a Gaussian Process Regressor with an ARD kernel.

    Kernel structure:

        k(x, x') = sigma_f² * k_base(x, x') + sigma_n² * delta(x, x')

    ARD is enabled because each input dimension has its own length-scale.
    Smaller length-scales indicate higher sensitivity to that input variable.
    """
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

    print(f"\nCreated GP model with kernel: {kernel_type}")
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
) -> GaussianProcessRegressor:
    """Train a Gaussian Process Regressor."""
    print("\n" + "=" * 70)
    print("TRAINING GAUSSIAN PROCESS")
    print("=" * 70)

    gp = create_gp_model(
        kernel_type=kernel_type,
        n_dims=X_train_scaled.shape[1],
        n_restarts=n_restarts,
        alpha=alpha,
        random_state=random_state,
    )

    print("\nFitting GP. This may take some time...")
    start_time = time.time()

    gp.fit(X_train_scaled, y_train_scaled)

    elapsed_time = time.time() - start_time
    log_marginal_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)

    print(f"Training completed in {elapsed_time:.1f} s")
    print(f"Log-marginal likelihood: {log_marginal_likelihood:.3f}")

    if verbose:
        print("\nOptimized kernel:")
        print(f"  {gp.kernel_}")

    return gp


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    gp: GaussianProcessRegressor,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> dict[str, Any]:
    """
    Evaluate the trained GP on training and test sets.

    RMSE and MAE are reported in meters. R² measures how well the surrogate
    explains the variance of the simulator output.
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

    train_rmse = float(np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
    train_mae = float(mean_absolute_error(y_train_true, y_train_pred))
    train_r2 = float(r2_score(y_train_true, y_train_pred))

    test_rmse = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
    test_mae = float(mean_absolute_error(y_test_true, y_test_pred))
    test_r2 = float(r2_score(y_test_true, y_test_pred))

    train_rmse_scaled = float(np.sqrt(mean_squared_error(y_train_scaled, y_train_pred_scaled)))
    test_rmse_scaled = float(np.sqrt(mean_squared_error(y_test_scaled, y_test_pred_scaled)))

    print("\nTraining set:")
    print(f"  RMSE: {train_rmse:.6f} m ({train_rmse * 1000.0:.2f} mm)")
    print(f"  MAE:  {train_mae:.6f} m ({train_mae * 1000.0:.2f} mm)")
    print(f"  R²:   {train_r2:.6f}")

    print("\nTest set:")
    print(f"  RMSE: {test_rmse:.6f} m ({test_rmse * 1000.0:.2f} mm)")
    print(f"  MAE:  {test_mae:.6f} m ({test_mae * 1000.0:.2f} mm)")
    print(f"  R²:   {test_r2:.6f}")

    print("\nPredictive uncertainty on test set:")
    print(f"  Mean std: {y_test_std.mean():.6f} m")
    print(f"  Max std:  {y_test_std.max():.6f} m")

    if test_r2 > 0.95:
        print("\nExcellent surrogate accuracy: R² > 0.95")
    elif test_r2 > 0.85:
        print("\nGood surrogate accuracy: R² > 0.85")
    elif test_r2 > 0.70:
        print("\nAcceptable accuracy, but the model can be improved.")
    else:
        print("\nLow accuracy. Consider increasing samples or changing kernel.")

    return {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "train_rmse_scaled": train_rmse_scaled,
        "test_rmse_scaled": test_rmse_scaled,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "y_train_std": y_train_std,
        "y_test_std": y_test_std,
        "y_train_true": y_train_true,
        "y_test_true": y_test_true,
    }


def analyze_test_outliers(
    X_test_scaled: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    scaler_X: StandardScaler,
    save_dir: Path = GP_RESULTS_DIR,
    top_n: int = 10,
) -> pd.DataFrame:
    """Analyze the largest prediction errors on the test set."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    param_names = ParameterRanges.get_param_names()
    X_test_original = scaler_X.inverse_transform(X_test_scaled)

    error = y_test_true - y_test_pred
    abs_error = np.abs(error)

    error_df = pd.DataFrame(X_test_original, columns=param_names)
    error_df["y_true"] = y_test_true
    error_df["y_pred"] = y_test_pred
    error_df["error"] = error
    error_df["abs_error"] = abs_error
    error_df["std"] = y_test_std

    error_df = error_df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    filepath = save_dir / "test_prediction_errors.csv"
    error_df.to_csv(filepath, index=False)

    print("\n" + "=" * 70)
    print("TEST OUTLIER ANALYSIS")
    print("=" * 70)
    print(f"Saved full error table to: {filepath}")

    print(f"\nTop {top_n} largest absolute test errors:")
    display_cols = [
        "y_true",
        "y_pred",
        "error",
        "abs_error",
        "std",
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
    print(error_df[display_cols].head(top_n).to_string(index=False))

    high_outreach = error_df[error_df["y_true"] > HIGH_OUTREACH_THRESHOLD]

    if len(high_outreach) > 0:
        high_rmse = float(np.sqrt(np.mean(high_outreach["error"] ** 2)))
        print(f"\nHigh-outreach region, y_true > {HIGH_OUTREACH_THRESHOLD:.2f} m:")
        print(f"  Samples:    {len(high_outreach)}")
        print(f"  Mean error: {high_outreach['error'].mean():.6f} m")
        print(f"  RMSE:       {high_rmse:.6f} m")
        print(f"  Max error:  {high_outreach['abs_error'].max():.6f} m")
    else:
        print(f"\nNo test samples with y_true > {HIGH_OUTREACH_THRESHOLD:.2f} m.")

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

    Smaller length-scales correspond to higher local sensitivity.
    """
    print("\n" + "=" * 70)
    print("ARD PARAMETER RELEVANCE")
    print("=" * 70)

    if param_names is None:
        param_names = ParameterRanges.get_param_names()

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

    print("\nTop 3 most influential parameters:")
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
    save_dir: Path = GP_FIGURES_DIR,
) -> None:
    """
    Generate GP diagnostic plots.

    Generated figures:
    - gp_parity_plot.png
    - gp_residuals.png
    - gp_uncertainty.png
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_parity(
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        metrics,
        save_dir,
    )

    plot_residuals(
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        metrics,
        save_dir,
    )

    plot_uncertainty(
        y_test_true,
        y_test_pred,
        y_test_std,
        save_dir,
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

    min_val = min(float(y_train_true.min()), float(y_train_pred.min()))
    max_val = max(float(y_train_true.max()), float(y_train_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, label="Identity")

    ax.set_xlabel("True peak outreach [m]")
    ax.set_ylabel("Predicted peak outreach [m]")
    ax.set_title(f"Training set (R² = {metrics['train_r2']:.4f})", fontweight="bold")
    ax.legend()
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

    min_val = min(float(y_test_true.min()), float(y_test_pred.min()))
    max_val = max(float(y_test_true.max()), float(y_test_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, label="Identity")

    ax.set_xlabel("True peak outreach [m]")
    ax.set_ylabel("Predicted peak outreach [m]")
    ax.set_title(f"Test set (R² = {metrics['test_r2']:.4f})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Parity plot: GP surrogate performance", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = save_dir / "gp_parity_plot.png"
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
    residuals_train = y_train_true - y_train_pred
    residuals_test = y_test_true - y_test_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.scatter(
        y_train_pred,
        residuals_train,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(0.0, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted peak outreach [m]")
    ax.set_ylabel("Residual [m]")
    ax.set_title(f"Training residuals (RMSE = {metrics['train_rmse']:.6f} m)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        y_test_pred,
        residuals_test,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(0.0, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted peak outreach [m]")
    ax.set_ylabel("Residual [m]")
    ax.set_title(f"Test residuals (RMSE = {metrics['test_rmse']:.6f} m)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Residual analysis", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = save_dir / "gp_residuals.png"
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

    ax.plot(x, y_true_sorted, linewidth=2, label="True")
    ax.plot(x, y_pred_sorted, linewidth=2, label="Predicted")
    ax.fill_between(
        x,
        y_pred_sorted - 1.96 * y_std_sorted,
        y_pred_sorted + 1.96 * y_std_sorted,
        alpha=0.3,
        label="95% predictive interval",
    )

    ax.set_xlabel("Test sample, sorted by true value")
    ax.set_ylabel("Peak outreach [m]")
    ax.set_title("GP predictions with predictive uncertainty", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = save_dir / "gp_uncertainty.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_ard_relevance(
    ard_df: pd.DataFrame,
    save_dir: Path = GP_FIGURES_DIR,
) -> None:
    """Generate ARD parameter relevance bar plot."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(
        ard_df["Parameter"],
        ard_df["Relevance"],
        edgecolor="black",
        linewidth=1.5,
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
    ax.set_title("ARD parameter importance ranking", fontweight="bold", pad=20)
    ax.set_ylim(0.0, ard_df["Relevance"].max() * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

    filepath = save_dir / "gp_ard_relevance.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


# =============================================================================
# SAVE AND LOAD MODEL
# =============================================================================

def save_model(
    gp: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    metrics: dict[str, Any],
    ard_df: pd.DataFrame,
    kernel_type: str,
    alpha: float,
    save_dir: Path = GP_RESULTS_DIR,
) -> None:
    """Save the trained GP model, scalers, metrics and ARD results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "gp_model.pkl"
    scaler_x_path = save_dir / "scaler_X.pkl"
    scaler_y_path = save_dir / "scaler_y.pkl"
    metrics_path = save_dir / "metrics.csv"
    ard_path = save_dir / "ard_relevance.csv"
    info_path = save_dir / "model_info.txt"

    joblib.dump(gp, model_path)
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    metrics_clean = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }

    pd.DataFrame([metrics_clean]).to_csv(metrics_path, index=False)
    ard_df.to_csv(ard_path, index=False)

    with info_path.open("w", encoding="utf-8") as file:
        file.write("Gaussian Process Surrogate Model\n")
        file.write("=" * 40 + "\n")
        file.write(f"Kernel type: {kernel_type}\n")
        file.write(f"Input parameters: {ParameterRanges.get_param_names()}\n")
        file.write(f"Alpha: {alpha}\n")
        file.write("Target: peak_y\n")
        file.write("\nOptimized kernel:\n")
        file.write(str(gp.kernel_) + "\n")
        file.write("\nTest metrics:\n")
        file.write(f"R2:   {metrics['test_r2']:.6f}\n")
        file.write(f"RMSE: {metrics['test_rmse']:.6f} m\n")
        file.write(f"MAE:  {metrics['test_mae']:.6f} m\n")

    print(f"GP model saved: {model_path}")
    print(f"Input scaler saved: {scaler_x_path}")
    print(f"Output scaler saved: {scaler_y_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"ARD relevance saved: {ard_path}")
    print(f"Model info saved: {info_path}")


def load_model(save_dir: Path | str = GP_RESULTS_DIR) -> tuple[
    GaussianProcessRegressor,
    StandardScaler,
    StandardScaler,
]:
    """Load the trained GP model and scalers."""
    save_dir = Path(save_dir)

    model_path = save_dir / "gp_model.pkl"
    scaler_x_path = save_dir / "scaler_X.pkl"
    scaler_y_path = save_dir / "scaler_y.pkl"

    require_file(model_path)
    require_file(scaler_x_path)
    require_file(scaler_y_path)

    gp = joblib.load(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    print(f"Model loaded from: {save_dir}")

    return gp, scaler_X, scaler_y


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a single-output GP surrogate for peak_y."
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
    """Run the GP training pipeline."""
    args = parse_args()
    ensure_dirs()
    require_file(args.dataset)

    print("\n" + "=" * 70)
    print("GAUSSIAN PROCESS REGRESSION - PEAK_Y TRAINING PIPELINE")
    print("=" * 70 + "\n")

    X, y = load_dataset(args.dataset)

    print_dataset_feasibility_summary(
        filepath=args.dataset,
        x_r_max=DEFAULT_X_R_MAX,
        tolerance=DEFAULT_CONSTRAINT_TOLERANCE,
    )

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

    gp = train_gp(
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
        save_dir=GP_RESULTS_DIR,
        top_n=10,
    )

    ard_df = analyze_ard_relevance(gp)

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
            save_dir=GP_FIGURES_DIR,
        )

        plot_ard_relevance(
            ard_df=ard_df,
            save_dir=GP_FIGURES_DIR,
        )

    save_model(
        gp=gp,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        metrics=metrics,
        ard_df=ard_df,
        kernel_type=args.kernel,
        alpha=args.alpha,
        save_dir=GP_RESULTS_DIR,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("\nFinal metrics:")
    print(f"  Test R²:   {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.6f} m ({metrics['test_rmse'] * 1000.0:.2f} mm)")
    print(f"  Test MAE:  {metrics['test_mae']:.6f} m ({metrics['test_mae'] * 1000.0:.2f} mm)")

    print("\nTop 3 influential parameters:")
    for i in range(min(3, len(ard_df))):
        print(
            f"  {i + 1}. {ard_df.iloc[i]['Parameter']:12s} "
            f"(relevance: {ard_df.iloc[i]['Relevance']:.3f})"
        )

    print("\nOutput:")
    print(f"  Model:      {GP_RESULTS_DIR / 'gp_model.pkl'}")
    print(f"  Scalers:    {GP_RESULTS_DIR / 'scaler_X.pkl'}, {GP_RESULTS_DIR / 'scaler_y.pkl'}")
    print(f"  Metrics:    {GP_RESULTS_DIR / 'metrics.csv'}")
    print(f"  ARD:        {GP_RESULTS_DIR / 'ard_relevance.csv'}")
    print(f"  Model info: {GP_RESULTS_DIR / 'model_info.txt'}")

    if not args.no_plots:
        print(f"  Plots:      {GP_FIGURES_DIR / 'gp_*.png'}")

    print("\nNext step:")
    print("  Train the constraint surrogate or run inverse optimization scripts.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()