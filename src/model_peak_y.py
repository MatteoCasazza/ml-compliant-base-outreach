"""
model_peak_y.py
=========
Gaussian Process Regression surrogate model for predicting peak outreach.

Workflow:
1. Load simulation dataset
2. Train/test split
3. Standardize input and output variables
4. Train a Gaussian Process Regressor
5. Evaluate performance using RMSE, MAE and R²
6. Analyze parameter relevance using ARD length-scales
7. Generate diagnostic plots
8. Save model, scalers, metrics and ARD results

Main supervised-learning task:
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> peak_y

The GP is used later as a fast surrogate model inside the inverse
optimization pipeline.

Author: MatteoCasazza
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time
from typing import Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel as C
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset import load_dataset, load_dataset_dataframe, ParameterRanges


# ============================================================================
# PROJECT PATHS
# ============================================================================

# This makes paths independent from the current working directory.
# The file is in src/, therefore parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

GP_RESULTS_DIR = RESULTS_DIR / "gp"
GP_FIGURES_DIR = FIGURES_DIR / "gp"

for directory in [GP_RESULTS_DIR, GP_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PREPROCESSING
# ============================================================================

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           StandardScaler, StandardScaler, np.ndarray, np.ndarray]:
    """
    Prepare data for GP training: train/test split and standardization.

    Inputs
    ------
    X : ndarray, shape (n_samples, 9)
        Input parameter matrix.
    y : ndarray, shape (n_samples,)
        Target vector, corresponding to peak_y.
    test_size : float
        Fraction of samples used for testing.
    random_state : int
        Random seed for reproducibility.

    Outputs
    -------
    X_train_scaled : ndarray
        Standardized training inputs.
    X_test_scaled : ndarray
        Standardized test inputs.
    y_train_scaled : ndarray
        Standardized training outputs.
    y_test_scaled : ndarray
        Standardized test outputs.
    scaler_X : StandardScaler
        Scaler fitted on X_train.
    scaler_y : StandardScaler
        Scaler fitted on y_train.
    y_train : ndarray
        Original, unscaled training targets.
    y_test : ndarray
        Original, unscaled test targets.

    Notes
    -----
    Standardization is essential for Gaussian Processes because:
    - it improves numerical conditioning;
    - it makes length-scales comparable across dimensions;
    - it helps the optimizer converge more reliably.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING DATASET")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"Train set: {len(y_train)} samples ({100 * (1 - test_size):.0f}%)")
    print(f"Test set:  {len(y_test)} samples ({100 * test_size:.0f}%)")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    print("\nInput standardization:")
    print(f"  Mean X_train, first 3 features: {X_train_scaled.mean(axis=0)[:3]}")
    print(f"  Std  X_train, first 3 features: {X_train_scaled.std(axis=0)[:3]}")

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print("\nOutput standardization:")
    print(f"  Mean y_train: {y_train_scaled.mean():.6f}")
    print(f"  Std  y_train: {y_train_scaled.std():.6f}")

    # More robust checks: per-feature mean and std.
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
        y_test
    )


def print_dataset_feasibility_summary(
    filepath: str | Path = DATA_DIR / 'dataset_augmented.csv',
    x_r_max: float = 0.5,
    tolerance: float = 0.002
) -> None:
    """
    Print feasibility statistics if the dataset contains physical metrics.

    Inputs
    ------
    filepath : str
        Dataset CSV path.
    x_r_max : float
        Nominal maximum robot relative position [m].
    tolerance : float
        Constraint tolerance [m].

    Notes
    -----
    These statistics are not used for GP training.
    They are printed only to document how many samples are physically feasible.
    """
    try:
        df = load_dataset_dataframe(filepath)

        required_cols = {'constraint_violation_abs', 'extra_reach', 'peak_y'}
        if not required_cols.issubset(df.columns):
            print("\nDataset feasibility metrics not found. Skipping feasibility summary.")
            return

        feasible = df[df['constraint_violation_abs'] <= tolerance]
        good = feasible[feasible['extra_reach'] > 0.0]

        print("\n" + "=" * 70)
        print("DATASET FEASIBILITY SUMMARY")
        print("=" * 70)
        print(f"x_r_max:                    {x_r_max:.3f} m")
        print(f"Constraint tolerance:        {tolerance:.4f} m")
        print(f"Total samples:               {len(df)}")
        print(f"Feasible abs samples:            {len(feasible)}")
        print(f"Feasible extra-reach cases:  {len(good)}")
        print(f"Max feasible peak_y:         {feasible['peak_y'].max():.6f} m")
        print(f"Max feasible extra_reach:    {good['extra_reach'].max():.6f} m"
              if len(good) > 0 else "Max feasible extra_reach:    n/a")
        print("=" * 70)

    except Exception as e:
        print(f"\nCould not compute feasibility summary: {e}")


# ============================================================================
# GAUSSIAN PROCESS TRAINING
# ============================================================================

def create_gp_model(
    kernel_type: str = 'matern52',
    n_dims: int = 9,
    length_scale_init: Optional[np.ndarray] = None,
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e3),
    noise_level: float = 1e-5,
    alpha: float = 1e-6,
    n_restarts: int = 10
) -> GaussianProcessRegressor:
    """
    Create a Gaussian Process Regressor with a configurable kernel.

    Inputs
    ------
    kernel_type : str
        Kernel type. Supported values:
            'matern32', 'matern52', 'rbf'
    n_dims : int
        Input dimensionality.
    length_scale_init : ndarray, optional
        Initial ARD length-scales.
    length_scale_bounds : tuple
        Lower and upper bounds for length-scale optimization.
    noise_level : float
        Initial noise level for the WhiteKernel.
    n_restarts : int
        Number of optimizer restarts.

    Output
    ------
    gp : GaussianProcessRegressor
        Untrained GP model.

    Notes
    -----
    Kernel structure:
        k(x, x') = sigma_f² * k_base(x, x') + sigma_n² * delta(x, x')

    ARD is enabled because each input dimension has its own length-scale.
    Smaller length-scale means higher relevance of that input variable.
    """
    if length_scale_init is None:
        length_scale_init = np.ones(n_dims)

    if kernel_type == 'matern32':
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=1.5
        )
    elif kernel_type == 'matern52':
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=2.5
        )
    elif kernel_type == 'rbf':
        from sklearn.gaussian_process.kernels import RBF
        base_kernel = RBF(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds
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
        random_state=42,
        alpha=alpha
    )

    print(f"\n✓ Created GP with kernel: {kernel_type.upper()}")
    print(f"  Kernel formula: {kernel}")
    print(f"  Optimizer restarts: {n_restarts}")
    print(f"  Alpha regularization: {alpha}")

    return gp


def train_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel_type: str = 'matern52',
    n_restarts: int = 10,
    alpha: float = 1e-6,
    verbose: bool = True
) -> GaussianProcessRegressor:
    """
    Train a Gaussian Process Regressor.

    Inputs
    ------
    X_train : ndarray
        Standardized training inputs.
    y_train : ndarray
        Standardized training targets.
    kernel_type : str
        Kernel type.
    n_restarts : int
        Number of optimizer restarts.
    verbose : bool
        If True, print optimized kernel details.

    Output
    ------
    gp : GaussianProcessRegressor
        Trained GP model.
    """
    print("\n" + "=" * 70)
    print("TRAINING GAUSSIAN PROCESS")
    print("=" * 70)

    gp = create_gp_model(
        kernel_type=kernel_type,
        n_dims=X_train.shape[1],
        n_restarts=n_restarts,
        alpha=alpha
    )

    print("\nFitting GP. This may take some time for 1000 samples...")
    start_time = time.time()

    gp.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"✓ Training completed in {elapsed:.1f} s")

    lml = gp.log_marginal_likelihood(gp.kernel_.theta)
    print(f"\nLog-Marginal-Likelihood: {lml:.3f}")
    print("  Higher values usually indicate a better model fit.")

    if verbose:
        print("\nOptimized kernel:")
        print(f"  {gp.kernel_}")

    return gp


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    gp: GaussianProcessRegressor,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler_y: StandardScaler
) -> Dict[str, float]:
    """
    Evaluate the trained GP on training and test sets.

    Inputs
    ------
    gp : GaussianProcessRegressor
        Trained model.
    X_train_scaled : ndarray
        Standardized training inputs.
    y_train_scaled : ndarray
        Standardized training outputs.
    X_test_scaled : ndarray
        Standardized test inputs.
    y_test_scaled : ndarray
        Standardized test outputs.
    scaler_y : StandardScaler
        Output scaler used to convert predictions back to meters.

    Output
    ------
    metrics : dict
        Performance metrics and prediction arrays.
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    y_train_pred_scaled, y_train_std_scaled = gp.predict(
        X_train_scaled, return_std=True
    )
    y_test_pred_scaled, y_test_std_scaled = gp.predict(
        X_test_scaled, return_std=True
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

    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_r2 = r2_score(y_train_true, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)

    train_rmse_scaled = np.sqrt(mean_squared_error(
        y_train_scaled, y_train_pred_scaled
    ))
    test_rmse_scaled = np.sqrt(mean_squared_error(
        y_test_scaled, y_test_pred_scaled
    ))

    print("\n--- TRAINING SET ---")
    print(f"  RMSE:  {train_rmse:.6f} m  ({train_rmse * 1000:.2f} mm)")
    print(f"  MAE:   {train_mae:.6f} m  ({train_mae * 1000:.2f} mm)")
    print(f"  R²:    {train_r2:.6f}")

    print("\n--- TEST SET ---")
    print(f"  RMSE:  {test_rmse:.6f} m  ({test_rmse * 1000:.2f} mm)")
    print(f"  MAE:   {test_mae:.6f} m  ({test_mae * 1000:.2f} mm)")
    print(f"  R²:    {test_r2:.6f}")

    print("\n--- PREDICTIVE UNCERTAINTY, TEST SET ---")
    print(f"  Mean std: {y_test_std.mean():.6f} m")
    print(f"  Max std:  {y_test_std.max():.6f} m")

    if test_r2 > 0.95:
        print("\n✓ Excellent surrogate accuracy: R² > 0.95")
    elif test_r2 > 0.85:
        print("\n✓ Good surrogate accuracy: R² > 0.85")
    elif test_r2 > 0.70:
        print("\n⚠️  Acceptable accuracy, but the model can be improved.")
    else:
        print("\n⚠️  Low accuracy. Consider increasing samples or changing kernel.")

    return {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_rmse_scaled': train_rmse_scaled,
        'test_rmse_scaled': test_rmse_scaled,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_std': y_train_std,
        'y_test_std': y_test_std,
        'y_train_true': y_train_true,
        'y_test_true': y_test_true
    }

def analyze_test_outliers(
    X_test_scaled: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    scaler_X: StandardScaler,
    save_dir: str | Path = GP_RESULTS_DIR,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze the largest prediction errors on the test set.

    Inputs
    ------
    X_test_scaled : ndarray
        Standardized test inputs.
    y_test_true : ndarray
        True test target values in physical units [m].
    y_test_pred : ndarray
        Predicted test target values in physical units [m].
    y_test_std : ndarray
        Predictive standard deviation in physical units [m].
    scaler_X : StandardScaler
        Input scaler used to recover original parameter values.
    save_dir : str
        Directory where the error table is saved.
    top_n : int
        Number of largest errors printed.

    Output
    ------
    error_df : DataFrame
        Test samples sorted by absolute prediction error.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    param_names = ParameterRanges.get_param_names()
    X_test_original = scaler_X.inverse_transform(X_test_scaled)

    error = y_test_true - y_test_pred
    abs_error = np.abs(error)

    error_df = pd.DataFrame(X_test_original, columns=param_names)
    error_df['y_true'] = y_test_true
    error_df['y_pred'] = y_test_pred
    error_df['error'] = error
    error_df['abs_error'] = abs_error
    error_df['std'] = y_test_std

    error_df = error_df.sort_values('abs_error', ascending=False).reset_index(drop=True)

    filepath = save_dir / 'test_prediction_errors.csv'
    error_df.to_csv(filepath, index=False)

    print("\n" + "=" * 70)
    print("TEST OUTLIER ANALYSIS")
    print("=" * 70)
    print(f"Saved full error table to: {filepath}")

    print(f"\nTop {top_n} largest absolute test errors:")
    cols = [
        'y_true', 'y_pred', 'error', 'abs_error', 'std',
        'Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start'
    ]
    print(error_df[cols].head(top_n).to_string(index=False))

    high_outreach = error_df[error_df['y_true'] > 0.60]
    if len(high_outreach) > 0:
        print("\nHigh-outreach region, y_true > 0.60:")
        print(f"  Samples:    {len(high_outreach)}")
        print(f"  Mean error: {high_outreach['error'].mean():.6f} m")
        print(f"  RMSE:       {np.sqrt(np.mean(high_outreach['error']**2)):.6f} m")
        print(f"  Max error:  {high_outreach['abs_error'].max():.6f} m")
    else:
        print("\nNo test samples with y_true > 0.60.")

    return error_df


# ============================================================================
# ARD ANALYSIS
# ============================================================================

def extract_length_scales(gp: GaussianProcessRegressor) -> np.ndarray:
    """
    Robustly extract ARD length-scales from the optimized GP kernel.

    Output
    ------
    length_scales : ndarray
        Optimized length-scale for each input dimension.
    """
    kernel = gp.kernel_

    # Expected structure:
    # (ConstantKernel * Matern/RBF) + WhiteKernel
    try:
        return np.asarray(kernel.k1.k2.length_scale, dtype=float)
    except AttributeError:
        pass

    # Alternative possible structure:
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
    param_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Analyze parameter relevance using ARD length-scales.

    Inputs
    ------
    gp : GaussianProcessRegressor
        Trained GP model with ARD kernel.
    param_names : list, optional
        Input parameter names.

    Output
    ------
    df : DataFrame
        Columns:
            Parameter
            LengthScale
            Relevance

    Notes
    -----
    Relevance is computed as:
        r_i = (1 / l_i) / sum_j(1 / l_j)

    where l_i is the optimized length-scale of input dimension i.
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

    df = pd.DataFrame({
        'Parameter': param_names,
        'LengthScale': length_scales,
        'Relevance': relevance_normalized
    })

    df = df.sort_values('Relevance', ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print(f"{'Rank':<6}{'Parameter':<15}{'Length-scale':<18}{'Relevance':<15}")
    print("-" * 70)

    for i, row in df.iterrows():
        print(
            f"{i + 1:<6}{row['Parameter']:<15}"
            f"{row['LengthScale']:<18.6f}{row['Relevance']:<15.4f}"
        )

    print("-" * 70)

    print("\n✓ Top 3 most influential parameters:")
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
    metrics: dict,
    save_dir: str | Path = GP_FIGURES_DIR
) -> None:
    """
    Generate GP diagnostic plots.

    Generated figures
    -----------------
    - gp_parity_plot.png
    - gp_residuals.png
    - gp_uncertainty.png
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: parity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.scatter(
        y_train_true,
        y_train_pred,
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5,
        label='Train'
    )
    ax1.plot(
        [y_train_true.min(), y_train_true.max()],
        [y_train_true.min(), y_train_true.max()],
        'r--',
        linewidth=2,
        label='Identity'
    )
    ax1.set_xlabel('True Peak Outreach [m]', fontsize=12)
    ax1.set_ylabel('Predicted Peak Outreach [m]', fontsize=12)
    ax1.set_title(
        f'Training Set (R² = {metrics["train_r2"]:.4f})',
        fontsize=13,
        fontweight='bold'
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        y_test_true,
        y_test_pred,
        alpha=0.6,
        s=50,
        c='orange',
        edgecolors='black',
        linewidth=0.5,
        label='Test'
    )
    ax2.plot(
        [y_test_true.min(), y_test_true.max()],
        [y_test_true.min(), y_test_true.max()],
        'r--',
        linewidth=2,
        label='Identity'
    )
    ax2.set_xlabel('True Peak Outreach [m]', fontsize=12)
    ax2.set_ylabel('Predicted Peak Outreach [m]', fontsize=12)
    ax2.set_title(
        f'Test Set (R² = {metrics["test_r2"]:.4f})',
        fontsize=13,
        fontweight='bold'
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Parity Plot: GP Surrogate Performance',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    filepath = save_dir / 'gp_parity_plot.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

    # Plot 2: residuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    residuals_train = y_train_true - y_train_pred
    residuals_test = y_test_true - y_test_pred

    ax1.scatter(
        y_train_pred,
        residuals_train,
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Peak Outreach [m]', fontsize=12)
    ax1.set_ylabel('Residuals [m]', fontsize=12)
    ax1.set_title(
        f'Training Residuals (RMSE = {metrics["train_rmse"]:.6f} m)',
        fontsize=13,
        fontweight='bold'
    )
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        y_test_pred,
        residuals_test,
        alpha=0.6,
        s=50,
        c='orange',
        edgecolors='black',
        linewidth=0.5
    )
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Peak Outreach [m]', fontsize=12)
    ax2.set_ylabel('Residuals [m]', fontsize=12)
    ax2.set_title(
        f'Test Residuals (RMSE = {metrics["test_rmse"]:.6f} m)',
        fontsize=13,
        fontweight='bold'
    )
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    filepath = save_dir / 'gp_residuals.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

    # Plot 3: predictive uncertainty
    fig, ax = plt.subplots(figsize=(14, 8))

    idx_sorted = np.argsort(y_test_true)
    y_true_sorted = y_test_true[idx_sorted]
    y_pred_sorted = y_test_pred[idx_sorted]
    y_std_sorted = y_test_std[idx_sorted]

    x = np.arange(len(y_test_true))

    ax.plot(x, y_true_sorted, 'k-', linewidth=2, label='True', alpha=0.8)
    ax.plot(x, y_pred_sorted, 'b-', linewidth=2, label='Predicted')
    ax.fill_between(
        x,
        y_pred_sorted - 1.96 * y_std_sorted,
        y_pred_sorted + 1.96 * y_std_sorted,
        alpha=0.3,
        color='blue',
        label='95% predictive interval'
    )

    ax.set_xlabel('Test Sample (sorted by true value)', fontsize=12)
    ax.set_ylabel('Peak Outreach [m]', fontsize=12)
    ax.set_title('GP Predictions with Predictive Uncertainty',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = save_dir / 'gp_uncertainty.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_ard_relevance(
    ard_df: pd.DataFrame,
    save_dir: str | Path = GP_FIGURES_DIR
) -> None:
    """
    Generate ARD parameter relevance bar plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ard_df)))

    bars = ax.bar(
        ard_df['Parameter'],
        ard_df['Relevance'],
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_ylabel('Normalized Relevance', fontsize=13, fontweight='bold')
    ax.set_title('ARD: Parameter Importance Ranking',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, ard_df['Relevance'].max() * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=11)

    plt.tight_layout()

    filepath = save_dir / 'gp_ard_relevance.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


# ============================================================================
# SAVE AND LOAD MODEL
# ============================================================================

def save_model(
    gp: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    metrics: dict,
    ard_df: pd.DataFrame,
    kernel_type: str,
    alpha: float,
    save_dir: str | Path = GP_RESULTS_DIR
) -> None:
    """
    Save the trained model, scalers, metrics and ARD results.

    Inputs
    ------
    gp : GaussianProcessRegressor
        Trained GP model.
    scaler_X : StandardScaler
        Input scaler.
    scaler_y : StandardScaler
        Output scaler.
    metrics : dict
        Evaluation metrics.
    ard_df : DataFrame
        ARD relevance table.
    kernel_type : str
        Kernel used for training.
    save_dir : str
        Output directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(gp, save_dir / 'gp_model.pkl')
    print(f"✓ GP saved: {save_dir / 'gp_model.pkl'}")

    joblib.dump(scaler_X, save_dir / 'scaler_X.pkl')
    joblib.dump(scaler_y, save_dir / 'scaler_y.pkl')
    print(f"✓ Scalers saved in: {save_dir}")

    metrics_clean = {
        k: v for k, v in metrics.items()
        if not isinstance(v, np.ndarray)
    }

    pd.DataFrame([metrics_clean]).to_csv(
        save_dir / 'metrics.csv',
        index=False
    )
    print(f"✓ Metrics saved: {save_dir / 'metrics.csv'}")

    ard_df.to_csv(save_dir / 'ard_relevance.csv', index=False)
    print(f"✓ ARD saved: {save_dir / 'ard_relevance.csv'}")

    with open(save_dir / 'model_info.txt', 'w') as f:
        f.write("Gaussian Process Surrogate Model\n")
        f.write("=" * 40 + "\n")
        f.write(f"Kernel type: {kernel_type}\n")
        f.write(f"Input parameters: {ParameterRanges.get_param_names()}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write("Target: peak_y\n")
        f.write("\nOptimized kernel:\n")
        f.write(str(gp.kernel_) + "\n")
        f.write("\nTest metrics:\n")
        f.write(f"R2:   {metrics['test_r2']:.6f}\n")
        f.write(f"RMSE: {metrics['test_rmse']:.6f} m\n")
        f.write(f"MAE:  {metrics['test_mae']:.6f} m\n")

    print(f"✓ Model info saved: {save_dir / 'model_info.txt'}")


def load_model(save_dir: str | Path = GP_RESULTS_DIR) -> Tuple:
    """
    Load the trained GP model and scalers.

    Output
    ------
    gp : GaussianProcessRegressor
    scaler_X : StandardScaler
    scaler_y : StandardScaler
    """
    save_dir = Path(save_dir)

    gp = joblib.load(save_dir / 'gp_model.pkl')
    scaler_X = joblib.load(save_dir / 'scaler_X.pkl')
    scaler_y = joblib.load(save_dir / 'scaler_y.pkl')

    print(f"✓ Model loaded from: {save_dir}")

    return gp, scaler_X, scaler_y


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GAUSSIAN PROCESS REGRESSION - TRAINING PIPELINE")
    print("=" * 70 + "\n")

    DATASET_PATH = DATA_DIR / 'dataset_augmented.csv'
    KERNEL_TYPE = 'matern52'
    ALPHA = 1e-6

    # Matern 5/2 is kept as the final kernel because it produced more stable
    # inverse optimization results in previous experiments.
    N_RESTARTS = 10

    # 1. Load dataset
    X, y = load_dataset(str(DATASET_PATH))

    print_dataset_feasibility_summary(
        filepath=str(DATASET_PATH),
        x_r_max=0.5,
        tolerance=0.002
    )

    # 2. Preprocessing
    (
        X_train_sc,
        X_test_sc,
        y_train_sc,
        y_test_sc,
        scaler_X,
        scaler_y,
        y_train,
        y_test
    ) = prepare_data(X, y, test_size=0.2, random_state=42)

    # 3. GP training
    gp = train_gp(
        X_train_sc,
        y_train_sc,
        kernel_type=KERNEL_TYPE,
        n_restarts=N_RESTARTS,
        alpha=ALPHA,
        verbose=True
    )

    # 4. Evaluation
    metrics = evaluate_model(
        gp,
        X_train_sc,
        y_train_sc,
        X_test_sc,
        y_test_sc,
        scaler_y
    )

    # ========== Extra: test outlier analysis ==========
    error_df = analyze_test_outliers(
        X_test_scaled=X_test_sc,
        y_test_true=metrics['y_test_true'],
        y_test_pred=metrics['y_test_pred'],
        y_test_std=metrics['y_test_std'],
        scaler_X=scaler_X,
        save_dir=GP_RESULTS_DIR,
        top_n=10
    )

    # 5. ARD analysis
    ard_df = analyze_ard_relevance(gp)

    # 6. Plots
    print("\nGenerating plots...")
    plot_training_results(
        metrics['y_train_true'],
        metrics['y_train_pred'],
        metrics['y_train_std'],
        metrics['y_test_true'],
        metrics['y_test_pred'],
        metrics['y_test_std'],
        metrics,
        save_dir=GP_FIGURES_DIR
    )

    plot_ard_relevance(ard_df, save_dir=GP_FIGURES_DIR)

    # 7. Save model
    save_model(
        gp,
        scaler_X,
        scaler_y,
        metrics,
        ard_df,
        kernel_type=KERNEL_TYPE,
        alpha=ALPHA,
        save_dir=GP_RESULTS_DIR
    )

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("\nFINAL METRICS:")
    print(f"  Test R²:   {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.6f} m "
          f"({metrics['test_rmse'] * 1000:.2f} mm)")
    print(f"  Test MAE:  {metrics['test_mae']:.6f} m "
          f"({metrics['test_mae'] * 1000:.2f} mm)")

    print("\nTOP 3 INFLUENTIAL PARAMETERS:")
    for i in range(3):
        print(
            f"  {i + 1}. {ard_df.iloc[i]['Parameter']:12s} "
            f"(relevance: {ard_df.iloc[i]['Relevance']:.3f})"
        )

    print("\nOUTPUT:")
    print(f"  Model:      {GP_RESULTS_DIR / 'gp_model.pkl'}")
    print(f"  Scalers:    {GP_RESULTS_DIR / 'scaler_*.pkl'}")
    print(f"  Metrics:    {GP_RESULTS_DIR / 'metrics.csv'}")
    print(f"  ARD:        {GP_RESULTS_DIR / 'ard_relevance.csv'}")
    print(f"  Model info: {GP_RESULTS_DIR / 'model_info.txt'}")
    print(f"  Plots:      {GP_FIGURES_DIR / 'gp_*.png'}")

    print("\nNext step: src/optimization.py")
    print("=" * 70 + "\n")