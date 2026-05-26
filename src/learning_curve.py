"""
learning_curve.py
=================

Learning curve analysis for the GP_peak_y surrogate.

Goal
----
Evaluate how the Gaussian Process surrogate performance changes as the
number of training samples increases.

This analysis helps justify:
1. the final augmented dataset size;
2. the targeted augmentation strategy;
3. the use of the final GP surrogate for inverse optimization.

Target mapping
--------------
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> peak_y

Generated outputs
-----------------
results/learning_curve/
    learning_curve_raw_results.csv
    learning_curve_summary.csv

figures/learning_curve/
    learning_curve_rmse.png
    learning_curve_r2.png
    learning_curve_mae.png
    learning_curve_training_time.png
    learning_curve_combined.png

Author: Matteo Casazza
Date: 2026
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "learning_curve"
FIGURES_DIR = PROJECT_ROOT / "figures" / "learning_curve"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SETTINGS
# ============================================================================

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

TARGET_COL = "peak_y"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Same final GP choice used in the main surrogate model.
ALPHA = 1e-6
N_RESTARTS = 1

# Increase to 3 for final report-quality results.
# If runtime is too high, set N_REPEATS = 2.
N_REPEATS = 3

# The final 80% training pool is automatically appended.
TRAIN_SIZES = [
    100,
    200,
    400,
    800,
    1200,
]

HIGH_OUTREACH_THRESHOLD = 0.60


# ============================================================================
# DATA LOADING
# ============================================================================

def load_learning_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the final augmented dataset.

    Returns
    -------
    X : ndarray
        Input parameter matrix.
    y : ndarray
        Target peak_y values.
    df : DataFrame
        Full dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = PARAM_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = df[PARAM_COLS].values
    y = df[TARGET_COL].values

    return X, y, df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print dataset statistics relevant to the learning curve."""
    print("\nDataset summary:")
    print(f"  Samples:                  {len(df)}")
    print(f"  Input parameters:          {len(PARAM_COLS)}")
    print(f"  Target column:             {TARGET_COL}")
    print(f"  Mean peak_y:               {df[TARGET_COL].mean():.6f} m")
    print(f"  Std peak_y:                {df[TARGET_COL].std():.6f} m")
    print(f"  Min peak_y:                {df[TARGET_COL].min():.6f} m")
    print(f"  Max peak_y:                {df[TARGET_COL].max():.6f} m")

    high_count = int((df[TARGET_COL] > HIGH_OUTREACH_THRESHOLD).sum())
    print(
        f"  High-outreach samples > {HIGH_OUTREACH_THRESHOLD:.2f} m: "
        f"{high_count} ({100 * high_count / len(df):.1f}%)"
    )

    if "constraint_violation" in df.columns:
        feasible_count = int((df["constraint_violation"] <= 0.0).sum())
        print(
            f"  Feasible samples:          "
            f"{feasible_count} ({100 * feasible_count / len(df):.1f}%)"
        )

    if "extra_reach" in df.columns:
        extra_count = int((df["extra_reach"] > 0.0).sum())
        print(
            f"  Extra-reach samples:       "
            f"{extra_count} ({100 * extra_count / len(df):.1f}%)"
        )


# ============================================================================
# GP MODEL
# ============================================================================

def create_gp(n_dims: int) -> GaussianProcessRegressor:
    """
    Create the GP model used in the learning curve.

    The kernel is consistent with the final GP_peak_y surrogate:
        Constant * Matern(5/2, ARD) + WhiteKernel
    """
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(1e-10, 1e-1),
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA,
        normalize_y=False,
        n_restarts_optimizer=N_RESTARTS,
        random_state=RANDOM_STATE,
    )

    return gp


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Train a GP on a subset and evaluate it on the fixed test set.
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(
        y_train.reshape(-1, 1)
    ).ravel()

    gp = create_gp(n_dims=X_train.shape[1])

    start_time = time.time()
    gp.fit(X_train_scaled, y_train_scaled)
    train_time_s = time.time() - start_time

    pred_start_time = time.time()
    y_pred_scaled, y_std_scaled = gp.predict(
        X_test_scaled,
        return_std=True,
    )
    prediction_time_s = time.time() - pred_start_time

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_std = y_std_scaled * scaler_y.scale_[0]

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    high_mask = y_test > HIGH_OUTREACH_THRESHOLD

    if np.any(high_mask):
        high_rmse = np.sqrt(
            mean_squared_error(y_test[high_mask], y_pred[high_mask])
        )
        high_mae = mean_absolute_error(
            y_test[high_mask],
            y_pred[high_mask],
        )
        high_r2 = r2_score(
            y_test[high_mask],
            y_pred[high_mask],
        )
        high_n = int(np.sum(high_mask))
    else:
        high_rmse = np.nan
        high_mae = np.nan
        high_r2 = np.nan
        high_n = 0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "high_rmse": float(high_rmse),
        "high_mae": float(high_mae),
        "high_r2": float(high_r2),
        "high_n": int(high_n),
        "mean_pred_std": float(np.mean(y_std)),
        "max_abs_error": float(np.max(np.abs(y_test - y_pred))),
        "train_time_s": float(train_time_s),
        "prediction_time_s": float(prediction_time_s),
        "optimized_kernel": str(gp.kernel_),
    }


# ============================================================================
# LEARNING CURVE
# ============================================================================

def run_learning_curve() -> None:
    """Run the complete learning curve analysis."""
    X, y, df = load_learning_dataset()

    print("\n" + "=" * 80)
    print("LEARNING CURVE ANALYSIS - GP_peak_y")
    print("=" * 80)
    print(f"Dataset path:        {DATA_PATH}")
    print(f"Alpha:               {ALPHA}")
    print(f"Kernel:              Matern 5/2 + WhiteKernel")
    print(f"Restarts:            {N_RESTARTS}")
    print(f"Repeats:             {N_REPEATS}")
    print("=" * 80)

    print_dataset_summary(df)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    max_train_size = len(X_pool)

    train_sizes = [n for n in TRAIN_SIZES if n <= max_train_size]

    if max_train_size not in train_sizes:
        train_sizes.append(max_train_size)

    train_sizes = sorted(list(set(train_sizes)))

    print("\nLearning curve setup:")
    print(f"  Training pool samples:    {len(X_pool)}")
    print(f"  Fixed test samples:       {len(X_test)}")
    print(f"  Training sizes:           {train_sizes}")

    high_test_count = int(np.sum(y_test > HIGH_OUTREACH_THRESHOLD))
    print(
        f"  High-outreach test cases: {high_test_count} "
        f"(y_test > {HIGH_OUTREACH_THRESHOLD:.2f} m)"
    )

    all_rows = []

    for n_train in train_sizes:
        print("\n" + "-" * 80)
        print(f"Training size: {n_train}")
        print("-" * 80)

        for repeat in range(N_REPEATS):
            rng = np.random.default_rng(seed=1000 + repeat)

            subset_idx = rng.choice(
                len(X_pool),
                size=n_train,
                replace=False,
            )

            X_train = X_pool[subset_idx]
            y_train = y_pool[subset_idx]

            metrics = train_and_evaluate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            row = {
                "n_train": n_train,
                "repeat": repeat + 1,
                **metrics,
            }

            all_rows.append(row)

            print(
                f"Repeat {repeat + 1}/{N_REPEATS} | "
                f"RMSE = {metrics['rmse'] * 1000:.2f} mm | "
                f"MAE = {metrics['mae'] * 1000:.2f} mm | "
                f"R² = {metrics['r2']:.4f} | "
                f"High RMSE = {metrics['high_rmse'] * 1000:.2f} mm | "
                f"Mean std = {metrics['mean_pred_std'] * 1000:.2f} mm | "
                f"Time = {metrics['train_time_s']:.1f} s"
            )

    raw_df = pd.DataFrame(all_rows)

    raw_path = RESULTS_DIR / "learning_curve_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_learning_curve(raw_df)

    summary_path = RESULTS_DIR / "learning_curve_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_learning_curves(summary_df)

    print_final_summary(summary_df, raw_path, summary_path)


def summarize_learning_curve(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs by training set size."""
    summary_df = (
        raw_df
        .groupby("n_train")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            high_rmse_mean=("high_rmse", "mean"),
            high_rmse_std=("high_rmse", "std"),
            high_mae_mean=("high_mae", "mean"),
            high_mae_std=("high_mae", "std"),
            high_r2_mean=("high_r2", "mean"),
            mean_pred_std_mean=("mean_pred_std", "mean"),
            mean_pred_std_std=("mean_pred_std", "std"),
            max_abs_error_mean=("max_abs_error", "mean"),
            train_time_s_mean=("train_time_s", "mean"),
            train_time_s_std=("train_time_s", "std"),
            prediction_time_s_mean=("prediction_time_s", "mean"),
        )
        .reset_index()
    )

    # Useful report units.
    for col in [
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "high_rmse_mean",
        "high_rmse_std",
        "high_mae_mean",
        "high_mae_std",
        "mean_pred_std_mean",
        "mean_pred_std_std",
        "max_abs_error_mean",
    ]:
        summary_df[f"{col}_mm"] = summary_df[col] * 1000.0

    return summary_df


# ============================================================================
# PLOTS
# ============================================================================

def plot_learning_curves(summary_df: pd.DataFrame) -> None:
    """Generate report-ready plots."""
    plot_rmse(summary_df)
    plot_r2(summary_df)
    plot_mae(summary_df)
    plot_training_time(summary_df)
    plot_combined_summary(summary_df)


def plot_rmse(summary_df: pd.DataFrame) -> None:
    """Plot global and high-outreach RMSE."""
    x = summary_df["n_train"].values

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        x,
        summary_df["rmse_mean_mm"],
        yerr=summary_df["rmse_std_mm"],
        marker="o",
        linewidth=2,
        capsize=5,
        label="Global RMSE",
    )

    plt.errorbar(
        x,
        summary_df["high_rmse_mean_mm"],
        yerr=summary_df["high_rmse_std_mm"],
        marker="s",
        linewidth=2,
        capsize=5,
        linestyle="--",
        label=f"High-outreach RMSE (y > {HIGH_OUTREACH_THRESHOLD:.2f} m)",
    )

    plt.xlabel("Number of training samples")
    plt.ylabel("Test RMSE [mm]")
    plt.title("Learning Curve: GP Error vs Training Set Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "learning_curve_rmse.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_r2(summary_df: pd.DataFrame) -> None:
    """Plot R² learning curve."""
    x = summary_df["n_train"].values

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        x,
        summary_df["r2_mean"],
        yerr=summary_df["r2_std"],
        marker="o",
        linewidth=2,
        capsize=5,
        label="Global R²",
    )

    plt.xlabel("Number of training samples")
    plt.ylabel("Test R²")
    plt.title("Learning Curve: GP R² vs Training Set Size")
    plt.ylim(0.80, 1.01)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "learning_curve_r2.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_mae(summary_df: pd.DataFrame) -> None:
    """Plot global and high-outreach MAE."""
    x = summary_df["n_train"].values

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        x,
        summary_df["mae_mean_mm"],
        yerr=summary_df["mae_std_mm"],
        marker="o",
        linewidth=2,
        capsize=5,
        label="Global MAE",
    )

    plt.errorbar(
        x,
        summary_df["high_mae_mean_mm"],
        yerr=summary_df["high_mae_std_mm"],
        marker="s",
        linewidth=2,
        capsize=5,
        linestyle="--",
        label=f"High-outreach MAE (y > {HIGH_OUTREACH_THRESHOLD:.2f} m)",
    )

    plt.xlabel("Number of training samples")
    plt.ylabel("Test MAE [mm]")
    plt.title("Learning Curve: GP MAE vs Training Set Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "learning_curve_mae.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_training_time(summary_df: pd.DataFrame) -> None:
    """Plot GP training time as a function of training size."""
    x = summary_df["n_train"].values

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        x,
        summary_df["train_time_s_mean"],
        yerr=summary_df["train_time_s_std"],
        marker="o",
        linewidth=2,
        capsize=5,
    )

    plt.xlabel("Number of training samples")
    plt.ylabel("Training time [s]")
    plt.title("Learning Curve: GP Training Time Scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = FIGURES_DIR / "learning_curve_training_time.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_combined_summary(summary_df: pd.DataFrame) -> None:
    """
    Plot one compact summary figure.

    This is useful for presentations because it combines the most important
    learning-curve information in a single figure.
    """
    x = summary_df["n_train"].values

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.errorbar(
        x,
        summary_df["rmse_mean_mm"],
        yerr=summary_df["rmse_std_mm"],
        marker="o",
        linewidth=2,
        capsize=5,
        label="Global RMSE",
    )
    ax.errorbar(
        x,
        summary_df["high_rmse_mean_mm"],
        yerr=summary_df["high_rmse_std_mm"],
        marker="s",
        linewidth=2,
        capsize=5,
        linestyle="--",
        label="High-outreach RMSE",
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Prediction error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.errorbar(
        x,
        summary_df["r2_mean"],
        yerr=summary_df["r2_std"],
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("R²")
    ax.set_ylim(0.80, 1.01)
    ax.set_title("Explained variance")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(
        x,
        summary_df["mean_pred_std_mean"] * 1000.0,
        yerr=summary_df["mean_pred_std_std"] * 1000.0,
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Mean predictive std [mm]")
    ax.set_title("Average GP uncertainty")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(
        x,
        summary_df["train_time_s_mean"],
        yerr=summary_df["train_time_s_std"],
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Training time [s]")
    ax.set_title("Training cost")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Learning Curve Summary for GP_peak_y",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / "learning_curve_combined.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_final_summary(
    summary_df: pd.DataFrame,
    raw_path: Path,
    summary_path: Path,
) -> None:
    """Print final results and generated files."""
    print("\n" + "=" * 80)
    print("LEARNING CURVE SUMMARY")
    print("=" * 80)

    display_cols = [
        "n_train",
        "rmse_mean_mm",
        "rmse_std_mm",
        "mae_mean_mm",
        "r2_mean",
        "high_rmse_mean_mm",
        "mean_pred_std_mean",
        "train_time_s_mean",
    ]

    print(summary_df[display_cols].to_string(index=False))

    first = summary_df.iloc[0]
    last = summary_df.iloc[-1]

    improvement_rmse = (
        100.0
        * (first["rmse_mean"] - last["rmse_mean"])
        / first["rmse_mean"]
    )

    improvement_high_rmse = (
        100.0
        * (first["high_rmse_mean"] - last["high_rmse_mean"])
        / first["high_rmse_mean"]
    )

    print("\nMain observations:")
    print(
        f"  Global RMSE improves from {first['rmse_mean_mm']:.2f} mm "
        f"to {last['rmse_mean_mm']:.2f} mm "
        f"({improvement_rmse:.1f}% reduction)."
    )
    print(
        f"  High-outreach RMSE improves from {first['high_rmse_mean_mm']:.2f} mm "
        f"to {last['high_rmse_mean_mm']:.2f} mm "
        f"({improvement_high_rmse:.1f}% reduction)."
    )
    print(
        f"  Final R² at n={int(last['n_train'])}: {last['r2_mean']:.4f}."
    )
    print(
        f"  Final mean predictive std: "
        f"{last['mean_pred_std_mean'] * 1000.0:.2f} mm."
    )

    print("\nSaved files:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {FIGURES_DIR / 'learning_curve_rmse.png'}")
    print(f"  {FIGURES_DIR / 'learning_curve_r2.png'}")
    print(f"  {FIGURES_DIR / 'learning_curve_mae.png'}")
    print(f"  {FIGURES_DIR / 'learning_curve_training_time.png'}")
    print(f"  {FIGURES_DIR / 'learning_curve_combined.png'}")
    print("=" * 80)


if __name__ == "__main__":
    run_learning_curve()