"""
cross_validation_gp.py
======================

Cross-validation study for Gaussian Process kernel and alpha selection.

Goal
----
Compare different Gaussian Process kernel families and numerical alpha values
on the final augmented dataset.

Target mapping
--------------
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> peak_y

Cross-validation setup
----------------------
- 5-fold cross-validation
- Kernels:
    1. RBF
    2. Matern 3/2
    3. Matern 5/2
    4. Rational Quadratic
- Alpha values:
    1. 1e-10
    2. 1e-6
- One optimizer restart per GP fit

Generated outputs
-----------------
results/cross_validation/
    cross_validation_raw_results.csv
    cross_validation_summary.csv
    cross_validation_report_table.csv
    best_kernel_summary.txt

figures/cross_validation/
    cv_rmse_by_kernel_alpha.png
    cv_high_rmse_by_kernel_alpha.png
    cv_r2_by_kernel_alpha.png
    cv_lml_by_kernel_alpha.png
    cv_train_time_by_kernel_alpha.png
    cv_warnings_by_kernel_alpha.png
    cv_summary.png

Author: Matteo Casazza
Date: 2026
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_validation"
FIGURES_DIR = PROJECT_ROOT / "figures" / "cross_validation"

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

N_SPLITS = 5
N_RESTARTS = 1
ALPHAS = [1e-10, 1e-6]

KERNEL_NAMES = [
    "RBF",
    "Matern_3/2",
    "Matern_5/2",
    "RationalQuadratic",
]

RANDOM_STATE = 42

HIGH_OUTREACH_THRESHOLD = 0.60


# ============================================================================
# DATA
# ============================================================================

def load_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the final augmented dataset.
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
    """
    Print compact dataset statistics.
    """
    print("\nDataset summary:")
    print(f"  Samples:                  {len(df)}")
    print(f"  Input dimensions:          {len(PARAM_COLS)}")
    print(f"  Target:                    {TARGET_COL}")
    print(f"  Mean peak_y:               {df[TARGET_COL].mean():.6f} m")
    print(f"  Std peak_y:                {df[TARGET_COL].std():.6f} m")
    print(f"  Min peak_y:                {df[TARGET_COL].min():.6f} m")
    print(f"  Max peak_y:                {df[TARGET_COL].max():.6f} m")

    high_count = int((df[TARGET_COL] > HIGH_OUTREACH_THRESHOLD).sum())
    print(
        f"  High-outreach samples:     {high_count} "
        f"({100 * high_count / len(df):.1f}%)"
    )

    if "constraint_violation_abs" in df.columns:
        feasible_abs_count = int((df["constraint_violation_abs"] <= 0.002).sum())
        print(
            f"  Feasible_abs samples:      {feasible_abs_count} "
            f"({100 * feasible_abs_count / len(df):.1f}%)"
        )

    if "feasible_abs" in df.columns:
        feasible_abs_flag_count = int(df["feasible_abs"].sum())
        print(
            f"  Feasible_abs flag samples: {feasible_abs_flag_count} "
            f"({100 * feasible_abs_flag_count / len(df):.1f}%)"
        )


# ============================================================================
# KERNELS
# ============================================================================

def create_kernel(kernel_name: str, n_dims: int):
    """
    Create a GP kernel by name.
    """
    length_scale_init = np.ones(n_dims)
    length_scale_bounds = (1e-2, 1e3)

    if kernel_name == "RBF":
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(
                length_scale=length_scale_init,
                length_scale_bounds=length_scale_bounds,
            )
            + WhiteKernel(
                noise_level=1e-5,
                noise_level_bounds=(1e-10, 1e-1),
            )
        )

    elif kernel_name == "Matern_3/2":
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=length_scale_init,
                length_scale_bounds=length_scale_bounds,
                nu=1.5,
            )
            + WhiteKernel(
                noise_level=1e-5,
                noise_level_bounds=(1e-10, 1e-1),
            )
        )

    elif kernel_name == "Matern_5/2":
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=length_scale_init,
                length_scale_bounds=length_scale_bounds,
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=1e-5,
                noise_level_bounds=(1e-10, 1e-1),
            )
        )

    elif kernel_name == "RationalQuadratic":
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=(1e-2, 1e3),
                alpha_bounds=(1e-3, 1e3),
            )
            + WhiteKernel(
                noise_level=1e-5,
                noise_level_bounds=(1e-10, 1e-1),
            )
        )

    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}")

    return kernel


# ============================================================================
# METRICS
# ============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
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


def compute_high_outreach_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics only on high-outreach validation samples.
    """
    mask = y_true > HIGH_OUTREACH_THRESHOLD

    if not np.any(mask):
        return {
            "high_n": 0,
            "high_rmse": np.nan,
            "high_mae": np.nan,
            "high_r2": np.nan,
            "high_median_abs_error": np.nan,
            "high_max_abs_error": np.nan,
        }

    y_true_high = y_true[mask]
    y_pred_high = y_pred[mask]

    errors = y_pred_high - y_true_high
    abs_errors = np.abs(errors)

    return {
        "high_n": int(np.sum(mask)),
        "high_rmse": float(np.sqrt(mean_squared_error(y_true_high, y_pred_high))),
        "high_mae": float(mean_absolute_error(y_true_high, y_pred_high)),
        "high_r2": float(r2_score(y_true_high, y_pred_high)),
        "high_median_abs_error": float(np.median(abs_errors)),
        "high_max_abs_error": float(np.max(abs_errors)),
    }


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """
    Run the full cross-validation study.

    Scaling is fitted only on the training fold to avoid data leakage.
    """
    n_samples, n_dims = X.shape

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    total_fits = len(KERNEL_NAMES) * len(ALPHAS) * N_SPLITS
    fit_counter = 0
    global_start_time = time.time()

    all_rows = []

    print("\n" + "=" * 90)
    print("GAUSSIAN PROCESS CROSS-VALIDATION")
    print("=" * 90)
    print(f"Samples:                   {n_samples}")
    print(f"Input dimensions:          {n_dims}")
    print(f"CV folds:                  {N_SPLITS}")
    print(f"Optimizer restarts:        {N_RESTARTS}")
    print(f"Alpha values:              {ALPHAS}")
    print(f"Kernels:                   {KERNEL_NAMES}")
    print(f"Total GP fits:             {total_fits}")
    print(f"High-outreach threshold:   {HIGH_OUTREACH_THRESHOLD:.3f} m")
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

                y_train_scaled = scaler_y.fit_transform(
                    y_train.reshape(-1, 1)
                ).ravel()

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
                    f"{kernel_name} | alpha={alpha:.0e} | "
                    f"fold {fold_idx}/{N_SPLITS}"
                )

                fit_start_time = time.time()

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(X_train_scaled, y_train_scaled)

                train_time_s = time.time() - fit_start_time

                n_convergence_warnings = sum(
                    issubclass(w.category, ConvergenceWarning)
                    for w in caught_warnings
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

                metrics = compute_regression_metrics(y_val, y_pred)
                high_metrics = compute_high_outreach_metrics(y_val, y_pred)

                row = {
                    "kernel": kernel_name,
                    "alpha": alpha,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "train_time_s": float(train_time_s),
                    "prediction_time_s": float(prediction_time_s),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "mean_error": metrics["mean_error"],
                    "std_error": metrics["std_error"],
                    "median_abs_error": metrics["median_abs_error"],
                    "max_abs_error": metrics["max_abs_error"],
                    "high_n": high_metrics["high_n"],
                    "high_rmse": high_metrics["high_rmse"],
                    "high_mae": high_metrics["high_mae"],
                    "high_r2": high_metrics["high_r2"],
                    "high_median_abs_error": high_metrics["high_median_abs_error"],
                    "high_max_abs_error": high_metrics["high_max_abs_error"],
                    "mean_pred_std": float(np.mean(y_std)),
                    "max_pred_std": float(np.max(y_std)),
                    "log_marginal_likelihood": float(
                        gp.log_marginal_likelihood_value_
                    ),
                    "n_convergence_warnings": int(n_convergence_warnings),
                    "optimized_kernel": str(gp.kernel_),
                }

                all_rows.append(row)

                elapsed_total_s = time.time() - global_start_time
                avg_fit_time_s = elapsed_total_s / fit_counter
                remaining_s = avg_fit_time_s * (total_fits - fit_counter)

                print(
                    f"  RMSE: {metrics['rmse'] * 1000:.3f} mm | "
                    f"MAE: {metrics['mae'] * 1000:.3f} mm | "
                    f"R²: {metrics['r2']:.5f} | "
                    f"High RMSE: {high_metrics['high_rmse'] * 1000:.3f} mm | "
                    f"LML: {gp.log_marginal_likelihood_value_:.2f} | "
                    f"Warnings: {n_convergence_warnings} | "
                    f"Time: {train_time_s:.1f} s"
                )

                print(
                    f"  Elapsed: {elapsed_total_s / 60:.1f} min | "
                    f"Estimated remaining: {remaining_s / 60:.1f} min"
                )

                # Save checkpoint after every fit.
                checkpoint_df = pd.DataFrame(all_rows)
                checkpoint_path = RESULTS_DIR / "cross_validation_raw_results_checkpoint.csv"
                checkpoint_df.to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_rows)


# ============================================================================
# SUMMARY
# ============================================================================

def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validation results by kernel and alpha.
    """
    summary_df = (
        raw_df
        .groupby(["kernel", "alpha"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mean_error_mean=("mean_error", "mean"),
            std_error_mean=("std_error", "mean"),
            median_abs_error_mean=("median_abs_error", "mean"),
            max_abs_error_mean=("max_abs_error", "mean"),
            high_n_mean=("high_n", "mean"),
            high_rmse_mean=("high_rmse", "mean"),
            high_rmse_std=("high_rmse", "std"),
            high_mae_mean=("high_mae", "mean"),
            high_mae_std=("high_mae", "std"),
            high_r2_mean=("high_r2", "mean"),
            high_r2_std=("high_r2", "std"),
            high_max_abs_error_mean=("high_max_abs_error", "mean"),
            mean_pred_std_mean=("mean_pred_std", "mean"),
            mean_pred_std_std=("mean_pred_std", "std"),
            max_pred_std_mean=("max_pred_std", "mean"),
            lml_mean=("log_marginal_likelihood", "mean"),
            lml_std=("log_marginal_likelihood", "std"),
            train_time_s_mean=("train_time_s", "mean"),
            train_time_s_std=("train_time_s", "std"),
            train_time_s_total=("train_time_s", "sum"),
            prediction_time_s_mean=("prediction_time_s", "mean"),
            convergence_warnings_total=("n_convergence_warnings", "sum"),
        )
        .reset_index()
    )

    summary_df = summary_df.sort_values(
        by=[
            "rmse_mean",
            "high_rmse_mean",
            "convergence_warnings_total",
            "lml_mean",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    metric_cols_m = [
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "mean_error_mean",
        "std_error_mean",
        "median_abs_error_mean",
        "max_abs_error_mean",
        "high_rmse_mean",
        "high_rmse_std",
        "high_mae_mean",
        "high_mae_std",
        "high_max_abs_error_mean",
        "mean_pred_std_mean",
        "mean_pred_std_std",
        "max_pred_std_mean",
    ]

    for col in metric_cols_m:
        summary_df[f"{col}_mm"] = summary_df[col] * 1000.0

    return summary_df


def make_report_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create compact report-friendly table.
    """
    report_df = pd.DataFrame({
        "Kernel": summary_df["kernel"],
        "Alpha": summary_df["alpha"].map(lambda x: f"{x:.0e}"),
        "RMSE [mm]": summary_df["rmse_mean_mm"].round(3),
        "RMSE std [mm]": summary_df["rmse_std_mm"].round(3),
        "MAE [mm]": summary_df["mae_mean_mm"].round(3),
        "R2": summary_df["r2_mean"].round(5),
        "High RMSE [mm]": summary_df["high_rmse_mean_mm"].round(3),
        "High RMSE std [mm]": summary_df["high_rmse_std_mm"].round(3),
        "Mean pred std [mm]": summary_df["mean_pred_std_mean_mm"].round(3),
        "LML": summary_df["lml_mean"].round(3),
        "Warnings": summary_df["convergence_warnings_total"].astype(int),
        "Total train time [min]": (summary_df["train_time_s_total"] / 60.0).round(2),
    })

    return report_df


def select_best_configuration(summary_df: pd.DataFrame) -> pd.Series:
    """
    Select the best configuration.

    Selection criterion:
    1. lowest global RMSE;
    2. lowest high-outreach RMSE;
    3. lowest number of convergence warnings;
    4. highest log-marginal likelihood.
    """
    ranked = summary_df.sort_values(
        by=[
            "rmse_mean",
            "high_rmse_mean",
            "convergence_warnings_total",
            "lml_mean",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    return ranked.iloc[0]


def save_best_kernel_summary(
    summary_df: pd.DataFrame,
    report_df: pd.DataFrame,
) -> Path:
    """
    Save text summary with selected kernel.
    """
    best = select_best_configuration(summary_df)

    best_high = summary_df.sort_values(
        by=[
            "high_rmse_mean",
            "rmse_mean",
            "convergence_warnings_total",
        ],
        ascending=[True, True, True],
    ).iloc[0]

    best_lml = summary_df.sort_values(
        by=["lml_mean"],
        ascending=False,
    ).iloc[0]

    lines = []
    lines.append("CROSS-VALIDATION SUMMARY")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Setup:")
    lines.append(f"  Dataset: {DATA_PATH}")
    lines.append(f"  Folds: {N_SPLITS}")
    lines.append(f"  Optimizer restarts: {N_RESTARTS}")
    lines.append(f"  Kernels: {', '.join(KERNEL_NAMES)}")
    lines.append(f"  Alpha values: {ALPHAS}")
    lines.append(f"  High-outreach threshold: {HIGH_OUTREACH_THRESHOLD:.3f} m")
    lines.append("")
    lines.append("Best configuration by global RMSE:")
    lines.append(f"  Kernel: {best['kernel']}")
    lines.append(f"  Alpha: {best['alpha']:.0e}")
    lines.append(f"  RMSE: {best['rmse_mean_mm']:.3f} ± {best['rmse_std_mm']:.3f} mm")
    lines.append(f"  MAE: {best['mae_mean_mm']:.3f} ± {best['mae_std_mm']:.3f} mm")
    lines.append(f"  R2: {best['r2_mean']:.5f} ± {best['r2_std']:.5f}")
    lines.append(
        f"  High-outreach RMSE: "
        f"{best['high_rmse_mean_mm']:.3f} ± {best['high_rmse_std_mm']:.3f} mm"
    )
    lines.append(f"  Mean predictive std: {best['mean_pred_std_mean_mm']:.3f} mm")
    lines.append(f"  Mean LML: {best['lml_mean']:.3f}")
    lines.append(f"  Convergence warnings: {int(best['convergence_warnings_total'])}")
    lines.append("")
    lines.append("Best configuration by high-outreach RMSE:")
    lines.append(f"  Kernel: {best_high['kernel']}")
    lines.append(f"  Alpha: {best_high['alpha']:.0e}")
    lines.append(
        f"  High-outreach RMSE: "
        f"{best_high['high_rmse_mean_mm']:.3f} mm"
    )
    lines.append(f"  Global RMSE: {best_high['rmse_mean_mm']:.3f} mm")
    lines.append("")
    lines.append("Best configuration by LML:")
    lines.append(f"  Kernel: {best_lml['kernel']}")
    lines.append(f"  Alpha: {best_lml['alpha']:.0e}")
    lines.append(f"  Mean LML: {best_lml['lml_mean']:.3f}")
    lines.append(f"  Global RMSE: {best_lml['rmse_mean_mm']:.3f} mm")
    lines.append("")
    lines.append("Recommended final configuration:")
    lines.append(f"  Kernel: {best['kernel']}")
    lines.append(f"  Alpha: {best['alpha']:.0e}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  The selected configuration provides the best or most competitive "
        "prediction accuracy while maintaining strong performance in the "
        "high-outreach region. The high-outreach metric is especially important "
        "because inverse optimization operates in the extra-reach regime."
    )
    lines.append(
        "  The alpha comparison also verifies whether the GP performance is "
        "sensitive to numerical regularization. Since a WhiteKernel is included, "
        "large differences between alpha values are not expected."
    )
    lines.append("")
    lines.append("Compact report table:")
    lines.append(report_df.to_string(index=False))

    output_path = RESULTS_DIR / "best_kernel_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return output_path


def print_summary(summary_df: pd.DataFrame, report_df: pd.DataFrame) -> None:
    """
    Print CV summary to terminal.
    """
    print("\n" + "=" * 90)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 90)
    print(report_df.to_string(index=False))

    best = select_best_configuration(summary_df)

    print("\nRecommended configuration:")
    print(f"  Kernel:             {best['kernel']}")
    print(f"  Alpha:              {best['alpha']:.0e}")
    print(f"  RMSE:               {best['rmse_mean_mm']:.3f} mm")
    print(f"  High RMSE:          {best['high_rmse_mean_mm']:.3f} mm")
    print(f"  R²:                 {best['r2_mean']:.5f}")
    print(f"  LML:                {best['lml_mean']:.3f}")
    print(f"  Warnings:           {int(best['convergence_warnings_total'])}")
    print("=" * 90)


# ============================================================================
# PLOTS
# ============================================================================

def make_labels(summary_df: pd.DataFrame) -> List[str]:
    """
    Make compact labels for kernel-alpha configurations.
    """
    return [
        f"{row.kernel}\nα={row.alpha:.0e}"
        for row in summary_df.itertuples()
    ]


def get_bar_colors(summary_df: pd.DataFrame):
    """
    Assign colors by kernel.
    """
    color_map = {
        "RBF": "#4C78A8",
        "Matern_3/2": "#F58518",
        "Matern_5/2": "#54A24B",
        "RationalQuadratic": "#B279A2",
    }

    return [color_map.get(kernel, "gray") for kernel in summary_df["kernel"]]


def plot_bar_metric(
    summary_df: pd.DataFrame,
    value_col: str,
    error_col: str,
    ylabel: str,
    title: str,
    filename: str,
    convert_to_mm: bool = False,
    lower_is_better: bool = True,
) -> None:
    """
    Generic bar plot for CV summary metrics.
    """
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    values = summary_df[value_col].values.copy()
    errors = summary_df[error_col].values.copy() if error_col else None

    if convert_to_mm:
        values = values * 1000.0
        if errors is not None:
            errors = errors * 1000.0

    fig, ax = plt.subplots(figsize=(13, 6))

    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.88,
    )

    best_idx = int(np.argmin(values) if lower_is_better else np.argmax(values))
    bars[best_idx].set_linewidth(2.6)
    bars[best_idx].set_edgecolor("black")

    for idx, (bar, value) in enumerate(zip(bars, values)):
        offset = 0.015 * (np.nanmax(values) - np.nanmin(values) + 1e-9)
        label = f"{value:.2f}" if convert_to_mm else f"{value:.4f}"

        if "LML" in ylabel:
            label = f"{value:.1f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if idx == best_idx else "normal",
            rotation=0,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_warnings(summary_df: pd.DataFrame) -> None:
    """
    Plot total convergence warnings.
    """
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    values = summary_df["convergence_warnings_total"].values

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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Total convergence warnings")
    ax.set_title("Cross-validation Convergence Warnings")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    plt.tight_layout()

    output_path = FIGURES_DIR / "cv_warnings_by_kernel_alpha.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_summary_figure(summary_df: pd.DataFrame) -> None:
    """
    Create one compact summary figure for presentation.
    """
    labels = make_labels(summary_df)
    colors = get_bar_colors(summary_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # RMSE
    ax = axes[0, 0]
    values = summary_df["rmse_mean_mm"].values
    errors = summary_df["rmse_std_mm"].values
    ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="black")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Global prediction error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    # High RMSE
    ax = axes[0, 1]
    values = summary_df["high_rmse_mean_mm"].values
    errors = summary_df["high_rmse_std_mm"].values
    ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="black")
    ax.set_ylabel("High-outreach RMSE [mm]")
    ax.set_title("High-outreach prediction error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    # R2
    ax = axes[1, 0]
    values = summary_df["r2_mean"].values
    errors = summary_df["r2_std"].values
    ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="black")
    ax.set_ylabel("R²")
    ax.set_title("Explained variance")
    ax.set_ylim(0.90, 1.01)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    # LML
    ax = axes[1, 1]
    values = summary_df["lml_mean"].values
    errors = summary_df["lml_std"].values
    ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="black")
    ax.set_ylabel("Mean LML")
    ax.set_title("Log-marginal likelihood")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=35)

    plt.suptitle(
        "Gaussian Process Cross-validation Summary",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = FIGURES_DIR / "cv_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def generate_plots(summary_df: pd.DataFrame) -> None:
    """
    Generate all CV plots.
    """
    print("\n" + "=" * 90)
    print("GENERATING CROSS-VALIDATION FIGURES")
    print("=" * 90)

    plot_bar_metric(
        summary_df=summary_df,
        value_col="rmse_mean",
        error_col="rmse_std",
        ylabel="RMSE [mm]",
        title="Cross-validation RMSE by Kernel and Alpha",
        filename="cv_rmse_by_kernel_alpha.png",
        convert_to_mm=True,
        lower_is_better=True,
    )

    plot_bar_metric(
        summary_df=summary_df,
        value_col="high_rmse_mean",
        error_col="high_rmse_std",
        ylabel="High-outreach RMSE [mm]",
        title="Cross-validation High-outreach RMSE by Kernel and Alpha",
        filename="cv_high_rmse_by_kernel_alpha.png",
        convert_to_mm=True,
        lower_is_better=True,
    )

    plot_bar_metric(
        summary_df=summary_df,
        value_col="r2_mean",
        error_col="r2_std",
        ylabel="R²",
        title="Cross-validation R² by Kernel and Alpha",
        filename="cv_r2_by_kernel_alpha.png",
        convert_to_mm=False,
        lower_is_better=False,
    )

    plot_bar_metric(
        summary_df=summary_df,
        value_col="lml_mean",
        error_col="lml_std",
        ylabel="Mean LML",
        title="Cross-validation Log-marginal Likelihood by Kernel and Alpha",
        filename="cv_lml_by_kernel_alpha.png",
        convert_to_mm=False,
        lower_is_better=False,
    )

    plot_bar_metric(
        summary_df=summary_df,
        value_col="train_time_s_total",
        error_col=None,
        ylabel="Total training time [s]",
        title="Cross-validation Training Time by Kernel and Alpha",
        filename="cv_train_time_by_kernel_alpha.png",
        convert_to_mm=False,
        lower_is_better=True,
    )

    plot_warnings(summary_df)

    plot_summary_figure(summary_df)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run full cross-validation pipeline.
    """
    print("\nLoading dataset...")
    X, y, df = load_dataset()

    print_dataset_summary(df)

    raw_df = run_cross_validation(X, y)

    raw_path = RESULTS_DIR / "cross_validation_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_results(raw_df)

    summary_path = RESULTS_DIR / "cross_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report_df = make_report_table(summary_df)

    report_path = RESULTS_DIR / "cross_validation_report_table.csv"
    report_df.to_csv(report_path, index=False)

    best_summary_path = save_best_kernel_summary(
        summary_df=summary_df,
        report_df=report_df,
    )

    print_summary(summary_df, report_df)

    generate_plots(summary_df)

    print("\n" + "=" * 90)
    print("CROSS-VALIDATION COMPLETED")
    print("=" * 90)
    print("Generated files:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")
    print(f"  {best_summary_path}")
    print(f"  {FIGURES_DIR / 'cv_rmse_by_kernel_alpha.png'}")
    print(f"  {FIGURES_DIR / 'cv_high_rmse_by_kernel_alpha.png'}")
    print(f"  {FIGURES_DIR / 'cv_r2_by_kernel_alpha.png'}")
    print(f"  {FIGURES_DIR / 'cv_lml_by_kernel_alpha.png'}")
    print(f"  {FIGURES_DIR / 'cv_train_time_by_kernel_alpha.png'}")
    print(f"  {FIGURES_DIR / 'cv_warnings_by_kernel_alpha.png'}")
    print(f"  {FIGURES_DIR / 'cv_summary.png'}")
    print("=" * 90)


if __name__ == "__main__":
    main()