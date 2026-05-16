"""
cross_validation_gp.py
======================
Cross-validation study for Gaussian Process kernel and alpha selection.

This script compares a small set of Gaussian Process kernels using 3-fold
cross-validation on the final augmented dataset.

The goal is to select the most robust kernel/alpha configuration for the final
surrogate model.

Compared kernels:
- RBF
- Matern 3/2
- Matern 5/2

Compared alpha values:
- 1e-10
- 1e-8
- 1e-6

The script also generates report-ready figures and tables:
- CV RMSE by kernel and alpha
- CV high-outreach RMSE by kernel and alpha
- CV log-marginal-likelihood by kernel and alpha
- Peak_y distribution before/after dataset augmentation
- Feasible extra-reach distribution before/after dataset augmentation
- Feasible high-outreach sample count before/after dataset augmentation

Author: MatteoCasazza
Date: 2026
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ConstantKernel as C,
    WhiteKernel,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from dataset import load_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "data/dataset_augmented.csv"
ORIGINAL_DATASET_PATH = "data/dataset_outreach.csv"

N_SPLITS = 3
N_RESTARTS = 3
ALPHAS = [1e-10, 1e-8, 1e-6]

RANDOM_STATE = 42

X_R_MAX = 0.5
CONSTRAINT_TOLERANCE = 0.002
HIGH_OUTREACH_THRESHOLD = 0.60

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================================
# KERNEL DEFINITIONS
# ============================================================================

def create_kernel_configs(n_dims: int) -> dict:
    """
    Create Gaussian Process kernel configurations.

    Parameters
    ----------
    n_dims : int
        Number of input dimensions.

    Returns
    -------
    dict
        Dictionary containing kernel name and kernel object.
    """
    ls_init = np.ones(n_dims)
    ls_bounds = (1e-2, 1e3)

    kernels = {
        "RBF": (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=ls_init, length_scale_bounds=ls_bounds)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),

        "Matern_3/2": (
            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=ls_init,
                length_scale_bounds=ls_bounds,
                nu=1.5,
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),

        "Matern_5/2": (
            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=ls_init,
                length_scale_bounds=ls_bounds,
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),
    }

    return kernels


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics in physical units.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    abs_error = np.abs(y_true - y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "median_abs_error": np.median(abs_error),
        "max_abs_error": np.max(abs_error),
    }


def compute_high_outreach_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> dict:
    """
    Compute metrics only in the high-outreach region.
    """
    mask = y_true > threshold

    if np.sum(mask) == 0:
        return {
            "high_n": 0,
            "high_rmse": np.nan,
            "high_mae": np.nan,
            "high_max_abs_error": np.nan,
        }

    y_true_high = y_true[mask]
    y_pred_high = y_pred[mask]

    abs_error = np.abs(y_true_high - y_pred_high)

    return {
        "high_n": int(np.sum(mask)),
        "high_rmse": np.sqrt(mean_squared_error(y_true_high, y_pred_high)),
        "high_mae": mean_absolute_error(y_true_high, y_pred_high),
        "high_max_abs_error": np.max(abs_error),
    }


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    n_restarts: int = 3,
    alphas: list[float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run K-fold cross-validation for different GP kernels and alpha values.

    Important
    ---------
    Scalers are fitted only on the training fold to avoid data leakage.
    """
    if alphas is None:
        alphas = [1e-10]

    n_samples, n_dims = X.shape
    kernels = create_kernel_configs(n_dims)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_results = []

    print("\n" + "=" * 90)
    print("GAUSSIAN PROCESS CROSS-VALIDATION")
    print("=" * 90)
    print(f"Dataset samples:          {n_samples}")
    print(f"Input dimensions:         {n_dims}")
    print(f"CV folds:                 {n_splits}")
    print(f"Restarts per fit:         {n_restarts}")
    print(f"Alpha values:             {alphas}")
    print(f"High-outreach threshold:  {HIGH_OUTREACH_THRESHOLD} m")
    print(f"Kernels:                  {list(kernels.keys())}")
    print("=" * 90)

    total_fits = len(alphas) * len(kernels) * n_splits
    fit_counter = 0
    global_start = time.time()

    for alpha in alphas:
        print("\n" + "#" * 90)
        print(f"ALPHA: {alpha}")
        print("#" * 90)

        for kernel_name, base_kernel in kernels.items():
            print("\n" + "-" * 90)
            print(f"KERNEL: {kernel_name} | ALPHA: {alpha}")
            print("-" * 90)

            for fold_idx, (train_idx, val_idx) in enumerate(
                kfold.split(X),
                start=1,
            ):
                fit_counter += 1

                print(
                    f"\nFit {fit_counter}/{total_fits} | "
                    f"Kernel: {kernel_name} | "
                    f"Alpha: {alpha} | "
                    f"Fold {fold_idx}/{n_splits}"
                )

                X_train = X[train_idx]
                X_val = X[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]

                # Fit scalers only on the training fold.
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_train_scaled = scaler_X.fit_transform(X_train)
                X_val_scaled = scaler_X.transform(X_val)

                y_train_scaled = scaler_y.fit_transform(
                    y_train.reshape(-1, 1)
                ).ravel()

                gp = GaussianProcessRegressor(
                    kernel=base_kernel,
                    alpha=alpha,
                    normalize_y=False,
                    n_restarts_optimizer=n_restarts,
                    random_state=random_state,
                )

                start_time = time.time()

                # Count convergence warnings without stopping the run.
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(X_train_scaled, y_train_scaled)

                elapsed = time.time() - start_time

                n_warnings = sum(
                    issubclass(w.category, ConvergenceWarning)
                    for w in caught_warnings
                )

                # Predict in scaled output space.
                y_pred_scaled, y_std_scaled = gp.predict(
                    X_val_scaled,
                    return_std=True,
                )

                # Convert predictions back to physical units.
                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).ravel()

                y_std = y_std_scaled * scaler_y.scale_[0]

                metrics = compute_metrics(y_val, y_pred)
                high_metrics = compute_high_outreach_metrics(
                    y_val,
                    y_pred,
                    threshold=HIGH_OUTREACH_THRESHOLD,
                )

                row = {
                    "kernel": kernel_name,
                    "alpha": alpha,
                    "fold": fold_idx,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "train_time_s": elapsed,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "median_abs_error": metrics["median_abs_error"],
                    "max_abs_error": metrics["max_abs_error"],
                    "mean_pred_std": np.mean(y_std),
                    "max_pred_std": np.max(y_std),
                    "high_n": high_metrics["high_n"],
                    "high_rmse": high_metrics["high_rmse"],
                    "high_mae": high_metrics["high_mae"],
                    "high_max_abs_error": high_metrics["high_max_abs_error"],
                    "log_marginal_likelihood": gp.log_marginal_likelihood_value_,
                    "n_convergence_warnings": n_warnings,
                    "optimized_kernel": str(gp.kernel_),
                }

                all_results.append(row)

                high_rmse = high_metrics["high_rmse"]
                high_rmse_str = (
                    f"{high_rmse * 1000:.3f} mm"
                    if not np.isnan(high_rmse)
                    else "nan"
                )

                print(
                    f"  RMSE: {metrics['rmse'] * 1000:.3f} mm | "
                    f"MAE: {metrics['mae'] * 1000:.3f} mm | "
                    f"R²: {metrics['r2']:.5f} | "
                    f"High RMSE: {high_rmse_str} | "
                    f"Warnings: {n_warnings} | "
                    f"Time: {elapsed:.1f} s"
                )

                elapsed_total = time.time() - global_start
                avg_time = elapsed_total / fit_counter
                remaining = avg_time * (total_fits - fit_counter)

                print(
                    f"  Elapsed total: {elapsed_total / 60:.1f} min | "
                    f"Estimated remaining: {remaining / 60:.1f} min"
                )

    results_df = pd.DataFrame(all_results)

    return results_df


# ============================================================================
# CV SUMMARY AND PLOTS
# ============================================================================

def summarize_cv_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validation results by kernel and alpha.
    """
    summary = results_df.groupby(["kernel", "alpha"]).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        median_abs_error_mean=("median_abs_error", "mean"),
        max_abs_error_mean=("max_abs_error", "mean"),
        high_n_mean=("high_n", "mean"),
        high_rmse_mean=("high_rmse", "mean"),
        high_rmse_std=("high_rmse", "std"),
        high_mae_mean=("high_mae", "mean"),
        high_max_abs_error_mean=("high_max_abs_error", "mean"),
        mean_pred_std_mean=("mean_pred_std", "mean"),
        max_pred_std_mean=("max_pred_std", "mean"),
        train_time_s_total=("train_time_s", "sum"),
        lml_mean=("log_marginal_likelihood", "mean"),
        lml_std=("log_marginal_likelihood", "std"),
        convergence_warnings_total=("n_convergence_warnings", "sum"),
    ).reset_index()

    summary = summary.sort_values(
        by=["rmse_mean", "high_rmse_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary


def print_summary(summary_df: pd.DataFrame) -> None:
    """
    Print compact CV summary.
    """
    print("\n" + "=" * 90)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 90)

    for _, row in summary_df.iterrows():
        print(f"\nKernel: {row['kernel']} | Alpha: {row['alpha']:.0e}")
        print(
            f"  RMSE:          {row['rmse_mean'] * 1000:.3f} "
            f"± {row['rmse_std'] * 1000:.3f} mm"
        )
        print(
            f"  MAE:           {row['mae_mean'] * 1000:.3f} "
            f"± {row['mae_std'] * 1000:.3f} mm"
        )
        print(
            f"  R²:            {row['r2_mean']:.5f} "
            f"± {row['r2_std']:.5f}"
        )
        print(
            f"  High RMSE:     {row['high_rmse_mean'] * 1000:.3f} "
            f"± {row['high_rmse_std'] * 1000:.3f} mm"
        )
        print(f"  High MAE:      {row['high_mae_mean'] * 1000:.3f} mm")
        print(f"  Mean pred std: {row['mean_pred_std_mean'] * 1000:.3f} mm")
        print(
            f"  Mean LML:      {row['lml_mean']:.3f} "
            f"± {row['lml_std']:.3f}"
        )
        print(f"  Warnings:      {int(row['convergence_warnings_total'])}")
        print(f"  Total time:    {row['train_time_s_total'] / 60:.2f} min")

    print("\n" + "-" * 90)
    print("RECOMMENDATION")
    print("-" * 90)

    best_rmse = summary_df.iloc[0]

    print(
        f"Best mean RMSE: {best_rmse['kernel']} "
        f"with alpha={best_rmse['alpha']:.0e}"
    )

    high_sorted = summary_df.sort_values(
        by=["high_rmse_mean", "rmse_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_high = high_sorted.iloc[0]

    print(
        f"Best high-outreach RMSE: {best_high['kernel']} "
        f"with alpha={best_high['alpha']:.0e}"
    )

    rmse_diff_mm = abs(best_rmse["rmse_mean"] - best_high["rmse_mean"]) * 1000
    print(f"RMSE difference between these two choices: {rmse_diff_mm:.3f} mm")

    if (
        best_rmse["kernel"] == best_high["kernel"]
        and best_rmse["alpha"] == best_high["alpha"]
    ):
        print(
            "The same configuration is best for both global RMSE and "
            "high-outreach RMSE."
        )
    else:
        print(
            "Global RMSE and high-outreach RMSE select different configurations. "
            "For this project, high-outreach behavior should be considered "
            "important because the goal is extra-reaching beyond nominal reach."
        )

    print("=" * 90)


def plot_cv_results(summary_df: pd.DataFrame) -> None:
    """
    Save bar plots for CV results.
    """
    labels = [
        f"{row.kernel}\nα={row.alpha:.0e}"
        for row in summary_df.itertuples()
    ]

    # RMSE plot
    plt.figure(figsize=(11, 5))
    plt.bar(
        labels,
        summary_df["rmse_mean"] * 1000,
        yerr=summary_df["rmse_std"] * 1000,
        capsize=5,
    )
    plt.ylabel("RMSE [mm]")
    plt.title("Cross-validation RMSE by kernel and alpha")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cv_kernel_alpha_rmse.png", dpi=300)
    plt.close()

    # High-outreach RMSE plot
    plt.figure(figsize=(11, 5))
    plt.bar(
        labels,
        summary_df["high_rmse_mean"] * 1000,
        yerr=summary_df["high_rmse_std"] * 1000,
        capsize=5,
    )
    plt.ylabel("High-outreach RMSE [mm]")
    plt.title("Cross-validation high-outreach RMSE by kernel and alpha")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cv_kernel_alpha_high_outreach_rmse.png", dpi=300)
    plt.close()

    # LML plot
    plt.figure(figsize=(11, 5))
    plt.bar(
        labels,
        summary_df["lml_mean"],
        yerr=summary_df["lml_std"],
        capsize=5,
    )
    plt.ylabel("Mean log-marginal-likelihood")
    plt.title("Cross-validation LML by kernel and alpha")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cv_kernel_alpha_lml.png", dpi=300)
    plt.close()


# ============================================================================
# DATASET AUGMENTATION REPORT FIGURES
# ============================================================================

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a dataset CSV file.

    The comment argument keeps compatibility with files that may contain
    commented metadata lines.
    """
    return pd.read_csv(path, comment="#")


def compute_dataset_summary(
    df: pd.DataFrame,
    dataset_name: str,
    tolerance: float = CONSTRAINT_TOLERANCE,
    high_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> dict:
    """
    Compute compact dataset statistics for reporting.
    """
    feasible = df[df["constraint_violation"] <= tolerance]
    feasible_extra = feasible[feasible["extra_reach"] > 0.0]
    feasible_high = feasible[feasible["peak_y"] > high_threshold]

    return {
        "dataset": dataset_name,
        "samples": len(df),
        "peak_y_mean_m": df["peak_y"].mean(),
        "peak_y_max_m": df["peak_y"].max(),
        "feasible_samples": len(feasible),
        "feasible_extra_reach": len(feasible_extra),
        "feasible_high_outreach": len(feasible_high),
        "max_feasible_peak_y_m": feasible["peak_y"].max(),
        "max_feasible_extra_reach_m": (
            feasible_extra["extra_reach"].max()
            if len(feasible_extra) > 0
            else np.nan
        ),
        "violation_rate_percent": 100.0 * (
            df["constraint_violation"] > tolerance
        ).mean(),
    }


def save_dataset_summary_table(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save dataset summary as CSV.
    """
    summary = pd.DataFrame([
        compute_dataset_summary(original_df, "Original"),
        compute_dataset_summary(augmented_df, "Augmented"),
    ])

    output_path = RESULTS_DIR / "dataset_augmentation_summary.csv"
    summary.to_csv(output_path, index=False)

    print("\nDataset augmentation summary:")
    print(summary.to_string(index=False))
    print(f"Saved: {output_path}")

    return summary


def plot_peak_y_distribution(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
) -> None:
    """
    Plot peak_y distribution before and after augmentation.
    """
    plt.figure(figsize=(8, 5))

    bins = np.linspace(
        min(original_df["peak_y"].min(), augmented_df["peak_y"].min()),
        max(original_df["peak_y"].max(), augmented_df["peak_y"].max()),
        35,
    )

    plt.hist(
        original_df["peak_y"],
        bins=bins,
        alpha=0.55,
        label="Original dataset",
        density=True,
    )
    plt.hist(
        augmented_df["peak_y"],
        bins=bins,
        alpha=0.55,
        label="Augmented dataset",
        density=True,
    )

    plt.axvline(
        X_R_MAX,
        linestyle="--",
        linewidth=2,
        label="Nominal reach",
    )
    plt.axvline(
        HIGH_OUTREACH_THRESHOLD,
        linestyle=":",
        linewidth=2,
        label="High-outreach threshold",
    )

    plt.xlabel("Peak outreach, peak_y [m]")
    plt.ylabel("Density")
    plt.title("Peak outreach distribution before and after augmentation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "augmentation_peak_y_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_extra_reach_distribution(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
) -> None:
    """
    Plot feasible extra-reach distribution before and after augmentation.
    """
    original_feasible = original_df[
        original_df["constraint_violation"] <= CONSTRAINT_TOLERANCE
    ]
    augmented_feasible = augmented_df[
        augmented_df["constraint_violation"] <= CONSTRAINT_TOLERANCE
    ]

    max_extra = max(
        original_feasible["extra_reach"].max(),
        augmented_feasible["extra_reach"].max(),
    )

    plt.figure(figsize=(8, 5))

    bins = np.linspace(0.0, max_extra, 30)

    plt.hist(
        original_feasible["extra_reach"],
        bins=bins,
        alpha=0.55,
        label="Original feasible samples",
        density=True,
    )
    plt.hist(
        augmented_feasible["extra_reach"],
        bins=bins,
        alpha=0.55,
        label="Augmented feasible samples",
        density=True,
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=2,
        label="No extra reach",
    )

    plt.xlabel("Extra reach [m]")
    plt.ylabel("Density")
    plt.title("Feasible extra-reach distribution before and after augmentation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "augmentation_extra_reach_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_feasible_high_outreach_counts(summary_df: pd.DataFrame) -> None:
    """
    Plot count of feasible high-outreach samples before and after augmentation.
    """
    plt.figure(figsize=(7, 5))

    labels = summary_df["dataset"]
    counts = summary_df["feasible_high_outreach"]

    bars = plt.bar(labels, counts)

    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Number of feasible high-outreach samples")
    plt.title("Effect of targeted dataset augmentation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "augmentation_feasible_high_outreach_count.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_cv_report_table(summary_df: pd.DataFrame) -> None:
    """
    Save a compact report-friendly CV table.
    """
    report_table = pd.DataFrame({
        "Kernel": summary_df["kernel"],
        "Alpha": summary_df["alpha"].map(lambda x: f"{x:.0e}"),
        "RMSE [mm]": (summary_df["rmse_mean"] * 1000).round(3),
        "RMSE std [mm]": (summary_df["rmse_std"] * 1000).round(3),
        "MAE [mm]": (summary_df["mae_mean"] * 1000).round(3),
        "R2": summary_df["r2_mean"].round(5),
        "High-outreach RMSE [mm]": (
            summary_df["high_rmse_mean"] * 1000
        ).round(3),
        "LML": summary_df["lml_mean"].round(3),
        "Warnings": summary_df["convergence_warnings_total"].astype(int),
    })

    output_path = RESULTS_DIR / "cv_report_table.csv"
    report_table.to_csv(output_path, index=False)

    print("\nCV report table:")
    print(report_table.to_string(index=False))
    print(f"Saved: {output_path}")


def generate_report_figures_and_tables(summary_df: pd.DataFrame) -> None:
    """
    Generate the most useful report figures and compact tables.
    """
    print("\n" + "=" * 90)
    print("GENERATING REPORT FIGURES AND TABLES")
    print("=" * 90)

    original_path = Path(ORIGINAL_DATASET_PATH)
    augmented_path = Path(DATASET_PATH)

    if not original_path.exists():
        print(f"Original dataset not found: {ORIGINAL_DATASET_PATH}")
        print("Skipping augmentation figures.")
        return

    if not augmented_path.exists():
        print(f"Augmented dataset not found: {DATASET_PATH}")
        print("Skipping augmentation figures.")
        return

    original_df = load_dataframe(ORIGINAL_DATASET_PATH)
    augmented_df = load_dataframe(DATASET_PATH)

    dataset_summary = save_dataset_summary_table(
        original_df,
        augmented_df,
    )

    plot_peak_y_distribution(original_df, augmented_df)
    plot_extra_reach_distribution(original_df, augmented_df)
    plot_feasible_high_outreach_counts(dataset_summary)
    save_cv_report_table(summary_df)

    print("=" * 90)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("\nLoading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print(f"Dataset path: {DATASET_PATH}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Peak_y range: {y.min():.4f} - {y.max():.4f} m")

    results_df = run_cross_validation(
        X=X,
        y=y,
        n_splits=N_SPLITS,
        n_restarts=N_RESTARTS,
        alphas=ALPHAS,
        random_state=RANDOM_STATE,
    )

    results_path = RESULTS_DIR / "cv_kernel_alpha_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = summarize_cv_results(results_df)

    summary_path = RESULTS_DIR / "cv_kernel_alpha_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print_summary(summary_df)
    plot_cv_results(summary_df)
    generate_report_figures_and_tables(summary_df)

    print("\nSaved files:")
    print(f"  - {results_path}")
    print(f"  - {summary_path}")
    print(f"  - {RESULTS_DIR / 'cv_report_table.csv'}")
    print(f"  - {RESULTS_DIR / 'dataset_augmentation_summary.csv'}")
    print(f"  - {FIGURES_DIR / 'cv_kernel_alpha_rmse.png'}")
    print(f"  - {FIGURES_DIR / 'cv_kernel_alpha_high_outreach_rmse.png'}")
    print(f"  - {FIGURES_DIR / 'cv_kernel_alpha_lml.png'}")
    print(f"  - {FIGURES_DIR / 'augmentation_peak_y_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'augmentation_extra_reach_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'augmentation_feasible_high_outreach_count.png'}")