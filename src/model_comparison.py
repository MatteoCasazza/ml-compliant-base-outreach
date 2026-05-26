"""
model_comparison.py
===================

Model comparison for the main GP_peak_y surrogate.

Goal
----
Compare the final Gaussian Process surrogate against a Random Forest baseline
on the main prediction task:

    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> peak_y

The comparison is used to justify the final choice of the Gaussian Process
surrogate for inverse optimization.

Why compare only peak_y?
------------------------
peak_y is the main performance output optimized in the inverse design problem.
The auxiliary max_xr surrogate is used only for constraint guidance and is
evaluated separately using constraint-specific metrics.

Generated outputs
-----------------
results/model_comparison/
    gp_vs_rf_metrics.csv
    gp_vs_rf_predictions.csv
    rf_feature_importance.csv
    model_comparison_summary.txt

figures/model_comparison/
    gp_vs_rf_metrics.png
    gp_vs_rf_parity.png
    gp_vs_rf_residuals.png
    rf_feature_importance.png
    gp_vs_rf_error_distribution.png

Author: Matteo Casazza
Date: 2026
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"

GP_DIR = PROJECT_ROOT / "results" / "gp"
GP_MODEL_PATH = GP_DIR / "gp_model.pkl"
GP_SCALER_X_PATH = GP_DIR / "scaler_X.pkl"
GP_SCALER_Y_PATH = GP_DIR / "scaler_y.pkl"

RESULTS_DIR = PROJECT_ROOT / "results" / "model_comparison"
FIGURES_DIR = PROJECT_ROOT / "figures" / "model_comparison"

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

HIGH_OUTREACH_THRESHOLD = 0.60

RF_CONFIG = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


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
    Print dataset statistics.
    """
    print("\nDataset summary:")
    print(f"  Samples:                  {len(df)}")
    print(f"  Target:                   {TARGET_COL}")
    print(f"  Mean peak_y:              {df[TARGET_COL].mean():.6f} m")
    print(f"  Std peak_y:               {df[TARGET_COL].std():.6f} m")
    print(f"  Min peak_y:               {df[TARGET_COL].min():.6f} m")
    print(f"  Max peak_y:               {df[TARGET_COL].max():.6f} m")

    high_count = int((df[TARGET_COL] > HIGH_OUTREACH_THRESHOLD).sum())
    print(
        f"  High-outreach samples:    {high_count} "
        f"({100 * high_count / len(df):.1f}%)"
    )

    if "constraint_violation" in df.columns:
        feasible_count = int((df["constraint_violation"] <= 0.0).sum())
        print(
            f"  Feasible samples:         {feasible_count} "
            f"({100 * feasible_count / len(df):.1f}%)"
        )


# ============================================================================
# GP LOADING AND PREDICTION
# ============================================================================

def load_final_gp():
    """
    Load the final trained GP_peak_y model and scalers.
    """
    missing_files = [
        path for path in [GP_MODEL_PATH, GP_SCALER_X_PATH, GP_SCALER_Y_PATH]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing GP files:\n" + "\n".join(str(p) for p in missing_files)
        )

    gp_model = joblib.load(GP_MODEL_PATH)
    scaler_X = joblib.load(GP_SCALER_X_PATH)
    scaler_y = joblib.load(GP_SCALER_Y_PATH)

    return gp_model, scaler_X, scaler_y


def predict_gp(
    gp_model,
    scaler_X,
    scaler_y,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Predict peak_y using the final GP surrogate.

    Returns
    -------
    y_pred : ndarray
        Mean prediction in physical units.
    y_std : ndarray
        Predictive standard deviation in physical units.
    prediction_time_s : float
        Prediction time.
    """
    X_scaled = scaler_X.transform(X)

    start_time = time.time()
    y_pred_scaled, y_std_scaled = gp_model.predict(
        X_scaled,
        return_std=True,
    )
    prediction_time_s = time.time() - start_time

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_std = y_std_scaled * scaler_y.scale_[0]

    return y_pred, y_std, prediction_time_s


# ============================================================================
# RANDOM FOREST
# ============================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[RandomForestRegressor, float]:
    """
    Train Random Forest baseline.

    Random Forest does not require feature scaling.
    """
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST BASELINE")
    print("=" * 80)
    print(f"n_estimators:       {RF_CONFIG['n_estimators']}")
    print(f"max_depth:          {RF_CONFIG['max_depth']}")
    print(f"min_samples_split:  {RF_CONFIG['min_samples_split']}")
    print(f"min_samples_leaf:   {RF_CONFIG['min_samples_leaf']}")
    print(f"max_features:       {RF_CONFIG['max_features']}")

    rf = RandomForestRegressor(**RF_CONFIG)

    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time_s = time.time() - start_time

    print(f"Training completed in {train_time_s:.2f} s")

    return rf, train_time_s


def predict_rf(
    rf_model: RandomForestRegressor,
    X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Predict with Random Forest and return prediction time.
    """
    start_time = time.time()
    y_pred = rf_model.predict(X)
    prediction_time_s = time.time() - start_time

    return y_pred, prediction_time_s


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    model_name: str,
    train_time_s: float,
    prediction_time_s: float,
    y_std_test: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute train/test metrics.
    """
    train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    train_mae = mean_absolute_error(y_true_train, y_pred_train)
    train_r2 = r2_score(y_true_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    test_mae = mean_absolute_error(y_true_test, y_pred_test)
    test_r2 = r2_score(y_true_test, y_pred_test)

    high_mask = y_true_test > HIGH_OUTREACH_THRESHOLD

    if np.any(high_mask):
        high_rmse = np.sqrt(
            mean_squared_error(y_true_test[high_mask], y_pred_test[high_mask])
        )
        high_mae = mean_absolute_error(
            y_true_test[high_mask],
            y_pred_test[high_mask],
        )
        high_r2 = r2_score(
            y_true_test[high_mask],
            y_pred_test[high_mask],
        )
        high_n = int(np.sum(high_mask))
    else:
        high_rmse = np.nan
        high_mae = np.nan
        high_r2 = np.nan
        high_n = 0

    errors_test = y_pred_test - y_true_test

    if y_std_test is not None:
        mean_pred_std = float(np.mean(y_std_test))
    else:
        mean_pred_std = np.nan

    return {
        "model": model_name,
        "train_rmse_m": float(train_rmse),
        "train_mae_m": float(train_mae),
        "train_r2": float(train_r2),
        "test_rmse_m": float(test_rmse),
        "test_mae_m": float(test_mae),
        "test_r2": float(test_r2),
        "high_rmse_m": float(high_rmse),
        "high_mae_m": float(high_mae),
        "high_r2": float(high_r2),
        "high_n": int(high_n),
        "mean_error_m": float(np.mean(errors_test)),
        "std_error_m": float(np.std(errors_test)),
        "max_abs_error_m": float(np.max(np.abs(errors_test))),
        "mean_pred_std_m": float(mean_pred_std),
        "train_time_s": float(train_time_s),
        "prediction_time_s": float(prediction_time_s),
        "has_uncertainty": bool(y_std_test is not None),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in readable form.
    """
    print("\n" + "-" * 80)
    print(metrics["model"])
    print("-" * 80)
    print("Training set:")
    print(f"  RMSE:              {metrics['train_rmse_m'] * 1000:.3f} mm")
    print(f"  MAE:               {metrics['train_mae_m'] * 1000:.3f} mm")
    print(f"  R²:                {metrics['train_r2']:.6f}")

    print("Test set:")
    print(f"  RMSE:              {metrics['test_rmse_m'] * 1000:.3f} mm")
    print(f"  MAE:               {metrics['test_mae_m'] * 1000:.3f} mm")
    print(f"  R²:                {metrics['test_r2']:.6f}")

    print(f"High-outreach test cases: {metrics['high_n']}")
    print(f"  High RMSE:         {metrics['high_rmse_m'] * 1000:.3f} mm")
    print(f"  High MAE:          {metrics['high_mae_m'] * 1000:.3f} mm")
    print(f"  High R²:           {metrics['high_r2']:.6f}")

    print("Error statistics:")
    print(f"  Mean error:        {metrics['mean_error_m'] * 1000:.3f} mm")
    print(f"  Error std:         {metrics['std_error_m'] * 1000:.3f} mm")
    print(f"  Max abs error:     {metrics['max_abs_error_m'] * 1000:.3f} mm")

    if metrics["has_uncertainty"]:
        print(f"  Mean pred std:     {metrics['mean_pred_std_m'] * 1000:.3f} mm")
    else:
        print("  Mean pred std:     not available")

    print(f"Training time:       {metrics['train_time_s']:.3f} s")
    print(f"Prediction time:     {metrics['prediction_time_s']:.6f} s")


# ============================================================================
# SAVING
# ============================================================================

def save_metrics(metrics_df: pd.DataFrame) -> Path:
    """
    Save model comparison metrics.
    """
    df = metrics_df.copy()

    metric_cols_m = [
        "train_rmse_m",
        "train_mae_m",
        "test_rmse_m",
        "test_mae_m",
        "high_rmse_m",
        "high_mae_m",
        "mean_error_m",
        "std_error_m",
        "max_abs_error_m",
        "mean_pred_std_m",
    ]

    for col in metric_cols_m:
        df[col.replace("_m", "_mm")] = df[col] * 1000.0

    path = RESULTS_DIR / "gp_vs_rf_metrics.csv"
    df.to_csv(path, index=False)

    print(f"Saved: {path}")
    return path


def save_predictions(
    y_train: np.ndarray,
    y_test: np.ndarray,
    gp_train_pred: np.ndarray,
    gp_test_pred: np.ndarray,
    rf_train_pred: np.ndarray,
    rf_test_pred: np.ndarray,
    gp_test_std: np.ndarray,
) -> Path:
    """
    Save train and test predictions.
    """
    train_df = pd.DataFrame({
        "split": "train",
        "y_true": y_train,
        "gp_pred": gp_train_pred,
        "rf_pred": rf_train_pred,
        "gp_std": np.nan,
        "gp_error": gp_train_pred - y_train,
        "rf_error": rf_train_pred - y_train,
    })

    test_df = pd.DataFrame({
        "split": "test",
        "y_true": y_test,
        "gp_pred": gp_test_pred,
        "rf_pred": rf_test_pred,
        "gp_std": gp_test_std,
        "gp_error": gp_test_pred - y_test,
        "rf_error": rf_test_pred - y_test,
    })

    pred_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    path = RESULTS_DIR / "gp_vs_rf_predictions.csv"
    pred_df.to_csv(path, index=False)

    print(f"Saved: {path}")
    return path


def save_rf_feature_importance(
    rf_model: RandomForestRegressor,
) -> pd.DataFrame:
    """
    Save Random Forest feature importances.
    """
    importance_df = pd.DataFrame({
        "feature": PARAM_COLS,
        "importance": rf_model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    path = RESULTS_DIR / "rf_feature_importance.csv"
    importance_df.to_csv(path, index=False)

    print(f"Saved: {path}")
    return importance_df


def save_text_summary(metrics_df: pd.DataFrame) -> Path:
    """
    Save a short text summary for report writing.
    """
    gp = metrics_df[metrics_df["model"] == "Gaussian Process"].iloc[0]
    rf = metrics_df[metrics_df["model"] == "Random Forest"].iloc[0]

    delta_rmse_mm = (gp["test_rmse_m"] - rf["test_rmse_m"]) * 1000.0
    delta_mae_mm = (gp["test_mae_m"] - rf["test_mae_m"]) * 1000.0
    delta_r2 = gp["test_r2"] - rf["test_r2"]

    lines = []
    lines.append("MODEL COMPARISON SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Task:")
    lines.append("  Main surrogate prediction: input parameters -> peak_y")
    lines.append("")
    lines.append("Test performance:")
    lines.append(
        f"  Gaussian Process: RMSE = {gp['test_rmse_m'] * 1000:.2f} mm, "
        f"MAE = {gp['test_mae_m'] * 1000:.2f} mm, "
        f"R2 = {gp['test_r2']:.4f}"
    )
    lines.append(
        f"  Random Forest:    RMSE = {rf['test_rmse_m'] * 1000:.2f} mm, "
        f"MAE = {rf['test_mae_m'] * 1000:.2f} mm, "
        f"R2 = {rf['test_r2']:.4f}"
    )
    lines.append("")
    lines.append("Difference GP - RF:")
    lines.append(f"  Delta RMSE = {delta_rmse_mm:+.2f} mm")
    lines.append(f"  Delta MAE  = {delta_mae_mm:+.2f} mm")
    lines.append(f"  Delta R2   = {delta_r2:+.4f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  The Random Forest is used as a non-parametric baseline. "
        "Even when its accuracy is competitive, the Gaussian Process is "
        "preferred for the final inverse optimization framework because it "
        "provides predictive uncertainty and ARD-based interpretability."
    )
    lines.append(
        "  The auxiliary max_xr surrogate is not included in this comparison, "
        "because it is a constraint-guidance model and is evaluated separately "
        "using constraint-specific metrics."
    )

    path = RESULTS_DIR / "model_comparison_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {path}")
    return path


# ============================================================================
# PLOTS
# ============================================================================

def plot_metrics(metrics_df: pd.DataFrame) -> None:
    """
    Plot model comparison metrics.
    """
    plot_df = metrics_df.copy()
    models = plot_df["model"].values

    colors = {
        "Gaussian Process": "#1f77b4",
        "Random Forest": "#ff7f0e",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    metric_specs = [
        ("test_rmse_m", "Test RMSE [mm]", "Lower is better"),
        ("test_mae_m", "Test MAE [mm]", "Lower is better"),
        ("test_r2", "Test R²", "Higher is better"),
    ]

    for ax, (metric, ylabel, subtitle) in zip(axes, metric_specs):
        if metric.endswith("_m"):
            values = plot_df[metric].values * 1000.0
        else:
            values = plot_df[metric].values

        bar_colors = [colors[m] for m in models]

        bars = ax.bar(
            models,
            values,
            color=bar_colors,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
        )

        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.grid(True, axis="y", alpha=0.3)

        if metric == "test_r2":
            ax.set_ylim(0.90, 1.00)

        for bar, value in zip(bars, values):
            if metric == "test_r2":
                label = f"{value:.4f}"
                offset = 0.002
            else:
                label = f"{value:.2f}"
                offset = max(values) * 0.02

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.suptitle(
        "Model Comparison: Gaussian Process vs Random Forest",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIGURES_DIR / "gp_vs_rf_metrics.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_parity(
    y_test: np.ndarray,
    gp_test_pred: np.ndarray,
    rf_test_pred: np.ndarray,
    gp_metrics: Dict[str, float],
    rf_metrics: Dict[str, float],
) -> None:
    """
    Plot GP and RF parity plots on the fixed test set.
    """
    y_min = min(y_test.min(), gp_test_pred.min(), rf_test_pred.min())
    y_max = max(y_test.max(), gp_test_pred.max(), rf_test_pred.max())

    margin = 0.03
    lims = [y_min - margin, y_max + margin]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = [
        (
            axes[0],
            gp_test_pred,
            "Gaussian Process",
            "#1f77b4",
            gp_metrics,
        ),
        (
            axes[1],
            rf_test_pred,
            "Random Forest",
            "#ff7f0e",
            rf_metrics,
        ),
    ]

    for ax, y_pred, title, color, metrics in configs:
        ax.scatter(
            y_test,
            y_pred,
            s=38,
            alpha=0.72,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.plot(
            lims,
            lims,
            linestyle="--",
            color="black",
            linewidth=1.8,
            label="Ideal prediction",
        )

        ax.axvline(
            HIGH_OUTREACH_THRESHOLD,
            linestyle=":",
            color="gray",
            linewidth=1.5,
            label="High-outreach threshold",
        )

        ax.axhline(
            HIGH_OUTREACH_THRESHOLD,
            linestyle=":",
            color="gray",
            linewidth=1.5,
        )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True peak_y [m]")
        ax.set_ylabel("Predicted peak_y [m]")
        ax.set_title(
            f"{title}\n"
            f"RMSE = {metrics['test_rmse_m'] * 1000:.2f} mm, "
            f"R² = {metrics['test_r2']:.4f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(
        "Parity Plot on Fixed Test Set",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = FIGURES_DIR / "gp_vs_rf_parity.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_residuals(
    y_test: np.ndarray,
    gp_test_pred: np.ndarray,
    rf_test_pred: np.ndarray,
) -> None:
    """
    Plot residuals against true peak_y.
    """
    gp_error_mm = (gp_test_pred - y_test) * 1000.0
    rf_error_mm = (rf_test_pred - y_test) * 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_test,
        gp_error_mm,
        s=35,
        alpha=0.70,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.3,
        label="Gaussian Process",
    )

    ax.scatter(
        y_test,
        rf_error_mm,
        s=35,
        alpha=0.70,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.3,
        label="Random Forest",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        color="black",
        linewidth=1.5,
    )

    ax.axvline(
        HIGH_OUTREACH_THRESHOLD,
        linestyle=":",
        color="gray",
        linewidth=1.5,
        label="High-outreach threshold",
    )

    ax.set_xlabel("True peak_y [m]")
    ax.set_ylabel("Prediction error [mm]")
    ax.set_title("Residuals on Fixed Test Set")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    path = FIGURES_DIR / "gp_vs_rf_residuals.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_error_distribution(
    y_test: np.ndarray,
    gp_test_pred: np.ndarray,
    rf_test_pred: np.ndarray,
) -> None:
    """
    Plot error distribution for both models.
    """
    gp_error_mm = (gp_test_pred - y_test) * 1000.0
    rf_error_mm = (rf_test_pred - y_test) * 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(gp_error_mm.min(), rf_error_mm.min()),
        max(gp_error_mm.max(), rf_error_mm.max()),
        30,
    )

    ax.hist(
        gp_error_mm,
        bins=bins,
        alpha=0.65,
        color="#1f77b4",
        edgecolor="black",
        label="Gaussian Process",
    )

    ax.hist(
        rf_error_mm,
        bins=bins,
        alpha=0.65,
        color="#ff7f0e",
        edgecolor="black",
        label="Random Forest",
    )

    ax.axvline(
        0.0,
        linestyle="--",
        color="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Prediction error [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    path = FIGURES_DIR / "gp_vs_rf_error_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_rf_feature_importance(rf_importance_df, save_path):
    """
    Plot Random Forest feature importance with a warm color gradient.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort from least important to most important so the biggest bar is on top after barh
    df_plot = rf_importance_df.sort_values("importance", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Warm gradient: light orange -> deep orange/red
    colors = plt.cm.YlOrRd(np.linspace(0.35, 0.90, len(df_plot)))

    bars = ax.barh(
        df_plot["feature"],
        df_plot["importance"],
        color=colors,
        edgecolor="black",
        linewidth=1.2
    )

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title(
        "Random Forest: Feature Importance Ranking",
        fontsize=16,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_xlim(0, max(df_plot["importance"]) * 1.12)

    # Make style cleaner
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run complete GP vs Random Forest comparison.
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: GAUSSIAN PROCESS vs RANDOM FOREST")
    print("=" * 80)
    print("Task: input parameters -> peak_y")
    print("=" * 80)

    X, y, df = load_dataset()
    print_dataset_summary(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\nTrain/test split:")
    print(f"  Train samples:            {len(y_train)}")
    print(f"  Test samples:             {len(y_test)}")
    print(f"  Random state:             {RANDOM_STATE}")

    # ----------------------------------------------------------------------
    # Gaussian Process prediction
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING FINAL GAUSSIAN PROCESS")
    print("=" * 80)

    gp_model, scaler_X, scaler_y = load_final_gp()

    print(f"Loaded GP model:     {GP_MODEL_PATH}")
    print(f"Loaded scaler_X:     {GP_SCALER_X_PATH}")
    print(f"Loaded scaler_y:     {GP_SCALER_Y_PATH}")

    gp_train_pred, _, gp_train_pred_time = predict_gp(
        gp_model,
        scaler_X,
        scaler_y,
        X_train,
    )

    gp_test_pred, gp_test_std, gp_test_pred_time = predict_gp(
        gp_model,
        scaler_X,
        scaler_y,
        X_test,
    )

    gp_prediction_time_s = gp_train_pred_time + gp_test_pred_time

    # The final GP was already trained in model_peak_y.py.
    # Here we only load it, so train_time_s is set to NaN.
    gp_metrics = compute_metrics(
        y_true_train=y_train,
        y_pred_train=gp_train_pred,
        y_true_test=y_test,
        y_pred_test=gp_test_pred,
        model_name="Gaussian Process",
        train_time_s=np.nan,
        prediction_time_s=gp_prediction_time_s,
        y_std_test=gp_test_std,
    )

    print_metrics(gp_metrics)

    # ----------------------------------------------------------------------
    # Random Forest training and prediction
    # ----------------------------------------------------------------------
    rf_model, rf_train_time_s = train_random_forest(
        X_train=X_train,
        y_train=y_train,
    )

    rf_train_pred, rf_train_pred_time = predict_rf(
        rf_model,
        X_train,
    )

    rf_test_pred, rf_test_pred_time = predict_rf(
        rf_model,
        X_test,
    )

    rf_prediction_time_s = rf_train_pred_time + rf_test_pred_time

    rf_metrics = compute_metrics(
        y_true_train=y_train,
        y_pred_train=rf_train_pred,
        y_true_test=y_test,
        y_pred_test=rf_test_pred,
        model_name="Random Forest",
        train_time_s=rf_train_time_s,
        prediction_time_s=rf_prediction_time_s,
        y_std_test=None,
    )

    print_metrics(rf_metrics)

    # ----------------------------------------------------------------------
    # Save numerical results
    # ----------------------------------------------------------------------
    metrics_df = pd.DataFrame([gp_metrics, rf_metrics])

    save_metrics(metrics_df)

    save_predictions(
        y_train=y_train,
        y_test=y_test,
        gp_train_pred=gp_train_pred,
        gp_test_pred=gp_test_pred,
        rf_train_pred=rf_train_pred,
        rf_test_pred=rf_test_pred,
        gp_test_std=gp_test_std,
    )

    importance_df = save_rf_feature_importance(rf_model)

    save_text_summary(metrics_df)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_metrics(metrics_df)

    plot_parity(
        y_test=y_test,
        gp_test_pred=gp_test_pred,
        rf_test_pred=rf_test_pred,
        gp_metrics=gp_metrics,
        rf_metrics=rf_metrics,
    )

    plot_residuals(
        y_test=y_test,
        gp_test_pred=gp_test_pred,
        rf_test_pred=rf_test_pred,
    )

    plot_error_distribution(
        y_test=y_test,
        gp_test_pred=gp_test_pred,
        rf_test_pred=rf_test_pred,
    )

    plot_rf_feature_importance(
        importance_df,
        save_path=FIGURES_DIR / "rf_feature_importance.png",
    )

    # ----------------------------------------------------------------------
    # Final interpretation
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL INTERPRETATION")
    print("=" * 80)

    delta_rmse_mm = (
        gp_metrics["test_rmse_m"] - rf_metrics["test_rmse_m"]
    ) * 1000.0

    delta_mae_mm = (
        gp_metrics["test_mae_m"] - rf_metrics["test_mae_m"]
    ) * 1000.0

    delta_r2 = gp_metrics["test_r2"] - rf_metrics["test_r2"]

    print("Difference GP - RF:")
    print(f"  Delta RMSE: {delta_rmse_mm:+.3f} mm")
    print(f"  Delta MAE:  {delta_mae_mm:+.3f} mm")
    print(f"  Delta R²:   {delta_r2:+.5f}")

    print("\nWhy the GP is selected:")
    print("  - It provides predictive uncertainty.")
    print("  - It provides ARD-based interpretability.")
    print("  - It is smooth and suitable for inverse optimization.")
    print("  - It is already integrated in the constraint-aware framework.")

    print("\nRole of Random Forest:")
    print("  - Strong non-parametric baseline.")
    print("  - Useful feature importance comparison.")
    print("  - No native predictive uncertainty for optimization.")

    print("\nGenerated files:")
    print(f"  {RESULTS_DIR / 'gp_vs_rf_metrics.csv'}")
    print(f"  {RESULTS_DIR / 'gp_vs_rf_predictions.csv'}")
    print(f"  {RESULTS_DIR / 'rf_feature_importance.csv'}")
    print(f"  {RESULTS_DIR / 'model_comparison_summary.txt'}")
    print(f"  {FIGURES_DIR / 'gp_vs_rf_metrics.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_rf_parity.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_rf_residuals.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_rf_error_distribution.png'}")
    print(f"  {FIGURES_DIR / 'rf_feature_importance.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()