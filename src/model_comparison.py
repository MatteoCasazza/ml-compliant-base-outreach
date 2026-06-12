"""
model_comparison.py
===================

Compare the Gaussian Process surrogate pair against the multi-output Neural
Network surrogate.

Models compared
---------------
1. GP pair
   - GP_peak_y:       input parameters -> peak_y
   - GP_max_abs_xr:   input parameters -> max_abs_xr

2. NN multi-output
   - NN: input parameters -> [peak_y, max_abs_xr]

The comparison is designed for report and presentation use. It collects the
main regression, constraint-classification, near-boundary and inference-time
metrics in one place and generates publication-ready figures.

Expected inputs
---------------
results/gp/metrics.csv
results/gp/test_prediction_errors.csv
results/gp_constraints/metrics_max_abs_xr.csv
results/nn/metrics.csv

Optional inputs for inference timing
------------------------------------
results/gp/gp_model.pkl
results/gp/scaler_X.pkl
results/gp/scaler_y.pkl
results/gp_constraints/gp_max_abs_xr_model.pkl
results/gp_constraints/scaler_X_max_abs_xr.pkl
results/gp_constraints/scaler_y_max_abs_xr.pkl
results/nn/nn_multioutput_model.pt
results/nn/scaler_X.pkl
results/nn/scaler_Y.pkl

data/dataset_augmented.csv

Generated outputs
-----------------
results/model_comparison/model_comparison_summary.csv
results/model_comparison/model_comparison_report_table.csv
results/model_comparison/model_comparison_report_table.md
results/model_comparison/inference_time_summary.csv
results/model_comparison/model_comparison_interpretation.txt

figures/model_comparison/model_comparison_peak_y_parity_gp_vs_nn.png
figures/model_comparison/model_comparison_peak_y_residuals_gp_vs_nn.png
figures/model_comparison/model_comparison_peak_y_error_distribution_gp_vs_nn.png
figures/model_comparison/model_comparison_rmse.png
figures/model_comparison/model_comparison_r2.png
figures/model_comparison/model_comparison_constraint_safety.png
figures/model_comparison/model_comparison_near_boundary.png
figures/model_comparison/model_comparison_inference_time.png
figures/model_comparison/model_comparison_summary.png

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - only used if torch is unavailable
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

GP_RESULTS_DIR = RESULTS_DIR / "gp"
GP_CONSTRAINT_RESULTS_DIR = RESULTS_DIR / "gp_constraints"
NN_RESULTS_DIR = RESULTS_DIR / "nn"

COMPARISON_RESULTS_DIR = RESULTS_DIR / "model_comparison"
COMPARISON_FIGURES_DIR = FIGURES_DIR / "model_comparison"

for directory in [COMPARISON_RESULTS_DIR, COMPARISON_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset_augmented.csv"


# =============================================================================
# SETTINGS
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

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495
HIGH_OUTREACH_THRESHOLD = 0.600
NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520

# Keep this small enough to be practical even on CPU.
INFERENCE_BATCH_SIZES = [1, 1000, 10000]
INFERENCE_REPEATS = 5
INFERENCE_WARMUP = 2


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def read_single_row_csv(path: str | Path, name: str) -> pd.Series:
    """
    Read a CSV expected to contain one row of metrics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")

    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"Empty metrics file for {name}: {path}")

    return df.iloc[0]


def get_value(row: pd.Series, candidates: Iterable[str], default: float = np.nan) -> float:
    """
    Return the first available metric among several possible column names.
    """
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                return float(value)
    return float(default)


def to_percent_if_fraction(value: float) -> float:
    """
    Convert a probability/fraction to percent when it appears to be in [0, 1].
    Leave already-percent values unchanged.
    """
    if pd.isna(value):
        return np.nan
    if abs(value) <= 1.0:
        return 100.0 * value
    return value


def save_table_markdown(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save a DataFrame as a markdown table without requiring external packages.
    """
    path = Path(path)
    lines = []
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")



def first_existing_path(paths: Iterable[str | Path]) -> Path:
    """
    Return the first existing path among several candidates.

    This keeps the comparison script robust when older training scripts used
    slightly different filenames, e.g. max_xr instead of max_abs_xr.
    """
    paths = [Path(path) for path in paths]
    for path in paths:
        if path.exists():
            return path
    return paths[0]


# =============================================================================
# METRIC COLLECTION
# =============================================================================

def compute_gp_peak_high_outreach_metrics() -> Dict[str, float]:
    """
    Compute high-outreach peak_y metrics from the GP test prediction table.

    The peak_y GP metrics.csv does not necessarily contain high-outreach metrics,
    so this function computes them from results/gp/test_prediction_errors.csv.
    """
    error_path = GP_RESULTS_DIR / "test_prediction_errors.csv"
    if not error_path.exists():
        return {
            "peak_y_high_outreach_n": np.nan,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    df = pd.read_csv(error_path)
    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
        return {
            "peak_y_high_outreach_n": np.nan,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    mask = df["y_true"] > HIGH_OUTREACH_THRESHOLD
    if not mask.any():
        return {
            "peak_y_high_outreach_n": 0,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    y_true = df.loc[mask, "y_true"].to_numpy()
    y_pred = df.loc[mask, "y_pred"].to_numpy()
    error = y_pred - y_true

    return {
        "peak_y_high_outreach_n": int(mask.sum()),
        "peak_y_high_outreach_rmse_mm": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
        "peak_y_high_outreach_mae_mm": float(np.mean(np.abs(error)) * 1000.0),
    }


def collect_metrics() -> pd.DataFrame:
    """
    Collect metrics from GP and NN result files into one comparison table.
    """
    gp_peak = read_single_row_csv(GP_RESULTS_DIR / "metrics.csv", "GP peak_y metrics")
    gp_constraint = read_single_row_csv(
        GP_CONSTRAINT_RESULTS_DIR / "metrics_max_abs_xr.csv",
        "GP max_abs_xr metrics",
    )
    nn_metrics = read_single_row_csv(NN_RESULTS_DIR / "metrics.csv", "NN metrics")

    gp_high = compute_gp_peak_high_outreach_metrics()

    rows = []

    # -------------------------------------------------------------------------
    # GP pair row
    # -------------------------------------------------------------------------
    rows.append(
        {
            "model": "GP pair",
            "description": "Two independent Gaussian Processes",
            "peak_y_rmse_mm": get_value(gp_peak, ["test_rmse", "test_rmse_m"]) * 1000.0,
            "peak_y_mae_mm": get_value(gp_peak, ["test_mae", "test_mae_m"]) * 1000.0,
            "peak_y_r2": get_value(gp_peak, ["test_r2"]),
            "peak_y_high_outreach_n": gp_high["peak_y_high_outreach_n"],
            "peak_y_high_outreach_rmse_mm": gp_high["peak_y_high_outreach_rmse_mm"],
            "peak_y_high_outreach_mae_mm": gp_high["peak_y_high_outreach_mae_mm"],
            "max_abs_xr_rmse_mm": get_value(gp_constraint, ["test_rmse_mm"]),
            "max_abs_xr_mae_mm": get_value(gp_constraint, ["test_mae_mm"]),
            "max_abs_xr_r2": get_value(gp_constraint, ["test_r2"]),
            "constraint_accuracy_percent": to_percent_if_fraction(
                get_value(gp_constraint, ["true_limit_classification_accuracy"])
            ),
            "false_feasible_percent": to_percent_if_fraction(
                get_value(gp_constraint, ["true_limit_false_feasible_rate"])
            ),
            "false_infeasible_percent": to_percent_if_fraction(
                get_value(gp_constraint, ["true_limit_false_infeasible_rate"])
            ),
            "near_boundary_n": get_value(gp_constraint, ["near_boundary_n_samples"]),
            "near_boundary_rmse_mm": get_value(gp_constraint, ["near_boundary_rmse_mm"]),
            "near_boundary_mae_mm": get_value(gp_constraint, ["near_boundary_mae_mm"]),
            "near_boundary_accuracy_percent": to_percent_if_fraction(
                get_value(gp_constraint, ["near_boundary_classification_accuracy_true_limit"])
            ),
            "near_boundary_false_feasible_percent": to_percent_if_fraction(
                get_value(gp_constraint, ["near_boundary_false_feasible_rate_true_limit"])
            ),
        }
    )

    # -------------------------------------------------------------------------
    # NN multi-output row
    # -------------------------------------------------------------------------
    rows.append(
        {
            "model": "NN multi-output",
            "description": "One PyTorch MLP with two outputs",
            "peak_y_rmse_mm": get_value(nn_metrics, ["test_peak_y_rmse_mm"]),
            "peak_y_mae_mm": get_value(nn_metrics, ["test_peak_y_mae_mm"]),
            "peak_y_r2": get_value(nn_metrics, ["test_peak_y_r2"]),
            "peak_y_high_outreach_n": get_value(nn_metrics, ["test_high_outreach_n_samples"]),
            "peak_y_high_outreach_rmse_mm": get_value(
                nn_metrics,
                ["test_high_outreach_peak_y_rmse_mm"],
            ),
            "peak_y_high_outreach_mae_mm": get_value(
                nn_metrics,
                ["test_high_outreach_peak_y_mae_mm"],
            ),
            "max_abs_xr_rmse_mm": get_value(nn_metrics, ["test_max_abs_xr_rmse_mm"]),
            "max_abs_xr_mae_mm": get_value(nn_metrics, ["test_max_abs_xr_mae_mm"]),
            "max_abs_xr_r2": get_value(nn_metrics, ["test_max_abs_xr_r2"]),
            "constraint_accuracy_percent": to_percent_if_fraction(
                get_value(nn_metrics, ["true_limit_classification_accuracy"])
            ),
            "false_feasible_percent": to_percent_if_fraction(
                get_value(nn_metrics, ["true_limit_false_feasible_rate"])
            ),
            "false_infeasible_percent": to_percent_if_fraction(
                get_value(nn_metrics, ["true_limit_false_infeasible_rate"])
            ),
            "near_boundary_n": get_value(nn_metrics, ["near_boundary_n_samples"]),
            "near_boundary_rmse_mm": get_value(nn_metrics, ["near_boundary_rmse_mm"]),
            "near_boundary_mae_mm": get_value(nn_metrics, ["near_boundary_mae_mm"]),
            "near_boundary_accuracy_percent": to_percent_if_fraction(
                get_value(nn_metrics, ["near_boundary_classification_accuracy_true_limit"])
            ),
            "near_boundary_false_feasible_percent": to_percent_if_fraction(
                get_value(nn_metrics, ["near_boundary_false_feasible_rate_true_limit"])
            ),
        }
    )

    comparison_df = pd.DataFrame(rows)
    return comparison_df


def make_report_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact table for the report/presentation.
    """
    table = comparison_df[
        [
            "model",
            "peak_y_rmse_mm",
            "peak_y_r2",
            "peak_y_high_outreach_rmse_mm",
            "max_abs_xr_rmse_mm",
            "max_abs_xr_r2",
            "constraint_accuracy_percent",
            "false_feasible_percent",
            "near_boundary_rmse_mm",
            "near_boundary_false_feasible_percent",
        ]
    ].copy()

    table = table.rename(
        columns={
            "model": "Model",
            "peak_y_rmse_mm": "peak_y RMSE [mm]",
            "peak_y_r2": "peak_y R2",
            "peak_y_high_outreach_rmse_mm": "High-outreach RMSE [mm]",
            "max_abs_xr_rmse_mm": "max_abs_xr RMSE [mm]",
            "max_abs_xr_r2": "max_abs_xr R2",
            "constraint_accuracy_percent": "Constraint accuracy [%]",
            "false_feasible_percent": "False feasible [%]",
            "near_boundary_rmse_mm": "Near-boundary RMSE [mm]",
            "near_boundary_false_feasible_percent": "Near-boundary false feasible [%]",
        }
    )

    rounded_cols = [col for col in table.columns if col != "Model"]
    for col in rounded_cols:
        table[col] = table[col].astype(float).round(3)

    return table


# =============================================================================
# PLOTS
# =============================================================================

def annotate_bars(ax, decimals: int = 2, suffix: str = "") -> None:
    """
    Add numeric labels above bars.
    """
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        label = f"{height:.{decimals}f}{suffix}"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_rmse_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Plot RMSE comparison for both outputs.
    """
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    peak_rmse = comparison_df["peak_y_rmse_mm"].to_numpy(dtype=float)
    constraint_rmse = comparison_df["max_abs_xr_rmse_mm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, peak_rmse, width, label="peak_y")
    ax.bar(x + width / 2, constraint_rmse, width, label="max_abs_xr")

    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Surrogate regression error: GP pair vs NN multi-output")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)

    plt.tight_layout()
    path = COMPARISON_FIGURES_DIR / "model_comparison_rmse.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_r2_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Plot R2 comparison for both outputs.
    """
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    peak_r2 = comparison_df["peak_y_r2"].to_numpy(dtype=float)
    constraint_r2 = comparison_df["max_abs_xr_r2"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, peak_r2, width, label="peak_y")
    ax.bar(x + width / 2, constraint_r2, width, label="max_abs_xr")

    ax.set_ylabel("R²")
    ax.set_title("Explained variance: GP pair vs NN multi-output")
    ax.set_ylim(max(0.90, np.nanmin([peak_r2.min(), constraint_r2.min()]) - 0.02), 1.005)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=9)

    plt.tight_layout()
    path = COMPARISON_FIGURES_DIR / "model_comparison_r2.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_constraint_safety(comparison_df: pd.DataFrame) -> None:
    """
    Plot constraint classification safety metrics.
    """
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    false_feasible = comparison_df["false_feasible_percent"].to_numpy(dtype=float)
    false_infeasible = comparison_df["false_infeasible_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, false_feasible, width, label="False feasible")
    ax.bar(x + width / 2, false_infeasible, width, label="False infeasible")

    ax.set_ylabel("Rate [%]")
    ax.set_title("Constraint classification errors at max_abs_xr ≤ 0.500 m")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)

    plt.tight_layout()
    path = COMPARISON_FIGURES_DIR / "model_comparison_constraint_safety.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_near_boundary(comparison_df: pd.DataFrame) -> None:
    """
    Plot near-boundary performance metrics.
    """
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    nb_rmse = comparison_df["near_boundary_rmse_mm"].to_numpy(dtype=float)
    nb_false_feasible = comparison_df["near_boundary_false_feasible_percent"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    bars1 = ax1.bar(x - width / 2, nb_rmse, width, label="RMSE [mm]")
    ax1.set_ylabel("Near-boundary RMSE [mm]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, nb_false_feasible, width, label="False feasible [%]", alpha=0.65)
    ax2.set_ylabel("Near-boundary false feasible [%]")

    ax1.set_title(
        f"Near-boundary constraint performance [{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
    )

    ax1.bar_label(bars1, fmt="%.2f", fontsize=9)
    ax2.bar_label(bars2, fmt="%.2f", fontsize=9)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    path = COMPARISON_FIGURES_DIR / "model_comparison_near_boundary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_summary_figure(comparison_df: pd.DataFrame) -> None:
    """
    Create a compact 2x2 summary figure for slides/report.
    """
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # peak_y RMSE
    ax = axes[0, 0]
    values = comparison_df["peak_y_rmse_mm"].to_numpy(dtype=float)
    ax.bar(labels, values, edgecolor="black")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("peak_y prediction error")
    ax.grid(True, axis="y", alpha=0.3)
    annotate_bars(ax, decimals=2)

    # max_abs_xr RMSE
    ax = axes[0, 1]
    values = comparison_df["max_abs_xr_rmse_mm"].to_numpy(dtype=float)
    ax.bar(labels, values, edgecolor="black")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("max_abs_xr prediction error")
    ax.grid(True, axis="y", alpha=0.3)
    annotate_bars(ax, decimals=2)

    # False feasible
    ax = axes[1, 0]
    values = comparison_df["false_feasible_percent"].to_numpy(dtype=float)
    ax.bar(labels, values, edgecolor="black")
    ax.set_ylabel("False feasible [%]")
    ax.set_title("Constraint safety error")
    ax.grid(True, axis="y", alpha=0.3)
    annotate_bars(ax, decimals=2)

    # Near-boundary RMSE
    ax = axes[1, 1]
    values = comparison_df["near_boundary_rmse_mm"].to_numpy(dtype=float)
    ax.bar(labels, values, edgecolor="black")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Near-boundary error")
    ax.grid(True, axis="y", alpha=0.3)
    annotate_bars(ax, decimals=2)

    plt.suptitle("Surrogate Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = COMPARISON_FIGURES_DIR / "model_comparison_summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")




def load_peak_y_prediction_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load saved peak_y predictions for GP and NN.

    The two prediction files are generated by model_peak_y.py and model_nn.py.
    They are assumed to use the same random_state=42 test split, but the GP
    error file may be sorted by absolute error; this does not affect parity,
    residual, or error-distribution plots.
    """
    gp_path = GP_RESULTS_DIR / "test_prediction_errors.csv"
    nn_path = NN_RESULTS_DIR / "test_predictions.csv"

    if not gp_path.exists():
        raise FileNotFoundError(
            f"Missing GP prediction file: {gp_path}. Run src/model_peak_y.py first."
        )
    if not nn_path.exists():
        raise FileNotFoundError(
            f"Missing NN prediction file: {nn_path}. Run src/model_nn.py first."
        )

    gp_df = pd.read_csv(gp_path)
    nn_df = pd.read_csv(nn_path)

    required_gp = {"y_true", "y_pred"}
    required_nn = {"true_peak_y", "pred_peak_y"}

    if not required_gp.issubset(gp_df.columns):
        raise ValueError(
            f"GP prediction file must contain columns {required_gp}, got {list(gp_df.columns)}"
        )
    if not required_nn.issubset(nn_df.columns):
        raise ValueError(
            f"NN prediction file must contain columns {required_nn}, got {list(nn_df.columns)}"
        )

    return gp_df, nn_df


def compute_rmse_mm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute RMSE in millimetres.
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) * 1000.0)


def compute_r2_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² without importing extra sklearn utilities here.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def plot_peak_y_parity_gp_vs_nn() -> None:
    """
    Plot GP and NN parity plots for peak_y on the fixed test set.
    """
    gp_df, nn_df = load_peak_y_prediction_tables()

    gp_true = gp_df["y_true"].to_numpy(dtype=float)
    gp_pred = gp_df["y_pred"].to_numpy(dtype=float)
    nn_true = nn_df["true_peak_y"].to_numpy(dtype=float)
    nn_pred = nn_df["pred_peak_y"].to_numpy(dtype=float)

    all_values = np.concatenate([gp_true, gp_pred, nn_true, nn_pred])
    pad = 0.02 * (all_values.max() - all_values.min())
    lim_min = all_values.min() - pad
    lim_max = all_values.max() + pad

    gp_rmse = compute_rmse_mm(gp_true, gp_pred)
    gp_r2 = compute_r2_simple(gp_true, gp_pred)
    nn_rmse = compute_rmse_mm(nn_true, nn_pred)
    nn_r2 = compute_r2_simple(nn_true, nn_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    plot_specs = [
        (axes[0], gp_true, gp_pred, "Gaussian Process", gp_rmse, gp_r2),
        (axes[1], nn_true, nn_pred, "Neural Network", nn_rmse, nn_r2),
    ]

    for ax, y_true, y_pred, title, rmse, r2 in plot_specs:
        ax.scatter(
            y_true,
            y_pred,
            alpha=0.65,
            s=42,
            edgecolors="black",
            linewidth=0.4,
        )
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=2, label="Ideal prediction")
        ax.axhline(HIGH_OUTREACH_THRESHOLD, linestyle=":", linewidth=1.8, label="High-outreach threshold")
        ax.axvline(HIGH_OUTREACH_THRESHOLD, linestyle=":", linewidth=1.8)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("True peak_y [m]")
        ax.set_title(f"{title}\nRMSE = {rmse:.2f} mm, R² = {r2:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Predicted peak_y [m]")
    plt.suptitle("Parity Plot on Fixed Test Set", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = COMPARISON_FIGURES_DIR / "model_comparison_peak_y_parity_gp_vs_nn.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_peak_y_residuals_gp_vs_nn() -> None:
    """
    Plot peak_y residuals for GP and NN on the fixed test set.
    """
    gp_df, nn_df = load_peak_y_prediction_tables()

    gp_true = gp_df["y_true"].to_numpy(dtype=float)
    gp_pred = gp_df["y_pred"].to_numpy(dtype=float)
    nn_true = nn_df["true_peak_y"].to_numpy(dtype=float)
    nn_pred = nn_df["pred_peak_y"].to_numpy(dtype=float)

    gp_error_mm = (gp_pred - gp_true) * 1000.0
    nn_error_mm = (nn_pred - nn_true) * 1000.0

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.scatter(
        gp_true,
        gp_error_mm,
        alpha=0.70,
        s=38,
        edgecolors="black",
        linewidth=0.35,
        label="Gaussian Process",
    )
    ax.scatter(
        nn_true,
        nn_error_mm,
        alpha=0.70,
        s=38,
        edgecolors="black",
        linewidth=0.35,
        label="Neural Network",
    )

    ax.axhline(0.0, linestyle="--", linewidth=2, color="black", label="Zero error")
    ax.axvline(HIGH_OUTREACH_THRESHOLD, linestyle=":", linewidth=1.8, label="High-outreach threshold")

    ax.set_xlabel("True peak_y [m]")
    ax.set_ylabel("Prediction error [mm]")
    ax.set_title("Residuals on Fixed Test Set")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = COMPARISON_FIGURES_DIR / "model_comparison_peak_y_residuals_gp_vs_nn.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_peak_y_error_distribution_gp_vs_nn() -> None:
    """
    Plot peak_y prediction error distributions for GP and NN.
    """
    gp_df, nn_df = load_peak_y_prediction_tables()

    gp_true = gp_df["y_true"].to_numpy(dtype=float)
    gp_pred = gp_df["y_pred"].to_numpy(dtype=float)
    nn_true = nn_df["true_peak_y"].to_numpy(dtype=float)
    nn_pred = nn_df["pred_peak_y"].to_numpy(dtype=float)

    gp_error_mm = (gp_pred - gp_true) * 1000.0
    nn_error_mm = (nn_pred - nn_true) * 1000.0

    all_errors = np.concatenate([gp_error_mm, nn_error_mm])
    low, high = np.nanpercentile(all_errors, [1, 99])
    margin = 0.15 * (high - low)
    bins = np.linspace(low - margin, high + margin, 32)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(gp_error_mm, bins=bins, alpha=0.65, edgecolor="black", label="Gaussian Process")
    ax.hist(nn_error_mm, bins=bins, alpha=0.65, edgecolor="black", label="Neural Network")
    ax.axvline(0.0, linestyle="--", linewidth=2, color="black", label="Zero error")

    ax.set_xlabel("Prediction error [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = COMPARISON_FIGURES_DIR / "model_comparison_peak_y_error_distribution_gp_vs_nn.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_peak_y_diagnostic_comparison() -> None:
    """
    Generate the main qualitative GP-vs-NN peak_y diagnostic plots.
    """
    plot_peak_y_parity_gp_vs_nn()
    plot_peak_y_residuals_gp_vs_nn()
    plot_peak_y_error_distribution_gp_vs_nn()

# =============================================================================
# INFERENCE TIME BENCHMARK
# =============================================================================

if TORCH_AVAILABLE:

    class MultiOutputMLP(nn.Module):
        """
        Same architecture class used in model_nn.py.
        """

        def __init__(self, input_dim: int, hidden_layers: Iterable[int], output_dim: int = 2) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


def load_dataset_inputs() -> np.ndarray:
    """
    Load input matrix for inference benchmarks.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    return df[INPUT_COLUMNS].to_numpy(dtype=np.float64)


def make_benchmark_batch(X: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Create a deterministic batch of exactly batch_size rows by tiling X.
    """
    if len(X) >= batch_size:
        return X[:batch_size]

    repeats = int(np.ceil(batch_size / len(X)))
    return np.tile(X, (repeats, 1))[:batch_size]


def benchmark_function(func, repeats: int = INFERENCE_REPEATS, warmup: int = INFERENCE_WARMUP) -> Tuple[float, float]:
    """
    Benchmark a callable and return mean and std elapsed time in milliseconds.
    """
    for _ in range(warmup):
        func()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

    return float(np.mean(times)), float(np.std(times))


def load_nn_for_timing():
    """
    Load the trained NN and scalers for timing.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")

    model_path = NN_RESULTS_DIR / "nn_multioutput_model.pt"
    scaler_x_path = NN_RESULTS_DIR / "scaler_X.pkl"
    scaler_y_path = NN_RESULTS_DIR / "scaler_Y.pkl"

    if not model_path.exists() or not scaler_x_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError("NN model or scalers not found. Run src/model_nn.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = MultiOutputMLP(
        input_dim=len(checkpoint["input_columns"]),
        hidden_layers=checkpoint["hidden_layers"],
        output_dim=len(checkpoint["target_columns"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)

    return model, scaler_X, scaler_Y, device


def run_inference_benchmark() -> pd.DataFrame:
    """
    Measure GP-pair and NN inference time for several batch sizes.

    The benchmark includes input scaling, model prediction and output inverse
    scaling, because that is what the optimization code will typically do.
    """
    print("\n" + "=" * 70)
    print("INFERENCE TIME BENCHMARK")
    print("=" * 70)

    X = load_dataset_inputs()

    rows = []

    # -------------------------------------------------------------------------
    # Load GP pair
    # -------------------------------------------------------------------------
    gp_peak_path = GP_RESULTS_DIR / "gp_model.pkl"
    gp_peak_scaler_x_path = GP_RESULTS_DIR / "scaler_X.pkl"
    gp_peak_scaler_y_path = GP_RESULTS_DIR / "scaler_y.pkl"

    gp_constraint_path = first_existing_path([
        GP_CONSTRAINT_RESULTS_DIR / "gp_max_abs_xr_model.pkl",
        GP_CONSTRAINT_RESULTS_DIR / "gp_max_xr_model.pkl",
    ])
    gp_constraint_scaler_x_path = first_existing_path([
        GP_CONSTRAINT_RESULTS_DIR / "scaler_X_max_abs_xr.pkl",
        GP_CONSTRAINT_RESULTS_DIR / "scaler_X_max_xr.pkl",
    ])
    gp_constraint_scaler_y_path = first_existing_path([
        GP_CONSTRAINT_RESULTS_DIR / "scaler_y_max_abs_xr.pkl",
        GP_CONSTRAINT_RESULTS_DIR / "scaler_y_max_xr.pkl",
    ])

    gp_required_paths = [
        gp_peak_path,
        gp_peak_scaler_x_path,
        gp_peak_scaler_y_path,
        gp_constraint_path,
        gp_constraint_scaler_x_path,
        gp_constraint_scaler_y_path,
    ]
    gp_available = all(path.exists() for path in gp_required_paths)

    nn_available = TORCH_AVAILABLE and all(
        path.exists()
        for path in [
            NN_RESULTS_DIR / "nn_multioutput_model.pt",
            NN_RESULTS_DIR / "scaler_X.pkl",
            NN_RESULTS_DIR / "scaler_Y.pkl",
        ]
    )

    if not gp_available:
        missing_paths = [str(path) for path in gp_required_paths if not path.exists()]
        print("⚠️  GP models/scalers not found. Skipping GP inference benchmark.")
        print(f"   Missing: {missing_paths}")
    else:
        gp_peak = joblib.load(gp_peak_path)
        gp_peak_scaler_X = joblib.load(gp_peak_scaler_x_path)
        gp_peak_scaler_y = joblib.load(gp_peak_scaler_y_path)

        gp_constraint = joblib.load(gp_constraint_path)
        gp_constraint_scaler_X = joblib.load(gp_constraint_scaler_x_path)
        gp_constraint_scaler_y = joblib.load(gp_constraint_scaler_y_path)

    if not nn_available:
        print("⚠️  NN model/scalers not found or PyTorch unavailable. Skipping NN inference benchmark.")
    else:
        nn_model, nn_scaler_X, nn_scaler_Y, nn_device = load_nn_for_timing()

    for batch_size in INFERENCE_BATCH_SIZES:
        X_batch = make_benchmark_batch(X, batch_size)

        if gp_available:
            def gp_pair_predict():
                X_peak_scaled = gp_peak_scaler_X.transform(X_batch)
                pred_peak_scaled = gp_peak.predict(X_peak_scaled).reshape(-1, 1)
                pred_peak = gp_peak_scaler_y.inverse_transform(pred_peak_scaled)

                X_constraint_scaled = gp_constraint_scaler_X.transform(X_batch)
                pred_constraint_scaled = gp_constraint.predict(X_constraint_scaled).reshape(-1, 1)
                pred_constraint = gp_constraint_scaler_y.inverse_transform(pred_constraint_scaled)

                return np.column_stack([pred_peak.ravel(), pred_constraint.ravel()])

            mean_ms, std_ms = benchmark_function(gp_pair_predict)
            rows.append(
                {
                    "model": "GP pair",
                    "batch_size": batch_size,
                    "mean_time_ms": mean_ms,
                    "std_time_ms": std_ms,
                    "time_per_sample_us": mean_ms * 1000.0 / batch_size,
                }
            )
            print(
                f"GP pair | batch {batch_size:5d}: "
                f"{mean_ms:.3f} ± {std_ms:.3f} ms "
                f"({mean_ms * 1000.0 / batch_size:.3f} µs/sample)"
            )

        if nn_available:
            def nn_predict():
                X_scaled = nn_scaler_X.transform(X_batch)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=nn_device)
                with torch.no_grad():
                    pred_scaled = nn_model(X_tensor).cpu().numpy()
                pred = nn_scaler_Y.inverse_transform(pred_scaled)
                return pred

            mean_ms, std_ms = benchmark_function(nn_predict)
            rows.append(
                {
                    "model": "NN multi-output",
                    "batch_size": batch_size,
                    "mean_time_ms": mean_ms,
                    "std_time_ms": std_ms,
                    "time_per_sample_us": mean_ms * 1000.0 / batch_size,
                }
            )
            print(
                f"NN      | batch {batch_size:5d}: "
                f"{mean_ms:.3f} ± {std_ms:.3f} ms "
                f"({mean_ms * 1000.0 / batch_size:.3f} µs/sample)"
            )

    timing_df = pd.DataFrame(rows)
    timing_path = COMPARISON_RESULTS_DIR / "inference_time_summary.csv"
    timing_df.to_csv(timing_path, index=False)
    print(f"✓ Inference timing saved: {timing_path}")

    return timing_df


def plot_inference_time(timing_df: pd.DataFrame) -> None:
    """
    Plot inference time per sample for GP pair and NN.
    """
    if timing_df.empty:
        print("No inference timing data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, group in timing_df.groupby("model"):
        group = group.sort_values("batch_size")
        ax.plot(
            group["batch_size"],
            group["time_per_sample_us"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Inference time per sample [µs]")
    ax.set_title("Inference efficiency: GP pair vs NN multi-output")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = COMPARISON_FIGURES_DIR / "model_comparison_inference_time.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# =============================================================================
# INTERPRETATION
# =============================================================================

def write_interpretation(comparison_df: pd.DataFrame, timing_df: Optional[pd.DataFrame]) -> Path:
    """
    Write a short interpretation file for the report.
    """
    gp = comparison_df[comparison_df["model"] == "GP pair"].iloc[0]
    nn_row = comparison_df[comparison_df["model"] == "NN multi-output"].iloc[0]

    lines = []
    lines.append("MODEL COMPARISON INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Compared models:")
    lines.append("  1. GP pair: independent GP models for peak_y and max_abs_xr.")
    lines.append("  2. NN multi-output: one PyTorch MLP predicting [peak_y, max_abs_xr].")
    lines.append("")
    lines.append("Main accuracy comparison:")
    lines.append(
        f"  peak_y RMSE: GP pair = {gp['peak_y_rmse_mm']:.2f} mm, "
        f"NN = {nn_row['peak_y_rmse_mm']:.2f} mm."
    )
    lines.append(
        f"  max_abs_xr RMSE: GP pair = {gp['max_abs_xr_rmse_mm']:.2f} mm, "
        f"NN = {nn_row['max_abs_xr_rmse_mm']:.2f} mm."
    )
    lines.append(
        f"  False feasible rate: GP pair = {gp['false_feasible_percent']:.2f}%, "
        f"NN = {nn_row['false_feasible_percent']:.2f}%."
    )
    lines.append(
        f"  Near-boundary RMSE: GP pair = {gp['near_boundary_rmse_mm']:.2f} mm, "
        f"NN = {nn_row['near_boundary_rmse_mm']:.2f} mm."
    )
    lines.append("")

    if gp["peak_y_rmse_mm"] < nn_row["peak_y_rmse_mm"] and gp["max_abs_xr_rmse_mm"] < nn_row["max_abs_xr_rmse_mm"]:
        lines.append(
            "Interpretation: with the current dataset size, the GP pair provides "
            "better predictive accuracy than the NN multi-output surrogate. This is "
            "consistent with Gaussian Processes being data-efficient for small to "
            "medium-sized datasets."
        )
    else:
        lines.append(
            "Interpretation: the NN is competitive with the GP pair on at least one "
            "of the predicted quantities. This suggests that the multi-output NN is "
            "a viable surrogate candidate, especially if the dataset is enlarged."
        )

    lines.append("")
    lines.append(
        "The NN remains useful because it is differentiable in PyTorch and can be "
        "used directly for gradient-based inverse optimization. The GP pair remains "
        "the more conservative choice for constraint-aware optimization when accuracy "
        "and low false-feasible rate are prioritized."
    )

    if timing_df is not None and not timing_df.empty:
        lines.append("")
        lines.append("Inference-time comparison:")
        for batch_size in sorted(timing_df["batch_size"].unique()):
            sub = timing_df[timing_df["batch_size"] == batch_size]
            entries = []
            for _, row in sub.iterrows():
                entries.append(
                    f"{row['model']} = {row['time_per_sample_us']:.3f} µs/sample"
                )
            lines.append(f"  batch={batch_size}: " + "; ".join(entries))

    output_path = COMPARISON_RESULTS_DIR / "model_comparison_interpretation.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Interpretation saved: {output_path}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("SURROGATE MODEL COMPARISON")
    print("=" * 70)

    print("\nCollecting saved model metrics...")
    comparison_df = collect_metrics()

    summary_path = COMPARISON_RESULTS_DIR / "model_comparison_summary.csv"
    comparison_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")

    report_table = make_report_table(comparison_df)
    report_path = COMPARISON_RESULTS_DIR / "model_comparison_report_table.csv"
    report_table.to_csv(report_path, index=False)
    print(f"✓ Report table saved: {report_path}")

    report_md_path = COMPARISON_RESULTS_DIR / "model_comparison_report_table.md"
    save_table_markdown(report_table, report_md_path)
    print(f"✓ Markdown report table saved: {report_md_path}")

    print("\nMODEL COMPARISON REPORT TABLE")
    print(report_table.to_string(index=False))

    print("\nGenerating comparison figures...")

    # Main qualitative GP-vs-NN diagnostics for peak_y.
    # These figures show how the models fail, not only the aggregate metrics.
    plot_peak_y_diagnostic_comparison()

    # Compact aggregate figures.
    plot_rmse_comparison(comparison_df)
    plot_r2_comparison(comparison_df)
    plot_constraint_safety(comparison_df)
    plot_near_boundary(comparison_df)
    plot_summary_figure(comparison_df)

    try:
        timing_df = run_inference_benchmark()
        plot_inference_time(timing_df)
    except Exception as exc:
        timing_df = pd.DataFrame()
        print(f"⚠️  Inference benchmark skipped: {exc}")

    interpretation_path = write_interpretation(comparison_df, timing_df)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)
    print("Generated files:")
    print(f"  {summary_path}")
    print(f"  {report_path}")
    print(f"  {report_md_path}")
    print(f"  {COMPARISON_RESULTS_DIR / 'inference_time_summary.csv'}")
    print(f"  {interpretation_path}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_peak_y_parity_gp_vs_nn.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_peak_y_residuals_gp_vs_nn.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_peak_y_error_distribution_gp_vs_nn.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_rmse.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_r2.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_constraint_safety.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_near_boundary.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_inference_time.png'}")
    print(f"  {COMPARISON_FIGURES_DIR / 'model_comparison_summary.png'}")
    print("\nNext step: regenerate the final 3000-sample dataset and retrain GP/NN, then rerun this comparison.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
