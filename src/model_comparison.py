"""
model_comparison.py
===================

Compare the Gaussian Process surrogate pair against the multi-output Neural
Network surrogate.

Models compared
---------------
1. GP pair
   - GP_peak_y:      input parameters -> peak_y
   - GP_max_abs_xr:  input parameters -> max_abs_xr

2. NN multi-output
   - NN ensemble: input parameters -> [peak_y, max_abs_xr]

The comparison is designed for report and presentation use. It collects the
main regression, constraint-classification, near-boundary and inference-time
metrics in one place and generates summary figures.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
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
NN_RESULTS_DIR = RESULTS_DIR / "nn_v2"

COMPARISON_RESULTS_DIR = RESULTS_DIR / "model_comparison"
COMPARISON_FIGURES_DIR = FIGURES_DIR / "model_comparison"

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

DEFAULT_INFERENCE_BATCH_SIZES = [1, 1000, 10000]
DEFAULT_INFERENCE_REPEATS = 5
DEFAULT_INFERENCE_WARMUP = 2


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, label: str) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            "Run the required previous pipeline step first."
        )


def read_single_row_csv(path: Path, label: str) -> pd.Series:
    """Read a CSV expected to contain one row of metrics."""
    require_file(path, label)

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError(f"Empty metrics file for {label}: {path}")

    return df.iloc[0]


def get_value(
    row: pd.Series,
    candidates: Iterable[str],
    default: float = np.nan,
) -> float:
    """Return the first available metric among several possible column names."""
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                return float(value)

    return float(default)


def get_metric_mm(
    row: pd.Series,
    mm_candidates: Iterable[str],
    meter_candidates: Iterable[str],
    default: float = np.nan,
) -> float:
    """
    Return a metric in millimeters.

    Some scripts save RMSE/MAE directly in millimeters, while older scripts save
    the same quantity in meters. This helper handles both cases.
    """
    value_mm = get_value(row, mm_candidates, default=np.nan)

    if pd.notna(value_mm):
        return float(value_mm)

    value_m = get_value(row, meter_candidates, default=np.nan)

    if pd.notna(value_m):
        return float(value_m * 1000.0)

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


def save_table_markdown(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a markdown table without external dependencies."""
    path = Path(path)

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def first_existing_path(paths: Iterable[Path]) -> Path:
    """
    Return the first existing path among several candidates.

    This keeps the script robust when older scripts used slightly different
    filenames, for example max_xr instead of max_abs_xr.
    """
    paths = list(paths)

    for path in paths:
        if path.exists():
            return path

    return paths[0]


def compute_rmse_mm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE in millimeters."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) * 1000.0)


def compute_mae_mm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE in millimeters."""
    return float(np.mean(np.abs(y_pred - y_true)) * 1000.0)


def compute_r2_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² without importing additional utilities."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot <= 0.0:
        return np.nan

    return 1.0 - ss_res / ss_tot


# =============================================================================
# METRIC COLLECTION
# =============================================================================

def compute_gp_peak_high_outreach_metrics() -> dict[str, float]:
    """
    Compute high-outreach peak_y metrics from the GP test prediction table.
    """
    error_path = GP_RESULTS_DIR / "test_prediction_errors.csv"

    if not error_path.exists():
        return {
            "peak_y_high_outreach_n": np.nan,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    df = pd.read_csv(error_path)

    required_cols = {"y_true", "y_pred"}
    if not required_cols.issubset(df.columns):
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

    y_true = df.loc[mask, "y_true"].to_numpy(dtype=float)
    y_pred = df.loc[mask, "y_pred"].to_numpy(dtype=float)

    return {
        "peak_y_high_outreach_n": int(mask.sum()),
        "peak_y_high_outreach_rmse_mm": compute_rmse_mm(y_true, y_pred),
        "peak_y_high_outreach_mae_mm": compute_mae_mm(y_true, y_pred),
    }


def compute_nn_peak_high_outreach_metrics() -> dict[str, float]:
    """
    Compute high-outreach peak_y metrics from the NN v2 test prediction table.
    """
    pred_path = NN_RESULTS_DIR / "test_predictions.csv"

    if not pred_path.exists():
        return {
            "peak_y_high_outreach_n": np.nan,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    df = pd.read_csv(pred_path)

    required_cols = {"true_peak_y", "pred_peak_y"}
    if not required_cols.issubset(df.columns):
        return {
            "peak_y_high_outreach_n": np.nan,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    mask = df["true_peak_y"] > HIGH_OUTREACH_THRESHOLD

    if not mask.any():
        return {
            "peak_y_high_outreach_n": 0,
            "peak_y_high_outreach_rmse_mm": np.nan,
            "peak_y_high_outreach_mae_mm": np.nan,
        }

    y_true = df.loc[mask, "true_peak_y"].to_numpy(dtype=float)
    y_pred = df.loc[mask, "pred_peak_y"].to_numpy(dtype=float)

    return {
        "peak_y_high_outreach_n": int(mask.sum()),
        "peak_y_high_outreach_rmse_mm": compute_rmse_mm(y_true, y_pred),
        "peak_y_high_outreach_mae_mm": compute_mae_mm(y_true, y_pred),
    }


def collect_metrics() -> pd.DataFrame:
    """Collect metrics from GP and NN result files into one comparison table."""
    gp_peak = read_single_row_csv(
        GP_RESULTS_DIR / "metrics.csv",
        "GP peak_y metrics",
    )

    gp_constraint = read_single_row_csv(
        GP_CONSTRAINT_RESULTS_DIR / "metrics_max_abs_xr.csv",
        "GP max_abs_xr metrics",
    )

    nn_metrics = read_single_row_csv(
        NN_RESULTS_DIR / "metrics.csv",
        "NN v2 metrics",
    )

    gp_high = compute_gp_peak_high_outreach_metrics()
    nn_high = compute_nn_peak_high_outreach_metrics()

    rows = [
        {
            "model": "GP pair",
            "description": "Two independent Gaussian Processes",
            "peak_y_rmse_mm": get_metric_mm(
                gp_peak,
                mm_candidates=["test_rmse_mm"],
                meter_candidates=["test_rmse", "test_rmse_m"],
            ),
            "peak_y_mae_mm": get_metric_mm(
                gp_peak,
                mm_candidates=["test_mae_mm"],
                meter_candidates=["test_mae", "test_mae_m"],
            ),
            "peak_y_r2": get_value(gp_peak, ["test_r2"]),
            "peak_y_high_outreach_n": gp_high["peak_y_high_outreach_n"],
            "peak_y_high_outreach_rmse_mm": gp_high["peak_y_high_outreach_rmse_mm"],
            "peak_y_high_outreach_mae_mm": gp_high["peak_y_high_outreach_mae_mm"],
            "max_abs_xr_rmse_mm": get_metric_mm(
                gp_constraint,
                mm_candidates=["test_rmse_mm"],
                meter_candidates=["test_rmse_m", "test_rmse"],
            ),
            "max_abs_xr_mae_mm": get_metric_mm(
                gp_constraint,
                mm_candidates=["test_mae_mm"],
                meter_candidates=["test_mae_m", "test_mae"],
            ),
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
        },
        {
            "model": "NN multi-output",
            "description": "One PyTorch MLP ensemble with two outputs",
            "peak_y_rmse_mm": get_value(nn_metrics, ["test_peak_y_rmse_mm"]),
            "peak_y_mae_mm": get_value(nn_metrics, ["test_peak_y_mae_mm"]),
            "peak_y_r2": get_value(nn_metrics, ["test_peak_y_r2"]),
            "peak_y_high_outreach_n": nn_high["peak_y_high_outreach_n"],
            "peak_y_high_outreach_rmse_mm": nn_high["peak_y_high_outreach_rmse_mm"],
            "peak_y_high_outreach_mae_mm": nn_high["peak_y_high_outreach_mae_mm"],
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
        },
    ]

    return pd.DataFrame(rows)


def make_report_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table for the report or presentation."""
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

    for col in table.columns:
        if col != "Model":
            table[col] = table[col].astype(float).round(3)

    return table


# =============================================================================
# PLOTS: AGGREGATE COMPARISON
# =============================================================================

def annotate_bars(ax: plt.Axes, decimals: int = 2, suffix: str = "") -> None:
    """Add numeric labels above bars."""
    for patch in ax.patches:
        height = patch.get_height()

        if np.isnan(height):
            continue

        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height,
            f"{height:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_rmse_comparison(comparison_df: pd.DataFrame) -> None:
    """Plot RMSE comparison for both outputs."""
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    peak_rmse = comparison_df["peak_y_rmse_mm"].to_numpy(dtype=float)
    constraint_rmse = comparison_df["max_abs_xr_rmse_mm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_peak = ax.bar(x - width / 2, peak_rmse, width, label="peak_y")
    bars_constraint = ax.bar(x + width / 2, constraint_rmse, width, label="max_abs_xr")

    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Surrogate regression error: GP pair vs NN multi-output")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax.bar_label(bars_peak, fmt="%.2f", fontsize=9)
    ax.bar_label(bars_constraint, fmt="%.2f", fontsize=9)

    path = COMPARISON_FIGURES_DIR / "model_comparison_rmse.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_r2_comparison(comparison_df: pd.DataFrame) -> None:
    """Plot R² comparison for both outputs."""
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    peak_r2 = comparison_df["peak_y_r2"].to_numpy(dtype=float)
    constraint_r2 = comparison_df["max_abs_xr_r2"].to_numpy(dtype=float)

    finite_values = np.concatenate([peak_r2[np.isfinite(peak_r2)], constraint_r2[np.isfinite(constraint_r2)]])

    if len(finite_values) > 0:
        y_min = max(0.0, float(np.min(finite_values)) - 0.02)
    else:
        y_min = 0.0

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_peak = ax.bar(x - width / 2, peak_r2, width, label="peak_y")
    bars_constraint = ax.bar(x + width / 2, constraint_r2, width, label="max_abs_xr")

    ax.set_ylabel("R²")
    ax.set_title("Explained variance: GP pair vs NN multi-output")
    ax.set_ylim(y_min, 1.005)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax.bar_label(bars_peak, fmt="%.4f", fontsize=9)
    ax.bar_label(bars_constraint, fmt="%.4f", fontsize=9)

    path = COMPARISON_FIGURES_DIR / "model_comparison_r2.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_constraint_safety(comparison_df: pd.DataFrame) -> None:
    """Plot constraint classification safety metrics."""
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    false_feasible = comparison_df["false_feasible_percent"].to_numpy(dtype=float)
    false_infeasible = comparison_df["false_infeasible_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_ff = ax.bar(x - width / 2, false_feasible, width, label="False feasible")
    bars_fi = ax.bar(x + width / 2, false_infeasible, width, label="False infeasible")

    ax.set_ylabel("Rate [%]")
    ax.set_title("Constraint classification errors at max_abs_xr <= 0.500 m")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax.bar_label(bars_ff, fmt="%.2f", fontsize=9)
    ax.bar_label(bars_fi, fmt="%.2f", fontsize=9)

    path = COMPARISON_FIGURES_DIR / "model_comparison_constraint_safety.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_near_boundary(comparison_df: pd.DataFrame) -> None:
    """Plot near-boundary constraint performance metrics."""
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    near_boundary_rmse = comparison_df["near_boundary_rmse_mm"].to_numpy(dtype=float)
    near_boundary_false_feasible = comparison_df["near_boundary_false_feasible_percent"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    bars_rmse = ax1.bar(
        x - width / 2,
        near_boundary_rmse,
        width,
        label="RMSE [mm]",
    )
    ax1.set_ylabel("Near-boundary RMSE [mm]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    bars_ff = ax2.bar(
        x + width / 2,
        near_boundary_false_feasible,
        width,
        alpha=0.65,
        label="False feasible [%]",
    )
    ax2.set_ylabel("Near-boundary false feasible [%]")

    ax1.set_title(
        f"Near-boundary constraint performance "
        f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
    )

    ax1.bar_label(bars_rmse, fmt="%.2f", fontsize=9)
    ax2.bar_label(bars_ff, fmt="%.2f", fontsize=9)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    path = COMPARISON_FIGURES_DIR / "model_comparison_near_boundary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_summary_figure(comparison_df: pd.DataFrame) -> None:
    """Create a compact 2x2 summary figure for slides or report."""
    labels = comparison_df["model"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    specs = [
        ("peak_y_rmse_mm", "peak_y prediction error", "RMSE [mm]"),
        ("max_abs_xr_rmse_mm", "max_abs_xr prediction error", "RMSE [mm]"),
        ("false_feasible_percent", "Constraint safety error", "False feasible [%]"),
        ("near_boundary_rmse_mm", "Near-boundary error", "RMSE [mm]"),
    ]

    for ax, (col, title, ylabel) in zip(axes.ravel(), specs):
        values = comparison_df[col].to_numpy(dtype=float)
        ax.bar(labels, values, edgecolor="black")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        annotate_bars(ax, decimals=2)

    fig.suptitle("Surrogate Model Comparison", fontsize=16, fontweight="bold")

    path = COMPARISON_FIGURES_DIR / "model_comparison_summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# =============================================================================
# PLOTS: PEAK_Y QUALITATIVE DIAGNOSTICS
# =============================================================================

def load_peak_y_prediction_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load saved peak_y predictions for GP and NN.

    The GP table may be sorted by absolute error. This does not affect parity,
    residual or error-distribution plots.
    """
    gp_path = GP_RESULTS_DIR / "test_prediction_errors.csv"
    nn_path = NN_RESULTS_DIR / "test_predictions.csv"

    require_file(gp_path, "GP peak_y prediction table")
    require_file(nn_path, "NN v2 prediction table")

    gp_df = pd.read_csv(gp_path)
    nn_df = pd.read_csv(nn_path)

    required_gp = {"y_true", "y_pred"}
    required_nn = {"true_peak_y", "pred_peak_y"}

    if not required_gp.issubset(gp_df.columns):
        raise ValueError(
            f"GP prediction file must contain columns {required_gp}, "
            f"got {list(gp_df.columns)}"
        )

    if not required_nn.issubset(nn_df.columns):
        raise ValueError(
            f"NN prediction file must contain columns {required_nn}, "
            f"got {list(nn_df.columns)}"
        )

    return gp_df, nn_df


def plot_peak_y_parity_gp_vs_nn() -> None:
    """Plot GP and NN parity plots for peak_y on the saved test sets."""
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
        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            linestyle="--",
            linewidth=2,
            label="Ideal prediction",
        )
        ax.axhline(
            HIGH_OUTREACH_THRESHOLD,
            linestyle=":",
            linewidth=1.8,
            label="High-outreach threshold",
        )
        ax.axvline(HIGH_OUTREACH_THRESHOLD, linestyle=":", linewidth=1.8)

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("True peak_y [m]")
        ax.set_title(f"{title}\nRMSE = {rmse:.2f} mm, R² = {r2:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Predicted peak_y [m]")
    fig.suptitle("Parity Plot on Test Set", fontsize=15, fontweight="bold")

    path = COMPARISON_FIGURES_DIR / "model_comparison_peak_y_parity_gp_vs_nn.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_peak_y_residuals_gp_vs_nn() -> None:
    """Plot peak_y residuals for GP and NN."""
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

    ax.axhline(0.0, linestyle="--", linewidth=2, label="Zero error")
    ax.axvline(
        HIGH_OUTREACH_THRESHOLD,
        linestyle=":",
        linewidth=1.8,
        label="High-outreach threshold",
    )

    ax.set_xlabel("True peak_y [m]")
    ax.set_ylabel("Prediction error [mm]")
    ax.set_title("Peak_y residuals on test set")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = COMPARISON_FIGURES_DIR / "model_comparison_peak_y_residuals_gp_vs_nn.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_peak_y_error_distribution_gp_vs_nn() -> None:
    """Plot peak_y prediction error distributions for GP and NN."""
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
    ax.axvline(0.0, linestyle="--", linewidth=2, label="Zero error")

    ax.set_xlabel("Prediction error [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Peak_y prediction error distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = COMPARISON_FIGURES_DIR / "model_comparison_peak_y_error_distribution_gp_vs_nn.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_peak_y_diagnostic_comparison() -> None:
    """Generate qualitative GP-vs-NN peak_y diagnostic plots."""
    plot_peak_y_parity_gp_vs_nn()
    plot_peak_y_residuals_gp_vs_nn()
    plot_peak_y_error_distribution_gp_vs_nn()


# =============================================================================
# INFERENCE TIME BENCHMARK
# =============================================================================

if TORCH_AVAILABLE:

    class MultiOutputMLP(nn.Module):
        """Architecture class used to reload the saved NN v2 checkpoint."""

        def __init__(
            self,
            input_dim: int,
            hidden_layers: Iterable[int],
            output_dim: int = 2,
        ) -> None:
            super().__init__()

            layers: list[nn.Module] = []
            previous_dim = input_dim

            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(previous_dim, int(hidden_dim)))
                layers.append(nn.ReLU())
                previous_dim = int(hidden_dim)

            layers.append(nn.Linear(previous_dim, output_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.network(x)


def load_dataset_inputs() -> np.ndarray:
    """Load input matrix for inference benchmarks."""
    require_file(DATASET_PATH, "augmented dataset")

    df = pd.read_csv(DATASET_PATH, comment="#")

    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns in dataset: {missing}")

    return df[INPUT_COLUMNS].to_numpy(dtype=np.float64)


def make_benchmark_batch(X: np.ndarray, batch_size: int) -> np.ndarray:
    """Create a deterministic batch of exactly batch_size rows by tiling X."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if len(X) >= batch_size:
        return X[:batch_size]

    repeats = int(np.ceil(batch_size / len(X)))

    return np.tile(X, (repeats, 1))[:batch_size]


def benchmark_function(
    func: Callable[[], Any],
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    """Benchmark a callable and return mean and std elapsed time in ms."""
    for _ in range(warmup):
        func()

    times_ms = []

    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def load_nn_for_timing():
    """Load the trained NN v2 ensemble and scalers for timing."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")

    model_path = NN_RESULTS_DIR / "nn_multioutput_model.pt"
    scaler_x_path = NN_RESULTS_DIR / "scaler_X.pkl"
    scaler_y_path = NN_RESULTS_DIR / "scaler_Y.pkl"

    require_file(model_path, "NN v2 checkpoint")
    require_file(scaler_x_path, "NN v2 input scaler")
    require_file(scaler_y_path, "NN v2 output scaler")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)

    hidden_layers = checkpoint["hidden_layers"]
    input_dim = len(checkpoint["input_columns"])
    output_dim = len(checkpoint["target_columns"])

    state_dicts = checkpoint.get("ensemble_state_dicts")

    if state_dicts is None:
        state_dicts = [checkpoint["model_state_dict"]]

    models = []

    for state_dict in state_dicts:
        model = MultiOutputMLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models, scaler_X, scaler_Y, device


def run_inference_benchmark(
    batch_sizes: list[int],
    repeats: int,
    warmup: int,
) -> pd.DataFrame:
    """
    Measure GP-pair and NN inference time for several batch sizes.

    The benchmark includes input scaling, model prediction and output inverse
    scaling, because that is what the optimization code typically does.
    """
    print("\n" + "=" * 70)
    print("INFERENCE TIME BENCHMARK")
    print("=" * 70)

    X = load_dataset_inputs()

    rows = []

    gp_peak_path = GP_RESULTS_DIR / "gp_model.pkl"
    gp_peak_scaler_x_path = GP_RESULTS_DIR / "scaler_X.pkl"
    gp_peak_scaler_y_path = GP_RESULTS_DIR / "scaler_y.pkl"

    gp_constraint_path = first_existing_path(
        [
            GP_CONSTRAINT_RESULTS_DIR / "gp_max_abs_xr_model.pkl",
            GP_CONSTRAINT_RESULTS_DIR / "gp_max_xr_model.pkl",
        ]
    )
    gp_constraint_scaler_x_path = first_existing_path(
        [
            GP_CONSTRAINT_RESULTS_DIR / "scaler_X_max_abs_xr.pkl",
            GP_CONSTRAINT_RESULTS_DIR / "scaler_X_max_xr.pkl",
        ]
    )
    gp_constraint_scaler_y_path = first_existing_path(
        [
            GP_CONSTRAINT_RESULTS_DIR / "scaler_y_max_abs_xr.pkl",
            GP_CONSTRAINT_RESULTS_DIR / "scaler_y_max_xr.pkl",
        ]
    )

    gp_required_paths = [
        gp_peak_path,
        gp_peak_scaler_x_path,
        gp_peak_scaler_y_path,
        gp_constraint_path,
        gp_constraint_scaler_x_path,
        gp_constraint_scaler_y_path,
    ]

    gp_available = all(path.exists() for path in gp_required_paths)

    nn_required_paths = [
        NN_RESULTS_DIR / "nn_multioutput_model.pt",
        NN_RESULTS_DIR / "scaler_X.pkl",
        NN_RESULTS_DIR / "scaler_Y.pkl",
    ]

    nn_available = TORCH_AVAILABLE and all(path.exists() for path in nn_required_paths)

    if gp_available:
        gp_peak = joblib.load(gp_peak_path)
        gp_peak_scaler_X = joblib.load(gp_peak_scaler_x_path)
        gp_peak_scaler_y = joblib.load(gp_peak_scaler_y_path)

        gp_constraint = joblib.load(gp_constraint_path)
        gp_constraint_scaler_X = joblib.load(gp_constraint_scaler_x_path)
        gp_constraint_scaler_y = joblib.load(gp_constraint_scaler_y_path)
    else:
        missing_paths = [str(path) for path in gp_required_paths if not path.exists()]
        print("GP models or scalers not found. Skipping GP inference benchmark.")
        print(f"Missing: {missing_paths}")

    if nn_available:
        nn_models, nn_scaler_X, nn_scaler_Y, nn_device = load_nn_for_timing()
    else:
        print("NN model/scalers not found or PyTorch unavailable. Skipping NN inference benchmark.")

    for batch_size in batch_sizes:
        X_batch = make_benchmark_batch(X, batch_size)

        if gp_available:

            def gp_pair_predict() -> np.ndarray:
                X_peak_scaled = gp_peak_scaler_X.transform(X_batch)
                pred_peak_scaled = gp_peak.predict(X_peak_scaled).reshape(-1, 1)
                pred_peak = gp_peak_scaler_y.inverse_transform(pred_peak_scaled)

                X_constraint_scaled = gp_constraint_scaler_X.transform(X_batch)
                pred_constraint_scaled = gp_constraint.predict(X_constraint_scaled).reshape(-1, 1)
                pred_constraint = gp_constraint_scaler_y.inverse_transform(pred_constraint_scaled)

                return np.column_stack([pred_peak.ravel(), pred_constraint.ravel()])

            mean_ms, std_ms = benchmark_function(
                gp_pair_predict,
                repeats=repeats,
                warmup=warmup,
            )

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
                f"GP pair | batch {batch_size:6d}: "
                f"{mean_ms:.3f} ± {std_ms:.3f} ms "
                f"({mean_ms * 1000.0 / batch_size:.3f} us/sample)"
            )

        if nn_available:

            def nn_predict() -> np.ndarray:
                X_scaled = nn_scaler_X.transform(X_batch)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=nn_device)

                with torch.no_grad():
                    pred_scaled_all = [
                        model(X_tensor).cpu().numpy()
                        for model in nn_models
                    ]

                pred_scaled = np.mean(pred_scaled_all, axis=0)
                return nn_scaler_Y.inverse_transform(pred_scaled)

            mean_ms, std_ms = benchmark_function(
                nn_predict,
                repeats=repeats,
                warmup=warmup,
            )

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
                f"NN      | batch {batch_size:6d}: "
                f"{mean_ms:.3f} ± {std_ms:.3f} ms "
                f"({mean_ms * 1000.0 / batch_size:.3f} us/sample)"
            )

    timing_df = pd.DataFrame(rows)

    timing_path = COMPARISON_RESULTS_DIR / "inference_time_summary.csv"
    timing_df.to_csv(timing_path, index=False)

    print(f"Inference timing saved: {timing_path}")

    return timing_df


def plot_inference_time(timing_df: pd.DataFrame) -> None:
    """Plot inference time per sample for GP pair and NN."""
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
    ax.set_ylabel("Inference time per sample [us]")
    ax.set_title("Inference efficiency: GP pair vs NN multi-output")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    path = COMPARISON_FIGURES_DIR / "model_comparison_inference_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# =============================================================================
# INTERPRETATION
# =============================================================================

def write_interpretation(
    comparison_df: pd.DataFrame,
    timing_df: pd.DataFrame | None,
) -> Path:
    """Write a short interpretation file for the report."""
    gp = comparison_df[comparison_df["model"] == "GP pair"].iloc[0]
    nn_row = comparison_df[comparison_df["model"] == "NN multi-output"].iloc[0]

    lines = [
        "MODEL COMPARISON INTERPRETATION",
        "=" * 80,
        "",
        "Compared models:",
        "  1. GP pair: independent GP models for peak_y and max_abs_xr.",
        "  2. NN multi-output: one PyTorch MLP ensemble predicting [peak_y, max_abs_xr].",
        "",
        "Main accuracy comparison:",
        (
            f"  peak_y RMSE: GP pair = {gp['peak_y_rmse_mm']:.2f} mm, "
            f"NN = {nn_row['peak_y_rmse_mm']:.2f} mm."
        ),
        (
            f"  max_abs_xr RMSE: GP pair = {gp['max_abs_xr_rmse_mm']:.2f} mm, "
            f"NN = {nn_row['max_abs_xr_rmse_mm']:.2f} mm."
        ),
        (
            f"  False feasible rate: GP pair = {gp['false_feasible_percent']:.2f}%, "
            f"NN = {nn_row['false_feasible_percent']:.2f}%."
        ),
        (
            f"  Near-boundary RMSE: GP pair = {gp['near_boundary_rmse_mm']:.2f} mm, "
            f"NN = {nn_row['near_boundary_rmse_mm']:.2f} mm."
        ),
        "",
    ]

    gp_better_peak = gp["peak_y_rmse_mm"] < nn_row["peak_y_rmse_mm"]
    gp_better_constraint = gp["max_abs_xr_rmse_mm"] < nn_row["max_abs_xr_rmse_mm"]

    if gp_better_peak and gp_better_constraint:
        lines.append(
            "Interpretation: with the current dataset size, the GP pair provides "
            "lower regression error on both predicted quantities. This is consistent "
            "with Gaussian Processes being data-efficient for small to medium datasets."
        )
    else:
        lines.append(
            "Interpretation: the NN is competitive with the GP pair on at least one "
            "predicted quantity. This supports its use as a differentiable surrogate "
            "for gradient-based inverse optimization."
        )

    lines.extend(
        [
            "",
            "The NN remains useful because it is differentiable in PyTorch and can be "
            "used directly for gradient-based inverse optimization. The GP pair remains "
            "useful when predictive uncertainty and conservative constraint handling "
            "are prioritized.",
        ]
    )

    if timing_df is not None and not timing_df.empty:
        lines.extend(["", "Inference-time comparison:"])

        for batch_size in sorted(timing_df["batch_size"].unique()):
            sub = timing_df[timing_df["batch_size"] == batch_size]
            entries = []

            for _, row in sub.iterrows():
                entries.append(
                    f"{row['model']} = {row['time_per_sample_us']:.3f} us/sample"
                )

            lines.append(f"  batch={batch_size}: " + "; ".join(entries))

    output_path = COMPARISON_RESULTS_DIR / "model_comparison_interpretation.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Interpretation saved: {output_path}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def parse_batch_sizes(raw: str) -> list[int]:
    """Parse comma-separated batch sizes."""
    values = [item.strip() for item in raw.split(",") if item.strip()]

    batch_sizes = [int(value) for value in values]

    if any(value <= 0 for value in batch_sizes):
        raise ValueError("All batch sizes must be positive.")

    return batch_sizes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare GP-pair and NN multi-output surrogate models."
    )

    parser.add_argument(
        "--skip_timing",
        action="store_true",
        help="Skip inference-time benchmark.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    parser.add_argument(
        "--batch_sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_INFERENCE_BATCH_SIZES),
        help="Comma-separated batch sizes for inference benchmark.",
    )

    parser.add_argument(
        "--timing_repeats",
        type=int,
        default=DEFAULT_INFERENCE_REPEATS,
        help=f"Number of timing repeats. Default: {DEFAULT_INFERENCE_REPEATS}.",
    )

    parser.add_argument(
        "--timing_warmup",
        type=int,
        default=DEFAULT_INFERENCE_WARMUP,
        help=f"Number of warmup calls. Default: {DEFAULT_INFERENCE_WARMUP}.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the surrogate model comparison."""
    args = parse_args()
    ensure_dirs()

    batch_sizes = parse_batch_sizes(args.batch_sizes)

    print("\n" + "=" * 70)
    print("SURROGATE MODEL COMPARISON")
    print("=" * 70)

    print("\nCollecting saved model metrics...")
    comparison_df = collect_metrics()

    summary_path = COMPARISON_RESULTS_DIR / "model_comparison_summary.csv"
    comparison_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    report_table = make_report_table(comparison_df)

    report_path = COMPARISON_RESULTS_DIR / "model_comparison_report_table.csv"
    report_table.to_csv(report_path, index=False)
    print(f"Report table saved: {report_path}")

    report_md_path = COMPARISON_RESULTS_DIR / "model_comparison_report_table.md"
    save_table_markdown(report_table, report_md_path)
    print(f"Markdown report table saved: {report_md_path}")

    print("\nMODEL COMPARISON REPORT TABLE")
    print(report_table.to_string(index=False))

    if not args.skip_plots:
        print("\nGenerating comparison figures...")

        plot_peak_y_diagnostic_comparison()
        plot_rmse_comparison(comparison_df)
        plot_r2_comparison(comparison_df)
        plot_constraint_safety(comparison_df)
        plot_near_boundary(comparison_df)
        plot_summary_figure(comparison_df)

    timing_df = pd.DataFrame()

    if not args.skip_timing:
        try:
            timing_df = run_inference_benchmark(
                batch_sizes=batch_sizes,
                repeats=args.timing_repeats,
                warmup=args.timing_warmup,
            )

            if not args.skip_plots:
                plot_inference_time(timing_df)

        except Exception as exc:
            print(f"Inference benchmark skipped: {exc}")

    interpretation_path = write_interpretation(comparison_df, timing_df)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)
    print("Generated files:")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")
    print(f"  - {report_md_path}")
    print(f"  - {interpretation_path}")

    if not args.skip_timing:
        print(f"  - {COMPARISON_RESULTS_DIR / 'inference_time_summary.csv'}")

    if not args.skip_plots:
        print(f"  - {COMPARISON_FIGURES_DIR / 'model_comparison_*.png'}")

    print("\nNext step:")
    print("  Run the constraint-aware inverse optimization scripts.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()