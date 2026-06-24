"""
physical_validation.py
======================

Physical validation of the final inverse-optimization solutions.

This script does NOT run a new optimization. It reloads the final selected
solutions from the completed optimization pipelines and re-simulates them with
the true dynamic simulator in order to generate time-response figures and
consistency checks.

Main idea
---------
For final comparison tables, BO and Random Search are reported as mean over
seeds. However, a mean over seeds is not a physical parameter vector and cannot
be simulated as a trajectory. Therefore, for physical validation trajectories,
this script selects the best feasible seed per target using the same criterion
used elsewhere:

    1. feasible solutions first;
    2. minimum true target error;
    3. if no feasible solution exists, minimum constraint violation;
    4. then minimum target error.

Outputs
-------
results/physical_validation/physical_validation_selected_candidates.csv
results/physical_validation/physical_validation_metrics.csv
results/physical_validation/physical_validation_consistency.csv
results/physical_validation/physical_validation_summary_overall.csv

figures/physical_validation/physical_validation_target0650.png
figures/physical_validation/physical_validation_target0750.png
figures/physical_validation/physical_validation_target0650_nn_only.png
figures/physical_validation/physical_validation_target0750_nn_only.png
figures/physical_validation/physical_validation_target0650_<method>.png
figures/physical_validation/physical_validation_target0750_<method>.png
figures/physical_validation/physical_validation_constraint_margin.png
figures/physical_validation/physical_validation_peak_summary.png
figures/physical_validation/physical_validation_extra_reach.png

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dynamics import simulate_system


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "physical_validation"
FIGURES_DIR = PROJECT_ROOT / "figures" / "physical_validation"

GP_DE_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "optimization_gp_de_v2"
    / "gp_de_results.csv"
)
NN_ADAM_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "optimization_nn_gradient_v2_safe"
    / "nn_gradient_results.csv"
)
BO_RESULTS_BY_SEED_PATH = (
    PROJECT_ROOT
    / "results"
    / "optimization_bo_final"
    / "bo_final_results_by_seed.csv"
)
RS_RESULTS_BY_SEED_PATH = (
    PROJECT_ROOT
    / "results"
    / "optimization_bo_final"
    / "random_search_final_results_by_seed.csv"
)

ROBOT_LIMIT_TRUE = 0.500
FEASIBILITY_TOLERANCE_M = 1e-9

REPRESENTATIVE_TARGETS = (0.65, 0.75)
INCLUDE_RANDOM_SEARCH = True

PARAM_COLUMNS = [
    "Kb",
    "Kr",
    "Mb",
    "hb",
    "hr",
    "f0",
    "f1",
    "A",
    "x_r_start",
    "Mr",
]
OPTIMIZED_COLUMNS = ["Kr", "hr", "f0", "f1", "A", "x_r_start"]
FIXED_DEFAULTS = {"Kb": 1000.0, "Mb": 20.0, "hb": 0.10, "Mr": 10.0}

METHOD_ORDER = [
    "GP+DE beta=0.5",
    "NN+Adam safe",
    "BO best seed",
    "Random Search best seed",
]

METHOD_COLORS = {
    "GP+DE beta=0.5": "#1f77b4",        # blu
    "NN+Adam safe": "#2ca02c",          # verde
    "BO best seed": "#ff7f0e",          # arancione
    "Random Search best seed": "#9467bd",  # viola
}

TARGET_COLORS = {
    0.55: "#1f77b4",   # blu
    0.60: "#17becf",   # ciano
    0.65: "#2ca02c",   # verde
    0.70: "#ff7f0e",   # arancione
    0.75: "#d62728",   # rosso
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run the corresponding optimization script first or check the path."
        )


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def complete_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed physical parameters if only optimized columns are stored."""
    out = df.copy()
    for col, value in FIXED_DEFAULTS.items():
        if col not in out.columns:
            out[col] = float(value)
    return out


def assert_has_parameters(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in PARAM_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} does not contain all simulator parameters. Missing: {missing}\n"
            f"Required columns: {PARAM_COLUMNS}"
        )


def row_to_params(row: pd.Series) -> Dict[str, float]:
    return {col: float(row[col]) for col in PARAM_COLUMNS}


def normalized_target(value: Any) -> float:
    return round(float(value), 6)


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def method_file_tag(method: str) -> str:
    """
    Convert method names into clean file-name tags.

    Examples:
        GP+DE beta=0.5          -> gp_de_beta05
        NN+Adam safe            -> nn_adam_safe
        BO best seed            -> bo_best_seed
        Random Search best seed -> random_search_best_seed
    """
    manual = {
        "GP+DE beta=0.5": "gp_de_beta05",
        "NN+Adam safe": "nn_adam_safe",
        "BO best seed": "bo_best_seed",
        "Random Search best seed": "random_search_best_seed",
    }
    if method in manual:
        return manual[method]

    tag = method.lower().strip()
    replacements = {
        "+": "_",
        " ": "_",
        "=": "",
        ".": "",
        "/": "_",
        "\\": "_",
        "-": "_",
        "β": "beta",
    }
    for old, new in replacements.items():
        tag = tag.replace(old, new)

    tag = "_".join(part for part in tag.split("_") if part)
    return tag

def get_method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def get_target_color(target: float) -> str:
    key = round(float(target), 2)
    return TARGET_COLORS.get(key, "#333333")

# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def selection_sort_key(row: pd.Series) -> Tuple[int, float, float]:
    """
    Select best physical candidate consistently across methods.

    Priority:
        feasible first, then target error, then constraint violation.
    If infeasible, minimize violation before target error.
    """
    feasible = as_bool(row.get("feasible_abs", False))

    target_error_m = float(row.get("target_error_m", np.nan))
    if not np.isfinite(target_error_m):
        target_error_m = float(row.get("target_error_mm", np.inf)) / 1000.0

    violation_m = float(row.get("constraint_violation_abs_m", np.nan))
    if not np.isfinite(violation_m):
        violation_m = float(row.get("constraint_violation_abs_mm", np.inf)) / 1000.0

    return (
        0 if feasible else 1,
        0.0 if feasible else violation_m,
        target_error_m,
    )


def select_best_per_target(df: pd.DataFrame, method_label: str) -> pd.DataFrame:
    """Select one best row per target using the standard feasibility-first rule."""
    if df.empty:
        return pd.DataFrame()

    work = complete_parameter_columns(df)
    assert_has_parameters(work, method_label)

    if "target" not in work.columns:
        raise KeyError(f"{method_label} file does not contain a 'target' column.")

    rows = []
    for target, group in work.groupby(work["target"].map(normalized_target)):
        group = group.copy()
        sorted_idx = sorted(group.index, key=lambda idx: selection_sort_key(group.loc[idx]))
        best = group.loc[sorted_idx[0]].copy()
        best["target"] = float(target)
        best["method"] = method_label
        rows.append(best)

    return pd.DataFrame(rows)


def load_selected_candidates() -> pd.DataFrame:
    """Load selected final solutions from all completed optimizers."""
    frames: List[pd.DataFrame] = []

    require_file(GP_DE_RESULTS_PATH)
    gp_df = pd.read_csv(GP_DE_RESULTS_PATH)
    gp_selected = select_best_per_target(gp_df, "GP+DE beta=0.5")
    frames.append(gp_selected)

    require_file(NN_ADAM_RESULTS_PATH)
    nn_df = pd.read_csv(NN_ADAM_RESULTS_PATH)
    nn_selected = select_best_per_target(nn_df, "NN+Adam safe")
    frames.append(nn_selected)

    require_file(BO_RESULTS_BY_SEED_PATH)
    bo_df = pd.read_csv(BO_RESULTS_BY_SEED_PATH)
    bo_selected = select_best_per_target(bo_df, "BO best seed")
    frames.append(bo_selected)

    if INCLUDE_RANDOM_SEARCH:
        if RS_RESULTS_BY_SEED_PATH.exists():
            rs_df = pd.read_csv(RS_RESULTS_BY_SEED_PATH)
            rs_selected = select_best_per_target(rs_df, "Random Search best seed")
            frames.append(rs_selected)
        else:
            print(f"⚠ Random Search file not found, skipping: {RS_RESULTS_BY_SEED_PATH}")

    selected = pd.concat(frames, ignore_index=True, sort=False)
    selected["target"] = selected["target"].astype(float)
    selected["method"] = selected["method"].astype(str)
    selected["method_order"] = selected["method"].map(method_sort_key)
    selected = selected.sort_values(["target", "method_order"]).reset_index(drop=True)
    selected = selected.drop(columns=["method_order"])

    if "residual_margin_mm" not in selected.columns and "max_abs_xr_true" in selected.columns:
        selected["residual_margin_mm"] = (
            ROBOT_LIMIT_TRUE - selected["max_abs_xr_true"].astype(float)
        ) * 1000.0

    if "constraint_violation_abs_mm" not in selected.columns and "max_abs_xr_true" in selected.columns:
        selected["constraint_violation_abs_mm"] = np.maximum(
            0.0,
            selected["max_abs_xr_true"].astype(float) - ROBOT_LIMIT_TRUE,
        ) * 1000.0

    path = RESULTS_DIR / "physical_validation_selected_candidates.csv"
    selected.to_csv(path, index=False)
    print(f"✓ Saved selected candidates: {path}")

    return selected


# =============================================================================
# TRUE SIMULATOR FULL TRAJECTORY
# =============================================================================

def extract_metrics_from_simulator_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)

    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict) and "peak_y" in item:
                return dict(item)
        if len(output) >= 2 and isinstance(output[1], dict):
            return dict(output[1])

    raise TypeError(
        "Could not extract metrics dictionary from simulate_system output. "
        f"Received type: {type(output)}"
    )


def complete_abs_metrics(
    metrics: Dict[str, Any],
    t: Optional[np.ndarray],
    y: Optional[np.ndarray],
    x_b: Optional[np.ndarray],
    x_r: Optional[np.ndarray],
) -> Dict[str, Any]:
    completed = dict(metrics)

    if "peak_y" not in completed:
        if y is None:
            raise KeyError(
                "Simulator metrics do not contain 'peak_y' and trajectory y is unavailable."
            )
        completed["peak_y"] = float(np.max(y))

    if "max_abs_xr" not in completed:
        if x_r is not None:
            completed["max_abs_xr"] = float(np.max(np.abs(x_r)))
        elif "max_xr" in completed and "min_xr" in completed:
            completed["max_abs_xr"] = max(
                abs(float(completed["max_xr"])),
                abs(float(completed["min_xr"])),
            )
        elif "max_xr" in completed:
            completed["max_abs_xr"] = abs(float(completed["max_xr"]))
        else:
            raise KeyError("Simulator metrics do not contain 'max_abs_xr' or x_r trajectory.")

    if "max_abs_xb" not in completed:
        if x_b is not None:
            completed["max_abs_xb"] = float(np.max(np.abs(x_b)))
        elif "max_xb" in completed and "min_xb" in completed:
            completed["max_abs_xb"] = max(
                abs(float(completed["max_xb"])),
                abs(float(completed["min_xb"])),
            )
        elif "max_xb" in completed:
            completed["max_abs_xb"] = abs(float(completed["max_xb"]))
        else:
            completed["max_abs_xb"] = np.nan

    completed["constraint_violation_abs"] = max(
        0.0,
        float(completed["max_abs_xr"]) - ROBOT_LIMIT_TRUE,
    )
    completed["constraint_violation_abs_mm"] = (
        completed["constraint_violation_abs"] * 1000.0
    )
    completed["feasible_abs"] = (
        completed["constraint_violation_abs"] <= FEASIBILITY_TOLERANCE_M
    )
    completed["residual_margin_mm"] = (
        ROBOT_LIMIT_TRUE - float(completed["max_abs_xr"])
    ) * 1000.0
    completed["extra_reach"] = float(completed["peak_y"]) - ROBOT_LIMIT_TRUE

    if x_r is not None:
        completed["max_xr"] = float(np.max(x_r))
        completed["min_xr"] = float(np.min(x_r))
    if x_b is not None:
        completed["max_xb"] = float(np.max(x_b))
        completed["min_xb"] = float(np.min(x_b))
    if y is not None:
        completed["min_y"] = float(np.min(y))

    return completed


def simulate_full_response(
    params: Dict[str, float],
    y_target: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run the true simulator and return t, y, x_b, x_r, metrics."""
    try:
        output = simulate_system(
            params,
            y_target=float(y_target),
            x_r_max=ROBOT_LIMIT_TRUE,
            return_metrics=True,
            return_full=True,
        )
    except TypeError:
        output = simulate_system(params, return_metrics=True, return_full=True)

    if not isinstance(output, tuple):
        raise TypeError(
            "simulate_system did not return a tuple with the full solution. "
            "Make sure dynamics.py supports return_full=True."
        )

    sol = None
    metrics = None

    for item in output:
        if hasattr(item, "t") and hasattr(item, "y"):
            sol = item
        elif isinstance(item, dict) and "peak_y" in item:
            metrics = dict(item)

    if sol is None:
        raise TypeError("Could not find a scipy-like solution object with .t and .y.")

    t = np.asarray(sol.t, dtype=float)

    try:
        x_b = np.asarray(sol.y[2], dtype=float)
        x_r = np.asarray(sol.y[3], dtype=float)
    except Exception as exc:
        raise RuntimeError(
            "Could not extract x_b=sol.y[2] and x_r=sol.y[3]. "
            "Check the state ordering in dynamics.py."
        ) from exc

    y = x_b + x_r

    if metrics is None:
        metrics = extract_metrics_from_simulator_output(output)

    completed = complete_abs_metrics(metrics, t, y, x_b, x_r)
    return t, y, x_b, x_r, completed


# =============================================================================
# VALIDATION RUN
# =============================================================================

def validate_all_candidates(
    selected: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
]:
    """Re-simulate all selected candidates and save metrics/consistency rows."""
    metric_rows: List[Dict[str, Any]] = []
    consistency_rows: List[Dict[str, Any]] = []
    trajectories: Dict[
        Tuple[str, float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ] = {}

    print("\nRe-simulating selected candidates with true dynamic simulator...")

    for _, row in selected.iterrows():
        method = str(row["method"])
        target = float(row["target"])
        params = row_to_params(row)

        t, y, x_b, x_r, metrics = simulate_full_response(params, target)
        trajectories[(method, round(target, 6))] = (t, y, x_b, x_r, metrics)

        peak_y = float(metrics["peak_y"])
        max_abs_xr = float(metrics["max_abs_xr"])
        target_error_mm = abs(peak_y - target) * 1000.0
        residual_margin_mm = (ROBOT_LIMIT_TRUE - max_abs_xr) * 1000.0
        violation_mm = max(0.0, max_abs_xr - ROBOT_LIMIT_TRUE) * 1000.0

        metric_row = {
            "method": method,
            "target": target,
            "peak_y_recomputed": peak_y,
            "target_error_recomputed_mm": target_error_mm,
            "max_abs_xr_recomputed": max_abs_xr,
            "max_abs_xb_recomputed": float(metrics.get("max_abs_xb", np.nan)),
            "residual_margin_recomputed_mm": residual_margin_mm,
            "constraint_violation_recomputed_mm": violation_mm,
            "feasible_recomputed_abs": bool(metrics["feasible_abs"]),
            "extra_reach_recomputed_m": float(metrics.get("extra_reach", np.nan)),
            "max_xr_recomputed": float(metrics.get("max_xr", np.nan)),
            "min_xr_recomputed": float(metrics.get("min_xr", np.nan)),
            "max_xb_recomputed": float(metrics.get("max_xb", np.nan)),
            "min_xb_recomputed": float(metrics.get("min_xb", np.nan)),
        }

        for col in PARAM_COLUMNS:
            metric_row[col] = params[col]

        metric_rows.append(metric_row)

        stored_peak = (
            float(row["peak_y_true"])
            if "peak_y_true" in row and pd.notna(row["peak_y_true"])
            else np.nan
        )
        stored_xr = (
            float(row["max_abs_xr_true"])
            if "max_abs_xr_true" in row and pd.notna(row["max_abs_xr_true"])
            else np.nan
        )
        stored_error = (
            float(row["target_error_mm"])
            if "target_error_mm" in row and pd.notna(row["target_error_mm"])
            else np.nan
        )
        stored_margin = (
            float(row["residual_margin_mm"])
            if "residual_margin_mm" in row and pd.notna(row["residual_margin_mm"])
            else np.nan
        )

        consistency_rows.append(
            {
                "method": method,
                "target": target,
                "stored_peak_y_true": stored_peak,
                "recomputed_peak_y": peak_y,
                "peak_y_diff_mm": (peak_y - stored_peak) * 1000.0
                if np.isfinite(stored_peak)
                else np.nan,
                "stored_max_abs_xr_true": stored_xr,
                "recomputed_max_abs_xr": max_abs_xr,
                "max_abs_xr_diff_mm": (max_abs_xr - stored_xr) * 1000.0
                if np.isfinite(stored_xr)
                else np.nan,
                "stored_target_error_mm": stored_error,
                "recomputed_target_error_mm": target_error_mm,
                "target_error_diff_mm": target_error_mm - stored_error
                if np.isfinite(stored_error)
                else np.nan,
                "stored_residual_margin_mm": stored_margin,
                "recomputed_residual_margin_mm": residual_margin_mm,
                "residual_margin_diff_mm": residual_margin_mm - stored_margin
                if np.isfinite(stored_margin)
                else np.nan,
            }
        )

        print(
            f"  {method:24s} target={target:.2f} | "
            f"peak_y={peak_y:.4f} m | err={target_error_mm:.2f} mm | "
            f"max_abs_xr={max_abs_xr:.4f} m | margin={residual_margin_mm:.2f} mm | "
            f"feasible={bool(metrics['feasible_abs'])}"
        )

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(["target", "method"])
        .reset_index(drop=True)
    )
    consistency_df = (
        pd.DataFrame(consistency_rows)
        .sort_values(["target", "method"])
        .reset_index(drop=True)
    )

    metrics_path = RESULTS_DIR / "physical_validation_metrics.csv"
    consistency_path = RESULTS_DIR / "physical_validation_consistency.csv"

    metrics_df.to_csv(metrics_path, index=False)
    consistency_df.to_csv(consistency_path, index=False)

    print(f"✓ Saved metrics: {metrics_path}")
    print(f"✓ Saved consistency check: {consistency_path}")

    return metrics_df, trajectories


# =============================================================================
# PLOTS
# =============================================================================

def available_methods_for_target(
    trajectories: Dict[
        Tuple[str, float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ],
    target: float,
) -> List[str]:
    target_key = round(float(target), 6)
    methods = [method for (method, t) in trajectories.keys() if np.isclose(t, target_key)]
    return sorted(methods, key=method_sort_key)


def plot_time_response(
    target: float,
    trajectories: Dict[
        Tuple[str, float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ],
    methods_to_plot: Sequence[str],
    output_path: Path,
    title: str,
    show_method_in_label: bool = True,
) -> None:
    """Plot y(t), x_r(t), and x_b(t) for selected methods."""
    target_key = round(float(target), 6)

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.5), sharex=True)

    plotted_any = False

    for method in methods_to_plot:
        key = (method, target_key)
        if key not in trajectories:
            print(f"⚠ Missing trajectory for {method}, target={target:.3f}")
            continue

        t, y, x_b, x_r, metrics = trajectories[key]

        peak_y = float(metrics["peak_y"])
        error_mm = abs(peak_y - target) * 1000.0

        if show_method_in_label:
            y_label = f"{method} (err={error_mm:.1f} mm)"
            xr_label = method
            xb_label = method
        else:
            y_label = f"y(t), err={error_mm:.1f} mm"
            xr_label = "x_r(t)"
            xb_label = "x_b(t)"

        axes[0].plot(t, y, linewidth=2.0, label=y_label)
        axes[1].plot(t, x_r, linewidth=2.0, label=xr_label)
        axes[2].plot(t, x_b, linewidth=2.0, label=xb_label)

        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    axes[0].axhline(
        target,
        linestyle=":",
        linewidth=2.2,
        label=f"Target {target:.2f} m",
    )
    axes[0].axhline(
        ROBOT_LIMIT_TRUE,
        linestyle="--",
        linewidth=1.8,
        label="Nominal robot reach 0.500 m",
    )

    axes[1].axhline(
        ROBOT_LIMIT_TRUE,
        linestyle="--",
        linewidth=2.0,
        label="+ robot limit",
    )
    axes[1].axhline(
        -ROBOT_LIMIT_TRUE,
        linestyle="--",
        linewidth=2.0,
        label="- robot limit",
    )

    axes[2].axhline(
        0.0,
        linestyle=":",
        linewidth=1.5,
        label="Zero reference",
    )

    axes[0].set_ylabel("y(t) = x_b + x_r [m]")
    axes[1].set_ylabel("x_r(t) [m]")
    axes[2].set_ylabel("x_b(t) [m]")
    axes[2].set_xlabel("Time [s]")

    axes[0].set_title(title)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_time_response_for_target(
    target: float,
    trajectories: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
) -> None:
    """Overlay y(t), x_r(t), x_b(t) for all methods for one target."""
    methods = available_methods_for_target(trajectories, target)
    if not methods:
        print(f"⚠ No trajectories available for target {target:.3f}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 11), sharex=True)

    for method in methods:
        t, y, x_b, x_r, metrics = trajectories[(method, round(float(target), 6))]
        color = get_method_color(method)
        label = f"{method} (err={abs(float(metrics['peak_y']) - target) * 1000.0:.1f} mm)"

        axes[0].plot(t, y, linewidth=2.2, color=color, label=label)
        axes[1].plot(t, x_r, linewidth=2.2, color=color, label=method)
        axes[2].plot(t, x_b, linewidth=2.2, color=color, label=method)

    # Target and limits
    axes[0].axhline(target, linestyle=":", linewidth=2.2, color="black", label=f"Target {target:.2f} m")
    axes[0].axhline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=1.8, color="gray", label="Nominal robot reach 0.500 m")
    axes[1].axhline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=2.0, color="gray", label="+ robot limit")
    axes[1].axhline(-ROBOT_LIMIT_TRUE, linestyle="--", linewidth=2.0, color="gray", label="- robot limit")
    axes[2].axhline(0.0, linestyle=":", linewidth=1.5, color="gray", label="Zero reference")

    axes[0].set_ylabel("y(t) = x_b + x_r [m]")
    axes[1].set_ylabel("x_r(t) [m]")
    axes[2].set_ylabel("x_b(t) [m]")
    axes[2].set_xlabel("Time [s]")

    axes[0].set_title(
        f"Physical validation time response, target = {target:.2f} m\n"
        "BO and Random Search trajectories use the best feasible seed, not mean-over-seeds values."
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    tag = int(round(target * 1000))
    path = FIGURES_DIR / f"physical_validation_target{tag:04d}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_time_response_single_method(
    target: float,
    method: str,
    trajectories: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
) -> None:
    key = (method, round(float(target), 6))
    if key not in trajectories:
        print(f"⚠ Missing trajectory for {method}, target={target:.3f}")
        return

    t, y, x_b, x_r, metrics = trajectories[key]
    color = get_method_color(method)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)

    axes[0].plot(t, y, linewidth=2.4, color=color, label=method)
    axes[1].plot(t, x_r, linewidth=2.4, color=color, label=method)
    axes[2].plot(t, x_b, linewidth=2.4, color=color, label=method)

    axes[0].axhline(target, linestyle=":", linewidth=2.0, color="black", label=f"Target {target:.2f} m")
    axes[0].axhline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=1.6, color="gray", label="Nominal robot reach 0.500 m")
    axes[1].axhline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=1.8, color="gray", label="+ robot limit")
    axes[1].axhline(-ROBOT_LIMIT_TRUE, linestyle="--", linewidth=1.8, color="gray", label="- robot limit")
    axes[2].axhline(0.0, linestyle=":", linewidth=1.4, color="gray")

    axes[0].set_ylabel("y(t) [m]")
    axes[1].set_ylabel("x_r(t) [m]")
    axes[2].set_ylabel("x_b(t) [m]")
    axes[2].set_xlabel("Time [s]")

    err_mm = abs(float(metrics["peak_y"]) - target) * 1000.0
    margin_mm = (ROBOT_LIMIT_TRUE - float(metrics["max_abs_xr"])) * 1000.0

    axes[0].set_title(
        f"{method} - physical validation, target = {target:.2f} m\n"
        f"target error = {err_mm:.2f} mm, residual margin = {margin_mm:.2f} mm"
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    tag = int(round(target * 1000))
    safe_method = method_file_tag(method)
    path = FIGURES_DIR / f"physical_validation_target{tag:04d}_{safe_method}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_peak_summary(metrics_df: pd.DataFrame) -> None:
    df = metrics_df.copy()
    methods = sorted(df["method"].unique(), key=method_sort_key)

    plt.figure(figsize=(9, 6))
    targets = sorted(df["target"].unique())

    plt.plot(targets, targets, linestyle="--", linewidth=2.0, label="Ideal tracking")
    plt.axhline(
        ROBOT_LIMIT_TRUE,
        linestyle=":",
        linewidth=1.8,
        label="Nominal robot reach 0.500 m",
    )

    for method in methods:
        sub = df[df["method"] == method].sort_values("target")
        plt.plot(
            sub["target"],
            sub["peak_y_recomputed"],
            marker="o",
            linewidth=2.0,
            label=method,
        )

    plt.xlabel("Target outreach [m]")
    plt.ylabel("Recomputed true peak_y [m]")
    plt.title("Physical validation peak outreach summary")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "physical_validation_peak_summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {path}")


def plot_constraint_margin(metrics_df: pd.DataFrame) -> None:
    df = metrics_df.copy()
    methods = sorted(df["method"].unique(), key=method_sort_key)
    targets = sorted(df["target"].unique())

    pivot = df.pivot_table(
        index="target",
        columns="method",
        values="residual_margin_recomputed_mm",
        aggfunc="mean",
    ).reindex(index=targets, columns=methods)

    x = np.arange(len(targets))
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)

    plt.figure(figsize=(10, 6))

    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2.0) * width
        values = pivot[method].to_numpy(dtype=float)

        plt.bar(
            x + offset,
            values,
            width=width,
            label=method,
            edgecolor="black",
            linewidth=0.5,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.5, label="Constraint boundary")
    plt.xticks(x, [f"{t:.2f}" for t in targets])
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Residual margin to true robot limit [mm]")
    plt.title("Physical validation residual constraint margin")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "physical_validation_constraint_margin.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {path}")

# =============================================================================
# NORMALIZED PARAMETER COMPARISON
# =============================================================================

PARAM_BOUNDS = {
    "Kr": (1500.0, 5000.0),
    "hr": (0.10, 0.45),
    "f0": (0.10, 0.45),
    "f1": (1.00, 4.00),
    "A": (0.09, 0.12),
    "x_r_start": (0.35, 0.40),
}

PARAM_LABELS = {
    "Kr": r"$K_r$",
    "hr": r"$h_r$",
    "f0": r"$f_0$",
    "f1": r"$f_1$",
    "A": r"$A$",
    "x_r_start": r"$x_{r,\mathrm{start}}$",
}

METHOD_COLORS = {
    "GP+DE beta=0.5": "#1f77b4",          # blue
    "NN+Adam safe": "#2ca02c",            # green
    "BO best seed": "#ff7f0e",            # orange
    "Random Search best seed": "#9467bd", # purple
}

METHOD_MARKERS = {
    "GP+DE beta=0.5": "o",
    "NN+Adam safe": "s",
    "BO best seed": "^",
    "Random Search best seed": "D",
}


def normalize_parameter(value: float, lower: float, upper: float) -> float:
    """Normalize a parameter value to [0, 1] using its optimization bounds."""
    if upper <= lower:
        return np.nan
    return (float(value) - lower) / (upper - lower)


def plot_normalized_parameter_comparison(metrics_df: pd.DataFrame) -> None:
    """
    Plot normalized optimized controllable parameters for all validated methods.

    Each subplot corresponds to one optimized parameter.
    Each line corresponds to one optimizer.
    Values close to 0 or 1 indicate parameters close to their lower or upper bounds.
    """
    df = metrics_df.copy()

    required_cols = ["method", "target"] + list(PARAM_BOUNDS.keys())
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Cannot plot normalized parameter comparison. "
            f"Missing columns: {missing}"
        )

    methods = sorted(df["method"].unique(), key=method_sort_key)
    targets = sorted(df["target"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, param in zip(axes, PARAM_BOUNDS.keys()):
        lower, upper = PARAM_BOUNDS[param]

        for method in methods:
            sub = df[df["method"] == method].sort_values("target")
            x = sub["target"].to_numpy(dtype=float)
            y = np.array(
                [
                    normalize_parameter(v, lower, upper)
                    for v in sub[param].to_numpy(dtype=float)
                ]
            )

            ax.plot(
                x,
                y,
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2.0,
                markersize=5.5,
                color=METHOD_COLORS.get(method, None),
                label=method,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
        ax.axhline(1.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
        ax.set_title(PARAM_LABELS[param])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    for ax in axes[3:]:
        ax.set_xlabel("Target outreach [m]")

    axes[0].set_ylabel("Normalized value")
    axes[3].set_ylabel("Normalized value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Normalized optimized controllable parameters for all validated methods",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    path = FIGURES_DIR / "physical_validation_normalized_parameters_all_methods.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {path}")


def plot_extra_reach_summary(metrics_df: pd.DataFrame) -> None:
    """Optional plot showing base-assisted extra reach above nominal 0.5 m."""
    df = metrics_df.copy()
    methods = sorted(df["method"].unique(), key=method_sort_key)
    targets = sorted(df["target"].unique())

    plt.figure(figsize=(9, 6))

    for method in methods:
        sub = df[df["method"] == method].sort_values("target")
        plt.plot(
            sub["target"],
            sub["extra_reach_recomputed_m"] * 1000.0,
            marker="o",
            linewidth=2.0,
            label=method,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Extra reach over nominal robot reach [mm]")
    plt.title("Physical validation extra reach from compliant-base dynamics")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "physical_validation_extra_reach.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {path}")


def generate_plots(
    metrics_df: pd.DataFrame,
    trajectories: Dict[
        Tuple[str, float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ],
) -> None:
    print("\nGenerating physical validation plots...")

    methods = sorted(metrics_df["method"].unique(), key=method_sort_key)

    for target in REPRESENTATIVE_TARGETS:
        tag = int(round(target * 1000))

        # Combined plot with all methods
        plot_time_response_for_target(target, trajectories)

        # Single plot for each method
        for method in methods:
            plot_time_response_single_method(target, method, trajectories)

        # NN-only alias for report
        if "NN+Adam safe" in methods:
            key = ("NN+Adam safe", round(float(target), 6))
            if key in trajectories:
                _, _, _, _, metrics = trajectories[key]

                peak_y = float(metrics["peak_y"])
                err_mm = abs(peak_y - target) * 1000.0
                margin_mm = (ROBOT_LIMIT_TRUE - float(metrics["max_abs_xr"])) * 1000.0

                nn_title = (
                    f"Physical validation time response, target = {target:.2f} m\n"
                    f"NN+Adam safe: peak_y = {peak_y:.4f} m, "
                    f"error = {err_mm:.1f} mm, margin = {margin_mm:.1f} mm"
                )

                plot_time_response(
                    target=target,
                    trajectories=trajectories,
                    methods_to_plot=["NN+Adam safe"],
                    output_path=FIGURES_DIR / f"physical_validation_target{tag:04d}_nn_only.png",
                    title=nn_title,
                    show_method_in_label=False,
                )

    plot_peak_summary(metrics_df)
    plot_constraint_margin(metrics_df)
    plot_extra_reach_summary(metrics_df)
    plot_normalized_parameter_comparison(metrics_df)
    
# =============================================================================
# MAIN
# =============================================================================

def print_summary(metrics_df: pd.DataFrame) -> None:
    display_cols = [
        "method",
        "target",
        "peak_y_recomputed",
        "target_error_recomputed_mm",
        "max_abs_xr_recomputed",
        "residual_margin_recomputed_mm",
        "constraint_violation_recomputed_mm",
        "feasible_recomputed_abs",
    ]

    print("\n" + "=" * 80)
    print("PHYSICAL VALIDATION SUMMARY")
    print("=" * 80)
    print(metrics_df[display_cols].to_string(index=False))

    overall = (
        metrics_df.groupby("method")
        .agg(
            feasible_rate_percent=(
                "feasible_recomputed_abs",
                lambda s: 100.0 * s.astype(bool).mean(),
            ),
            mean_target_error_mm=("target_error_recomputed_mm", "mean"),
            max_target_error_mm=("target_error_recomputed_mm", "max"),
            mean_residual_margin_mm=("residual_margin_recomputed_mm", "mean"),
            min_residual_margin_mm=("residual_margin_recomputed_mm", "min"),
            max_abs_xr_m=("max_abs_xr_recomputed", "max"),
        )
        .reset_index()
    )

    overall["method_order"] = overall["method"].map(method_sort_key)
    overall = overall.sort_values("method_order").drop(columns="method_order")

    overall_path = RESULTS_DIR / "physical_validation_summary_overall.csv"
    overall.to_csv(overall_path, index=False)

    print("\nOverall:")
    print(overall.to_string(index=False))
    print(f"\n✓ Saved overall summary: {overall_path}")
    print("=" * 80 + "\n")


def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("PHYSICAL VALIDATION OF FINAL OPTIMIZATION SOLUTIONS")
    print("=" * 80)
    print("Representative time-response targets:", REPRESENTATIVE_TARGETS)
    print("BO/Random Search trajectory selection: best feasible seed per target")
    print("True robot displacement limit: ±0.500 m")
    print("=" * 80)

    selected = load_selected_candidates()
    metrics_df, trajectories = validate_all_candidates(selected)

    generate_plots(metrics_df, trajectories)
    print_summary(metrics_df)

    print("Saved result files:")
    for path in sorted(RESULTS_DIR.glob("physical_validation_*.csv")):
        print(f"  {path}")

    print("\nSaved figures:")
    for path in sorted(FIGURES_DIR.glob("physical_validation_*.png")):
        print(f"  {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()