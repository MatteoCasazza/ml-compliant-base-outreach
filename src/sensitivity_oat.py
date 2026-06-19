"""
sensitivity_oat.py
==================

One-at-a-time sensitivity analysis around final optimized solutions.

This script is intended to be run after:
    - final optimization comparison
    - physical_validation.py

It does NOT run a new optimization. It takes one selected final solution,
perturbs one parameter at a time, and re-simulates the true dynamic system.

Default reference solution
--------------------------
    Method:  NN+Adam safe
    Targets: 0.65 m and 0.75 m

Rationale
---------
The target 0.65 m represents a reachable case with very small tracking error.
The target 0.75 m represents a saturated high-outreach case where the robot
workspace constraint limits the achievable outreach.

Perturbations
-------------
Controllable optimized parameters are varied by +/- 5% and +/- 10% of their
allowed optimization range:
    Kr, hr, f0, f1, A, x_r_start

Physical/model parameters are varied by +/- 5% and +/- 10% relative to their
nominal value:
    Kb, Mb, hb, Mr

Outputs
-------
results/sensitivity_oat/sensitivity_oat_results.csv
results/sensitivity_oat/sensitivity_oat_summary.csv
results/sensitivity_oat/sensitivity_oat_nominal_metrics.csv

figures/sensitivity_oat/sensitivity_peak_y_tornado_target0650.png
figures/sensitivity_oat/sensitivity_max_abs_xr_tornado_target0650.png
figures/sensitivity_oat/sensitivity_residual_margin_tornado_target0650.png
figures/sensitivity_oat/sensitivity_curves_peak_y_target0650.png
figures/sensitivity_oat/sensitivity_curves_max_abs_xr_target0650.png
figures/sensitivity_oat/sensitivity_curves_max_abs_xr_absolute_target0650.png
figures/sensitivity_oat/sensitivity_curves_residual_margin_target0650.png

and the same set for target0750.

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dynamics import simulate_system


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "sensitivity_oat"
FIGURES_DIR = PROJECT_ROOT / "figures" / "sensitivity_oat"

# Prefer the candidates already selected by physical_validation.py, because that
# file consistently contains the final physical parameter vectors for all methods.
PHYSICAL_VALIDATION_SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "physical_validation"
    / "physical_validation_selected_candidates.csv"
)

# Fallback if physical_validation.py has not been run yet.
NN_ADAM_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "optimization_nn_gradient_v2_safe"
    / "nn_gradient_results.csv"
)

REFERENCE_METHOD = "NN+Adam safe"
TARGETS_TO_ANALYZE = (0.65, 0.75)

ROBOT_LIMIT_TRUE = 0.500
FEASIBILITY_TOLERANCE_M = 1e-9

PARAM_COLUMNS = ["Kb", "Kr", "Mb", "hb", "hr", "f0", "f1", "A", "x_r_start", "Mr"]
FIXED_DEFAULTS = {"Kb": 1000.0, "Mb": 20.0, "hb": 0.10, "Mr": 10.0}

CONTROLLABLE_PARAMS = ["Kr", "hr", "f0", "f1", "A", "x_r_start"]
PHYSICAL_PARAMS = ["Kb", "Mb", "hb", "Mr"]
PARAMETERS_TO_ANALYZE = CONTROLLABLE_PARAMS + PHYSICAL_PARAMS

# Optimization bounds used throughout the project.
CONTROLLABLE_BOUNDS = {
    "Kr": (1500.0, 5000.0),
    "hr": (0.10, 0.45),
    "f0": (0.10, 0.45),
    "f1": (1.00, 4.00),
    "A": (0.09, 0.12),
    "x_r_start": (0.35, 0.40),
}

# +/- 10%, +/- 5%, nominal. For controllable parameters this is interpreted as
# percent of allowed range. For physical parameters this is interpreted as
# relative percent of the nominal value.
PERTURBATION_LEVELS = (-0.10, -0.05, 0.0, 0.05, 0.10)

# Plot order: from most directly excitation-related to physical/model params.
PARAMETER_DISPLAY_ORDER = [
    "A",
    "f0",
    "f1",
    "x_r_start",
    "Kr",
    "hr",
    "Kb",
    "Mb",
    "hb",
    "Mr",
]


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def target_tag(target: float) -> str:
    return f"target{int(round(float(target) * 1000)):04d}"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run the previous pipeline step first or check the path."
        )


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def complete_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def selection_sort_key(row: pd.Series) -> Tuple[int, float, float]:
    """Feasible first, then target error; if infeasible, minimize violation."""
    feasible = as_bool(row.get("feasible_abs", False))

    target_error_m = row.get("target_error_m", np.nan)
    if not np.isfinite(float(target_error_m)):
        target_error_m = float(row.get("target_error_mm", np.inf)) / 1000.0

    violation_m = row.get("constraint_violation_abs_m", np.nan)
    if not np.isfinite(float(violation_m)):
        violation_m = float(row.get("constraint_violation_abs_mm", np.inf)) / 1000.0

    return (
        0 if feasible else 1,
        0.0 if feasible else float(violation_m),
        float(target_error_m),
    )


# =============================================================================
# LOAD REFERENCE SOLUTIONS
# =============================================================================

def load_reference_solutions() -> pd.DataFrame:
    """Load one final selected solution per target for the reference method."""
    if PHYSICAL_VALIDATION_SELECTED_PATH.exists():
        df = pd.read_csv(PHYSICAL_VALIDATION_SELECTED_PATH)
        source = PHYSICAL_VALIDATION_SELECTED_PATH
    else:
        require_file(NN_ADAM_RESULTS_PATH)
        df = pd.read_csv(NN_ADAM_RESULTS_PATH)
        if "method" not in df.columns:
            df["method"] = REFERENCE_METHOD
        source = NN_ADAM_RESULTS_PATH

    df = complete_parameter_columns(df)
    assert_has_parameters(df, str(source))

    if "method" in df.columns:
        df = df[df["method"].astype(str) == REFERENCE_METHOD].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for REFERENCE_METHOD={REFERENCE_METHOD!r} in {source}."
        )

    rows = []
    for target in TARGETS_TO_ANALYZE:
        group = df[np.isclose(df["target"].astype(float), float(target))].copy()
        if group.empty:
            raise ValueError(f"No reference solution found for target={target:.3f}.")
        sorted_idx = sorted(group.index, key=lambda idx: selection_sort_key(group.loc[idx]))
        best = group.loc[sorted_idx[0]].copy()
        best["target"] = float(target)
        best["method"] = REFERENCE_METHOD
        rows.append(best)

    selected = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)

    path = RESULTS_DIR / "sensitivity_oat_reference_solutions.csv"
    selected.to_csv(path, index=False)
    print(f"✓ Saved reference solutions: {path}")

    return selected


# =============================================================================
# SIMULATOR HELPERS
# =============================================================================

def _extract_metrics_from_simulator_output(output: Any) -> Dict[str, Any]:
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


def _complete_metrics(metrics: Dict[str, Any], target: float) -> Dict[str, Any]:
    completed = dict(metrics)

    if "peak_y" not in completed:
        raise KeyError("Simulator metrics do not contain 'peak_y'.")

    if "max_abs_xr" not in completed:
        if "max_xr" in completed and "min_xr" in completed:
            completed["max_abs_xr"] = max(
                abs(float(completed["max_xr"])),
                abs(float(completed["min_xr"])),
            )
        elif "max_xr" in completed:
            completed["max_abs_xr"] = abs(float(completed["max_xr"]))
        else:
            raise KeyError(
                "Simulator metrics do not contain 'max_abs_xr' or usable max_xr/min_xr."
            )

    if "max_abs_xb" not in completed:
        if "max_xb" in completed and "min_xb" in completed:
            completed["max_abs_xb"] = max(
                abs(float(completed["max_xb"])),
                abs(float(completed["min_xb"])),
            )
        elif "max_xb" in completed:
            completed["max_abs_xb"] = abs(float(completed["max_xb"]))
        else:
            completed["max_abs_xb"] = np.nan

    peak_y = float(completed["peak_y"])
    max_abs_xr = float(completed["max_abs_xr"])

    completed["target_error_m"] = abs(peak_y - float(target))
    completed["target_error_mm"] = completed["target_error_m"] * 1000.0
    completed["residual_margin_m"] = ROBOT_LIMIT_TRUE - max_abs_xr
    completed["residual_margin_mm"] = completed["residual_margin_m"] * 1000.0
    completed["constraint_violation_abs_m"] = max(0.0, max_abs_xr - ROBOT_LIMIT_TRUE)
    completed["constraint_violation_abs_mm"] = completed["constraint_violation_abs_m"] * 1000.0
    completed["feasible_abs"] = completed["constraint_violation_abs_m"] <= FEASIBILITY_TOLERANCE_M
    completed["extra_reach"] = peak_y - ROBOT_LIMIT_TRUE

    return completed


_SIM_CACHE: Dict[Tuple[float, Tuple[Tuple[str, float], ...]], Dict[str, Any]] = {}


def simulate_candidate(params: Dict[str, float], target: float) -> Dict[str, Any]:
    """Run true simulator with caching to avoid duplicate nominal simulations."""
    key = (
        round(float(target), 8),
        tuple((name, round(float(params[name]), 12)) for name in PARAM_COLUMNS),
    )
    if key in _SIM_CACHE:
        return dict(_SIM_CACHE[key])

    try:
        output = simulate_system(
            params,
            y_target=float(target),
            x_r_max=ROBOT_LIMIT_TRUE,
            return_metrics=True,
        )
    except TypeError:
        try:
            output = simulate_system(params, return_metrics=True)
        except TypeError:
            # Very old interface fallback. It may only return peak_y, so this is
            # intentionally strict because this analysis needs max_abs_xr.
            output = simulate_system(params)

    metrics = _extract_metrics_from_simulator_output(output)
    metrics = _complete_metrics(metrics, target=float(target))
    _SIM_CACHE[key] = dict(metrics)
    return metrics


# =============================================================================
# PERTURBATION LOGIC
# =============================================================================

def perturb_parameter(
    nominal_params: Dict[str, float],
    parameter: str,
    perturbation_level: float,
) -> Tuple[Dict[str, float], float, bool, str]:
    """
    Return perturbed params, tested value, whether clipping occurred, and mode.

    Controllable parameters are perturbed by percentage of allowed optimization
    range. Physical parameters are perturbed by relative percentage of nominal
    value.
    """
    params = dict(nominal_params)
    nominal_value = float(nominal_params[parameter])

    if parameter in CONTROLLABLE_BOUNDS:
        low, high = CONTROLLABLE_BOUNDS[parameter]
        raw_value = nominal_value + float(perturbation_level) * (high - low)
        tested_value = min(max(raw_value, low), high)
        clipped = not np.isclose(raw_value, tested_value)
        mode = "percent_of_allowed_range"
    else:
        raw_value = nominal_value * (1.0 + float(perturbation_level))
        tested_value = raw_value
        clipped = False
        mode = "relative_percent_of_nominal"

    params[parameter] = float(tested_value)
    return params, float(tested_value), bool(clipped), mode


# =============================================================================
# MAIN SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity(reference_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: List[Dict[str, Any]] = []
    nominal_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("ONE-AT-A-TIME SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Reference method: {REFERENCE_METHOD}")
    print(f"Targets:          {TARGETS_TO_ANALYZE}")
    print(f"Parameters:       {PARAMETERS_TO_ANALYZE}")
    print(f"Perturbations:    {[int(100*p) for p in PERTURBATION_LEVELS]}%")
    print("=" * 80)

    for _, ref_row in reference_df.iterrows():
        target = float(ref_row["target"])
        nominal_params = row_to_params(ref_row)
        nominal_metrics = simulate_candidate(nominal_params, target)

        print(
            f"\nNominal {REFERENCE_METHOD}, target={target:.2f}: "
            f"peak_y={nominal_metrics['peak_y']:.4f} m | "
            f"err={nominal_metrics['target_error_mm']:.2f} mm | "
            f"max_abs_xr={nominal_metrics['max_abs_xr']:.4f} m | "
            f"margin={nominal_metrics['residual_margin_mm']:.2f} mm | "
            f"feasible={nominal_metrics['feasible_abs']}"
        )

        nominal_record = {
            "method": REFERENCE_METHOD,
            "target": target,
            **{f"nominal_{p}": nominal_params[p] for p in PARAM_COLUMNS},
            "nominal_peak_y": float(nominal_metrics["peak_y"]),
            "nominal_target_error_mm": float(nominal_metrics["target_error_mm"]),
            "nominal_max_abs_xr": float(nominal_metrics["max_abs_xr"]),
            "nominal_residual_margin_mm": float(nominal_metrics["residual_margin_mm"]),
            "nominal_feasible_abs": bool(nominal_metrics["feasible_abs"]),
        }
        nominal_rows.append(nominal_record)

        for parameter in PARAMETERS_TO_ANALYZE:
            parameter_group = "controllable" if parameter in CONTROLLABLE_PARAMS else "physical_model"
            nominal_value = nominal_params[parameter]

            print(f"  Varying {parameter:10s} ({parameter_group})")

            for perturbation in PERTURBATION_LEVELS:
                perturbed_params, tested_value, clipped, mode = perturb_parameter(
                    nominal_params=nominal_params,
                    parameter=parameter,
                    perturbation_level=perturbation,
                )
                metrics = simulate_candidate(perturbed_params, target)

                peak_y = float(metrics["peak_y"])
                max_abs_xr = float(metrics["max_abs_xr"])

                row = {
                    "method": REFERENCE_METHOD,
                    "target": target,
                    "parameter": parameter,
                    "parameter_group": parameter_group,
                    "variation_mode": mode,
                    "perturbation_fraction": float(perturbation),
                    "perturbation_percent": float(perturbation * 100.0),
                    "nominal_value": float(nominal_value),
                    "tested_value": float(tested_value),
                    "clipped_to_bound": bool(clipped),
                    "peak_y": peak_y,
                    "delta_peak_y_m": peak_y - float(nominal_metrics["peak_y"]),
                    "delta_peak_y_mm": (peak_y - float(nominal_metrics["peak_y"])) * 1000.0,
                    "target_error_mm": float(metrics["target_error_mm"]),
                    "max_abs_xr": max_abs_xr,
                    "delta_max_abs_xr_m": max_abs_xr - float(nominal_metrics["max_abs_xr"]),
                    "delta_max_abs_xr_mm": (max_abs_xr - float(nominal_metrics["max_abs_xr"])) * 1000.0,
                    "residual_margin_mm": float(metrics["residual_margin_mm"]),
                    "constraint_violation_mm": float(metrics["constraint_violation_abs_mm"]),
                    "feasible_abs": bool(metrics["feasible_abs"]),
                    "extra_reach": float(metrics["extra_reach"]),
                    "max_abs_xb": float(metrics.get("max_abs_xb", np.nan)),
                    "nominal_peak_y": float(nominal_metrics["peak_y"]),
                    "nominal_max_abs_xr": float(nominal_metrics["max_abs_xr"]),
                    "nominal_residual_margin_mm": float(nominal_metrics["residual_margin_mm"]),
                }

                for param_name in PARAM_COLUMNS:
                    row[param_name] = float(perturbed_params[param_name])

                all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    nominal_df = pd.DataFrame(nominal_rows)

    def _row_at_percent(group: pd.DataFrame, pct: float) -> pd.Series | None:
        if group.empty:
            return None
        idx = (group["perturbation_percent"].astype(float) - float(pct)).abs().idxmin()
        if abs(float(group.loc[idx, "perturbation_percent"]) - float(pct)) > 1e-9:
            return None
        return group.loc[idx]

    def _value_at_percent(group: pd.DataFrame, pct: float, col: str) -> float:
        row = _row_at_percent(group, pct)
        if row is None or col not in row:
            return float("nan")
        return float(row[col])

    def _central_slope(group: pd.DataFrame, col: str) -> float:
        # Slope between -5% and +5%, expressed as metric units per 1% perturbation.
        minus = _value_at_percent(group, -5.0, col)
        plus = _value_at_percent(group, 5.0, col)
        if not (np.isfinite(minus) and np.isfinite(plus)):
            return float("nan")
        return float((plus - minus) / 10.0)

    def _risk_label(min_margin_mm: float) -> str:
        if min_margin_mm < 0.0:
            return "critical"
        if min_margin_mm < 2.0:
            return "warning"
        return "safe"

    summary_rows = []
    for (target, parameter), group in results_df.groupby(["target", "parameter"]):
        group = group.sort_values("perturbation_percent").copy()

        peak_values = group["peak_y"].to_numpy(dtype=float)
        xr_values = group["max_abs_xr"].to_numpy(dtype=float)
        err_values = group["target_error_mm"].to_numpy(dtype=float)
        margin_values = group["residual_margin_mm"].to_numpy(dtype=float)

        param_group = str(group["parameter_group"].iloc[0])
        nominal_peak = float(group["nominal_peak_y"].iloc[0])
        nominal_xr = float(group["nominal_max_abs_xr"].iloc[0])

        worst_idx = group["residual_margin_mm"].astype(float).idxmin()
        worst = group.loc[worst_idx]
        min_margin_mm = float(worst["residual_margin_mm"])

        delta_peak_minus10 = _value_at_percent(group, -10.0, "delta_peak_y_mm")
        delta_peak_minus5 = _value_at_percent(group, -5.0, "delta_peak_y_mm")
        delta_peak_plus5 = _value_at_percent(group, 5.0, "delta_peak_y_mm")
        delta_peak_plus10 = _value_at_percent(group, 10.0, "delta_peak_y_mm")

        delta_xr_minus10 = _value_at_percent(group, -10.0, "delta_max_abs_xr_mm")
        delta_xr_minus5 = _value_at_percent(group, -5.0, "delta_max_abs_xr_mm")
        delta_xr_plus5 = _value_at_percent(group, 5.0, "delta_max_abs_xr_mm")
        delta_xr_plus10 = _value_at_percent(group, 10.0, "delta_max_abs_xr_mm")

        margin_minus10 = _value_at_percent(group, -10.0, "residual_margin_mm")
        margin_plus10 = _value_at_percent(group, 10.0, "residual_margin_mm")

        summary_rows.append(
            {
                "method": REFERENCE_METHOD,
                "target": float(target),
                "parameter": parameter,
                "parameter_group": param_group,
                "peak_y_range_mm": float((peak_values.max() - peak_values.min()) * 1000.0),
                "max_abs_delta_peak_y_mm": float(np.max(np.abs((peak_values - nominal_peak) * 1000.0))),
                "delta_peak_y_minus10_mm": delta_peak_minus10,
                "delta_peak_y_minus5_mm": delta_peak_minus5,
                "delta_peak_y_plus5_mm": delta_peak_plus5,
                "delta_peak_y_plus10_mm": delta_peak_plus10,
                "local_slope_peak_y_mm_per_percent": _central_slope(group, "delta_peak_y_mm"),
                "max_abs_xr_range_mm": float((xr_values.max() - xr_values.min()) * 1000.0),
                "max_abs_delta_max_abs_xr_mm": float(np.max(np.abs((xr_values - nominal_xr) * 1000.0))),
                "delta_max_abs_xr_minus10_mm": delta_xr_minus10,
                "delta_max_abs_xr_minus5_mm": delta_xr_minus5,
                "delta_max_abs_xr_plus5_mm": delta_xr_plus5,
                "delta_max_abs_xr_plus10_mm": delta_xr_plus10,
                "local_slope_max_abs_xr_mm_per_percent": _central_slope(group, "delta_max_abs_xr_mm"),
                "local_slope_residual_margin_mm_per_percent": _central_slope(group, "residual_margin_mm"),
                "target_error_range_mm": float(err_values.max() - err_values.min()),
                "max_target_error_mm": float(err_values.max()),
                "min_residual_margin_mm": min_margin_mm,
                "residual_margin_minus10_mm": margin_minus10,
                "residual_margin_plus10_mm": margin_plus10,
                "worst_margin_perturbation_percent": float(worst["perturbation_percent"]),
                "worst_margin_tested_value": float(worst["tested_value"]),
                "worst_margin_peak_y": float(worst["peak_y"]),
                "worst_margin_max_abs_xr": float(worst["max_abs_xr"]),
                "worst_margin_target_error_mm": float(worst["target_error_mm"]),
                "worst_margin_feasible_abs": bool(worst["feasible_abs"]),
                "max_constraint_violation_mm": float(group["constraint_violation_mm"].max()),
                "feasibility_rate_percent": float(group["feasible_abs"].mean() * 100.0),
                "n_infeasible_points": int((~group["feasible_abs"].astype(bool)).sum()),
                "n_tested_points": int(len(group)),
                "n_clipped_points": int(group["clipped_to_bound"].sum()),
                "risk_label": _risk_label(min_margin_mm),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["parameter_order"] = summary_df["parameter"].map(
        lambda p: PARAMETER_DISPLAY_ORDER.index(p) if p in PARAMETER_DISPLAY_ORDER else 999
    )
    summary_df = summary_df.sort_values(["target", "parameter_order"]).drop(columns="parameter_order")

    results_path = RESULTS_DIR / "sensitivity_oat_results.csv"
    summary_path = RESULTS_DIR / "sensitivity_oat_summary.csv"
    nominal_path = RESULTS_DIR / "sensitivity_oat_nominal_metrics.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    nominal_df.to_csv(nominal_path, index=False)

    print(f"\n✓ Saved detailed results: {results_path}")
    print(f"✓ Saved summary:          {summary_path}")
    print(f"✓ Saved nominal metrics:  {nominal_path}")

    return results_df, summary_df, nominal_df


# =============================================================================
# PLOTS
# =============================================================================

def ordered_summary_for_target(summary_df: pd.DataFrame, target: float, metric: str) -> pd.DataFrame:
    df = summary_df[np.isclose(summary_df["target"], target)].copy()
    df = df.sort_values(metric, ascending=True)
    return df


def plot_tornado(summary_df: pd.DataFrame, target: float, metric: str, title: str, filename: str) -> None:
    df = ordered_summary_for_target(summary_df, target, metric)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.barh(df["parameter"], df[metric], edgecolor="black")
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("Perturbed parameter")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    if metric == "min_residual_margin_mm":
        ax.axvline(0.0, linestyle="--", linewidth=2.0, label="Constraint boundary")
        ax.axvspan(min(df[metric].min(), 0.0), 0.0, alpha=0.10, label="Infeasible margin")
        ax.legend(fontsize=8, loc="lower right")

    # Add group labels visually through text annotations.
    max_abs_metric = max(float(np.nanmax(np.abs(df[metric]))), 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        label = "C" if row["parameter_group"] == "controllable" else "P"
        value = float(row[metric])
        offset = 0.02 * max_abs_metric
        x_text = value + offset if value >= 0 else value - offset
        ha = "left" if value >= 0 else "right"
        ax.text(x_text, i, label, va="center", ha=ha, fontsize=9)

    ax.text(
        0.98,
        0.02,
        "C = controllable, P = physical/model",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_metric_curves(
    results_df: pd.DataFrame,
    target: float,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    df_t = results_df[np.isclose(results_df["target"], target)].copy()

    params = [p for p in PARAMETER_DISPLAY_ORDER if p in df_t["parameter"].unique()]
    n_params = len(params)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.4 * n_rows), sharex=False)
    axes = np.ravel(axes)

    for ax, param in zip(axes, params):
        df_p = df_t[df_t["parameter"] == param].sort_values("perturbation_percent")
        ax.plot(df_p["perturbation_percent"], df_p[metric], marker="o", linewidth=2)
        ax.axvline(0.0, linestyle="--", linewidth=1.3)
        ax.axhline(0.0, linestyle=":", linewidth=1.1)

        if metric == "residual_margin_mm":
            ax.axhline(0.0, linestyle="--", linewidth=1.5, label="Constraint boundary")

        if metric == "max_abs_xr":
            ax.axhline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=1.5, label="True limit")

        group = str(df_p["parameter_group"].iloc[0])
        mode = str(df_p["variation_mode"].iloc[0])
        ax.set_title(f"{param} ({group})", fontsize=10)
        ax.set_xlabel("Perturbation [%]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if df_p["clipped_to_bound"].any():
            clipped = df_p[df_p["clipped_to_bound"]]
            ax.scatter(
                clipped["perturbation_percent"],
                clipped[metric],
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                label="Clipped",
                zorder=5,
            )

        if mode == "percent_of_allowed_range":
            mode_label = "% of bounds range"
        else:
            mode_label = "% of nominal"
        ax.text(
            0.03,
            0.95,
            mode_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )

        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def generate_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\nGenerating sensitivity plots...")
    for target in TARGETS_TO_ANALYZE:
        tag = target_tag(target)

        plot_tornado(
            summary_df=summary_df,
            target=target,
            metric="peak_y_range_mm",
            title=f"OAT sensitivity of peak_y, target = {target:.2f} m",
            filename=f"sensitivity_peak_y_tornado_{tag}.png",
        )
        plot_tornado(
            summary_df=summary_df,
            target=target,
            metric="max_abs_xr_range_mm",
            title=f"OAT sensitivity of max_abs_xr, target = {target:.2f} m",
            filename=f"sensitivity_max_abs_xr_tornado_{tag}.png",
        )
        plot_tornado(
            summary_df=summary_df,
            target=target,
            metric="min_residual_margin_mm",
            title=f"Minimum residual margin under OAT perturbations, target = {target:.2f} m",
            filename=f"sensitivity_residual_margin_tornado_{tag}.png",
        )

        plot_metric_curves(
            results_df=results_df,
            target=target,
            metric="delta_peak_y_mm",
            ylabel="Delta peak_y [mm]",
            title=f"OAT sensitivity curves for peak_y, target = {target:.2f} m",
            filename=f"sensitivity_curves_peak_y_{tag}.png",
        )
        plot_metric_curves(
            results_df=results_df,
            target=target,
            metric="delta_max_abs_xr_mm",
            ylabel="Delta max_abs_xr [mm]",
            title=f"OAT sensitivity curves for max_abs_xr variation, target = {target:.2f} m",
            filename=f"sensitivity_curves_max_abs_xr_{tag}.png",
        )
        plot_metric_curves(
            results_df=results_df,
            target=target,
            metric="max_abs_xr",
            ylabel="max_abs_xr [m]",
            title=f"Absolute robot-displacement response under OAT perturbations, target = {target:.2f} m",
            filename=f"sensitivity_curves_max_abs_xr_absolute_{tag}.png",
        )
        plot_metric_curves(
            results_df=results_df,
            target=target,
            metric="residual_margin_mm",
            ylabel="Residual margin [mm]",
            title=f"Residual constraint margin under OAT perturbations, target = {target:.2f} m",
            filename=f"sensitivity_curves_residual_margin_{tag}.png",
        )


def print_final_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("OAT SENSITIVITY SUMMARY")
    print("=" * 80)

    for target in TARGETS_TO_ANALYZE:
        df_t = summary_df[np.isclose(summary_df["target"], target)].copy()
        print(f"\nTarget {target:.2f} m")
        cols = [
            "parameter",
            "parameter_group",
            "peak_y_range_mm",
            "max_abs_xr_range_mm",
            "local_slope_peak_y_mm_per_percent",
            "local_slope_max_abs_xr_mm_per_percent",
            "min_residual_margin_mm",
            "worst_margin_perturbation_percent",
            "feasibility_rate_percent",
            "risk_label",
        ]
        # Sort by peak_y impact for compact terminal reading.
        df_print = df_t.sort_values("peak_y_range_mm", ascending=False)
        print(df_print[cols].to_string(index=False))

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("ONE-AT-A-TIME SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Reference method:       {REFERENCE_METHOD}")
    print(f"Targets to analyze:     {TARGETS_TO_ANALYZE}")
    print(f"True robot limit:       ±{ROBOT_LIMIT_TRUE:.3f} m")
    print("Controllable variation: ±5/10% of parameter bounds range")
    print("Physical variation:     ±5/10% relative to nominal value")
    print("=" * 80)

    reference_df = load_reference_solutions()
    results_df, summary_df, _ = run_sensitivity(reference_df)
    generate_plots(results_df, summary_df)
    print_final_summary(summary_df)

    print("\nSaved files:")
    print(f"  {RESULTS_DIR / 'sensitivity_oat_results.csv'}")
    print(f"  {RESULTS_DIR / 'sensitivity_oat_summary.csv'}")
    print(f"  {RESULTS_DIR / 'sensitivity_oat_nominal_metrics.csv'}")
    print(f"  {FIGURES_DIR}")
    print("\nNext step: inspect sensitivity results, then run Monte Carlo robustness.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
