"""
optimizer_comparison.py
=======================

Comparison of optimization algorithms for constraint-aware inverse design.

Goal
----
Compare three optimizers on the same constraint-aware inverse optimization task:

1. Random Search
2. Powell Multi-Start
3. Differential Evolution

The objective uses:
- GP_peak_y surrogate for target outreach prediction
- GP_max_xr surrogate for robot displacement constraint prediction

All final optimizer candidates are validated with the true dynamic simulator.

Optimized variables
-------------------
    Kr, hr, f0, f1, A, x_r_start

Fixed variables
---------------
    Kb, Mb, hb, Mr

Generated outputs
-----------------
results/optimizer_comparison/
    optimizer_comparison_results.csv
    optimizer_comparison_summary.csv
    best_optimizer_summary.txt

figures/optimizer_comparison/
    optimizer_comparison_error.png
    optimizer_comparison_violation.png
    optimizer_comparison_feasibility.png
    optimizer_comparison_time.png
    optimizer_comparison_evaluations.png
    optimizer_comparison_achieved_vs_target.png
    optimizer_comparison_summary.png

Author: Matteo Casazza
Date: 2026
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize

from dynamics import simulate_system


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "optimizer_comparison"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimizer_comparison"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GP_PEAK_DIR = PROJECT_ROOT / "results" / "gp"
GP_MAX_XR_DIR = PROJECT_ROOT / "results" / "gp_constraints"

GP_PEAK_MODEL_PATH = GP_PEAK_DIR / "gp_model.pkl"
GP_PEAK_SCALER_X_PATH = GP_PEAK_DIR / "scaler_X.pkl"
GP_PEAK_SCALER_Y_PATH = GP_PEAK_DIR / "scaler_y.pkl"

GP_MAX_XR_MODEL_PATH = GP_MAX_XR_DIR / "gp_max_xr_model.pkl"
GP_MAX_XR_SCALER_X_PATH = GP_MAX_XR_DIR / "scaler_X_max_xr.pkl"
GP_MAX_XR_SCALER_Y_PATH = GP_MAX_XR_DIR / "scaler_y_max_xr.pkl"


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TARGETS = [0.52, 0.58, 0.64]

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

OPT_COLS = [
    "Kr",
    "hr",
    "f0",
    "f1",
    "A",
    "x_r_start",
]

FIXED_PARAMS = {
    "Kb": 1000.0,
    "Mb": 20.0,
    "hb": 0.10,
    "Mr": 10.0,
}

BOUNDS = {
    "Kr": (1500.0, 5000.0),
    "hr": (0.10, 0.45),
    "f0": (0.10, 0.45),
    "f1": (1.00, 4.00),
    "A": (0.09, 0.12),
    "x_r_start": (0.35, 0.40),
}

X_R_MAX_TRUE = 0.500
X_R_MAX_OPT = 0.495

VALIDATION_ERROR_TOL_MM = 10.0

LAMBDA_UNCERTAINTY = 0.15
LAMBDA_CONSTRAINT = 500.0
LAMBDA_CONSTRAINT_UNCERTAINTY = 0.05

# Random Search
N_RANDOM_SAMPLES = 1500

# Powell Multi-Start
N_POWELL_STARTS = 20
POWELL_MAXITER = 350

# Differential Evolution
DE_MAXITER = 80
DE_POPSIZE = 12
DE_TOL = 1e-7
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_POLISH = True


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_gp_bundle(
    model_path: Path,
    scaler_x_path: Path,
    scaler_y_path: Path,
):
    """
    Load a trained GP model and its scalers.
    """
    missing = [
        path for path in [model_path, scaler_x_path, scaler_y_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(str(path) for path in missing)
        )

    model = joblib.load(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_X, scaler_y


def load_models() -> Dict:
    """
    Load GP_peak_y and GP_max_xr models.
    """
    print("\n" + "=" * 80)
    print("LOADING SURROGATE MODELS")
    print("=" * 80)

    gp_peak, scaler_X_peak, scaler_y_peak = load_gp_bundle(
        GP_PEAK_MODEL_PATH,
        GP_PEAK_SCALER_X_PATH,
        GP_PEAK_SCALER_Y_PATH,
    )

    gp_max_xr, scaler_X_xr, scaler_y_xr = load_gp_bundle(
        GP_MAX_XR_MODEL_PATH,
        GP_MAX_XR_SCALER_X_PATH,
        GP_MAX_XR_SCALER_Y_PATH,
    )

    print(f"Loaded GP_peak_y: {GP_PEAK_MODEL_PATH}")
    print(f"Loaded GP_max_xr: {GP_MAX_XR_MODEL_PATH}")

    return {
        "gp_peak": gp_peak,
        "scaler_X_peak": scaler_X_peak,
        "scaler_y_peak": scaler_y_peak,
        "gp_max_xr": gp_max_xr,
        "scaler_X_xr": scaler_X_xr,
        "scaler_y_xr": scaler_y_xr,
    }


# ============================================================================
# PARAMETER UTILITIES
# ============================================================================

def bounds_as_list() -> List[Tuple[float, float]]:
    """
    Return optimizer bounds in OPT_COLS order.
    """
    return [BOUNDS[name] for name in OPT_COLS]


def vector_to_params(x: np.ndarray) -> Dict[str, float]:
    """
    Convert optimizer vector to full parameter dictionary.
    """
    params = dict(FIXED_PARAMS)

    for name, value in zip(OPT_COLS, x):
        params[name] = float(value)

    return params


def params_to_feature_row(params: Dict[str, float]) -> np.ndarray:
    """
    Convert full parameter dictionary to GP input row.
    """
    return np.array([[params[name] for name in PARAM_COLS]], dtype=float)


def random_sample(bounds: List[Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    """
    Sample one random point inside the bounds.
    """
    return np.array(
        [rng.uniform(low, high) for low, high in bounds],
        dtype=float,
    )


# ============================================================================
# SURROGATE PREDICTION
# ============================================================================

def predict_gp_physical(
    model,
    scaler_X,
    scaler_y,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with a GP and convert mean/std back to physical units.
    """
    X_scaled = scaler_X.transform(X)

    y_pred_scaled, y_std_scaled = model.predict(
        X_scaled,
        return_std=True,
    )

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_std = y_std_scaled * scaler_y.scale_[0]

    return y_pred, y_std


def predict_candidate(
    x: np.ndarray,
    models: Dict,
) -> Dict[str, float]:
    """
    Predict peak_y and max_xr for a candidate optimizer vector.
    """
    params = vector_to_params(x)
    X_row = params_to_feature_row(params)

    peak_pred, peak_std = predict_gp_physical(
        models["gp_peak"],
        models["scaler_X_peak"],
        models["scaler_y_peak"],
        X_row,
    )

    xr_pred, xr_std = predict_gp_physical(
        models["gp_max_xr"],
        models["scaler_X_xr"],
        models["scaler_y_xr"],
        X_row,
    )

    return {
        "gp_peak_y_pred": float(peak_pred[0]),
        "gp_peak_y_std": float(peak_std[0]),
        "gp_max_xr_pred": float(xr_pred[0]),
        "gp_max_xr_std": float(xr_std[0]),
    }


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

class ObjectiveCounter:
    """
    Callable constraint-aware objective with evaluation counter.
    """

    def __init__(self, y_target: float, models: Dict):
        self.y_target = float(y_target)
        self.models = models
        self.n_evaluations = 0
        self.best_value = np.inf
        self.best_x = None

    def __call__(self, x: np.ndarray) -> float:
        self.n_evaluations += 1

        pred = predict_candidate(x, self.models)

        y_pred = pred["gp_peak_y_pred"]
        y_std = pred["gp_peak_y_std"]

        max_xr_pred = pred["gp_max_xr_pred"]
        max_xr_std = pred["gp_max_xr_std"]

        target_error = (y_pred - self.y_target) ** 2
        uncertainty_penalty = LAMBDA_UNCERTAINTY * (y_std ** 2)

        predicted_violation = max(0.0, max_xr_pred - X_R_MAX_OPT)
        constraint_penalty = LAMBDA_CONSTRAINT * (predicted_violation ** 2)

        constraint_uncertainty_penalty = (
            LAMBDA_CONSTRAINT_UNCERTAINTY * (max_xr_std ** 2)
        )

        objective = (
            target_error
            + uncertainty_penalty
            + constraint_penalty
            + constraint_uncertainty_penalty
        )

        if objective < self.best_value:
            self.best_value = float(objective)
            self.best_x = np.array(x, dtype=float)

        return float(objective)


# ============================================================================
# OPTIMIZERS
# ============================================================================

def optimize_random_search(
    y_target: float,
    models: Dict,
    rng: np.random.Generator,
) -> Dict:
    """
    Random Search optimizer.
    """
    bounds = bounds_as_list()
    objective = ObjectiveCounter(y_target, models)

    start_time = time.time()

    best_x = None
    best_value = np.inf

    for _ in range(N_RANDOM_SAMPLES):
        x = random_sample(bounds, rng)
        value = objective(x)

        if value < best_value:
            best_value = value
            best_x = x.copy()

    optimization_time_s = time.time() - start_time

    return {
        "method": "Random Search",
        "x_opt": best_x,
        "objective_value": float(best_value),
        "success": True,
        "message": "Completed random search",
        "n_evaluations": int(objective.n_evaluations),
        "optimization_time_s": float(optimization_time_s),
    }


def optimize_powell_multistart(
    y_target: float,
    models: Dict,
    rng: np.random.Generator,
) -> Dict:
    """
    Powell local optimization from multiple random starts.
    """
    bounds = bounds_as_list()
    scipy_bounds = bounds

    global_objective = ObjectiveCounter(y_target, models)

    start_time = time.time()

    best_x = None
    best_value = np.inf
    best_success = False
    best_message = ""

    for start_idx in range(N_POWELL_STARTS):
        x0 = random_sample(bounds, rng)

        result = minimize(
            fun=global_objective,
            x0=x0,
            method="Powell",
            bounds=scipy_bounds,
            options={
                "maxiter": POWELL_MAXITER,
                "xtol": 1e-6,
                "ftol": 1e-8,
                "disp": False,
            },
        )

        if result.fun < best_value:
            best_value = float(result.fun)
            best_x = np.array(result.x, dtype=float)
            best_success = bool(result.success)
            best_message = str(result.message)

        print(
            f"    Powell start {start_idx + 1}/{N_POWELL_STARTS} | "
            f"best objective so far = {best_value:.6e}"
        )

    optimization_time_s = time.time() - start_time

    return {
        "method": "Powell Multi-Start",
        "x_opt": best_x,
        "objective_value": float(best_value),
        "success": bool(best_success),
        "message": best_message,
        "n_evaluations": int(global_objective.n_evaluations),
        "optimization_time_s": float(optimization_time_s),
    }


def optimize_differential_evolution(
    y_target: float,
    models: Dict,
) -> Dict:
    """
    Differential Evolution optimizer.
    """
    bounds = bounds_as_list()
    objective = ObjectiveCounter(y_target, models)

    start_time = time.time()

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        polish=DE_POLISH,
        seed=RANDOM_STATE,
        workers=1,
        updating="immediate",
        disp=False,
    )

    optimization_time_s = time.time() - start_time

    return {
        "method": "Differential Evolution",
        "x_opt": np.array(result.x, dtype=float),
        "objective_value": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "n_evaluations": int(objective.n_evaluations),
        "optimization_time_s": float(optimization_time_s),
    }


# ============================================================================
# TRUE SIMULATOR VALIDATION
# ============================================================================

def extract_metrics_from_sim_output(output) -> Dict[str, float]:
    """
    Extract metrics from simulate_system output.

    This function is intentionally robust to different return formats.
    """
    if isinstance(output, dict):
        return output

    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict):
                return item

        numeric_items = [
            float(item)
            for item in output
            if isinstance(item, (int, float, np.floating))
        ]

        if len(numeric_items) >= 1:
            return {
                "peak_y": numeric_items[0],
            }

    if isinstance(output, (int, float, np.floating)):
        return {
            "peak_y": float(output),
        }

    raise RuntimeError(
        "Could not extract metrics from simulate_system output. "
        f"Output type: {type(output)}"
    )


def validate_with_true_simulator(
    params: Dict[str, float],
    y_target: float,
) -> Dict[str, float]:
    """
    Validate candidate using the true dynamic simulator.
    """
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=X_R_MAX_TRUE,
            return_metrics=True,
        )
    except TypeError:
        output = simulate_system(
            params,
            T_sim=60.0,
            dt=0.001,
            return_metrics=True,
        )

    metrics = extract_metrics_from_sim_output(output)

    peak_y = float(metrics.get("peak_y", metrics.get("y_sim", np.nan)))
    max_xr = float(metrics.get("max_xr", np.nan))
    max_xb = float(metrics.get("max_xb", np.nan))

    if np.isnan(peak_y):
        raise RuntimeError("Simulation metrics do not contain peak_y.")

    if np.isnan(max_xr):
        max_xr = float(metrics.get("robot_max", np.nan))

    constraint_violation = float(
        metrics.get(
            "constraint_violation",
            max(0.0, max_xr - X_R_MAX_TRUE) if not np.isnan(max_xr) else np.nan,
        )
    )

    extra_reach = float(
        metrics.get(
            "extra_reach",
            peak_y - X_R_MAX_TRUE,
        )
    )

    error_sim = abs(peak_y - y_target)

    feasible = bool(
        constraint_violation <= 1e-9
        if not np.isnan(constraint_violation)
        else False
    )

    return {
        "y_sim": peak_y,
        "error_sim": error_sim,
        "error_sim_mm": error_sim * 1000.0,
        "max_xr_sim": max_xr,
        "max_xb_sim": max_xb,
        "extra_reach": extra_reach,
        "constraint_violation": constraint_violation,
        "constraint_violation_mm": constraint_violation * 1000.0,
        "feasible": feasible,
    }


# ============================================================================
# RESULT ROW
# ============================================================================

def build_result_row(
    y_target: float,
    opt_result: Dict,
    models: Dict,
) -> Dict:
    """
    Build one result row after optimizer and true simulator validation.
    """
    x_opt = opt_result["x_opt"]
    params = vector_to_params(x_opt)

    surrogate_pred = predict_candidate(x_opt, models)
    sim_metrics = validate_with_true_simulator(params, y_target)

    validated_success = (
        bool(sim_metrics["feasible"])
        and float(sim_metrics["error_sim_mm"]) <= VALIDATION_ERROR_TOL_MM
    )

    row = {
        "method": opt_result["method"],
        "target": float(y_target),
        "optimizer_internal_success": bool(opt_result["success"]),
        "validated_success": bool(validated_success),
        "message": opt_result["message"],
        "objective_value": float(opt_result["objective_value"]),
        "n_evaluations": int(opt_result["n_evaluations"]),
        "optimization_time_s": float(opt_result["optimization_time_s"]),
    }

    for name in PARAM_COLS:
        row[name] = float(params[name])

    row.update(surrogate_pred)
    row.update(sim_metrics)

    return row


# ============================================================================
# SUMMARY
# ============================================================================

def summarize_optimizer_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize optimizer comparison by method.
    """
    summary_df = (
        results_df
        .groupby("method")
        .agg(
            mean_error_mm=("error_sim_mm", "mean"),
            std_error_mm=("error_sim_mm", "std"),
            max_error_mm=("error_sim_mm", "max"),
            mean_violation_mm=("constraint_violation_mm", "mean"),
            max_violation_mm=("constraint_violation_mm", "max"),
            feasibility_rate_percent=("feasible", lambda x: 100.0 * np.mean(x)),
            mean_extra_reach_m=("extra_reach", "mean"),
            max_extra_reach_m=("extra_reach", "max"),
            mean_optimization_time_s=("optimization_time_s", "mean"),
            total_optimization_time_s=("optimization_time_s", "sum"),
            mean_n_evaluations=("n_evaluations", "mean"),
            total_n_evaluations=("n_evaluations", "sum"),
            optimizer_internal_successful_runs=("optimizer_internal_success", "sum"),
            validated_successful_runs=("validated_success", "sum"),
            validated_success_rate_percent=("validated_success", lambda x: 100.0 * np.mean(x)),
            n_targets=("target", "count"),
        )
        .reset_index()
    )

    summary_df = summary_df.sort_values(
        by=[
            "validated_success_rate_percent",
            "feasibility_rate_percent",
            "max_violation_mm",
            "mean_error_mm",
            "mean_optimization_time_s",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    return summary_df


def save_best_optimizer_summary(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Path:
    """
    Save text summary with final optimizer recommendation.
    """
    best = summary_df.iloc[0]

    lines = []
    lines.append("OPTIMIZER COMPARISON SUMMARY")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Setup:")
    lines.append(f"  Targets: {TARGETS}")
    lines.append(f"  Optimized variables: {OPT_COLS}")
    lines.append(f"  True robot limit: {X_R_MAX_TRUE:.3f} m")
    lines.append(f"  Optimization safety limit: {X_R_MAX_OPT:.3f} m")
    lines.append(f"  Validation success tolerance: {VALIDATION_ERROR_TOL_MM:.1f} mm")
    lines.append("")
    lines.append("Compared methods:")
    lines.append(f"  Random Search: {N_RANDOM_SAMPLES} samples")
    lines.append(
        f"  Powell Multi-Start: {N_POWELL_STARTS} starts, "
        f"maxiter={POWELL_MAXITER}"
    )
    lines.append(
        f"  Differential Evolution: maxiter={DE_MAXITER}, "
        f"popsize={DE_POPSIZE}, polish={DE_POLISH}"
    )
    lines.append("")
    lines.append("Summary by method:")
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("Recommended optimizer:")
    lines.append(f"  {best['method']}")
    lines.append("")
    lines.append("Reason:")
    lines.append(
        "  The recommended optimizer is selected by prioritizing validated "
        "physical success, feasibility, constraint violation, and finally "
        "simulation tracking error and computational time. A validated success "
        f"is defined as a feasible solution with simulation error below "
        f"{VALIDATION_ERROR_TOL_MM:.1f} mm."
    )
    lines.append("")
    lines.append("Detailed results:")
    lines.append(results_df.to_string(index=False))

    path = RESULTS_DIR / "best_optimizer_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {path}")

    return path


# ============================================================================
# PLOTS
# ============================================================================

def method_colors() -> Dict[str, str]:
    """
    Consistent colors for methods.
    """
    return {
        "Random Search": "#4C78A8",
        "Powell Multi-Start": "#F58518",
        "Differential Evolution": "#54A24B",
    }


def plot_grouped_bars(
    results_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """
    Grouped bar plot by target and method.
    """
    methods = list(results_df["method"].unique())
    targets = sorted(results_df["target"].unique())
    colors = method_colors()

    x = np.arange(len(targets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        values = []

        for target in targets:
            row = results_df[
                (results_df["method"] == method)
                & (results_df["target"] == target)
            ]

            if len(row) == 0:
                values.append(np.nan)
            else:
                values.append(float(row[value_col].iloc[0]))

        offset = (i - (len(methods) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=method,
            color=colors.get(method, "gray"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.88,
        )

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(1.0, np.nanmax(values)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{target:.2f}" for target in targets])
    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_feasibility(summary_df: pd.DataFrame) -> None:
    """
    Plot feasibility rate by method.
    """
    colors = method_colors()
    methods = summary_df["method"].values
    values = summary_df["feasibility_rate_percent"].values

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        methods,
        values,
        color=[colors.get(m, "gray") for m in methods],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.88,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Feasibility rate [%]")
    ax.set_ylim(0, 110)
    ax.set_title("Optimizer Feasibility Rate")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    path = FIGURES_DIR / "optimizer_comparison_feasibility.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_summary(summary_df: pd.DataFrame) -> None:
    """
    Compact 2x2 summary figure.
    """
    colors = method_colors()
    methods = summary_df["method"].values
    bar_colors = [colors.get(m, "gray") for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    specs = [
        ("mean_error_mm", "Mean error [mm]", "Target tracking error"),
        ("max_violation_mm", "Max violation [mm]", "Worst constraint violation"),
        ("validated_success_rate_percent", "Validated success [%]", "Validated physical success"),
        ("mean_optimization_time_s", "Mean time [s]", "Optimization time"),
    ]

    for ax, (col, ylabel, title) in zip(axes.ravel(), specs):
        values = summary_df[col].values

        bars = ax.bar(
            methods,
            values,
            color=bar_colors,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.88,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(1.0, np.nanmax(values)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20)

    plt.suptitle(
        "Optimizer Comparison Summary",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / "optimizer_comparison_summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_achieved_vs_target(results_df: pd.DataFrame) -> None:
    """
    Plot simulated achieved outreach against target.
    """
    colors = method_colors()

    fig, ax = plt.subplots(figsize=(8, 7))

    targets = results_df["target"].values
    achieved = results_df["y_sim"].values

    min_val = min(targets.min(), achieved.min()) - 0.02
    max_val = max(targets.max(), achieved.max()) + 0.02

    for method in results_df["method"].unique():
        df_m = results_df[results_df["method"] == method]

        ax.scatter(
            df_m["target"],
            df_m["y_sim"],
            s=90,
            color=colors.get(method, "gray"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            label=method,
        )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Ideal target tracking",
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("Simulated peak_y [m]")
    ax.set_title("Achieved Outreach vs Target")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    path = FIGURES_DIR / "optimizer_comparison_achieved_vs_target.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def generate_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """
    Generate all optimizer comparison plots.
    """
    print("\n" + "=" * 80)
    print("GENERATING OPTIMIZER COMPARISON PLOTS")
    print("=" * 80)

    plot_grouped_bars(
        results_df=results_df,
        value_col="error_sim_mm",
        ylabel="Simulation error [mm]",
        title="Optimizer Comparison: Simulation Error",
        filename="optimizer_comparison_error.png",
    )

    plot_grouped_bars(
        results_df=results_df,
        value_col="constraint_violation_mm",
        ylabel="Constraint violation [mm]",
        title="Optimizer Comparison: Constraint Violation",
        filename="optimizer_comparison_violation.png",
    )

    plot_grouped_bars(
        results_df=results_df,
        value_col="optimization_time_s",
        ylabel="Optimization time [s]",
        title="Optimizer Comparison: Runtime",
        filename="optimizer_comparison_time.png",
    )

    plot_grouped_bars(
        results_df=results_df,
        value_col="n_evaluations",
        ylabel="Objective evaluations",
        title="Optimizer Comparison: Objective Evaluations",
        filename="optimizer_comparison_evaluations.png",
    )

    plot_feasibility(summary_df)
    plot_achieved_vs_target(results_df)
    plot_summary(summary_df)


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_optimizer_comparison() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run optimizer comparison for all targets and methods.
    """
    models = load_models()

    rng = np.random.default_rng(RANDOM_STATE)

    all_rows = []

    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON")
    print("=" * 80)
    print(f"Targets:              {TARGETS}")
    print(f"Optimized variables:  {OPT_COLS}")
    print(f"Fixed parameters:     {FIXED_PARAMS}")
    print(f"Safety limit:         {X_R_MAX_OPT:.3f} m")
    print(f"True limit:           {X_R_MAX_TRUE:.3f} m")
    print("=" * 80)

    optimizer_functions = [
        ("Random Search", optimize_random_search),
        ("Powell Multi-Start", optimize_powell_multistart),
        ("Differential Evolution", optimize_differential_evolution),
    ]

    for y_target in TARGETS:
        print("\n" + "#" * 80)
        print(f"TARGET: {y_target:.3f} m")
        print("#" * 80)

        for method_name, optimizer_fn in optimizer_functions:
            print("\n" + "-" * 80)
            print(f"Running {method_name} for target {y_target:.3f} m")
            print("-" * 80)

            if method_name == "Differential Evolution":
                opt_result = optimizer_fn(y_target, models)
            else:
                opt_result = optimizer_fn(y_target, models, rng)

            row = build_result_row(
                y_target=y_target,
                opt_result=opt_result,
                models=models,
            )

            all_rows.append(row)

            print(
                f"\nResult | {method_name} | target={y_target:.3f} m\n"
                f"  GP peak_y pred:       {row['gp_peak_y_pred']:.6f} m "
                f"± {row['gp_peak_y_std']:.6f} m\n"
                f"  GP max_xr pred:       {row['gp_max_xr_pred']:.6f} m "
                f"± {row['gp_max_xr_std']:.6f} m\n"
                f"  Simulated peak_y:     {row['y_sim']:.6f} m\n"
                f"  Simulation error:     {row['error_sim_mm']:.3f} mm\n"
                f"  max_xr simulation:    {row['max_xr_sim']:.6f} m\n"
                f"  Violation:            {row['constraint_violation_mm']:.3f} mm\n"
                f"  Feasible:             {row['feasible']}\n"
                f"  Time:                 {row['optimization_time_s']:.2f} s\n"
                f"  Evaluations:          {row['n_evaluations']}"
            )

            checkpoint_df = pd.DataFrame(all_rows)
            checkpoint_path = RESULTS_DIR / "optimizer_comparison_results_checkpoint.csv"
            checkpoint_df.to_csv(checkpoint_path, index=False)

    results_df = pd.DataFrame(all_rows)

    results_path = RESULTS_DIR / "optimizer_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = summarize_optimizer_results(results_df)

    summary_path = RESULTS_DIR / "optimizer_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    save_best_optimizer_summary(results_df, summary_df)

    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    generate_plots(results_df, summary_df)

    print("\nGenerated files:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {RESULTS_DIR / 'best_optimizer_summary.txt'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_error.png'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_violation.png'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_feasibility.png'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_time.png'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_evaluations.png'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_achieved_vs_target.png'}")
    print(f"  {FIGURES_DIR / 'optimizer_comparison_summary.png'}")

    return results_df, summary_df


if __name__ == "__main__":
    run_optimizer_comparison()