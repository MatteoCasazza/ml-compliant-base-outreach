"""
Constraint-aware inverse optimization.

This script compares two surrogate-based inverse optimization formulations:

1. Baseline formulation:
   - uses only the GP surrogate for peak_y;
   - minimizes target tracking error;
   - checks the robot displacement constraint only after true simulation.

2. Constraint-aware formulation:
   - uses the GP surrogate for peak_y;
   - uses an auxiliary GP surrogate for max_xr;
   - penalizes predicted robot constraint violations inside the objective.

Main goal
---------
Given a desired target outreach, find controllable parameters:

    [Kr, hr, f0, f1, A, x_r_start]

while keeping the robot relative displacement below its physical limit.

The final candidates are always validated using the true dynamic simulator.

Main outputs
------------
results/constraint_aware/baseline_vs_constraint_results.csv
results/constraint_aware/baseline_vs_constraint_all_attempts.csv
results/constraint_aware/baseline_vs_constraint_summary.csv
results/constraint_aware/report_summary_baseline_vs_constraint.csv
results/constraint_aware/final_solution_target064.csv

figures/constraint_aware/target_tracking_baseline_vs_constraint.png
figures/constraint_aware/constraint_violation_baseline_vs_constraint.png
figures/constraint_aware/simulation_error_baseline_vs_constraint.png
figures/constraint_aware/max_xr_baseline_vs_constraint.png
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from dynamics import simulate_system


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_GP_DIR = PROJECT_ROOT / "results" / "gp"
CONSTRAINT_GP_DIR = PROJECT_ROOT / "results" / "gp_constraints"

RESULTS_DIR = PROJECT_ROOT / "results" / "constraint_aware"
FIGURES_DIR = PROJECT_ROOT / "figures" / "constraint_aware"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConstraintAwareConfig:
    """
    Configuration for the constraint-aware inverse optimization experiment.

    The optimized variables are:

        [Kr, hr, f0, f1, A, x_r_start]

    The fixed variables are:

        [Kb, Mb, hb, Mr]
    """

    # Fixed passive-system parameters
    Kb: float = 1000.0
    Mb: float = 20.0
    hb: float = 0.10
    Mr: float = 10.0

    # True robot stroke limit used in final simulation validation
    x_r_max_true: float = 0.50

    # Conservative robot stroke limit used inside the surrogate objective
    x_r_max_opt: float = 0.495

    # Objective weights
    lambda_constraint: float = 500.0
    lambda_uncertainty: float = 0.15
    lambda_constraint_uncertainty: float = 0.05

    # Targets used to compare baseline and constraint-aware optimization
    targets: Tuple[float, ...] = (0.52, 0.58, 0.64, 0.70)

    # Final solution selected for detailed visualization/sensitivity/robustness
    final_selected_target: float = 0.64

    # Bounds for optimized parameters:
    # [Kr, hr, f0, f1, A, x_r_start]
    Kr_min: float = 1500.0
    Kr_max: float = 5000.0

    hr_min: float = 0.10
    hr_max: float = 0.45

    f0_min: float = 0.10
    f0_max: float = 0.45

    f1_min: float = 1.00
    f1_max: float = 4.00

    A_min: float = 0.09
    A_max: float = 0.12

    x_r_start_min: float = 0.35
    x_r_start_max: float = 0.40

    # Optimization settings
    n_optimization_attempts: int = 5

    de_strategy: str = "best1bin"
    de_maxiter: int = 120
    de_popsize: int = 18
    de_tol: float = 1e-7
    de_mutation: Tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7
    de_polish: bool = True
    de_workers: int = 1
    de_updating: str = "immediate"

    @staticmethod
    def input_columns() -> List[str]:
        """Return full GP input column order."""
        return [
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

    @staticmethod
    def optimized_columns() -> List[str]:
        """Return optimized parameter names in optimizer vector order."""
        return [
            "Kr",
            "hr",
            "f0",
            "f1",
            "A",
            "x_r_start",
        ]

    def fixed_params(self) -> Dict[str, float]:
        """Return fixed physical parameters."""
        return {
            "Kb": self.Kb,
            "Mb": self.Mb,
            "hb": self.hb,
            "Mr": self.Mr,
        }

    def bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for optimized variables."""
        return [
            (self.Kr_min, self.Kr_max),
            (self.hr_min, self.hr_max),
            (self.f0_min, self.f0_max),
            (self.f1_min, self.f1_max),
            (self.A_min, self.A_max),
            (self.x_r_start_min, self.x_r_start_max),
        ]

    def de_settings(self) -> Dict[str, Any]:
        """Return Differential Evolution settings."""
        return {
            "strategy": self.de_strategy,
            "maxiter": self.de_maxiter,
            "popsize": self.de_popsize,
            "tol": self.de_tol,
            "mutation": self.de_mutation,
            "recombination": self.de_recombination,
            "polish": self.de_polish,
            "workers": self.de_workers,
            "updating": self.de_updating,
        }


# =============================================================================
# Utility functions
# =============================================================================

def ensure_output_dirs() -> None:
    """Create output folders used by this script."""
    for directory in [RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_model_file(path: Path) -> Any:
    """
    Load a saved model or scaler.

    Some files may have been saved with joblib, while others may have been saved
    with pickle. This function supports both formats.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_surrogates() -> Dict[str, Any]:
    """Load the main peak_y GP and the auxiliary max_xr GP."""
    print("Loading surrogate models...")

    gp_peak_y = load_model_file(MAIN_GP_DIR / "gp_model.pkl")
    scaler_X_peak_y = load_model_file(MAIN_GP_DIR / "scaler_X.pkl")
    scaler_y_peak_y = load_model_file(MAIN_GP_DIR / "scaler_y.pkl")

    gp_max_xr = load_model_file(CONSTRAINT_GP_DIR / "gp_max_xr_model.pkl")
    scaler_X_max_xr = load_model_file(CONSTRAINT_GP_DIR / "scaler_X_max_xr.pkl")
    scaler_y_max_xr = load_model_file(CONSTRAINT_GP_DIR / "scaler_y_max_xr.pkl")

    print("Models loaded.")
    print(f"  Main GP directory:       {MAIN_GP_DIR}")
    print(f"  Constraint GP directory: {CONSTRAINT_GP_DIR}")
    print()

    return {
        "gp_peak_y": gp_peak_y,
        "scaler_X_peak_y": scaler_X_peak_y,
        "scaler_y_peak_y": scaler_y_peak_y,
        "gp_max_xr": gp_max_xr,
        "scaler_X_max_xr": scaler_X_max_xr,
        "scaler_y_max_xr": scaler_y_max_xr,
    }


def vector_to_params(x: np.ndarray, config: ConstraintAwareConfig) -> Dict[str, float]:
    """
    Convert optimizer vector to a full parameter dictionary.

    Optimized vector:
        [Kr, hr, f0, f1, A, x_r_start]
    """
    Kr, hr, f0, f1, A, x_r_start = x

    params = {
        "Kb": float(config.Kb),
        "Kr": float(Kr),
        "Mb": float(config.Mb),
        "hb": float(config.hb),
        "hr": float(hr),
        "f0": float(f0),
        "f1": float(f1),
        "A": float(A),
        "x_r_start": float(x_r_start),
        "Mr": float(config.Mr),
    }

    return params


def params_to_feature_vector(
    params: Dict[str, float],
    config: ConstraintAwareConfig,
) -> np.ndarray:
    """Convert parameter dictionary to ML feature vector."""
    return np.array(
        [[params[col] for col in config.input_columns()]],
        dtype=float,
    )


def predict_peak_y(
    params: Dict[str, float],
    models: Dict[str, Any],
    config: ConstraintAwareConfig,
    return_std: bool = True,
) -> Tuple[float, float] | float:
    """Predict peak_y using the main GP surrogate."""
    X = params_to_feature_vector(params, config)
    X_scaled = models["scaler_X_peak_y"].transform(X)

    if return_std:
        y_scaled, std_scaled = models["gp_peak_y"].predict(
            X_scaled,
            return_std=True,
        )

        y = models["scaler_y_peak_y"].inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()[0]

        y_scale = models["scaler_y_peak_y"].scale_[0]
        std = std_scaled[0] * y_scale

        return float(y), float(std)

    y_scaled = models["gp_peak_y"].predict(X_scaled)
    y = models["scaler_y_peak_y"].inverse_transform(
        y_scaled.reshape(-1, 1)
    ).ravel()[0]

    return float(y)


def predict_max_xr(
    params: Dict[str, float],
    models: Dict[str, Any],
    config: ConstraintAwareConfig,
    return_std: bool = True,
) -> Tuple[float, float] | float:
    """Predict max_xr using the auxiliary constraint GP surrogate."""
    X = params_to_feature_vector(params, config)
    X_scaled = models["scaler_X_max_xr"].transform(X)

    if return_std:
        y_scaled, std_scaled = models["gp_max_xr"].predict(
            X_scaled,
            return_std=True,
        )

        y = models["scaler_y_max_xr"].inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()[0]

        y_scale = models["scaler_y_max_xr"].scale_[0]
        std = std_scaled[0] * y_scale

        return float(y), float(std)

    y_scaled = models["gp_max_xr"].predict(X_scaled)
    y = models["scaler_y_max_xr"].inverse_transform(
        y_scaled.reshape(-1, 1)
    ).ravel()[0]

    return float(y)


# =============================================================================
# Objectives
# =============================================================================

def objective_baseline(
    x: np.ndarray,
    y_target: float,
    models: Dict[str, Any],
    config: ConstraintAwareConfig,
) -> float:
    """
    Baseline objective.

    Uses only the peak_y GP surrogate. The robot displacement constraint is not
    included in the objective and is checked only after true simulation.
    """
    params = vector_to_params(x, config)

    y_pred, y_std = predict_peak_y(
        params=params,
        models=models,
        config=config,
        return_std=True,
    )

    tracking_cost = (y_pred - y_target) ** 2
    uncertainty_cost = config.lambda_uncertainty * y_std ** 2

    return float(tracking_cost + uncertainty_cost)


def objective_constraint_aware(
    x: np.ndarray,
    y_target: float,
    models: Dict[str, Any],
    config: ConstraintAwareConfig,
) -> float:
    """
    Constraint-aware objective.

    Uses:
    - GP_peak_y for target tracking;
    - GP_max_xr for predicted robot displacement;
    - a penalty on predicted robot stroke violation.
    """
    params = vector_to_params(x, config)

    y_pred, y_std = predict_peak_y(
        params=params,
        models=models,
        config=config,
        return_std=True,
    )

    max_xr_pred, max_xr_std = predict_max_xr(
        params=params,
        models=models,
        config=config,
        return_std=True,
    )

    tracking_cost = (y_pred - y_target) ** 2
    uncertainty_cost = config.lambda_uncertainty * y_std ** 2

    predicted_violation = max(0.0, max_xr_pred - config.x_r_max_opt)
    constraint_cost = config.lambda_constraint * predicted_violation ** 2

    constraint_uncertainty_cost = (
        config.lambda_constraint_uncertainty * max_xr_std ** 2
    )

    total_cost = (
        tracking_cost
        + uncertainty_cost
        + constraint_cost
        + constraint_uncertainty_cost
    )

    return float(total_cost)


# =============================================================================
# Optimization and validation
# =============================================================================

def run_de_optimization(
    objective_func: Callable,
    y_target: float,
    models: Dict[str, Any],
    config: ConstraintAwareConfig,
    attempt_id: int,
) -> Any:
    """Run one Differential Evolution optimization attempt."""
    seed = 1000 + 100 * attempt_id + int(y_target * 1000)

    result = differential_evolution(
        objective_func,
        bounds=config.bounds(),
        args=(y_target, models, config),
        seed=seed,
        **config.de_settings(),
    )

    return result


def simulate_candidate(
    params: Dict[str, float],
    y_target: float,
    config: ConstraintAwareConfig,
) -> Dict[str, float]:
    """
    Validate candidate parameters using the true dynamic simulator.

    The simulator may return:
    - a metrics dictionary directly;
    - a tuple containing metrics and/or solution data.

    This function extracts and returns the metrics dictionary.
    """
    output = simulate_system(
        params,
        y_target=y_target,
        x_r_max=config.x_r_max_true,
        return_metrics=True,
    )

    if isinstance(output, dict):
        return output

    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict) and "peak_y" in item:
                return item

        # Common case: (peak_y, metrics)
        if len(output) >= 2 and isinstance(output[1], dict):
            return output[1]

    raise TypeError(
        "Could not extract metrics dictionary from simulate_system output. "
        f"Received type: {type(output)}"
    )


def build_attempt_record(
    method_name: str,
    y_target: float,
    attempt: int,
    result: Any,
    opt_time: float,
    models: Dict[str, Any],
    config: ConstraintAwareConfig,
) -> Dict[str, Any]:
    """Build a complete record for one optimization attempt."""
    params = vector_to_params(result.x, config)

    y_gp_pred, y_gp_std = predict_peak_y(
        params=params,
        models=models,
        config=config,
        return_std=True,
    )

    max_xr_gp_pred, max_xr_gp_std = predict_max_xr(
        params=params,
        models=models,
        config=config,
        return_std=True,
    )

    sim_metrics = simulate_candidate(
        params=params,
        y_target=y_target,
        config=config,
    )

    y_sim = float(sim_metrics["peak_y"])
    max_xr_sim = float(sim_metrics["max_xr"])
    max_xb_sim = float(sim_metrics["max_xb"])
    extra_reach = float(sim_metrics["extra_reach"])
    constraint_violation = float(sim_metrics["constraint_violation"])

    error_sim = abs(y_sim - y_target)
    error_gp = abs(y_gp_pred - y_target)
    error_gp_vs_sim = abs(y_gp_pred - y_sim)

    feasible = constraint_violation <= 1e-9

    predicted_violation_gp = max(0.0, max_xr_gp_pred - config.x_r_max_opt)

    record = {
        "method": method_name,
        "method_type": (
            "baseline_peak_y_only"
            if method_name == "baseline"
            else "constraint_aware_peak_y_max_xr"
        ),
        "target": float(y_target),
        "y_target": float(y_target),
        "attempt": attempt + 1,
        "objective_value": float(result.fun),
        "optimization_time_s": float(opt_time),
        "n_function_evaluations": int(getattr(result, "nfev", -1)),
        "success": bool(result.success),
        "message": str(result.message),
        "y_gp_pred": float(y_gp_pred),
        "y_gp_std": float(y_gp_std),
        "max_xr_gp_pred": float(max_xr_gp_pred),
        "max_xr_gp_std": float(max_xr_gp_std),
        "predicted_constraint_violation_gp": float(predicted_violation_gp),
        "y_sim": float(y_sim),
        "error_gp": float(error_gp),
        "error_sim": float(error_sim),
        "error_gp_vs_sim": float(error_gp_vs_sim),
        "max_xr_sim": float(max_xr_sim),
        "max_xb_sim": float(max_xb_sim),
        "extra_reach": float(extra_reach),
        "constraint_violation": float(constraint_violation),
        "feasible": bool(feasible),
        **params,
    }

    return record


def is_better_record(
    candidate: Dict[str, Any],
    current_best: Dict[str, Any],
) -> bool:
    """
    Compare two validated records.

    Selection logic:
    1. prefer feasible solutions;
    2. among feasible solutions, minimize simulation error;
    3. if none are feasible, minimize constraint violation first, then error.
    """
    candidate_key = (
        0 if candidate["feasible"] else 1,
        candidate["constraint_violation"],
        candidate["error_sim"],
    )

    best_key = (
        0 if current_best["feasible"] else 1,
        current_best["constraint_violation"],
        current_best["error_sim"],
    )

    return candidate_key < best_key


def optimize_for_target(
    y_target: float,
    models: Dict[str, Any],
    method_name: str,
    config: ConstraintAwareConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optimize parameters for a single target using either baseline or
    constraint-aware objective.
    """
    if method_name == "baseline":
        objective_func = objective_baseline
    elif method_name == "constraint_aware":
        objective_func = objective_constraint_aware
    else:
        raise ValueError(f"Unknown method: {method_name}")

    print("-" * 80)
    print(f"Method: {method_name}")
    print(f"Target: {y_target:.3f} m")
    print("-" * 80)

    best_record = None
    all_attempts = []

    for attempt in range(config.n_optimization_attempts):
        print(f"  Attempt {attempt + 1}/{config.n_optimization_attempts}...")

        start_time = time.time()

        result = run_de_optimization(
            objective_func=objective_func,
            y_target=y_target,
            models=models,
            config=config,
            attempt_id=attempt,
        )

        opt_time = time.time() - start_time

        record = build_attempt_record(
            method_name=method_name,
            y_target=y_target,
            attempt=attempt,
            result=result,
            opt_time=opt_time,
            models=models,
            config=config,
        )

        all_attempts.append(record)

        print(
            f"    y_gp={record['y_gp_pred']:.4f} m | "
            f"xr_gp={record['max_xr_gp_pred']:.4f} m | "
            f"y_sim={record['y_sim']:.4f} m | "
            f"xr_sim={record['max_xr_sim']:.4f} m | "
            f"violation={record['constraint_violation'] * 1000:.2f} mm | "
            f"error={record['error_sim'] * 1000:.2f} mm | "
            f"feasible={record['feasible']}"
        )

        if best_record is None or is_better_record(record, best_record):
            best_record = record

    print()
    print("  Best selected solution:")
    print(
        f"    y_sim={best_record['y_sim']:.4f} m | "
        f"error={best_record['error_sim'] * 1000:.2f} mm | "
        f"max_xr={best_record['max_xr_sim']:.4f} m | "
        f"violation={best_record['constraint_violation'] * 1000:.2f} mm | "
        f"feasible={best_record['feasible']}"
    )
    print()

    return best_record, all_attempts


# =============================================================================
# Tables and plots
# =============================================================================

def create_report_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for the report."""
    rows = []

    for method in results_df["method"].unique():
        df_m = results_df[results_df["method"] == method]

        rows.append(
            {
                "method": method,
                "n_targets": len(df_m),
                "mean_error_mm": df_m["error_sim"].mean() * 1000.0,
                "max_error_mm": df_m["error_sim"].max() * 1000.0,
                "feasibility_rate_percent": df_m["feasible"].mean() * 100.0,
                "mean_constraint_violation_mm": (
                    df_m["constraint_violation"].mean() * 1000.0
                ),
                "max_constraint_violation_mm": (
                    df_m["constraint_violation"].max() * 1000.0
                ),
                "mean_max_xr_m": df_m["max_xr_sim"].mean(),
                "max_max_xr_m": df_m["max_xr_sim"].max(),
                "mean_extra_reach_m": df_m["extra_reach"].mean(),
                "max_extra_reach_m": df_m["extra_reach"].max(),
                "mean_optimization_time_s": df_m["optimization_time_s"].mean(),
                "mean_function_evaluations": df_m["n_function_evaluations"].mean(),
            }
        )

    return pd.DataFrame(rows)


def save_final_selected_solution(
    results_df: pd.DataFrame,
    config: ConstraintAwareConfig,
) -> Path:
    """
    Save the final selected constraint-aware solution.

    By default, the selected detailed case is target = 0.64 m.
    This file is intended to be used later by visualization.py.
    """
    target = config.final_selected_target

    df_final = results_df[
        (results_df["method"] == "constraint_aware")
        & (np.isclose(results_df["target"], target))
    ].copy()

    if df_final.empty:
        raise ValueError(
            f"No constraint-aware solution found for target {target:.3f} m."
        )

    if len(df_final) > 1:
        df_final = df_final.sort_values(
            ["feasible", "constraint_violation", "error_sim"],
            ascending=[False, True, True],
        ).head(1)

    final_path = RESULTS_DIR / "final_solution_target064.csv"
    df_final.to_csv(final_path, index=False)

    return final_path


def create_summary_plots(
    results_df: pd.DataFrame,
    config: ConstraintAwareConfig,
) -> None:
    """Create comparison plots between baseline and constraint-aware optimization."""
    methods = list(results_df["method"].unique())
    targets = np.sort(results_df["target"].unique())
    width = 0.018

    # -------------------------------------------------------------------------
    # Target tracking plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for method in methods:
        df_m = results_df[results_df["method"] == method].sort_values("target")

        plt.plot(
            df_m["target"],
            df_m["y_sim"],
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.plot(
        [targets.min(), targets.max()],
        [targets.min(), targets.max()],
        "--",
        linewidth=2,
        label="Ideal tracking",
    )

    plt.axhline(
        config.x_r_max_true,
        linestyle=":",
        linewidth=2,
        label="Nominal robot reach",
    )

    plt.xlabel("Target outreach [m]")
    plt.ylabel("Simulated peak outreach [m]")
    plt.title("Target tracking: baseline vs constraint-aware")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "target_tracking_baseline_vs_constraint.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Constraint violation plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for i, method in enumerate(methods):
        df_m = results_df[results_df["method"] == method].sort_values("target")
        offset = (i - 0.5) * width

        plt.bar(
            df_m["target"] + offset,
            df_m["constraint_violation"] * 1000.0,
            width=width,
            label=method,
            edgecolor="black",
        )

    plt.axhline(0.0, linestyle="--", linewidth=2)
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Constraint violation [mm]")
    plt.title("Robot constraint violation")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "constraint_violation_baseline_vs_constraint.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Simulation error plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for i, method in enumerate(methods):
        df_m = results_df[results_df["method"] == method].sort_values("target")
        offset = (i - 0.5) * width

        plt.bar(
            df_m["target"] + offset,
            df_m["error_sim"] * 1000.0,
            width=width,
            label=method,
            edgecolor="black",
        )

    plt.xlabel("Target outreach [m]")
    plt.ylabel("Simulation absolute error [mm]")
    plt.title("Simulation target error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "simulation_error_baseline_vs_constraint.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # max_xr plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for method in methods:
        df_m = results_df[results_df["method"] == method].sort_values("target")

        plt.plot(
            df_m["target"],
            df_m["max_xr_sim"],
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.axhline(
        config.x_r_max_true,
        linestyle="--",
        linewidth=2,
        label="True robot limit",
    )

    plt.axhline(
        config.x_r_max_opt,
        linestyle=":",
        linewidth=2,
        label="Optimization safety limit",
    )

    plt.xlabel("Target outreach [m]")
    plt.ylabel("Simulated max_xr [m]")
    plt.title("Maximum robot displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "max_xr_baseline_vs_constraint.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Optional legacy names for compatibility with older notes/scripts
    # -------------------------------------------------------------------------
    legacy_map = {
        "target_tracking_baseline_vs_constraint.png": "baseline_vs_constraint_targets.png",
        "constraint_violation_baseline_vs_constraint.png": "baseline_vs_constraint_violation.png",
        "simulation_error_baseline_vs_constraint.png": "baseline_vs_constraint_error.png",
        "max_xr_baseline_vs_constraint.png": "baseline_vs_constraint_max_xr.png",
    }

    for new_name, old_name in legacy_map.items():
        new_path = FIGURES_DIR / new_name
        old_path = FIGURES_DIR / old_name

        if new_path.exists():
            old_path.write_bytes(new_path.read_bytes())

# =============================================================================
# Additional report / presentation plots
# =============================================================================

def get_params_from_result_row(row: pd.Series) -> Dict[str, float]:
    """
    Reconstruct a parameter dictionary from a results DataFrame row.

    The constraint-aware results table stores physical parameters directly as:
    Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start, Mr.
    """
    return {
        "Kb": float(row["Kb"]),
        "Kr": float(row["Kr"]),
        "Mb": float(row["Mb"]),
        "hb": float(row["hb"]),
        "hr": float(row["hr"]),
        "f0": float(row["f0"]),
        "f1": float(row["f1"]),
        "A": float(row["A"]),
        "x_r_start": float(row["x_r_start"]),
        "Mr": float(row["Mr"]),
    }


def plot_optimized_parameters_comparison(
    results_df: pd.DataFrame,
    config: ConstraintAwareConfig,
) -> None:
    """
    Plot optimized controllable parameters for baseline vs constraint-aware
    solutions across targets.

    Parameters are normalized to [0, 1] using their optimization bounds so that
    parameters with different units can be compared in one figure.
    """
    save_path = FIGURES_DIR / "optimized_parameters_baseline_vs_constraint.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    controllable_names = config.optimized_columns()
    bounds = config.bounds()

    methods = ["baseline", "constraint_aware"]
    method_labels = {
        "baseline": "Baseline",
        "constraint_aware": "Constraint-aware",
    }

    method_styles = {
        "baseline": {"linestyle": "--", "marker": "o"},
        "constraint_aware": {"linestyle": "-", "marker": "s"},
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, param_name, bound in zip(axes, controllable_names, bounds):
        lower, upper = bound

        for method in methods:
            df_m = results_df[results_df["method"] == method].sort_values("target")

            if df_m.empty:
                continue

            values = df_m[param_name].values.astype(float)
            values_norm = (values - lower) / (upper - lower)

            ax.plot(
                df_m["target"],
                values_norm,
                linestyle=method_styles[method]["linestyle"],
                marker=method_styles[method]["marker"],
                linewidth=2,
                markersize=7,
                label=method_labels[method],
            )

        ax.set_title(param_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Target outreach [m]")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.03,
            0.06,
            f"Bounds: [{lower:.3g}, {upper:.3g}]",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11)

    fig.suptitle(
        "Optimized Controllable Parameters: Baseline vs Constraint-aware",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def simulate_candidate_with_solution(
    params: Dict[str, float],
    y_target: float,
    config: ConstraintAwareConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run the true dynamic simulator and return full time response.

    This version is compatible with dynamics.py versions using return_full=True.

    Returns:
        t, y, x_b, x_r, metrics
    """
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=config.x_r_max_true,
            return_metrics=True,
            return_full=True,
        )
    except TypeError:
        output = simulate_system(
            params,
            T_sim=60.0,
            dt=0.001,
            return_full=True,
        )

    if not isinstance(output, tuple):
        raise RuntimeError(
            "Unexpected simulate_system output. Expected tuple with solution."
        )

    sol = None
    metrics = None
    peak_y = None

    for item in output:
        if hasattr(item, "t") and hasattr(item, "y"):
            sol = item
        elif isinstance(item, dict) and "peak_y" in item:
            metrics = item
        elif isinstance(item, (float, int, np.floating)):
            peak_y = float(item)

    if sol is None:
        raise RuntimeError(
            "Could not extract ODE solution from simulate_system output."
        )

    t = sol.t
    x_b = sol.y[2]
    x_r = sol.y[3]
    y = x_b + x_r

    if metrics is None:
        peak_y_calc = float(np.max(y))
        max_xr = float(np.max(x_r))
        max_xb = float(np.max(x_b))

        metrics = {
            "peak_y": peak_y_calc if peak_y is None else peak_y,
            "max_xr": max_xr,
            "max_xb": max_xb,
            "extra_reach": float(peak_y_calc - config.x_r_max_true),
            "constraint_violation": float(max(0.0, max_xr - config.x_r_max_true)),
            "target_error": float(abs(peak_y_calc - y_target)),
        }

    return t, y, x_b, x_r, metrics

def plot_time_responses_baseline_vs_constraint(
    results_df: pd.DataFrame,
    config: ConstraintAwareConfig,
) -> None:
    """
    Generate one time-response comparison figure per target.

    Each figure compares baseline and constraint-aware solutions using the true
    dynamic simulator.

    Saved examples:
        time_response_target0520_baseline_vs_constraint.png
        time_response_target0580_baseline_vs_constraint.png
        time_response_target0640_baseline_vs_constraint.png
        time_response_target0700_baseline_vs_constraint.png
    """
    methods = ["baseline", "constraint_aware"]

    method_labels = {
        "baseline": "Baseline",
        "constraint_aware": "Constraint-aware",
    }

    method_styles = {
        "baseline": "--",
        "constraint_aware": "-",
    }

    for target in np.sort(results_df["target"].unique()):
        df_target = results_df[np.isclose(results_df["target"], target)]

        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

        for method in methods:
            df_m = df_target[df_target["method"] == method]

            if df_m.empty:
                continue

            row = df_m.iloc[0]
            params = get_params_from_result_row(row)

            t, y, x_b, x_r, metrics = simulate_candidate_with_solution(
                params=params,
                y_target=float(target),
                config=config,
            )

            label = method_labels[method]
            linestyle = method_styles[method]

            axes[0].plot(
                t,
                y,
                linestyle=linestyle,
                linewidth=2,
                label=(
                    f"{label} "
                    f"(peak={metrics['peak_y']:.3f} m, "
                    f"err={abs(metrics['peak_y'] - target) * 1000:.1f} mm)"
                ),
            )

            axes[1].plot(
                t,
                x_r,
                linestyle=linestyle,
                linewidth=2,
                label=(
                    f"{label} "
                    f"(max_xr={metrics['max_xr']:.3f} m)"
                ),
            )

            axes[2].plot(
                t,
                x_b,
                linestyle=linestyle,
                linewidth=2,
                label=(
                    f"{label} "
                    f"(max_xb={metrics['max_xb']:.3f} m)"
                ),
            )

        axes[0].axhline(
            target,
            linestyle=":",
            linewidth=2,
            label=f"Target = {target:.3f} m",
        )

        axes[0].axhline(
            config.x_r_max_true,
            linestyle="--",
            linewidth=1.8,
            label="Nominal robot reach",
        )

        axes[1].axhline(
            config.x_r_max_true,
            linestyle=":",
            linewidth=2,
            label=f"Robot limit = {config.x_r_max_true:.3f} m",
        )

        axes[0].set_ylabel("Total outreach y(t) [m]")
        axes[1].set_ylabel("Robot displacement x_r(t) [m]")
        axes[2].set_ylabel("Base displacement x_b(t) [m]")
        axes[2].set_xlabel("Time [s]")

        axes[0].set_title(
            f"Time Response Comparison for Target = {target:.3f} m",
            fontsize=14,
            fontweight="bold",
        )

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        plt.tight_layout()

        target_tag = f"{int(round(target * 1000)):04d}"
        save_path = (
            FIGURES_DIR
            / f"time_response_target{target_tag}_baseline_vs_constraint.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")


def print_final_summary(
    results_df: pd.DataFrame,
    report_summary_df: pd.DataFrame,
    final_solution_path: Path,
    config: ConstraintAwareConfig,
) -> None:
    """Print final comparison summary."""
    print("=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)

    df_print = results_df.copy()
    df_print["error_sim_mm"] = df_print["error_sim"] * 1000.0
    df_print["constraint_violation_mm"] = (
        df_print["constraint_violation"] * 1000.0
    )

    display_cols = [
        "method",
        "target",
        "y_sim",
        "error_sim_mm",
        "max_xr_sim",
        "constraint_violation_mm",
        "feasible",
    ]

    print(df_print[display_cols].to_string(index=False))
    print()

    for _, row in report_summary_df.iterrows():
        print(f"{row['method']}:")
        print(f"  Mean simulation error:     {row['mean_error_mm']:.2f} mm")
        print(f"  Max simulation error:      {row['max_error_mm']:.2f} mm")
        print(f"  Feasibility rate:          {row['feasibility_rate_percent']:.1f}%")
        print(
            f"  Mean constraint violation: "
            f"{row['mean_constraint_violation_mm']:.2f} mm"
        )
        print(
            f"  Max constraint violation:  "
            f"{row['max_constraint_violation_mm']:.2f} mm"
        )
        print(f"  Mean max_xr:               {row['mean_max_xr_m']:.4f} m")
        print(f"  Max max_xr:                {row['max_max_xr_m']:.4f} m")
        print(f"  Mean extra reach:          {row['mean_extra_reach_m']:.4f} m")
        print(f"  Max extra reach:           {row['max_extra_reach_m']:.4f} m")
        print()

    print("Final selected solution:")
    print(f"  Method:                    constraint_aware")
    print(f"  Target:                    {config.final_selected_target:.3f} m")
    print(f"  Saved to:                  {final_solution_path}")
    print()


def save_outputs(
    results_df: pd.DataFrame,
    attempts_df: pd.DataFrame,
    config: ConstraintAwareConfig,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Save result tables, report summary and final selected solution."""
    results_path = RESULTS_DIR / "baseline_vs_constraint_results.csv"
    attempts_path = RESULTS_DIR / "baseline_vs_constraint_all_attempts.csv"
    summary_path = RESULTS_DIR / "baseline_vs_constraint_summary.csv"
    report_summary_path = RESULTS_DIR / "report_summary_baseline_vs_constraint.csv"

    results_df.to_csv(results_path, index=False)
    attempts_df.to_csv(attempts_path, index=False)

    report_summary_df = create_report_summary(results_df)
    report_summary_df.to_csv(report_summary_path, index=False)

    # Keep a second summary filename for compatibility with previous version.
    report_summary_df.to_csv(summary_path, index=False)

    final_solution_path = save_final_selected_solution(results_df, config)

    return (
        results_path,
        attempts_path,
        summary_path,
        report_summary_path,
        final_solution_path,
    )


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    """Run the full constraint-aware inverse optimization comparison."""
    ensure_output_dirs()

    config = ConstraintAwareConfig()

    print("=" * 80)
    print("CONSTRAINT-AWARE INVERSE OPTIMIZATION")
    print("=" * 80)
    print()

    print("Settings:")
    print(f"  Targets:                  {np.array(config.targets)}")
    print(f"  True robot limit:         {config.x_r_max_true:.3f} m")
    print(f"  Optimization robot limit: {config.x_r_max_opt:.3f} m")
    print(f"  Constraint lambda:        {config.lambda_constraint}")
    print(f"  Uncertainty lambda:       {config.lambda_uncertainty}")
    print(f"  Fixed parameters:         {config.fixed_params()}")
    print(f"  Bounds:                   {config.bounds()}")
    print(f"  Optimization attempts:    {config.n_optimization_attempts}")
    print(f"  Final selected target:    {config.final_selected_target:.3f} m")
    print()

    models = load_surrogates()

    best_records = []
    all_attempt_records = []

    for y_target in config.targets:
        for method_name in ["baseline", "constraint_aware"]:
            best_record, all_attempts = optimize_for_target(
                y_target=float(y_target),
                models=models,
                method_name=method_name,
                config=config,
            )

            best_records.append(best_record)
            all_attempt_records.extend(all_attempts)

    results_df = pd.DataFrame(best_records)
    attempts_df = pd.DataFrame(all_attempt_records)

    (
        results_path,
        attempts_path,
        summary_path,
        report_summary_path,
        final_solution_path,
    ) = save_outputs(
        results_df=results_df,
        attempts_df=attempts_df,
        config=config,
    )

    report_summary_df = pd.read_csv(report_summary_path)

    create_summary_plots(
        results_df=results_df,
        config=config,
    )

    plot_optimized_parameters_comparison(
        results_df=results_df,
        config=config,
    )

    plot_time_responses_baseline_vs_constraint(
        results_df=results_df,
        config=config,
    )

    print_final_summary(
        results_df=results_df,
        report_summary_df=report_summary_df,
        final_solution_path=final_solution_path,
        config=config,
    )

    print("Saved files:")
    print(f"  {results_path}")
    print(f"  {attempts_path}")
    print(f"  {summary_path}")
    print(f"  {report_summary_path}")
    print(f"  {final_solution_path}")
    print(f"  {FIGURES_DIR / 'target_tracking_baseline_vs_constraint.png'}")
    print(f"  {FIGURES_DIR / 'constraint_violation_baseline_vs_constraint.png'}")
    print(f"  {FIGURES_DIR / 'simulation_error_baseline_vs_constraint.png'}")
    print(f"  {FIGURES_DIR / 'max_xr_baseline_vs_constraint.png'}")
    print(f"  {FIGURES_DIR / 'optimized_parameters_baseline_vs_constraint.png'}")

    for target in np.sort(results_df["target"].unique()):
        target_tag = f"{int(round(target * 1000)):04d}"
        print(
            f"  {FIGURES_DIR / f'time_response_target{target_tag}_baseline_vs_constraint.png'}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()