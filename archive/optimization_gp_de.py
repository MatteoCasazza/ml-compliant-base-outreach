"""
optimization_gp_de.py
=====================

Constraint-aware inverse optimization using the trained Gaussian Process pair
and Differential Evolution.

Main purpose
------------
Given a desired outreach target y_target, find controllable parameters

    [Kr, hr, f0, f1, A, x_r_start]

while keeping the true absolute robot displacement within the physical limit

    max_abs_xr <= 0.500 m.

The optimization itself uses trained GP surrogate models:

    GP_peak_y       : predicts peak_y
    GP_max_abs_xr   : predicts max_abs_xr

A conservative safety limit is used inside the surrogate objective:

    max_abs_xr_pred <= 0.495 m

All final candidates are validated with the true dynamic simulator and the true
physical limit of 0.500 m.

Why Differential Evolution?
---------------------------
The optimization landscape can be non-convex because of dynamic amplification,
frequency effects and constraint boundaries. Differential Evolution is a robust
global derivative-free optimizer for low-dimensional bounded problems.

Outputs
-------
results/optimization_gp_de/gp_de_results.csv
results/optimization_gp_de/gp_de_all_attempts.csv
results/optimization_gp_de/gp_de_convergence.csv
results/optimization_gp_de/gp_de_summary.csv
results/optimization_gp_de/gp_de_objective_settings.csv
results/optimization_gp_de/gp_de_uncertainty_diagnostics.csv

figures/optimization_gp_de/gp_de_target_tracking.png
figures/optimization_gp_de/gp_de_target_error.png
figures/optimization_gp_de/gp_de_constraint_validation.png
figures/optimization_gp_de/gp_de_predicted_vs_true.png
figures/optimization_gp_de/gp_de_optimized_parameters.png
figures/optimization_gp_de/gp_de_convergence.png
figures/optimization_gp_de/gp_de_uncertainty_diagnostics.png
figures/optimization_gp_de/time_response_target*.png

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import copy
import pickle
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from dynamics import simulate_system


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_GP_DIR = PROJECT_ROOT / "results" / "gp"
CONSTRAINT_GP_DIR = PROJECT_ROOT / "results" / "gp_constraints"
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_gp_de"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_gp_de"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GPDEConfig:
    """
    Configuration for GP + Differential Evolution inverse optimization.

    Optimized variables:
        [Kr, hr, f0, f1, A, x_r_start]

    Fixed variables:
        [Kb, Mb, hb, Mr]
    """

    # ------------------------------------------------------------------
    # Fixed physical parameters
    # ------------------------------------------------------------------
    Kb: float = 1000.0
    Mb: float = 20.0
    hb: float = 0.10
    Mr: float = 10.0

    # ------------------------------------------------------------------
    # Physical and optimization limits
    # ------------------------------------------------------------------
    robot_limit_true: float = 0.500
    robot_limit_opt: float = 0.495
    feasibility_tolerance_m: float = 1e-9

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    targets: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)

    # ------------------------------------------------------------------
    # Bounds for optimized variables: [Kr, hr, f0, f1, A, x_r_start]
    # These bounds match the high-outreach optimization-relevant region.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Normalized objective scales
    # ------------------------------------------------------------------
    # 10 mm target error gives tracking cost = 1.
    y_error_scale_m: float = 0.010

    # 5 mm predicted constraint violation gives unweighted violation cost = 1.
    x_constraint_scale_m: float = 0.005

    # Constraint penalty. With lambda=10, a 5 mm violation costs 10,
    # equivalent to about 31.6 mm of target error.
    lambda_constraint: float = 10.0

    # Small uncertainty penalties available because GP returns predictive std.
    # Set to zero if you want the simpler objective with no uncertainty terms.
    lambda_uncertainty_peak_y: float = 0.05
    lambda_uncertainty_max_abs_xr: float = 0.10

    # Optional conservative constraint: if beta > 0, the constraint penalty uses
    # max_abs_xr_pred + beta * std_max_abs_xr instead of only the mean prediction.
    # Default beta=0 keeps the objective simple; uncertainty is already penalized.
    beta_constraint_std: float = 0.0

    # ------------------------------------------------------------------
    # Differential Evolution settings
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    run_uncertainty_diagnostics: bool = True
    n_uncertainty_samples: int = 2500

    # Optional preliminary lambda sweep for the hardest target.
    # Disabled by default because it adds extra runtime. Enable only if you want
    # to empirically calibrate lambda_constraint before the final run.
    run_lambda_sweep: bool = True
    lambda_sweep_values: Tuple[float, ...] = (3.0, 5.0, 10.0, 30.0)
    lambda_sweep_target: float = 0.65
    lambda_sweep_attempts: int = 2
    lambda_sweep_maxiter: int = 70
    lambda_sweep_popsize: int = 12

    @staticmethod
    def input_columns() -> List[str]:
        """Full GP input order used during training."""
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
        """Optimizer vector order."""
        return [
            "Kr",
            "hr",
            "f0",
            "f1",
            "A",
            "x_r_start",
        ]

    def fixed_params(self) -> Dict[str, float]:
        return {
            "Kb": self.Kb,
            "Mb": self.Mb,
            "hb": self.hb,
            "Mr": self.Mr,
        }

    def bounds(self) -> List[Tuple[float, float]]:
        return [
            (self.Kr_min, self.Kr_max),
            (self.hr_min, self.hr_max),
            (self.f0_min, self.f0_max),
            (self.f1_min, self.f1_max),
            (self.A_min, self.A_max),
            (self.x_r_start_min, self.x_r_start_max),
        ]

    def de_settings(self) -> Dict[str, Any]:
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
# BASIC UTILITIES
# =============================================================================

def ensure_output_dirs() -> None:
    for directory in [RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_model_file(path: str | Path) -> Any:
    """Load a joblib/pickle model or scaler file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_surrogates() -> Dict[str, Any]:
    """Load final GP_peak_y and GP_max_abs_xr models and scalers."""
    print("Loading GP surrogate models...")

    models = {
        "gp_peak_y": load_model_file(MAIN_GP_DIR / "gp_model.pkl"),
        "scaler_X_peak_y": load_model_file(MAIN_GP_DIR / "scaler_X.pkl"),
        "scaler_y_peak_y": load_model_file(MAIN_GP_DIR / "scaler_y.pkl"),
        "gp_max_abs_xr": load_model_file(CONSTRAINT_GP_DIR / "gp_max_abs_xr_model.pkl"),
        "scaler_X_max_abs_xr": load_model_file(CONSTRAINT_GP_DIR / "scaler_X_max_abs_xr.pkl"),
        "scaler_y_max_abs_xr": load_model_file(CONSTRAINT_GP_DIR / "scaler_y_max_abs_xr.pkl"),
    }

    print("✓ Models loaded")
    print(f"  GP_peak_y directory:       {MAIN_GP_DIR}")
    print(f"  GP_max_abs_xr directory:   {CONSTRAINT_GP_DIR}")
    print()

    return models


def vector_to_params(x: np.ndarray, config: GPDEConfig) -> Dict[str, float]:
    """Convert optimizer vector to full parameter dictionary."""
    Kr, hr, f0, f1, A, x_r_start = x

    return {
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


def params_to_feature_vector(params: Dict[str, float], config: GPDEConfig) -> np.ndarray:
    """Convert full params dictionary to GP feature vector."""
    return np.array([[params[col] for col in config.input_columns()]], dtype=float)


def vectors_to_feature_matrix(X_opt: np.ndarray, config: GPDEConfig) -> np.ndarray:
    """Convert many optimizer vectors to full GP feature matrix."""
    X_opt = np.asarray(X_opt, dtype=float)
    if X_opt.ndim == 1:
        X_opt = X_opt.reshape(1, -1)

    rows = []
    for x in X_opt:
        params = vector_to_params(x, config)
        rows.append([params[col] for col in config.input_columns()])

    return np.asarray(rows, dtype=float)


# =============================================================================
# GP PREDICTION
# =============================================================================

def predict_gp_single(
    params: Dict[str, float],
    models: Dict[str, Any],
    config: GPDEConfig,
) -> Dict[str, float]:
    """Predict peak_y and max_abs_xr with associated GP standard deviations."""
    X = params_to_feature_vector(params, config)

    # Peak_y GP
    X_peak_scaled = models["scaler_X_peak_y"].transform(X)
    y_peak_scaled, std_peak_scaled = models["gp_peak_y"].predict(
        X_peak_scaled,
        return_std=True,
    )
    peak_y = models["scaler_y_peak_y"].inverse_transform(
        y_peak_scaled.reshape(-1, 1)
    ).ravel()[0]
    peak_y_std = std_peak_scaled[0] * models["scaler_y_peak_y"].scale_[0]

    # max_abs_xr GP
    X_xr_scaled = models["scaler_X_max_abs_xr"].transform(X)
    xr_scaled, std_xr_scaled = models["gp_max_abs_xr"].predict(
        X_xr_scaled,
        return_std=True,
    )
    max_abs_xr = models["scaler_y_max_abs_xr"].inverse_transform(
        xr_scaled.reshape(-1, 1)
    ).ravel()[0]
    max_abs_xr_std = std_xr_scaled[0] * models["scaler_y_max_abs_xr"].scale_[0]

    return {
        "peak_y_pred": float(peak_y),
        "peak_y_std": float(peak_y_std),
        "max_abs_xr_pred": float(max_abs_xr),
        "max_abs_xr_std": float(max_abs_xr_std),
    }


def predict_gp_batch(
    X_opt: np.ndarray,
    models: Dict[str, Any],
    config: GPDEConfig,
) -> pd.DataFrame:
    """Vectorized GP prediction for many optimizer vectors."""
    X_full = vectors_to_feature_matrix(X_opt, config)

    X_peak_scaled = models["scaler_X_peak_y"].transform(X_full)
    y_peak_scaled, std_peak_scaled = models["gp_peak_y"].predict(
        X_peak_scaled,
        return_std=True,
    )
    peak_y = models["scaler_y_peak_y"].inverse_transform(
        y_peak_scaled.reshape(-1, 1)
    ).ravel()
    peak_y_std = std_peak_scaled * models["scaler_y_peak_y"].scale_[0]

    X_xr_scaled = models["scaler_X_max_abs_xr"].transform(X_full)
    xr_scaled, std_xr_scaled = models["gp_max_abs_xr"].predict(
        X_xr_scaled,
        return_std=True,
    )
    max_abs_xr = models["scaler_y_max_abs_xr"].inverse_transform(
        xr_scaled.reshape(-1, 1)
    ).ravel()
    max_abs_xr_std = std_xr_scaled * models["scaler_y_max_abs_xr"].scale_[0]

    df = pd.DataFrame(X_full, columns=config.input_columns())
    df["peak_y_pred"] = peak_y
    df["peak_y_std"] = peak_y_std
    df["max_abs_xr_pred"] = max_abs_xr
    df["max_abs_xr_std"] = max_abs_xr_std

    return df


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective_terms(
    prediction: Dict[str, float],
    y_target: float,
    config: GPDEConfig,
) -> Dict[str, float]:
    """
    Compute normalized objective terms for GP + DE.

    J = tracking
        + lambda_constraint * constraint_violation
        + lambda_uncertainty_peak_y * peak_y_uncertainty
        + lambda_uncertainty_max_abs_xr * max_abs_xr_uncertainty

    where every term is dimensionless.
    """
    peak_y_pred = prediction["peak_y_pred"]
    peak_y_std = prediction["peak_y_std"]
    max_abs_xr_pred = prediction["max_abs_xr_pred"]
    max_abs_xr_std = prediction["max_abs_xr_std"]

    tracking_error_scaled = (peak_y_pred - y_target) / config.y_error_scale_m
    tracking_cost = tracking_error_scaled ** 2

    constraint_value = max_abs_xr_pred + config.beta_constraint_std * max_abs_xr_std
    predicted_violation_m = max(0.0, constraint_value - config.robot_limit_opt)
    predicted_violation_scaled = predicted_violation_m / config.x_constraint_scale_m
    constraint_cost = config.lambda_constraint * predicted_violation_scaled ** 2

    peak_y_uncertainty_scaled = peak_y_std / config.y_error_scale_m
    peak_y_uncertainty_cost = (
        config.lambda_uncertainty_peak_y * peak_y_uncertainty_scaled ** 2
    )

    max_abs_xr_uncertainty_scaled = max_abs_xr_std / config.x_constraint_scale_m
    max_abs_xr_uncertainty_cost = (
        config.lambda_uncertainty_max_abs_xr * max_abs_xr_uncertainty_scaled ** 2
    )

    total = (
        tracking_cost
        + constraint_cost
        + peak_y_uncertainty_cost
        + max_abs_xr_uncertainty_cost
    )

    return {
        "objective_value": float(total),
        "tracking_cost": float(tracking_cost),
        "constraint_cost": float(constraint_cost),
        "peak_y_uncertainty_cost": float(peak_y_uncertainty_cost),
        "max_abs_xr_uncertainty_cost": float(max_abs_xr_uncertainty_cost),
        "tracking_error_scaled": float(tracking_error_scaled),
        "predicted_violation_m": float(predicted_violation_m),
        "predicted_violation_scaled": float(predicted_violation_scaled),
        "constraint_value_m": float(constraint_value),
    }


def objective_gp_de(
    x: np.ndarray,
    y_target: float,
    models: Dict[str, Any],
    config: GPDEConfig,
) -> float:
    """Objective minimized by Differential Evolution."""
    params = vector_to_params(x, config)
    prediction = predict_gp_single(params, models, config)
    terms = objective_terms(prediction, y_target, config)
    return float(terms["objective_value"])


# =============================================================================
# TRUE SIMULATOR VALIDATION
# =============================================================================

def _extract_metrics_from_simulator_output(output: Any) -> Dict[str, Any]:
    """Extract a metrics dictionary from different simulate_system return formats."""
    if isinstance(output, dict):
        return dict(output)

    if isinstance(output, tuple):
        # Prefer an explicit metrics dictionary containing peak_y.
        for item in output:
            if isinstance(item, dict) and "peak_y" in item:
                return dict(item)

        # Fallback: some old versions returned (peak_y, metrics).
        if len(output) >= 2 and isinstance(output[1], dict):
            return dict(output[1])

    raise TypeError(
        "Could not extract metrics dictionary from simulate_system output. "
        f"Received type: {type(output)}"
    )


def _complete_abs_metrics(metrics: Dict[str, Any], config: GPDEConfig) -> Dict[str, Any]:
    """
    Ensure the simulator metrics include absolute constraint quantities.
    """
    completed = dict(metrics)

    # peak_y is mandatory.
    if "peak_y" not in completed:
        raise KeyError("Simulator metrics do not contain 'peak_y'.")

    # max_abs_xr preferred; otherwise reconstruct from max_xr/min_xr if possible.
    if "max_abs_xr" not in completed:
        if "max_xr" in completed and "min_xr" in completed:
            completed["max_abs_xr"] = max(abs(float(completed["max_xr"])), abs(float(completed["min_xr"])))
        elif "max_xr" in completed:
            # Backward-compatible fallback. Less complete if min_xr is absent.
            completed["max_abs_xr"] = abs(float(completed["max_xr"]))
        else:
            raise KeyError(
                "Simulator metrics do not contain 'max_abs_xr' or a usable "
                "combination of 'max_xr'/'min_xr'."
            )

    if "constraint_violation_abs" not in completed:
        completed["constraint_violation_abs"] = max(
            0.0,
            float(completed["max_abs_xr"]) - config.robot_limit_true,
        )

    if "feasible_abs" not in completed:
        completed["feasible_abs"] = (
            float(completed["constraint_violation_abs"]) <= config.feasibility_tolerance_m
        )

    if "extra_reach" not in completed:
        completed["extra_reach"] = float(completed["peak_y"]) - config.robot_limit_true

    if "max_abs_xb" not in completed:
        if "max_xb" in completed and "min_xb" in completed:
            completed["max_abs_xb"] = max(abs(float(completed["max_xb"])), abs(float(completed["min_xb"])))
        elif "max_xb" in completed:
            completed["max_abs_xb"] = abs(float(completed["max_xb"]))
        else:
            completed["max_abs_xb"] = np.nan

    return completed


def simulate_candidate(
    params: Dict[str, float],
    y_target: float,
    config: GPDEConfig,
) -> Dict[str, Any]:
    """Validate a candidate with the true dynamic simulator."""
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=config.robot_limit_true,
            return_metrics=True,
        )
    except TypeError:
        # Backward-compatible fallback for older dynamics.py versions.
        output = simulate_system(
            params,
            return_metrics=True,
        )

    metrics = _extract_metrics_from_simulator_output(output)
    return _complete_abs_metrics(metrics, config)


def simulate_candidate_with_solution(
    params: Dict[str, float],
    y_target: float,
    config: GPDEConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Run the true simulator and try to return full time response.

    Returns None if the local dynamics.py does not support return_full=True.
    """
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=config.robot_limit_true,
            return_metrics=True,
            return_full=True,
        )
    except TypeError:
        try:
            output = simulate_system(params, return_metrics=True, return_full=True)
        except TypeError:
            return None

    if not isinstance(output, tuple):
        return None

    sol = None
    metrics = None

    for item in output:
        if hasattr(item, "t") and hasattr(item, "y"):
            sol = item
        elif isinstance(item, dict) and "peak_y" in item:
            metrics = dict(item)

    if sol is None:
        return None

    t = np.asarray(sol.t)

    try:
        x_b = np.asarray(sol.y[2])
        x_r = np.asarray(sol.y[3])
    except Exception:
        return None

    y = x_b + x_r

    if metrics is None:
        metrics = {
            "peak_y": float(np.max(y)),
            "max_abs_xr": float(np.max(np.abs(x_r))),
            "max_abs_xb": float(np.max(np.abs(x_b))),
            "extra_reach": float(np.max(y) - config.robot_limit_true),
            "constraint_violation_abs": float(max(0.0, np.max(np.abs(x_r)) - config.robot_limit_true)),
        }

    metrics = _complete_abs_metrics(metrics, config)
    return t, y, x_b, x_r, metrics


# =============================================================================
# DIFFERENTIAL EVOLUTION OPTIMIZATION
# =============================================================================

def run_de_optimization(
    y_target: float,
    models: Dict[str, Any],
    config: GPDEConfig,
    attempt_id: int,
) -> Tuple[Any, pd.DataFrame]:
    """Run one Differential Evolution attempt and log best-J convergence."""
    seed = 1000 + 100 * attempt_id + int(round(y_target * 1000))
    convergence_rows: List[Dict[str, Any]] = []
    generation_counter = {"generation": 0}

    def callback(xk: np.ndarray, convergence: float) -> bool:
        generation_counter["generation"] += 1
        j_value = objective_gp_de(xk, y_target, models, config)
        convergence_rows.append(
            {
                "target": float(y_target),
                "attempt": attempt_id + 1,
                "generation": generation_counter["generation"],
                "best_objective_so_far": float(j_value),
                "scipy_convergence": float(convergence),
            }
        )
        return False

    result = differential_evolution(
        objective_gp_de,
        bounds=config.bounds(),
        args=(y_target, models, config),
        seed=seed,
        callback=callback,
        **config.de_settings(),
    )

    # Include final polished point even if callback did not record it.
    convergence_rows.append(
        {
            "target": float(y_target),
            "attempt": attempt_id + 1,
            "generation": generation_counter["generation"] + 1,
            "best_objective_so_far": float(result.fun),
            "scipy_convergence": np.nan,
        }
    )

    return result, pd.DataFrame(convergence_rows)


def build_attempt_record(
    y_target: float,
    attempt_id: int,
    result: Any,
    optimization_time_s: float,
    models: Dict[str, Any],
    config: GPDEConfig,
) -> Dict[str, Any]:
    """Build one full record including surrogate prediction and true validation."""
    params = vector_to_params(result.x, config)
    prediction = predict_gp_single(params, models, config)
    terms = objective_terms(prediction, y_target, config)
    sim_metrics = simulate_candidate(params, y_target, config)

    peak_y_true = float(sim_metrics["peak_y"])
    max_abs_xr_true = float(sim_metrics["max_abs_xr"])
    constraint_violation_abs = float(sim_metrics["constraint_violation_abs"])
    feasible_abs = bool(sim_metrics["feasible_abs"])

    peak_y_error_true = abs(peak_y_true - y_target)
    peak_y_error_pred = abs(prediction["peak_y_pred"] - y_target)

    predicted_safety_violation_m = max(
        0.0,
        float(prediction["max_abs_xr_pred"]) - config.robot_limit_opt,
    )

    peak_y_pred_error_signed_m = float(prediction["peak_y_pred"] - peak_y_true)
    max_abs_xr_pred_error_signed_m = float(
        prediction["max_abs_xr_pred"] - max_abs_xr_true
    )

    reachability_gap_m = max(0.0, float(y_target - peak_y_true))

    record = {
        "method": "GP_DE",
        "target": float(y_target),
        "attempt": attempt_id + 1,
        "success": bool(result.success),
        "message": str(result.message),
        "objective_value": float(result.fun),
        "optimization_time_s": float(optimization_time_s),
        "surrogate_function_evaluations": int(getattr(result, "nfev", -1)),
        "true_simulator_calls_online": 1,
        "offline_dataset_calls": 3000,
        # Predicted quantities
        "peak_y_pred": float(prediction["peak_y_pred"]),
        "peak_y_std": float(prediction["peak_y_std"]),
        "max_abs_xr_pred": float(prediction["max_abs_xr_pred"]),
        "max_abs_xr_std": float(prediction["max_abs_xr_std"]),
        "constraint_value_m": float(terms["constraint_value_m"]),
        "predicted_violation_m": float(terms["predicted_violation_m"]),
        "predicted_safety_violation_m": float(predicted_safety_violation_m),
        "predicted_safety_violation_mm": float(predicted_safety_violation_m * 1000.0),  
        # Objective terms
        "tracking_cost": float(terms["tracking_cost"]),
        "constraint_cost": float(terms["constraint_cost"]),
        "peak_y_uncertainty_cost": float(terms["peak_y_uncertainty_cost"]),
        "max_abs_xr_uncertainty_cost": float(terms["max_abs_xr_uncertainty_cost"]),
        "tracking_error_scaled": float(terms["tracking_error_scaled"]),
        "predicted_violation_scaled": float(terms["predicted_violation_scaled"]),
        # True simulator quantities
        "peak_y_true": peak_y_true,
        "target_error_m": float(peak_y_error_true),
        "target_error_mm": float(peak_y_error_true * 1000.0),
        "target_error_pred_m": float(peak_y_error_pred),
        "gp_peak_y_error_vs_true_m": float(abs(prediction["peak_y_pred"] - peak_y_true)),
        "gp_peak_y_error_vs_true_mm": float(abs(prediction["peak_y_pred"] - peak_y_true) * 1000.0),
        "gp_peak_y_error_signed_mm": float(peak_y_pred_error_signed_m * 1000.0),
        "gp_max_abs_xr_error_vs_true_mm": float(abs(prediction["max_abs_xr_pred"] - max_abs_xr_true) * 1000.0),
        "gp_max_abs_xr_error_signed_mm": float(max_abs_xr_pred_error_signed_m * 1000.0),
        "reachability_gap_mm": float(reachability_gap_m * 1000.0),
        "max_abs_xr_true": max_abs_xr_true,
        "constraint_violation_abs_m": constraint_violation_abs,
        "constraint_violation_abs_mm": constraint_violation_abs * 1000.0,
        "feasible_abs": feasible_abs,
        "extra_reach_true": float(sim_metrics.get("extra_reach", np.nan)),
        "max_abs_xb_true": float(sim_metrics.get("max_abs_xb", np.nan)),
        # Objective settings copied for traceability
        "robot_limit_true": config.robot_limit_true,
        "robot_limit_opt": config.robot_limit_opt,
        "lambda_constraint": config.lambda_constraint,
        "lambda_uncertainty_peak_y": config.lambda_uncertainty_peak_y,
        "lambda_uncertainty_max_abs_xr": config.lambda_uncertainty_max_abs_xr,
        "beta_constraint_std": config.beta_constraint_std,
        **params,
    }

    return record


def is_better_record(candidate: Dict[str, Any], current_best: Dict[str, Any]) -> bool:
    """
    Select the best true-validated candidate.

    Priority:
    1. feasible solutions first;
    2. among feasible solutions, minimum true target error;
    3. if none are feasible, minimum true constraint violation;
    4. then minimum true target error.
    """
    candidate_key = (
        0 if candidate["feasible_abs"] else 1,
        0.0 if candidate["feasible_abs"] else candidate["constraint_violation_abs_m"],
        candidate["target_error_m"],
    )
    best_key = (
        0 if current_best["feasible_abs"] else 1,
        0.0 if current_best["feasible_abs"] else current_best["constraint_violation_abs_m"],
        current_best["target_error_m"],
    )
    return candidate_key < best_key


def optimize_target(
    y_target: float,
    models: Dict[str, Any],
    config: GPDEConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Optimize one target with multiple DE attempts."""
    print("-" * 80)
    print(f"GP + DE target: {y_target:.3f} m")
    print("-" * 80)

    attempt_records: List[Dict[str, Any]] = []
    convergence_tables: List[pd.DataFrame] = []
    best_record: Optional[Dict[str, Any]] = None

    for attempt_id in range(config.n_optimization_attempts):
        print(f"  Attempt {attempt_id + 1}/{config.n_optimization_attempts}...")
        start = time.time()

        result, convergence_df = run_de_optimization(
            y_target=y_target,
            models=models,
            config=config,
            attempt_id=attempt_id,
        )

        opt_time = time.time() - start

        record = build_attempt_record(
            y_target=y_target,
            attempt_id=attempt_id,
            result=result,
            optimization_time_s=opt_time,
            models=models,
            config=config,
        )

        attempt_records.append(record)
        convergence_tables.append(convergence_df)

        print(
            f"    pred_y={record['peak_y_pred']:.4f} m | "
            f"pred_xr={record['max_abs_xr_pred']:.4f} m | "
            f"true_y={record['peak_y_true']:.4f} m | "
            f"true_xr_abs={record['max_abs_xr_true']:.4f} m | "
            f"err={record['target_error_mm']:.2f} mm | "
            f"viol={record['constraint_violation_abs_mm']:.2f} mm | "
            f"feasible={record['feasible_abs']} | "
            f"nfev={record['surrogate_function_evaluations']} | "
            f"time={opt_time:.1f} s"
        )

        if best_record is None or is_better_record(record, best_record):
            best_record = record

    assert best_record is not None

    if not any(row["feasible_abs"] for row in attempt_records):
        print(
            "  ⚠ No feasible true-simulator candidate found for this target. "
            "Best solution minimizes violation, then target error."
        )

    print("  Best selected true-validated solution:")
    print(
        f"    true_y={best_record['peak_y_true']:.4f} m | "
        f"target_error={best_record['target_error_mm']:.2f} mm | "
        f"true_xr_abs={best_record['max_abs_xr_true']:.4f} m | "
        f"violation={best_record['constraint_violation_abs_mm']:.2f} mm | "
        f"feasible={best_record['feasible_abs']}"
    )
    print()

    attempts_df = pd.DataFrame(attempt_records)
    convergence_df_all = pd.concat(convergence_tables, ignore_index=True)
    return best_record, attempts_df, convergence_df_all


# =============================================================================
# OPTIONAL DIAGNOSTICS
# =============================================================================

def sample_random_optimizer_vectors(config: GPDEConfig, n_samples: int, seed: int = 2026) -> np.ndarray:
    """Uniformly sample optimizer vectors within DE bounds."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(config.bounds(), dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low=low, high=high, size=(n_samples, len(bounds)))


def run_uncertainty_diagnostics(models: Dict[str, Any], config: GPDEConfig) -> pd.DataFrame:
    """
    Check whether GP predictive uncertainty varies across the DE search box.
    """
    print("\n" + "=" * 80)
    print("GP UNCERTAINTY DIAGNOSTICS OVER DE BOUNDS")
    print("=" * 80)

    X_opt = sample_random_optimizer_vectors(
        config=config,
        n_samples=config.n_uncertainty_samples,
        seed=2026,
    )
    pred_df = predict_gp_batch(X_opt, models, config)

    summary_rows = []
    for col in ["peak_y_std", "max_abs_xr_std"]:
        values_mm = pred_df[col].to_numpy() * 1000.0
        summary_rows.append(
            {
                "quantity": col,
                "mean_mm": np.mean(values_mm),
                "std_mm": np.std(values_mm),
                "min_mm": np.min(values_mm),
                "p05_mm": np.percentile(values_mm, 5),
                "p50_mm": np.percentile(values_mm, 50),
                "p95_mm": np.percentile(values_mm, 95),
                "max_mm": np.max(values_mm),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "gp_de_uncertainty_diagnostics_summary.csv", index=False)
    pred_df.to_csv(RESULTS_DIR / "gp_de_uncertainty_diagnostics.csv", index=False)

    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pred_df["peak_y_std"] * 1000.0, bins=40, edgecolor="black")
    axes[0].set_xlabel("GP peak_y predictive std [mm]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Uncertainty distribution: peak_y")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(pred_df["max_abs_xr_std"] * 1000.0, bins=40, edgecolor="black")
    axes[1].set_xlabel("GP max_abs_xr predictive std [mm]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Uncertainty distribution: max_abs_xr")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "gp_de_uncertainty_diagnostics.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    return summary_df


def run_lambda_sweep(models: Dict[str, Any], base_config: GPDEConfig) -> pd.DataFrame:
    """
    Optional lightweight lambda_constraint sweep on a difficult target.
    """
    print("\n" + "=" * 80)
    print("OPTIONAL LAMBDA_CONSTRAINT SWEEP")
    print("=" * 80)
    print(f"Target: {base_config.lambda_sweep_target:.3f} m")
    print(f"Values: {base_config.lambda_sweep_values}")

    rows = []

    for lambda_value in base_config.lambda_sweep_values:
        sweep_config = replace(
            base_config,
            lambda_constraint=float(lambda_value),
            n_optimization_attempts=base_config.lambda_sweep_attempts,
            de_maxiter=base_config.lambda_sweep_maxiter,
            de_popsize=base_config.lambda_sweep_popsize,
            run_uncertainty_diagnostics=False,
            run_lambda_sweep=False,
        )

        print(f"\nLambda constraint = {lambda_value:g}")
        best_record, attempts_df, _ = optimize_target(
            y_target=base_config.lambda_sweep_target,
            models=models,
            config=sweep_config,
        )

        row = dict(best_record)
        row["sweep_lambda_constraint"] = float(lambda_value)
        row["sweep_n_attempts"] = sweep_config.n_optimization_attempts
        row["sweep_maxiter"] = sweep_config.de_maxiter
        row["sweep_popsize"] = sweep_config.de_popsize
        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    path = RESULTS_DIR / "gp_de_lambda_sweep.csv"
    sweep_df.to_csv(path, index=False)
    print(f"\n✓ Lambda sweep saved: {path}")
    return sweep_df


def plot_lambda_sweep(sweep_df: pd.DataFrame, config: GPDEConfig) -> None:
    """Plot lambda_constraint sensitivity for the selected target."""
    if sweep_df.empty:
        return

    df = sweep_df.sort_values("sweep_lambda_constraint").copy()

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(
        df["sweep_lambda_constraint"],
        df["target_error_mm"],
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Target error",
    )
    ax1.set_xlabel("lambda_constraint")
    ax1.set_ylabel("Target error [mm]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        df["sweep_lambda_constraint"],
        df["max_abs_xr_true"],
        marker="s",
        linewidth=2,
        color="tab:orange",
        label="True max_abs_xr",
    )
    ax2.axhline(
        config.robot_limit_true,
        linestyle="--",
        linewidth=2,
        color="tab:red",
        label="True limit",
    )
    ax2.axhline(
        config.robot_limit_opt,
        linestyle=":",
        linewidth=2,
        color="tab:gray",
        label="Safety limit",
    )
    ax2.set_ylabel("True max_abs_xr [m]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(
        f"Lambda sensitivity for target = {config.lambda_sweep_target:.3f} m"
    )
    plt.tight_layout()

    path = FIGURES_DIR / "gp_de_lambda_sensitivity_target065.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {path}")

# =============================================================================
# SUMMARY TABLES
# =============================================================================

def create_summary_table(results_df: pd.DataFrame, attempts_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact report summary for GP + DE."""
    return pd.DataFrame(
        [
            {
                "method": "GP_DE",
                "n_targets": int(len(results_df)),
                "feasible_targets": int(results_df["feasible_abs"].sum()),
                "feasibility_rate_percent": float(results_df["feasible_abs"].mean() * 100.0),
                "mean_target_error_mm": float(results_df["target_error_mm"].mean()),
                "max_target_error_mm": float(results_df["target_error_mm"].max()),
                "mean_true_max_abs_xr_m": float(results_df["max_abs_xr_true"].mean()),
                "max_true_max_abs_xr_m": float(results_df["max_abs_xr_true"].max()),
                "mean_constraint_violation_abs_mm": float(results_df["constraint_violation_abs_mm"].mean()),
                "max_constraint_violation_abs_mm": float(results_df["constraint_violation_abs_mm"].max()),
                "mean_optimization_time_s_best_attempts": float(results_df["optimization_time_s"].mean()),
                "total_optimization_time_s_all_attempts": float(attempts_df["optimization_time_s"].sum()),
                "mean_surrogate_function_evaluations_best_attempts": float(results_df["surrogate_function_evaluations"].mean()),
                "total_surrogate_function_evaluations_all_attempts": int(attempts_df["surrogate_function_evaluations"].sum()),
                "online_true_simulator_calls": int(len(attempts_df)),
                "offline_dataset_calls": 3000,
            }
        ]
    )


def save_objective_settings(config: GPDEConfig) -> None:
    """Save objective and optimizer settings for traceability."""
    settings = asdict(config)
    settings_path = RESULTS_DIR / "gp_de_objective_settings.csv"
    pd.DataFrame([settings]).to_csv(settings_path, index=False)

    text_path = RESULTS_DIR / "gp_de_objective_settings.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("GP + Differential Evolution Objective Settings\n")
        f.write("=" * 60 + "\n\n")
        f.write("Objective:\n")
        f.write("J = ((peak_y_pred - y_target) / y_error_scale)^2\n")
        f.write("    + lambda_constraint * max(0, (max_abs_xr_pred - robot_limit_opt) / x_constraint_scale)^2\n")
        f.write("    + lambda_uncertainty_peak_y * (std_peak_y / y_error_scale)^2\n")
        f.write("    + lambda_uncertainty_max_abs_xr * (std_max_abs_xr / x_constraint_scale)^2\n\n")
        f.write("If beta_constraint_std > 0, the constraint term uses max_abs_xr_pred + beta * std_max_abs_xr.\n\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")


# =============================================================================
# PLOTS
# =============================================================================

def plot_target_tracking(results_df: pd.DataFrame, config: GPDEConfig) -> None:
    df = results_df.sort_values("target")

    plt.figure(figsize=(8, 6))
    plt.plot(df["target"], df["peak_y_true"], marker="o", linewidth=2, label="True simulator")
    plt.plot(df["target"], df["peak_y_pred"], marker="s", linestyle="--", linewidth=2, label="GP prediction")
    plt.plot(
        [df["target"].min(), df["target"].max()],
        [df["target"].min(), df["target"].max()],
        linestyle=":" ,
        linewidth=2,
        label="Ideal tracking",
    )
    plt.axhline(config.robot_limit_true, linestyle="--", linewidth=1.5, label="Nominal robot reach")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Peak outreach [m]")
    plt.title("GP + DE target tracking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "gp_de_target_tracking.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_target_error(results_df: pd.DataFrame) -> None:
    df = results_df.sort_values("target")

    plt.figure(figsize=(8, 6))
    plt.bar(df["target"].astype(str), df["target_error_mm"], edgecolor="black")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("True target error [mm]")
    plt.title("GP + DE final true-simulator target error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = FIGURES_DIR / "gp_de_target_error.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_constraint_validation(results_df: pd.DataFrame, config: GPDEConfig) -> None:
    df = results_df.sort_values("target")

    plt.figure(figsize=(8, 6))
    plt.plot(df["target"], df["max_abs_xr_true"], marker="o", linewidth=2, label="True max_abs_xr")
    plt.plot(df["target"], df["max_abs_xr_pred"], marker="s", linestyle="--", linewidth=2, label="GP predicted max_abs_xr")
    plt.axhline(config.robot_limit_true, linestyle="--", linewidth=2, label="True limit 0.500 m")
    plt.axhline(config.robot_limit_opt, linestyle=":", linewidth=2, label="Safety limit 0.495 m")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("max_abs_xr [m]")
    plt.title("Constraint validation after true simulation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "gp_de_constraint_validation.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_predicted_vs_true(results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # peak_y
    y_true = results_df["peak_y_true"].to_numpy()
    y_pred = results_df["peak_y_pred"].to_numpy()
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    axes[0].scatter(y_true, y_pred, s=70, edgecolor="black")
    for _, row in results_df.iterrows():
        axes[0].annotate(
            f"{row['target']:.2f}",
            (row["peak_y_true"], row["peak_y_pred"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    axes[0].plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    axes[0].set_xlabel("True peak_y [m]")
    axes[0].set_ylabel("GP predicted peak_y [m]")
    axes[0].set_title("peak_y prediction at optimized candidates")
    axes[0].grid(True, alpha=0.3)

    # max_abs_xr
    x_true = results_df["max_abs_xr_true"].to_numpy()
    x_pred = results_df["max_abs_xr_pred"].to_numpy()
    mn = min(x_true.min(), x_pred.min())
    mx = max(x_true.max(), x_pred.max())
    axes[1].scatter(x_true, x_pred, s=70, edgecolor="black")
    for _, row in results_df.iterrows():
        axes[1].annotate(
            f"{row['target']:.2f}",
            (row["max_abs_xr_true"], row["max_abs_xr_pred"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    axes[1].plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    axes[1].set_xlabel("True max_abs_xr [m]")
    axes[1].set_ylabel("GP predicted max_abs_xr [m]")
    axes[1].set_title("max_abs_xr prediction at optimized candidates")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "gp_de_predicted_vs_true.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_optimized_parameters(results_df: pd.DataFrame, config: GPDEConfig) -> None:
    df = results_df.sort_values("target")
    params = config.optimized_columns()
    bounds = config.bounds()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()

    for ax, param, (low, high) in zip(axes, params, bounds):
        values = df[param].to_numpy(dtype=float)
        values_norm = (values - low) / (high - low)
        ax.plot(df["target"], values_norm, marker="o", linewidth=2)
        ax.set_title(param)
        ax.set_xlabel("Target [m]")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.03,
            0.06,
            f"Bounds: [{low:.3g}, {high:.3g}]",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.suptitle("Optimized controllable parameters: GP + DE", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = FIGURES_DIR / "gp_de_optimized_parameters.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_convergence(convergence_df: pd.DataFrame) -> None:
    if convergence_df.empty:
        return

    plt.figure(figsize=(10, 6))

    # Plot the best attempt per target according to final objective.
    for target in sorted(convergence_df["target"].unique()):
        df_t = convergence_df[np.isclose(convergence_df["target"], target)].copy()
        # Select attempt with lowest last objective.
        last_by_attempt = df_t.sort_values("generation").groupby("attempt").tail(1)
        best_attempt = int(last_by_attempt.sort_values("best_objective_so_far").iloc[0]["attempt"])
        df_plot = df_t[df_t["attempt"] == best_attempt].sort_values("generation")
        plt.plot(
            df_plot["generation"],
            df_plot["best_objective_so_far"],
            linewidth=2,
            label=f"target {target:.2f} m, attempt {best_attempt}",
        )

    plt.yscale("log")
    plt.xlabel("DE generation")
    plt.ylabel("Best objective J")
    plt.title("Differential Evolution convergence")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "gp_de_convergence.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_time_responses(results_df: pd.DataFrame, config: GPDEConfig) -> None:
    """Create one true-simulator time response plot per target if supported."""
    for _, row in results_df.sort_values("target").iterrows():
        target = float(row["target"])
        params = {
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

        response = simulate_candidate_with_solution(params, target, config)
        if response is None:
            print(
                f"⚠ Time-response plot skipped for target {target:.3f}: "
                "simulate_system does not expose full solution in this environment."
            )
            continue

        t, y, x_b, x_r, metrics = response

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

        color_y = "tab:blue"
        color_xr = "tab:orange"
        color_xb = "tab:green"
        color_target = "black"
        color_limit = "tab:red"
        color_nominal = "tab:gray"

        axes[0].plot(
            t,
            y,
            linewidth=2.2,
            color=color_y,
            label="Total outreach y(t)",
        )
        axes[0].axhline(
            target,
            linestyle=":",
            linewidth=2.2,
            color=color_target,
            label=f"Target {target:.3f} m",
        )
        axes[0].axhline(
            config.robot_limit_true,
            linestyle="--",
            linewidth=1.8,
            color=color_nominal,
            label="Nominal robot reach",
        )
        axes[0].set_ylabel("y(t) [m]")
        axes[0].set_title(
            f"GP + DE true time response, target = {target:.3f} m\n"
            f"peak_y = {metrics['peak_y']:.3f} m, "
            f"error = {abs(metrics['peak_y'] - target) * 1000.0:.1f} mm, "
            f"max_abs_xr = {metrics['max_abs_xr']:.3f} m"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)

        axes[1].plot(
            t,
            x_r,
            linewidth=2.2,
            color=color_xr,
            label="Robot displacement x_r(t)",
        )
        axes[1].axhline(
            config.robot_limit_true,
            linestyle="--",
            linewidth=2,
            color=color_limit,
            label="+ robot limit",
        )
        axes[1].axhline(
            -config.robot_limit_true,
            linestyle="--",
            linewidth=2,
            color=color_limit,
            label="- robot limit",
        )
        axes[1].set_ylabel("x_r(t) [m]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

        axes[2].plot(
            t,
            x_b,
            linewidth=2.2,
            color=color_xb,
            label="Base displacement x_b(t)",
        )
        axes[2].axhline(
            0.0,
            linestyle=":",
            linewidth=1.5,
            color=color_target,
            label="Zero reference",
        )
        axes[2].set_ylabel("x_b(t) [m]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        target_tag = f"{int(round(target * 1000)):04d}"
        path = FIGURES_DIR / f"time_response_target{target_tag}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def generate_all_plots(
    results_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    config: GPDEConfig,
) -> None:
    print("\nGenerating GP + DE plots...")
    plot_target_tracking(results_df, config)
    plot_target_error(results_df)
    plot_constraint_validation(results_df, config)
    plot_predicted_vs_true(results_df)
    plot_optimized_parameters(results_df, config)
    plot_convergence(convergence_df)
    plot_time_responses(results_df, config)


# =============================================================================
# MAIN
# =============================================================================

def print_settings(config: GPDEConfig) -> None:
    print("=" * 80)
    print("GP + DIFFERENTIAL EVOLUTION CONSTRAINT-AWARE OPTIMIZATION")
    print("=" * 80)
    print("Settings:")
    print(f"  Targets:                         {np.array(config.targets)}")
    print(f"  True robot limit:                {config.robot_limit_true:.3f} m")
    print(f"  Surrogate safety limit:          {config.robot_limit_opt:.3f} m")
    print(f"  y error scale:                   {config.y_error_scale_m * 1000:.1f} mm")
    print(f"  x constraint scale:              {config.x_constraint_scale_m * 1000:.1f} mm")
    print(f"  lambda constraint:               {config.lambda_constraint:g}")
    print(f"  lambda uncertainty peak_y:       {config.lambda_uncertainty_peak_y:g}")
    print(f"  lambda uncertainty max_abs_xr:   {config.lambda_uncertainty_max_abs_xr:g}")
    print(f"  beta constraint std:             {config.beta_constraint_std:g}")
    print(f"  Fixed params:                    {config.fixed_params()}")
    print(f"  Optimized params:                {config.optimized_columns()}")
    print(f"  Bounds:                          {config.bounds()}")
    print(f"  DE attempts per target:          {config.n_optimization_attempts}")
    print(f"  DE maxiter / popsize:            {config.de_maxiter} / {config.de_popsize}")
    print(f"  DE mutation / recombination:     {config.de_mutation} / {config.de_recombination}")
    print("=" * 80 + "\n")


def print_final_summary(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FINAL GP + DE SUMMARY")
    print("=" * 80)

    display_df = results_df.sort_values("target").copy()
    display_cols = [
        "target",
        "peak_y_true",
        "peak_y_pred",
        "target_error_mm",
        "reachability_gap_mm",
        "gp_peak_y_error_signed_mm",
        "max_abs_xr_true",
        "max_abs_xr_pred",
        "predicted_safety_violation_mm",
        "constraint_violation_abs_mm",
        "feasible_abs",
        "surrogate_function_evaluations",
    ]
    print(display_df[display_cols].to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")


def main() -> None:
    ensure_output_dirs()
    config = GPDEConfig()
    print_settings(config)
    save_objective_settings(config)

    models = load_surrogates()

    if config.run_uncertainty_diagnostics:
        run_uncertainty_diagnostics(models, config)

    if config.run_lambda_sweep:
        lambda_sweep_df = run_lambda_sweep(models, config)
        plot_lambda_sweep(lambda_sweep_df, config)
        print(
            "\nLambda sweep completed. Review results/optimization_gp_de/"
            "gp_de_lambda_sweep.csv and "
            "figures/optimization_gp_de/gp_de_lambda_sensitivity_target065.png.\n"
        )

    best_records: List[Dict[str, Any]] = []
    all_attempt_tables: List[pd.DataFrame] = []
    all_convergence_tables: List[pd.DataFrame] = []

    total_start = time.time()

    for target in config.targets:
        best_record, attempts_df, convergence_df = optimize_target(
            y_target=float(target),
            models=models,
            config=config,
        )
        best_records.append(best_record)
        all_attempt_tables.append(attempts_df)
        all_convergence_tables.append(convergence_df)

    total_time_s = time.time() - total_start

    results_df = pd.DataFrame(best_records).sort_values("target").reset_index(drop=True)
    attempts_df = pd.concat(all_attempt_tables, ignore_index=True)
    convergence_df = pd.concat(all_convergence_tables, ignore_index=True)

    summary_df = create_summary_table(results_df, attempts_df)
    summary_df["total_wall_time_s"] = total_time_s

    # Save tables.
    results_path = RESULTS_DIR / "gp_de_results.csv"
    attempts_path = RESULTS_DIR / "gp_de_all_attempts.csv"
    convergence_path = RESULTS_DIR / "gp_de_convergence.csv"
    summary_path = RESULTS_DIR / "gp_de_summary.csv"
    report_table_path = RESULTS_DIR / "gp_de_report_table.csv"

    results_df.to_csv(results_path, index=False)
    attempts_df.to_csv(attempts_path, index=False)
    convergence_df.to_csv(convergence_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report_cols = [
        "target",
        "peak_y_true",
        "peak_y_pred",
        "target_error_mm",
        "reachability_gap_mm",
        "gp_peak_y_error_signed_mm",
        "gp_peak_y_error_vs_true_mm",
        "max_abs_xr_true",
        "max_abs_xr_pred",
        "gp_max_abs_xr_error_signed_mm",
        "predicted_safety_violation_mm",
        "constraint_violation_abs_mm",
        "feasible_abs",
        "surrogate_function_evaluations",
        "optimization_time_s",
    ]
    results_df[report_cols].to_csv(report_table_path, index=False)

    generate_all_plots(results_df, convergence_df, config)

    print_final_summary(results_df, summary_df)

    print("Saved files:")
    print(f"  {results_path}")
    print(f"  {attempts_path}")
    print(f"  {convergence_path}")
    print(f"  {summary_path}")
    print(f"  {report_table_path}")
    print(f"  {RESULTS_DIR / 'gp_de_objective_settings.csv'}")
    print(f"  {RESULTS_DIR / 'gp_de_objective_settings.txt'}")
    print(f"  {FIGURES_DIR / 'gp_de_target_tracking.png'}")
    print(f"  {FIGURES_DIR / 'gp_de_target_error.png'}")
    print(f"  {FIGURES_DIR / 'gp_de_constraint_validation.png'}")
    print(f"  {FIGURES_DIR / 'gp_de_predicted_vs_true.png'}")
    print(f"  {FIGURES_DIR / 'gp_de_optimized_parameters.png'}")
    print(f"  {FIGURES_DIR / 'gp_de_convergence.png'}")
    print("\nNext step: inspect GP + DE results, then implement NN + gradient optimization.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
