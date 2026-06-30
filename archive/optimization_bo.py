"""
optimization_bo.py
==================

Constraint-aware Bayesian Optimization baseline using the TRUE dynamic simulator.

Main purpose
------------
Given a desired outreach target y_target, find controllable excitation/robot
parameters

    [Kr, hr, f0, f1, A, x_r_start]

while respecting the physical robot displacement constraint

    max_abs_xr <= 0.500 m.

Unlike GP+DE and NN+Adam, this script does NOT use a pre-trained surrogate.
Bayesian Optimization (BO) builds a small online Gaussian Process model of the
scalar true objective J_true using direct calls to the dynamic simulator.

Methodological role
-------------------
BO is used as an online black-box baseline:

    GP+DE   : offline surrogate + global derivative-free optimizer
    NN+Adam : offline surrogate + differentiable optimizer
    BO      : no offline surrogate, sequential true-simulator optimization

Important fairness note
-----------------------
BO uses zero offline dataset calls, but many online true simulator calls.
The surrogate-based methods use the 3000-call offline dataset and only a small
number of online validation calls. Therefore, comparisons should report both
offline and online simulator costs.

Main BO configuration
---------------------
- Internal model: GaussianProcessRegressor on scalar J_true
- Kernel: Matern 5/2 + WhiteKernel
- Acquisition: Expected Improvement (EI)
- Initialization: Latin Hypercube Sampling (LHS)
- Candidate selection: global random/LHS pool + local perturbations around best
- True constraint threshold in BO objective: 0.500 m

Recommended workflow
--------------------
By default, the script runs only a pilot BO run first. Inspect the timing output.
If acceptable, set `stop_after_pilot = False` in BOConfig and run the full BO.

Outputs
-------
results/optimization_bo/bo_pilot_timing_target065.csv
results/optimization_bo/bo_results.csv
results/optimization_bo/bo_history.csv
results/optimization_bo/bo_summary.csv
results/optimization_bo/bo_random_search_results.csv
results/optimization_bo/bo_random_search_history.csv
results/optimization_bo/bo_random_search_seed_summary_target065.csv
results/optimization_bo/bo_kernel_sensitivity_target065.csv
results/optimization_bo/bo_acquisition_sensitivity_target065.csv
results/optimization_bo/bo_budget_sweep_target065.csv
results/optimization_bo/bo_three_way_target065.csv

figures/optimization_bo/bo_target_tracking.png
figures/optimization_bo/bo_target_error.png
figures/optimization_bo/bo_constraint_validation.png
figures/optimization_bo/bo_residual_margin.png
figures/optimization_bo/bo_convergence_objective.png
figures/optimization_bo/bo_convergence_feasible_error.png
figures/optimization_bo/bo_vs_random_search.png
figures/optimization_bo/bo_optimized_parameters.png
figures/optimization_bo/bo_kernel_sensitivity_target065.png
figures/optimization_bo/bo_acquisition_sensitivity_target065.png
figures/optimization_bo/bo_budget_sweep_target065.png
figures/optimization_bo/bo_pilot_timing_target065.png
figures/optimization_bo/time_response_target*.png

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel

from dynamics import simulate_system


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo"

GP_DE_RESULTS_PATH = PROJECT_ROOT / "results" / "optimization_gp_de" / "gp_de_results.csv"
NN_ADAM_RESULTS_PATH = PROJECT_ROOT / "results" / "optimization_nn_gradient" / "nn_gradient_results.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BOConfig:
    """Configuration for true-simulator Bayesian Optimization baseline."""

    # ------------------------------------------------------------------
    # Fixed physical parameters
    # ------------------------------------------------------------------
    Kb: float = 1000.0
    Mb: float = 20.0
    hb: float = 0.10
    Mr: float = 10.0

    # ------------------------------------------------------------------
    # True physical constraint and numerical tolerance
    # ------------------------------------------------------------------
    robot_limit_true: float = 0.500
    feasibility_tolerance_m: float = 1e-9

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    targets: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)
    sensitivity_target: float = 0.65

    # ------------------------------------------------------------------
    # Bounds for optimized variables: [Kr, hr, f0, f1, A, x_r_start]
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
    # Objective normalization scales
    # ------------------------------------------------------------------
    # 10 mm target error gives tracking cost = 1.
    y_error_scale_m: float = 0.010

    # 5 mm true constraint violation gives unweighted violation cost = 1.
    x_constraint_scale_m: float = 0.005

    # Keep equal to GP+DE and NN+Adam tracking/constraint trade-off.
    lambda_constraint: float = 10.0

    # ------------------------------------------------------------------
    # Pilot BO run: run first to estimate runtime.
    # ------------------------------------------------------------------
    run_pilot: bool = False
    stop_after_pilot: bool = False
    pilot_target: float = 0.65
    pilot_total_evaluations: int = 30
    pilot_initial_points: int = 10

    # ------------------------------------------------------------------
    # Main BO run.
    # If pilot runtime is too high, reduce these before the full run.
    # ------------------------------------------------------------------
    run_main_bo: bool = True
    main_total_evaluations: int = 100
    main_initial_points: int = 20

    # ------------------------------------------------------------------
    # Internal GP settings.
    # For speed, keep restarts low. n <= 100, so this remains manageable.
    # ------------------------------------------------------------------
    main_kernel: str = "matern"          # "matern" or "rbf"
    main_acquisition: str = "lcb"         # "ei" or "lcb"
    gp_alpha: float = 1e-8
    gp_restarts: int = 1
    gp_length_scale_bounds_low: float = 1e-2
    gp_length_scale_bounds_high: float = 1e2
    white_noise_level: float = 1e-6
    white_noise_bounds_low: float = 1e-9
    white_noise_bounds_high: float = 1e-3

    # Acquisition settings.
    ei_xi: float = 0.01
    lcb_kappa: float = 2.0

    # ------------------------------------------------------------------
    # Candidate pool for acquisition optimization.
    # Pool = global candidates + local perturbations around best-so-far.
    # ------------------------------------------------------------------
    candidate_pool_global: int = 4000
    candidate_pool_local: int = 1000
    local_sigma_unit: float = 0.07

    # ------------------------------------------------------------------
    # Random Search baseline.
    # ------------------------------------------------------------------
    run_random_search: bool = True
    random_search_total_evaluations: int = 100
    random_search_main_seed: int = 2026
    random_search_extra_seeds_target065: Tuple[int, ...] = (2027, 2031)

    # ------------------------------------------------------------------
    # Sensitivity analyses, all on target = sensitivity_target.
    # The main Matern+EI full-budget run is reused when possible.
    # ------------------------------------------------------------------
    run_kernel_sensitivity: bool = False
    kernel_sensitivity_values: Tuple[str, ...] = ("matern", "rbf")

    run_acquisition_sensitivity: bool = False
    acquisition_sensitivity_values: Tuple[str, ...] = ("ei", "lcb")

    run_budget_sweep: bool = False
    budget_sweep_total_evaluations: Tuple[int, ...] = (25, 50, 100)
    budget_sweep_initial_points: int = 10

    # ------------------------------------------------------------------
    # Plotting and time responses.
    # All are saved, but in the report use only 0.65 and 0.75 in the main text.
    # ------------------------------------------------------------------
    make_time_response_plots: bool = True

    # ------------------------------------------------------------------
    # Reproducibility.
    # ------------------------------------------------------------------
    random_seed: int = 12345

    # ------------------------------------------------------------------
    # Known simulator-cost accounting for final comparisons.
    # ------------------------------------------------------------------
    offline_dataset_calls_surrogate_methods: int = 3000
    online_validation_calls_gp_de_per_target: int = 5
    online_validation_calls_nn_adam_per_target: int = 5

    def optimized_columns(self) -> List[str]:
        return ["Kr", "hr", "f0", "f1", "A", "x_r_start"]

    def bounds_array(self) -> np.ndarray:
        return np.array(
            [
                [self.Kr_min, self.Kr_max],
                [self.hr_min, self.hr_max],
                [self.f0_min, self.f0_max],
                [self.f1_min, self.f1_max],
                [self.A_min, self.A_max],
                [self.x_r_start_min, self.x_r_start_max],
            ],
            dtype=float,
        )

    def fixed_params(self) -> Dict[str, float]:
        return {
            "Kb": float(self.Kb),
            "Mb": float(self.Mb),
            "hb": float(self.hb),
            "Mr": float(self.Mr),
        }


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def unit_to_physical(X_unit: np.ndarray, config: BOConfig) -> np.ndarray:
    """Map variables from unit cube [0,1]^d to physical bounds."""
    X_unit = np.asarray(X_unit, dtype=float)
    bounds = config.bounds_array()
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + X_unit * (upper - lower)


def physical_to_unit(X_phys: np.ndarray, config: BOConfig) -> np.ndarray:
    """Map physical variables to unit cube [0,1]^d."""
    X_phys = np.asarray(X_phys, dtype=float)
    bounds = config.bounds_array()
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return np.clip((X_phys - lower) / (upper - lower), 0.0, 1.0)


def physical_vector_to_params(x_phys: Sequence[float], config: BOConfig) -> Dict[str, float]:
    """Build full simulator parameter dictionary from optimized physical vector."""
    params = config.fixed_params()
    for name, value in zip(config.optimized_columns(), x_phys):
        params[name] = float(value)
    return params


def row_to_params(row: pd.Series, config: BOConfig) -> Dict[str, float]:
    params = config.fixed_params()
    for name in config.optimized_columns():
        params[name] = float(row[name])
    return params


def lhs_unit(n_samples: int, dim: int, seed: int) -> np.ndarray:
    """Latin Hypercube samples in [0,1]^dim."""
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return sampler.random(n_samples)


def get_metric(metrics: Dict[str, Any], key: str, default: Optional[float] = None) -> float:
    value = metrics.get(key, default)
    if value is None:
        raise KeyError(f"Metric {key!r} not found in simulator metrics.")
    return float(value)


def get_max_abs_xr(metrics: Dict[str, Any]) -> float:
    if "max_abs_xr" in metrics:
        return float(metrics["max_abs_xr"])
    if "max_xr" in metrics and "min_xr" in metrics:
        return float(max(abs(float(metrics["max_xr"])), abs(float(metrics["min_xr"]))))
    if "max_xr" in metrics:
        return abs(float(metrics["max_xr"]))
    raise KeyError("Could not infer max_abs_xr from metrics.")


def compute_objective_from_metrics(
    metrics: Dict[str, Any],
    y_target: float,
    config: BOConfig,
) -> Dict[str, Any]:
    """Compute BO true objective and physical performance metrics."""
    peak_y_true = get_metric(metrics, "peak_y")
    max_abs_xr_true = get_max_abs_xr(metrics)

    target_error_m = abs(peak_y_true - y_target)
    reachability_gap_m = max(0.0, y_target - peak_y_true)
    constraint_violation_abs_m = max(0.0, max_abs_xr_true - config.robot_limit_true)
    feasible_abs = bool(constraint_violation_abs_m <= config.feasibility_tolerance_m)

    tracking_cost = (target_error_m / config.y_error_scale_m) ** 2
    constraint_cost = config.lambda_constraint * (
        constraint_violation_abs_m / config.x_constraint_scale_m
    ) ** 2
    J_true = tracking_cost + constraint_cost

    residual_margin_m = config.robot_limit_true - max_abs_xr_true

    return {
        "peak_y_true": float(peak_y_true),
        "target_error_m": float(target_error_m),
        "target_error_mm": float(target_error_m * 1000.0),
        "reachability_gap_m": float(reachability_gap_m),
        "reachability_gap_mm": float(reachability_gap_m * 1000.0),
        "max_abs_xr_true": float(max_abs_xr_true),
        "constraint_violation_abs_m": float(constraint_violation_abs_m),
        "constraint_violation_abs_mm": float(constraint_violation_abs_m * 1000.0),
        "residual_margin_m": float(residual_margin_m),
        "residual_margin_mm": float(residual_margin_m * 1000.0),
        "feasible_abs": feasible_abs,
        "tracking_cost": float(tracking_cost),
        "constraint_cost": float(constraint_cost),
        "J_true": float(J_true),
    }


def simulate_true_candidate(
    x_unit: np.ndarray,
    y_target: float,
    config: BOConfig,
) -> Tuple[Dict[str, Any], float]:
    """Simulate one BO candidate and return result record plus simulation time."""
    x_phys = unit_to_physical(np.asarray(x_unit, dtype=float).reshape(1, -1), config)[0]
    params = physical_vector_to_params(x_phys, config)

    t0 = time.perf_counter()
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=config.robot_limit_true,
            return_metrics=True,
        )
    except TypeError:
        output = simulate_system(params, return_metrics=True)
    simulation_time_s = time.perf_counter() - t0

    metrics: Optional[Dict[str, Any]] = None
    if isinstance(output, dict):
        metrics = output
    elif isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict) and "peak_y" in item:
                metrics = item
                break
        if metrics is None and len(output) >= 2 and isinstance(output[1], dict):
            metrics = output[1]

    if metrics is None:
        raise TypeError(f"Could not extract metrics dictionary from simulate_system output: {type(output)}")

    objective = compute_objective_from_metrics(metrics, y_target, config)

    record: Dict[str, Any] = {}
    for name, value in zip(config.optimized_columns(), x_phys):
        record[name] = float(value)
    for i, value in enumerate(np.asarray(x_unit, dtype=float)):
        record[f"u_{config.optimized_columns()[i]}"] = float(value)
    record.update(objective)

    return record, simulation_time_s


def select_best_from_history(history_df: pd.DataFrame) -> pd.Series:
    """Select final solution by true feasibility first, target error second."""
    df = history_df.copy()
    feasible = df[df["feasible_abs"].astype(bool)].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(["target_error_mm", "J_true", "iteration"])
        return feasible.iloc[0]
    df = df.sort_values(["constraint_violation_abs_mm", "target_error_mm", "J_true", "iteration"])
    return df.iloc[0]


def add_cumulative_best_columns(history_df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative best objective and feasible-error columns per run."""
    if history_df.empty:
        return history_df

    all_groups = []
    group_cols = ["run_label", "target", "method", "kernel", "acquisition", "seed"]
    for _, group in history_df.groupby(group_cols, dropna=False):
        group = group.sort_values("iteration").copy()
        best_j_so_far = []
        best_feasible_error_so_far = []
        best_feasible_margin_so_far = []
        current_best_flags = []

        current_best_j = math.inf
        current_best_feasible_error = math.inf
        current_best_feasible_margin = np.nan
        best_j_index = None

        for idx, row in group.iterrows():
            J = float(row["J_true"])
            if J < current_best_j:
                current_best_j = J
                best_j_index = idx
            best_j_so_far.append(current_best_j)
            current_best_flags.append(bool(idx == best_j_index))

            if bool(row["feasible_abs"]):
                err = float(row["target_error_mm"])
                if err < current_best_feasible_error:
                    current_best_feasible_error = err
                    current_best_feasible_margin = float(row["residual_margin_mm"])

            best_feasible_error_so_far.append(
                np.nan if math.isinf(current_best_feasible_error) else current_best_feasible_error
            )
            best_feasible_margin_so_far.append(current_best_feasible_margin)

        group["best_J_so_far"] = best_j_so_far
        group["best_feasible_error_so_far_mm"] = best_feasible_error_so_far
        group["best_feasible_margin_so_far_mm"] = best_feasible_margin_so_far
        group["is_current_best_J"] = current_best_flags
        all_groups.append(group)

    return pd.concat(all_groups, ignore_index=True)


# =============================================================================
# GAUSSIAN PROCESS AND ACQUISITION
# =============================================================================

def build_kernel(kernel_kind: str, config: BOConfig):
    """Build internal GP kernel for BO."""
    dim = len(config.optimized_columns())
    ls_bounds = (config.gp_length_scale_bounds_low, config.gp_length_scale_bounds_high)
    noise_bounds = (config.white_noise_bounds_low, config.white_noise_bounds_high)

    if kernel_kind.lower() == "matern":
        base = Matern(length_scale=np.ones(dim), length_scale_bounds=ls_bounds, nu=2.5)
    elif kernel_kind.lower() == "rbf":
        base = RBF(length_scale=np.ones(dim), length_scale_bounds=ls_bounds)
    else:
        raise ValueError(f"Unsupported kernel_kind: {kernel_kind!r}")

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * base
        + WhiteKernel(
            noise_level=config.white_noise_level,
            noise_level_bounds=noise_bounds,
        )
    )
    return kernel


def fit_internal_gp(
    X_unit: np.ndarray,
    y: np.ndarray,
    kernel_kind: str,
    config: BOConfig,
    random_state: int,
) -> Tuple[GaussianProcessRegressor, float]:
    """Fit internal BO GP and return model plus fit time."""
    kernel = build_kernel(kernel_kind, config)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=config.gp_alpha,
        normalize_y=True,
        n_restarts_optimizer=config.gp_restarts,
        random_state=random_state,
    )

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gp.fit(X_unit, y)
    fit_time_s = time.perf_counter() - t0
    return gp, fit_time_s


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement for minimization."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)

    improvement = best_y - mu - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return np.maximum(ei, 0.0)


def acquisition_scores(
    gp: GaussianProcessRegressor,
    X_cand: np.ndarray,
    best_y: float,
    acquisition: str,
    config: BOConfig,
) -> np.ndarray:
    """Compute acquisition scores. Higher score is better."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    acquisition = acquisition.lower()

    if acquisition == "ei":
        return expected_improvement(mu, sigma, best_y, xi=config.ei_xi)

    if acquisition == "lcb":
        # For minimization, LCB = mu - kappa*sigma. Choose smallest LCB.
        # Return negative LCB so that higher score is better.
        lcb = mu - config.lcb_kappa * sigma
        return -lcb

    raise ValueError(f"Unsupported acquisition: {acquisition!r}")


def generate_candidate_pool(
    rng: np.random.Generator,
    best_x_unit: Optional[np.ndarray],
    config: BOConfig,
) -> np.ndarray:
    """Generate global + local candidates in unit cube."""
    dim = len(config.optimized_columns())

    X_global = rng.random((config.candidate_pool_global, dim))

    if best_x_unit is None or config.candidate_pool_local <= 0:
        return X_global

    best_x_unit = np.asarray(best_x_unit, dtype=float).reshape(1, -1)
    X_local = best_x_unit + rng.normal(
        loc=0.0,
        scale=config.local_sigma_unit,
        size=(config.candidate_pool_local, dim),
    )
    X_local = np.clip(X_local, 0.0, 1.0)

    return np.vstack([X_global, X_local])


def choose_next_candidate(
    gp: GaussianProcessRegressor,
    X_evaluated: np.ndarray,
    y_evaluated: np.ndarray,
    rng: np.random.Generator,
    acquisition: str,
    config: BOConfig,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Choose next BO candidate from acquisition candidate pool."""
    best_idx = int(np.argmin(y_evaluated))
    best_x = X_evaluated[best_idx]
    best_y = float(y_evaluated[best_idx])

    t0 = time.perf_counter()
    X_pool = generate_candidate_pool(rng, best_x, config)
    scores = acquisition_scores(gp, X_pool, best_y, acquisition, config)
    selected_idx = int(np.argmax(scores))
    acquisition_time_s = time.perf_counter() - t0

    selected = X_pool[selected_idx]
    diagnostics = {
        "acquisition_score": float(scores[selected_idx]),
        "candidate_pool_size": int(X_pool.shape[0]),
        "candidate_pool_global": int(config.candidate_pool_global),
        "candidate_pool_local": int(config.candidate_pool_local),
    }
    return selected, acquisition_time_s, diagnostics


# =============================================================================
# BO AND RANDOM SEARCH RUNNERS
# =============================================================================

def run_bo_single_target(
    y_target: float,
    config: BOConfig,
    run_label: str,
    total_evaluations: int,
    n_initial_points: int,
    kernel_kind: str,
    acquisition: str,
    seed: int,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """Run BO for one target and return best row, history, and timing summary."""
    if n_initial_points >= total_evaluations:
        raise ValueError("n_initial_points must be smaller than total_evaluations.")

    dim = len(config.optimized_columns())
    rng = np.random.default_rng(seed)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    records: List[Dict[str, Any]] = []

    wall_t0 = time.perf_counter()

    # Initial LHS design.
    X_init = lhs_unit(n_initial_points, dim, seed=seed)

    for i in range(n_initial_points):
        iter_t0 = time.perf_counter()
        record, sim_time_s = simulate_true_candidate(X_init[i], y_target, config)
        iter_time_s = time.perf_counter() - iter_t0

        X_list.append(X_init[i])
        y_list.append(float(record["J_true"]))

        record.update(
            {
                "run_label": run_label,
                "method": "BO",
                "target": float(y_target),
                "iteration": int(i + 1),
                "phase": "initial_lhs",
                "kernel": kernel_kind,
                "acquisition": acquisition,
                "seed": int(seed),
                "total_budget": int(total_evaluations),
                "initial_points": int(n_initial_points),
                "simulation_time_s": float(sim_time_s),
                "gp_fit_time_s": 0.0,
                "acquisition_time_s": 0.0,
                "iteration_time_s": float(iter_time_s),
                "acquisition_score": np.nan,
                "candidate_pool_size": 0,
                "candidate_pool_global": 0,
                "candidate_pool_local": 0,
            }
        )
        records.append(record)

    # Sequential BO iterations.
    for eval_idx in range(n_initial_points, total_evaluations):
        iter_t0 = time.perf_counter()
        X_eval = np.asarray(X_list, dtype=float)
        y_eval = np.asarray(y_list, dtype=float)

        gp, gp_fit_time_s = fit_internal_gp(
            X_eval,
            y_eval,
            kernel_kind=kernel_kind,
            config=config,
            random_state=seed + eval_idx,
        )

        x_next, acquisition_time_s, acq_diag = choose_next_candidate(
            gp=gp,
            X_evaluated=X_eval,
            y_evaluated=y_eval,
            rng=rng,
            acquisition=acquisition,
            config=config,
        )

        record, sim_time_s = simulate_true_candidate(x_next, y_target, config)
        iter_time_s = time.perf_counter() - iter_t0

        X_list.append(x_next)
        y_list.append(float(record["J_true"]))

        record.update(
            {
                "run_label": run_label,
                "method": "BO",
                "target": float(y_target),
                "iteration": int(eval_idx + 1),
                "phase": "bo",
                "kernel": kernel_kind,
                "acquisition": acquisition,
                "seed": int(seed),
                "total_budget": int(total_evaluations),
                "initial_points": int(n_initial_points),
                "simulation_time_s": float(sim_time_s),
                "gp_fit_time_s": float(gp_fit_time_s),
                "acquisition_time_s": float(acquisition_time_s),
                "iteration_time_s": float(iter_time_s),
                **acq_diag,
            }
        )
        records.append(record)

    history = pd.DataFrame(records)
    history = add_cumulative_best_columns(history)
    best_row = select_best_from_history(history)

    total_wall_time_s = time.perf_counter() - wall_t0
    timing = {
        "run_label": run_label,
        "target": float(y_target),
        "kernel": kernel_kind,
        "acquisition": acquisition,
        "seed": int(seed),
        "total_evaluations": int(total_evaluations),
        "initial_points": int(n_initial_points),
        "bo_iterations": int(total_evaluations - n_initial_points),
        "total_wall_time_s": float(total_wall_time_s),
        "mean_iteration_time_s": float(history["iteration_time_s"].mean()),
        "mean_simulation_time_s": float(history["simulation_time_s"].mean()),
        "mean_gp_fit_time_s": float(history["gp_fit_time_s"].mean()),
        "mean_acquisition_time_s": float(history["acquisition_time_s"].mean()),
        "best_target_error_mm": float(best_row["target_error_mm"]),
        "best_peak_y_true": float(best_row["peak_y_true"]),
        "best_max_abs_xr_true": float(best_row["max_abs_xr_true"]),
        "best_residual_margin_mm": float(best_row["residual_margin_mm"]),
        "best_feasible_abs": bool(best_row["feasible_abs"]),
    }

    return best_row, history, timing


def run_random_search_single_target(
    y_target: float,
    config: BOConfig,
    run_label: str,
    total_evaluations: int,
    seed: int,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """Run random search baseline for one target."""
    dim = len(config.optimized_columns())
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []
    wall_t0 = time.perf_counter()

    X = rng.random((total_evaluations, dim))

    for i in range(total_evaluations):
        iter_t0 = time.perf_counter()
        record, sim_time_s = simulate_true_candidate(X[i], y_target, config)
        iter_time_s = time.perf_counter() - iter_t0

        record.update(
            {
                "run_label": run_label,
                "method": "RandomSearch",
                "target": float(y_target),
                "iteration": int(i + 1),
                "phase": "random",
                "kernel": "none",
                "acquisition": "none",
                "seed": int(seed),
                "total_budget": int(total_evaluations),
                "initial_points": 0,
                "simulation_time_s": float(sim_time_s),
                "gp_fit_time_s": 0.0,
                "acquisition_time_s": 0.0,
                "iteration_time_s": float(iter_time_s),
                "acquisition_score": np.nan,
                "candidate_pool_size": 0,
                "candidate_pool_global": 0,
                "candidate_pool_local": 0,
            }
        )
        records.append(record)

    history = pd.DataFrame(records)
    history = add_cumulative_best_columns(history)
    best_row = select_best_from_history(history)

    total_wall_time_s = time.perf_counter() - wall_t0
    timing = {
        "run_label": run_label,
        "target": float(y_target),
        "seed": int(seed),
        "total_evaluations": int(total_evaluations),
        "total_wall_time_s": float(total_wall_time_s),
        "mean_iteration_time_s": float(history["iteration_time_s"].mean()),
        "mean_simulation_time_s": float(history["simulation_time_s"].mean()),
        "best_target_error_mm": float(best_row["target_error_mm"]),
        "best_peak_y_true": float(best_row["peak_y_true"]),
        "best_max_abs_xr_true": float(best_row["max_abs_xr_true"]),
        "best_residual_margin_mm": float(best_row["residual_margin_mm"]),
        "best_feasible_abs": bool(best_row["feasible_abs"]),
    }

    return best_row, history, timing


def build_result_row(best_row: pd.Series, config: BOConfig) -> Dict[str, Any]:
    """Build compact final-results row from selected best history row."""
    result = {
        "method": str(best_row.get("method", "BO")),
        "run_label": str(best_row.get("run_label", "main")),
        "target": float(best_row["target"]),
        "peak_y_true": float(best_row["peak_y_true"]),
        "target_error_mm": float(best_row["target_error_mm"]),
        "reachability_gap_mm": float(best_row["reachability_gap_mm"]),
        "max_abs_xr_true": float(best_row["max_abs_xr_true"]),
        "constraint_violation_abs_mm": float(best_row["constraint_violation_abs_mm"]),
        "residual_margin_mm": float(best_row["residual_margin_mm"]),
        "feasible_abs": bool(best_row["feasible_abs"]),
        "J_true": float(best_row["J_true"]),
        "tracking_cost": float(best_row["tracking_cost"]),
        "constraint_cost": float(best_row["constraint_cost"]),
        "best_iteration": int(best_row["iteration"]),
        "n_true_simulator_evaluations": int(best_row["total_budget"]),
        "kernel": str(best_row.get("kernel", "none")),
        "acquisition": str(best_row.get("acquisition", "none")),
        "seed": int(best_row.get("seed", -1)),
    }
    for col in config.optimized_columns():
        result[col] = float(best_row[col])
    return result


def summarize_results(results_df: pd.DataFrame, method_label: str) -> pd.DataFrame:
    """Summarize compact final results over targets."""
    if results_df.empty:
        return pd.DataFrame()
    summary = {
        "method": method_label,
        "n_targets": int(len(results_df)),
        "feasible_targets": int(results_df["feasible_abs"].sum()),
        "feasibility_rate_percent": float(100.0 * results_df["feasible_abs"].mean()),
        "mean_target_error_mm": float(results_df["target_error_mm"].mean()),
        "max_target_error_mm": float(results_df["target_error_mm"].max()),
        "mean_true_max_abs_xr_m": float(results_df["max_abs_xr_true"].mean()),
        "max_true_max_abs_xr_m": float(results_df["max_abs_xr_true"].max()),
        "mean_residual_margin_mm": float(results_df["residual_margin_mm"].mean()),
        "min_residual_margin_mm": float(results_df["residual_margin_mm"].min()),
        "mean_constraint_violation_abs_mm": float(results_df["constraint_violation_abs_mm"].mean()),
        "max_constraint_violation_abs_mm": float(results_df["constraint_violation_abs_mm"].max()),
        "total_true_simulator_calls": int(results_df["n_true_simulator_evaluations"].sum()),
    }
    if "optimization_time_s" in results_df.columns:
        summary["mean_optimization_time_s"] = float(results_df["optimization_time_s"].mean())
        summary["total_optimization_time_s"] = float(results_df["optimization_time_s"].sum())
    return pd.DataFrame([summary])


# =============================================================================
# PLOTTING
# =============================================================================

def _target_labels(targets: Iterable[float]) -> List[str]:
    return [f"{float(t):.2f}" for t in targets]


def plot_bo_results(results_df: pd.DataFrame, config: BOConfig) -> None:
    """Generate main BO result plots."""
    if results_df.empty:
        return
    df = results_df.sort_values("target").copy()

    # 1. Target tracking.
    plt.figure(figsize=(8, 6))
    plt.plot(df["target"], df["target"], linestyle="--", linewidth=2, color="black", label="Ideal tracking")
    plt.plot(df["target"], df["peak_y_true"], marker="o", linewidth=2.5, color="tab:blue", label="BO true peak_y")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("True achieved peak_y [m]")
    plt.title("BO target tracking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "bo_target_tracking.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # 2. Target error.
    plt.figure(figsize=(8, 6))
    bars = plt.bar(_target_labels(df["target"]), df["target_error_mm"], edgecolor="black")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Target error [mm]")
    plt.title("BO target error after true simulation")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, df["target_error_mm"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "bo_target_error.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # 3. Constraint validation.
    plt.figure(figsize=(8, 6))
    plt.plot(df["target"], df["max_abs_xr_true"], marker="o", linewidth=2.5, color="tab:orange", label="True max_abs_xr")
    plt.axhline(config.robot_limit_true, linestyle="--", linewidth=2, color="tab:red", label="True limit 0.500 m")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("True max_abs_xr [m]")
    plt.title("BO true constraint validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "bo_constraint_validation.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # 4. Residual margin.
    plt.figure(figsize=(8, 6))
    bars = plt.bar(_target_labels(df["target"]), df["residual_margin_mm"], edgecolor="black")
    plt.axhline(0.0, linestyle="--", linewidth=2, color="tab:red", label="Constraint boundary")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Residual margin to true limit [mm]")
    plt.title("BO residual margin: 0.500 - true max_abs_xr")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    for bar, value in zip(bars, df["residual_margin_mm"]):
        y_text = value if value >= 0 else 0
        plt.text(bar.get_x() + bar.get_width() / 2, y_text, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "bo_residual_margin.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # 5. Optimized parameters normalized to bounds.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.ravel()
    bounds = config.bounds_array()
    for i, col in enumerate(config.optimized_columns()):
        ax = axes[i]
        lower, upper = bounds[i]
        normalized = (df[col] - lower) / (upper - lower)
        ax.plot(df["target"], normalized, marker="o", linewidth=2)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Target outreach [m]")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.text(0.03, 0.06, f"Bounds: [{lower:.3g}, {upper:.3g}]", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.suptitle("BO optimized controllable parameters", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIGURES_DIR / "bo_optimized_parameters.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_bo_convergence(history_df: pd.DataFrame) -> None:
    """Plot BO convergence by true objective and feasible target error."""
    if history_df.empty:
        return
    df = history_df[(history_df["run_label"] == "main") & (history_df["method"] == "BO")].copy()
    if df.empty:
        return

    plt.figure(figsize=(9, 6))
    for target, df_t in df.groupby("target"):
        df_t = df_t.sort_values("iteration")
        plt.plot(df_t["iteration"], df_t["best_J_so_far"], linewidth=2, label=f"target {target:.2f}")
    plt.xlabel("True simulator calls")
    plt.ylabel("Best J_true so far")
    plt.title("BO convergence: best true objective")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "bo_convergence_objective.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    plt.figure(figsize=(9, 6))
    for target, df_t in df.groupby("target"):
        df_t = df_t.sort_values("iteration")
        plt.plot(df_t["iteration"], df_t["best_feasible_error_so_far_mm"], linewidth=2, label=f"target {target:.2f}")
    plt.xlabel("True simulator calls")
    plt.ylabel("Best feasible target error so far [mm]")
    plt.title("BO convergence: best feasible target error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "bo_convergence_feasible_error.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_bo_vs_random(bo_results_df: pd.DataFrame, random_results_df: pd.DataFrame) -> None:
    """Plot BO vs random search final target errors."""
    if bo_results_df.empty or random_results_df.empty:
        return

    bo = bo_results_df.sort_values("target").copy()
    rs = random_results_df[random_results_df["run_label"] == "random_main"].sort_values("target").copy()
    if rs.empty:
        return

    targets = bo["target"].values
    x = np.arange(len(targets))
    width = 0.36

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, bo["target_error_mm"], width, label="BO (Matern + EI)", edgecolor="black")
    plt.bar(x + width / 2, rs["target_error_mm"], width, label="Random Search", edgecolor="black")
    plt.xticks(x, _target_labels(targets))
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Best feasible target error [mm]")
    plt.title("BO vs Random Search at equal simulator budget")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "bo_vs_random_search.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_sensitivity_bar(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    filename: str,
    xlabel: str,
) -> None:
    """Generic sensitivity plot: target error and residual margin."""
    if df.empty or x_col not in df.columns:
        return
    df = df.copy()
    df[x_col] = df[x_col].astype(str)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(df[x_col], df["target_error_mm"], edgecolor="black", alpha=0.85, label="Target error")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Target error [mm]")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df[x_col], df["residual_margin_mm"], marker="o", linewidth=2, color="tab:orange", label="Residual margin")
    ax2.axhline(0.0, linestyle="--", linewidth=1.8, color="tab:red")
    ax2.set_ylabel("Residual margin [mm]")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    plt.title(title)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_budget_sweep(budget_df: pd.DataFrame) -> None:
    if budget_df.empty:
        return
    df = budget_df.sort_values("budget_total_evaluations").copy()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(df["budget_total_evaluations"], df["target_error_mm"], marker="o", linewidth=2.5, color="tab:blue", label="Target error")
    ax1.set_xlabel("True simulator calls")
    ax1.set_ylabel("Best feasible target error [mm]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["budget_total_evaluations"], df["residual_margin_mm"], marker="s", linewidth=2.5, color="tab:orange", label="Residual margin")
    ax2.axhline(0.0, linestyle="--", linewidth=1.8, color="tab:red")
    ax2.set_ylabel("Residual margin [mm]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    plt.title("BO budget sensitivity, target = 0.65 m")
    plt.tight_layout()
    path = FIGURES_DIR / "bo_budget_sweep_target065.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_pilot_timing(pilot_df: pd.DataFrame) -> None:
    if pilot_df.empty:
        return
    timing_cols = ["mean_simulation_time_s", "mean_gp_fit_time_s", "mean_acquisition_time_s"]
    labels = ["Simulator", "GP fit", "Acquisition"]
    values = [float(pilot_df.iloc[0][col]) for col in timing_cols]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, edgecolor="black")
    plt.ylabel("Mean time per BO call [s]")
    plt.title("BO pilot timing breakdown, target = 0.65 m")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "bo_pilot_timing_target065.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# =============================================================================
# TIME RESPONSE PLOTS
# =============================================================================

def simulate_candidate_with_solution(
    params: Dict[str, float],
    y_target: float,
    config: BOConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run true simulator and return t, y, x_b, x_r, metrics."""
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=config.robot_limit_true,
            return_metrics=True,
            return_full=True,
        )
    except TypeError:
        output = simulate_system(params, T_sim=60.0, dt=0.001, return_full=True)

    if not isinstance(output, tuple):
        raise RuntimeError("Unexpected simulate_system output. Expected tuple with solution.")

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
        raise RuntimeError("Could not extract ODE solution from simulate_system output.")

    t = sol.t
    x_b = sol.y[2]
    x_r = sol.y[3]
    y = x_b + x_r

    if metrics is None:
        max_abs_xr = float(np.max(np.abs(x_r)))
        metrics = {
            "peak_y": float(np.max(y)) if peak_y is None else peak_y,
            "max_abs_xr": max_abs_xr,
            "constraint_violation_abs": float(max(0.0, max_abs_xr - config.robot_limit_true)),
        }
    elif "max_abs_xr" not in metrics:
        metrics["max_abs_xr"] = get_max_abs_xr(metrics)

    return t, y, x_b, x_r, metrics


def plot_time_responses(results_df: pd.DataFrame, config: BOConfig) -> None:
    """Create true time-response plots for best BO solution per target."""
    if results_df.empty:
        return

    for _, row in results_df.sort_values("target").iterrows():
        target = float(row["target"])
        params = row_to_params(row, config)
        t, y, x_b, x_r, metrics = simulate_candidate_with_solution(params, target, config)

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

        axes[0].plot(t, y, linewidth=2.2, color="tab:blue", label="Total outreach y(t)")
        axes[0].axhline(target, linestyle="--", linewidth=2.0, color="black", label="Target")
        axes[0].axhline(config.robot_limit_true, linestyle=":", linewidth=2.0, color="tab:gray", label="Nominal reach")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title(
            f"BO true response, target = {target:.3f} m | peak_y = {metrics.get('peak_y', np.max(y)):.3f} m"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t, x_r, linewidth=2.0, color="tab:orange", label="Robot relative displacement x_r(t)")
        axes[1].axhline(config.robot_limit_true, linestyle="--", linewidth=2.0, color="tab:red", label="+ robot limit")
        axes[1].axhline(-config.robot_limit_true, linestyle="--", linewidth=2.0, color="tab:red", label="- robot limit")
        axes[1].set_ylabel("x_r [m]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(t, x_b, linewidth=2.0, color="tab:green", label="Base displacement x_b(t)")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("x_b [m]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        fig.suptitle(
            f"BO time response | error = {float(row['target_error_mm']):.2f} mm | "
            f"max_abs_xr = {float(row['max_abs_xr_true']):.4f} m | "
            f"margin = {float(row['residual_margin_mm']):.2f} mm",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = FIGURES_DIR / f"time_response_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


# =============================================================================
# CROSS-METHOD TARGET 0.65 COMPARISON TABLE
# =============================================================================

def make_three_way_target065_table(bo_results_df: pd.DataFrame, config: BOConfig) -> pd.DataFrame:
    """Create GP+DE / NN+Adam / BO comparison table for target 0.65 if files exist."""
    rows: List[Dict[str, Any]] = []
    target = config.sensitivity_target

    def _get_target_row(path: Path, method: str) -> Optional[pd.Series]:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "target" not in df.columns:
            return None
        df_t = df[np.isclose(df["target"].astype(float), target)]
        if df_t.empty:
            return None
        return df_t.iloc[0]

    gp_row = _get_target_row(GP_DE_RESULTS_PATH, "GP_DE")
    if gp_row is not None:
        max_xr = float(gp_row.get("max_abs_xr_true", np.nan))
        rows.append(
            {
                "method": "GP+DE",
                "peak_y_true": float(gp_row.get("peak_y_true", np.nan)),
                "target_error_mm": float(gp_row.get("target_error_mm", np.nan)),
                "max_abs_xr_true": max_xr,
                "residual_margin_mm": float((config.robot_limit_true - max_xr) * 1000.0),
                "feasible_abs": bool(gp_row.get("feasible_abs", True)),
                "offline_simulator_calls": config.offline_dataset_calls_surrogate_methods,
                "online_simulator_calls": config.online_validation_calls_gp_de_per_target,
                "surrogate_function_evaluations": int(gp_row.get("surrogate_function_evaluations", 0)),
                "optimization_time_s": float(gp_row.get("optimization_time_s", np.nan)) if "optimization_time_s" in gp_row.index else np.nan,
                "role": "offline GP surrogate + global DE",
            }
        )

    nn_row = _get_target_row(NN_ADAM_RESULTS_PATH, "NN_Adam")
    if nn_row is not None:
        max_xr = float(nn_row.get("max_abs_xr_true", np.nan))
        rows.append(
            {
                "method": "NN+Adam",
                "peak_y_true": float(nn_row.get("peak_y_true", np.nan)),
                "target_error_mm": float(nn_row.get("target_error_mm", np.nan)),
                "max_abs_xr_true": max_xr,
                "residual_margin_mm": float((config.robot_limit_true - max_xr) * 1000.0),
                "feasible_abs": bool(nn_row.get("feasible_abs", True)),
                "offline_simulator_calls": config.offline_dataset_calls_surrogate_methods,
                "online_simulator_calls": config.online_validation_calls_nn_adam_per_target,
                "surrogate_function_evaluations": int(nn_row.get("surrogate_function_evaluations", 0)),
                "optimization_time_s": float(nn_row.get("optimization_time_s", np.nan)) if "optimization_time_s" in nn_row.index else np.nan,
                "role": "offline NN surrogate + batched Adam",
            }
        )

    if not bo_results_df.empty:
        df_t = bo_results_df[np.isclose(bo_results_df["target"].astype(float), target)]
        if not df_t.empty:
            bo_row = df_t.iloc[0]
            rows.append(
                {
                    "method": "BO",
                    "peak_y_true": float(bo_row.get("peak_y_true", np.nan)),
                    "target_error_mm": float(bo_row.get("target_error_mm", np.nan)),
                    "max_abs_xr_true": float(bo_row.get("max_abs_xr_true", np.nan)),
                    "residual_margin_mm": float(bo_row.get("residual_margin_mm", np.nan)),
                    "feasible_abs": bool(bo_row.get("feasible_abs", True)),
                    "offline_simulator_calls": 0,
                    "online_simulator_calls": int(bo_row.get("n_true_simulator_evaluations", config.main_total_evaluations)),
                    "surrogate_function_evaluations": 0,
                    "optimization_time_s": float(bo_row.get("optimization_time_s", np.nan)) if "optimization_time_s" in bo_row.index else np.nan,
                    "role": "online true-simulator BO baseline",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(RESULTS_DIR / "bo_three_way_target065.csv", index=False)
    return out


# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def print_settings(config: BOConfig) -> None:
    print("=" * 80)
    print("TRUE-SIMULATOR BAYESIAN OPTIMIZATION BASELINE")
    print("=" * 80)
    print("Settings:")
    print(f"  Targets:                         {np.array(config.targets)}")
    print(f"  True robot limit:                {config.robot_limit_true:.3f} m")
    print(f"  Objective threshold:             {config.robot_limit_true:.3f} m")
    print(f"  y error scale:                   {config.y_error_scale_m * 1000:.1f} mm")
    print(f"  x constraint scale:              {config.x_constraint_scale_m * 1000:.1f} mm")
    print(f"  lambda constraint:               {config.lambda_constraint:g}")
    print(f"  Optimized params:                {config.optimized_columns()}")
    print(f"  Bounds:                          {list(map(tuple, config.bounds_array()))}")
    print(f"  Main kernel/acquisition:         {config.main_kernel} / {config.main_acquisition}")
    print(f"  Main budget:                     {config.main_total_evaluations} calls/target")
    print(f"  Initial LHS points:              {config.main_initial_points}")
    print(f"  Candidate pool global/local:     {config.candidate_pool_global} / {config.candidate_pool_local}")
    print(f"  GP restarts:                     {config.gp_restarts}")
    print(f"  Pilot run:                       {config.run_pilot}, stop_after_pilot={config.stop_after_pilot}")
    print("=" * 80)


def run_pilot(config: BOConfig) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PILOT BO RUN FOR RUNTIME ESTIMATION")
    print("=" * 80)
    print(
        f"Target={config.pilot_target:.3f} m | "
        f"total calls={config.pilot_total_evaluations} | "
        f"initial={config.pilot_initial_points} | "
        f"kernel={config.main_kernel} | acquisition={config.main_acquisition}"
    )

    best, hist, timing = run_bo_single_target(
        y_target=config.pilot_target,
        config=config,
        run_label="pilot",
        total_evaluations=config.pilot_total_evaluations,
        n_initial_points=config.pilot_initial_points,
        kernel_kind=config.main_kernel,
        acquisition=config.main_acquisition,
        seed=config.random_seed,
    )

    hist.to_csv(RESULTS_DIR / "bo_pilot_history_target065.csv", index=False)
    pilot_df = pd.DataFrame([timing])
    pilot_df.to_csv(RESULTS_DIR / "bo_pilot_timing_target065.csv", index=False)
    plot_pilot_timing(pilot_df)

    projected = timing["mean_iteration_time_s"] * config.main_total_evaluations * len(config.targets)
    projected_min = projected / 60.0

    print("\nPilot result:")
    print(
        f"  best true_y={timing['best_peak_y_true']:.4f} m | "
        f"err={timing['best_target_error_mm']:.2f} mm | "
        f"true max_abs_xr={timing['best_max_abs_xr_true']:.4f} m | "
        f"margin={timing['best_residual_margin_mm']:.2f} mm | "
        f"feasible={timing['best_feasible_abs']}"
    )
    print("Timing:")
    print(f"  total pilot wall time:           {timing['total_wall_time_s']:.2f} s")
    print(f"  mean iteration time:             {timing['mean_iteration_time_s']:.3f} s")
    print(f"  mean simulator time:             {timing['mean_simulation_time_s']:.3f} s")
    print(f"  mean GP fit time:                {timing['mean_gp_fit_time_s']:.3f} s")
    print(f"  mean acquisition time:           {timing['mean_acquisition_time_s']:.3f} s")
    print(f"  projected main BO time estimate: {projected_min:.1f} min")
    print(f"✓ Pilot saved: {RESULTS_DIR / 'bo_pilot_timing_target065.csv'}")

    return pilot_df


def run_main_bo(config: BOConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("MAIN BO OPTIMIZATION")
    print("=" * 80)

    result_rows = []
    history_rows = []
    timing_rows = []

    for i, target in enumerate(config.targets):
        seed = config.random_seed + 1000 + i
        print("-" * 80)
        print(f"BO target: {target:.3f} m")
        print("-" * 80)
        best, hist, timing = run_bo_single_target(
            y_target=target,
            config=config,
            run_label="main",
            total_evaluations=config.main_total_evaluations,
            n_initial_points=config.main_initial_points,
            kernel_kind=config.main_kernel,
            acquisition=config.main_acquisition,
            seed=seed,
        )
        result = build_result_row(best, config)
        result["optimization_time_s"] = float(timing["total_wall_time_s"])
        result_rows.append(result)
        history_rows.append(hist)
        timing_rows.append(timing)

        print(
            f"  Best true-validated BO solution: true_y={result['peak_y_true']:.4f} m | "
            f"err={result['target_error_mm']:.2f} mm | "
            f"true_xr_abs={result['max_abs_xr_true']:.4f} m | "
            f"margin={result['residual_margin_mm']:.2f} mm | "
            f"feasible={result['feasible_abs']} | "
            f"best_iter={result['best_iteration']} | "
            f"time={result['optimization_time_s']:.1f} s"
        )

    results_df = pd.DataFrame(result_rows).sort_values("target")
    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    timing_df = pd.DataFrame(timing_rows)

    results_df.to_csv(RESULTS_DIR / "bo_results.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "bo_history.csv", index=False)
    timing_df.to_csv(RESULTS_DIR / "bo_timing.csv", index=False)

    summary_df = summarize_results(results_df, "BO")
    summary_df.to_csv(RESULTS_DIR / "bo_summary.csv", index=False)

    print("\nMain BO summary:")
    print(results_df[["target", "peak_y_true", "target_error_mm", "max_abs_xr_true", "residual_margin_mm", "feasible_abs", "best_iteration", "optimization_time_s"]].to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    return results_df, history_df, timing_df


def run_random_search_baseline(config: BOConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("RANDOM SEARCH BASELINE")
    print("=" * 80)

    result_rows = []
    history_rows = []
    timing_rows = []

    # One main seed for all targets.
    for i, target in enumerate(config.targets):
        seed = config.random_search_main_seed + i
        print(f"Random Search target {target:.3f} m, seed={seed}")
        best, hist, timing = run_random_search_single_target(
            y_target=target,
            config=config,
            run_label="random_main",
            total_evaluations=config.random_search_total_evaluations,
            seed=seed,
        )
        result = build_result_row(best, config)
        result["optimization_time_s"] = float(timing["total_wall_time_s"])
        result_rows.append(result)
        history_rows.append(hist)
        timing_rows.append(timing)

    # Extra seeds only for target 0.65.
    extra_seed_rows = []
    for extra_seed in config.random_search_extra_seeds_target065:
        print(f"Random Search extra target {config.sensitivity_target:.3f} m, seed={extra_seed}")
        best, hist, timing = run_random_search_single_target(
            y_target=config.sensitivity_target,
            config=config,
            run_label="random_extra_target065",
            total_evaluations=config.random_search_total_evaluations,
            seed=extra_seed,
        )
        result = build_result_row(best, config)
        result["optimization_time_s"] = float(timing["total_wall_time_s"])
        extra_seed_rows.append(result)
        history_rows.append(hist)
        timing_rows.append(timing)

    results_df = pd.DataFrame(result_rows + extra_seed_rows)
    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    timing_df = pd.DataFrame(timing_rows)

    results_df.to_csv(RESULTS_DIR / "bo_random_search_results.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "bo_random_search_history.csv", index=False)
    timing_df.to_csv(RESULTS_DIR / "bo_random_search_timing.csv", index=False)

    # Seed summary target 0.65.
    target065 = results_df[np.isclose(results_df["target"], config.sensitivity_target)].copy()
    if not target065.empty:
        seed_summary = pd.DataFrame(
            [
                {
                    "target": config.sensitivity_target,
                    "n_seeds": int(len(target065)),
                    "mean_target_error_mm": float(target065["target_error_mm"].mean()),
                    "min_target_error_mm": float(target065["target_error_mm"].min()),
                    "max_target_error_mm": float(target065["target_error_mm"].max()),
                    "mean_residual_margin_mm": float(target065["residual_margin_mm"].mean()),
                    "feasible_rate_percent": float(100.0 * target065["feasible_abs"].mean()),
                }
            ]
        )
        seed_summary.to_csv(RESULTS_DIR / "bo_random_search_seed_summary_target065.csv", index=False)

    print("✓ Random Search baseline saved")
    return results_df, history_df, timing_df


def get_reusable_main_target065_result(
    bo_results_df: pd.DataFrame,
    bo_history_df: pd.DataFrame,
    config: BOConfig,
) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """Return main Matern+EI full-budget target 0.65 result/history if available."""
    if bo_results_df.empty:
        return None, None
    mask = (
        np.isclose(bo_results_df["target"].astype(float), config.sensitivity_target)
        & (bo_results_df["kernel"].astype(str).str.lower() == config.main_kernel.lower())
        & (bo_results_df["acquisition"].astype(str).str.lower() == config.main_acquisition.lower())
        & (bo_results_df["n_true_simulator_evaluations"].astype(int) == config.main_total_evaluations)
    )
    if not mask.any():
        return None, None
    result_row = bo_results_df[mask].iloc[0]
    hist = bo_history_df[
        (bo_history_df["run_label"] == "main")
        & np.isclose(bo_history_df["target"].astype(float), config.sensitivity_target)
    ].copy()
    return result_row, hist


def run_sensitivity_analyses(
    config: BOConfig,
    bo_results_df: pd.DataFrame,
    bo_history_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("BO SENSITIVITY ANALYSES ON TARGET 0.65")
    print("=" * 80)

    reusable_result, reusable_hist = get_reusable_main_target065_result(bo_results_df, bo_history_df, config)

    # ------------------------------------------------------------------
    # Kernel sensitivity: Matern vs RBF, EI acquisition.
    # ------------------------------------------------------------------
    kernel_rows = []
    kernel_histories = []
    if config.run_kernel_sensitivity:
        print("\nKernel sensitivity: Matern vs RBF")
        for kernel_kind in config.kernel_sensitivity_values:
            use_reusable = (
                reusable_result is not None
                and kernel_kind.lower() == config.main_kernel.lower()
                and config.main_acquisition.lower() == "ei"
                and config.main_total_evaluations == config.main_total_evaluations
            )
            if use_reusable:
                result = dict(reusable_result)
                result["sensitivity_type"] = "kernel"
                result["sensitivity_value"] = kernel_kind
                result["budget_total_evaluations"] = int(config.main_total_evaluations)
                kernel_rows.append(result)
                if reusable_hist is not None:
                    kernel_histories.append(reusable_hist)
                print(f"  Reused main run for kernel={kernel_kind}")
                continue

            best, hist, timing = run_bo_single_target(
                y_target=config.sensitivity_target,
                config=config,
                run_label="kernel_sensitivity",
                total_evaluations=config.main_total_evaluations,
                n_initial_points=config.main_initial_points,
                kernel_kind=kernel_kind,
                acquisition="ei",
                seed=config.random_seed + 3000 + len(kernel_rows),
            )
            result = build_result_row(best, config)
            result["optimization_time_s"] = float(timing["total_wall_time_s"])
            result["sensitivity_type"] = "kernel"
            result["sensitivity_value"] = kernel_kind
            result["budget_total_evaluations"] = int(config.main_total_evaluations)
            kernel_rows.append(result)
            kernel_histories.append(hist)
            print(
                f"  {kernel_kind}: err={result['target_error_mm']:.2f} mm | "
                f"margin={result['residual_margin_mm']:.2f} mm | feasible={result['feasible_abs']}"
            )

    kernel_df = pd.DataFrame(kernel_rows)
    if not kernel_df.empty:
        kernel_df.to_csv(RESULTS_DIR / "bo_kernel_sensitivity_target065.csv", index=False)
        plot_sensitivity_bar(
            kernel_df.rename(columns={"kernel": "kernel_label"}),
            x_col="kernel_label",
            title="BO kernel sensitivity, target = 0.65 m",
            filename="bo_kernel_sensitivity_target065.png",
            xlabel="Kernel",
        )

    # ------------------------------------------------------------------
    # Acquisition sensitivity: EI vs LCB, Matern kernel.
    # ------------------------------------------------------------------
    acquisition_rows = []
    acquisition_histories = []
    if config.run_acquisition_sensitivity:
        print("\nAcquisition sensitivity: EI vs LCB")
        for acquisition in config.acquisition_sensitivity_values:
            use_reusable = (
                reusable_result is not None
                and config.main_kernel.lower() == "matern"
                and acquisition.lower() == config.main_acquisition.lower()
            )
            if use_reusable:
                result = dict(reusable_result)
                result["sensitivity_type"] = "acquisition"
                result["sensitivity_value"] = acquisition
                result["budget_total_evaluations"] = int(config.main_total_evaluations)
                acquisition_rows.append(result)
                if reusable_hist is not None:
                    acquisition_histories.append(reusable_hist)
                print(f"  Reused main run for acquisition={acquisition}")
                continue

            best, hist, timing = run_bo_single_target(
                y_target=config.sensitivity_target,
                config=config,
                run_label="acquisition_sensitivity",
                total_evaluations=config.main_total_evaluations,
                n_initial_points=config.main_initial_points,
                kernel_kind="matern",
                acquisition=acquisition,
                seed=config.random_seed + 4000 + len(acquisition_rows),
            )
            result = build_result_row(best, config)
            result["optimization_time_s"] = float(timing["total_wall_time_s"])
            result["sensitivity_type"] = "acquisition"
            result["sensitivity_value"] = acquisition
            result["budget_total_evaluations"] = int(config.main_total_evaluations)
            acquisition_rows.append(result)
            acquisition_histories.append(hist)
            print(
                f"  {acquisition}: err={result['target_error_mm']:.2f} mm | "
                f"margin={result['residual_margin_mm']:.2f} mm | feasible={result['feasible_abs']}"
            )

    acquisition_df = pd.DataFrame(acquisition_rows)
    if not acquisition_df.empty:
        acquisition_df.to_csv(RESULTS_DIR / "bo_acquisition_sensitivity_target065.csv", index=False)
        plot_sensitivity_bar(
            acquisition_df.rename(columns={"acquisition": "acquisition_label"}),
            x_col="acquisition_label",
            title="BO acquisition sensitivity, target = 0.65 m",
            filename="bo_acquisition_sensitivity_target065.png",
            xlabel="Acquisition",
        )

    # ------------------------------------------------------------------
    # Budget sweep: 25, 50, full budget. Reuse main full-budget if possible.
    # ------------------------------------------------------------------
    budget_rows = []
    budget_histories = []
    if config.run_budget_sweep:
        print("\nBudget sweep")
        budgets = sorted(set(int(v) for v in config.budget_sweep_total_evaluations))
        for budget in budgets:
            use_reusable = (
                reusable_result is not None
                and budget == config.main_total_evaluations
                and config.main_kernel.lower() == "matern"
                and config.main_acquisition.lower() == "ei"
            )
            if use_reusable:
                result = dict(reusable_result)
                result["sensitivity_type"] = "budget"
                result["sensitivity_value"] = budget
                result["budget_total_evaluations"] = int(budget)
                budget_rows.append(result)
                if reusable_hist is not None:
                    budget_histories.append(reusable_hist)
                print(f"  Reused main run for budget={budget}")
                continue

            n_initial = min(config.budget_sweep_initial_points, max(5, budget // 2))
            if n_initial >= budget:
                n_initial = max(2, budget - 1)

            best, hist, timing = run_bo_single_target(
                y_target=config.sensitivity_target,
                config=config,
                run_label="budget_sweep",
                total_evaluations=budget,
                n_initial_points=n_initial,
                kernel_kind="matern",
                acquisition="ei",
                seed=config.random_seed + 5000 + budget,
            )
            result = build_result_row(best, config)
            result["optimization_time_s"] = float(timing["total_wall_time_s"])
            result["sensitivity_type"] = "budget"
            result["sensitivity_value"] = budget
            result["budget_total_evaluations"] = int(budget)
            budget_rows.append(result)
            budget_histories.append(hist)
            print(
                f"  budget={budget}: err={result['target_error_mm']:.2f} mm | "
                f"margin={result['residual_margin_mm']:.2f} mm | feasible={result['feasible_abs']}"
            )

    budget_df = pd.DataFrame(budget_rows)
    if not budget_df.empty:
        budget_df.to_csv(RESULTS_DIR / "bo_budget_sweep_target065.csv", index=False)
        plot_budget_sweep(budget_df)

    # Save sensitivity histories for reproducibility.
    all_sens_histories = []
    for frames in [kernel_histories, acquisition_histories, budget_histories]:
        for frame in frames:
            if frame is not None and not frame.empty:
                all_sens_histories.append(frame)
    if all_sens_histories:
        pd.concat(all_sens_histories, ignore_index=True).to_csv(
            RESULTS_DIR / "bo_sensitivity_histories_target065.csv", index=False
        )

    return kernel_df, acquisition_df, budget_df


def save_objective_settings(config: BOConfig) -> None:
    settings = asdict(config)
    with open(RESULTS_DIR / "bo_objective_settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    config = BOConfig()
    ensure_dirs()
    save_objective_settings(config)
    print_settings(config)

    pilot_df = pd.DataFrame()
    if config.run_pilot:
        pilot_df = run_pilot(config)
        if config.stop_after_pilot:
            print("\n" + "=" * 80)
            print("STOPPING AFTER PILOT RUN")
            print("=" * 80)
            print("Review the pilot timing first.")
            print("If the projected time is acceptable, set:")
            print("    stop_after_pilot = False")
            print("inside BOConfig and run the script again.")
            return

    bo_results_df = pd.DataFrame()
    bo_history_df = pd.DataFrame()
    bo_timing_df = pd.DataFrame()

    if config.run_main_bo:
        bo_results_df, bo_history_df, bo_timing_df = run_main_bo(config)

        print("\nGenerating BO plots...")
        plot_bo_results(bo_results_df, config)
        plot_bo_convergence(bo_history_df)
        if config.make_time_response_plots:
            plot_time_responses(bo_results_df, config)

    random_results_df = pd.DataFrame()
    random_history_df = pd.DataFrame()
    random_timing_df = pd.DataFrame()
    if config.run_random_search:
        random_results_df, random_history_df, random_timing_df = run_random_search_baseline(config)
        if not bo_results_df.empty:
            plot_bo_vs_random(bo_results_df, random_results_df)

    kernel_df = pd.DataFrame()
    acquisition_df = pd.DataFrame()
    budget_df = pd.DataFrame()
    if not bo_results_df.empty and (
        config.run_kernel_sensitivity or config.run_acquisition_sensitivity or config.run_budget_sweep
    ):
        kernel_df, acquisition_df, budget_df = run_sensitivity_analyses(config, bo_results_df, bo_history_df)

    if not bo_results_df.empty:
        three_way_df = make_three_way_target065_table(bo_results_df, config)
        if not three_way_df.empty:
            print("\nThree-way target 0.65 comparison:")
            print(three_way_df.to_string(index=False))
            print(f"✓ Saved: {RESULTS_DIR / 'bo_three_way_target065.csv'}")

    print("\n" + "=" * 80)
    print("BO PIPELINE COMPLETED")
    print("=" * 80)
    if not bo_results_df.empty:
        print(f"✓ Results: {RESULTS_DIR / 'bo_results.csv'}")
        print(f"✓ History: {RESULTS_DIR / 'bo_history.csv'}")
        print(f"✓ Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
