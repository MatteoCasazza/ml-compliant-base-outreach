"""
optimization_nn_gradient.py
===========================

Constraint-aware inverse optimization using the trained multi-output Neural
Network surrogate and batched multi-start Adam optimization.

Main purpose
------------
Given a desired outreach target y_target, find controllable parameters

    [Kr, hr, f0, f1, A, x_r_start]

while respecting the physical robot displacement constraint

    max_abs_xr <= 0.500 m.

The optimization loop uses the trained NN ensemble surrogate:

    NN ensemble mean : predicts [peak_y, max_abs_xr]

A conservative safety threshold is used inside the surrogate objective:

    max_abs_xr_pred <= 0.490 m

All selected candidates are finally validated with the true dynamic simulator.

Why gradient-based optimization?
--------------------------------
Unlike the GP pair, the NN is differentiable inside PyTorch. Therefore the
inverse-design variables can be optimized directly through automatic
differentiation. Because the objective is non-convex, a batched multi-start Adam
strategy is used.

Important methodological note
-----------------------------
The NN ensemble mean is differentiable because the outputs of all seed models are
averaged inside the PyTorch graph. The ensemble is used as a robustness-oriented
deterministic surrogate; no GP-style uncertainty penalty is added.

Absolute objective values should not be compared directly between GP+DE and
NN+Adam. Only final physical metrics after true-simulator validation should be
compared.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dynamics import simulate_system


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NN_RESULTS_DIR = PROJECT_ROOT / "results" / "nn"
GP_DE_RESULTS_PATH = PROJECT_ROOT / "results" / "optimization_gp_de" / "gp_de_results.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_nn_gradient"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_nn_gradient"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NNGradientConfig:
    """
    Configuration for NN + batched multi-start Adam inverse optimization.

    Optimized variables:
        [Kr, hr, f0, f1, A, x_r_start]

    Fixed variables:
        [Kb, Mb, hb, Mr]
    """

    # Fixed physical parameters
    Kb: float = 1000.0
    Mb: float = 20.0
    hb: float = 0.10
    Mr: float = 10.0

    # Physical and optimization limits
    robot_limit_true: float = 0.500
    robot_limit_opt: float = 0.490
    feasibility_tolerance_m: float = 1e-9

    # Targets
    targets: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)

    # Bounds for optimized variables: [Kr, hr, f0, f1, A, x_r_start]
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

    # Normalized objective scales
    y_error_scale_m: float = 0.010
    x_constraint_scale_m: float = 0.003
    lambda_constraint: float = 100.0

    # Batched Adam settings
    n_random_starts: int = 200
    adam_steps: int = 800
    learning_rate: float = 0.03
    top_k_validate: int = 50

    # Random initialization avoids exact sigmoid saturation at initialization.
    init_u_low: float = 0.02
    init_u_high: float = 0.98

    # Optional warm start from the GP+DE solution.
    include_gp_de_start: bool = False

    # Diagnostics for sigmoid-bound saturation.
    near_bound_threshold: float = 0.02

    # Reproducibility.
    random_seed: int = 2026

    # Convergence logging frequency.
    log_every: int = 10

    # Optional lightweight sensitivity analyses
    run_learning_rate_sweep: bool = False
    lr_sweep_target: float = 0.65
    lr_sweep_values: tuple[float, ...] = (0.01, 0.03, 0.05)
    lr_sweep_n_starts: int = 50
    lr_sweep_steps: int = 300
    lr_sweep_top_k_validate: int = 3

    run_nstarts_sweep: bool = False
    nstarts_sweep_target: float = 0.65
    nstarts_sweep_values: tuple[int, ...] = (50, 100, 150, 200)
    nstarts_sweep_steps: int = 300
    nstarts_sweep_top_k_validate: int = 3

    @staticmethod
    def input_columns() -> list[str]:
        """Full NN input order used during training."""
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
    def target_columns() -> list[str]:
        """NN output order used during training."""
        return ["peak_y", "max_abs_xr"]

    @staticmethod
    def optimized_columns() -> list[str]:
        """Optimizer vector order."""
        return ["Kr", "hr", "f0", "f1", "A", "x_r_start"]

    def fixed_params(self) -> dict[str, float]:
        """Return fixed physical parameters."""
        return {
            "Kb": self.Kb,
            "Mb": self.Mb,
            "hb": self.hb,
            "Mr": self.Mr,
        }

    def bounds(self) -> list[tuple[float, float]]:
        """Return physical bounds for optimized variables."""
        return [
            (self.Kr_min, self.Kr_max),
            (self.hr_min, self.hr_max),
            (self.f0_min, self.f0_max),
            (self.f1_min, self.f1_max),
            (self.A_min, self.A_max),
            (self.x_r_start_min, self.x_r_start_max),
        ]


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class MultiOutputMLP(nn.Module):
    """Fully connected multi-output neural surrogate."""

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


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_output_dirs() -> None:
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run the previous pipeline step first."
        )


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a torch checkpoint with compatibility across PyTorch versions."""
    require_file(path)

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def logit_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable numpy logit."""
    u = np.clip(u, eps, 1.0 - eps)
    return np.log(u / (1.0 - u))


def physical_to_unit(x: np.ndarray, config: NNGradientConfig) -> np.ndarray:
    """Map physical optimized variables to the unit interval [0, 1]."""
    bounds = np.asarray(config.bounds(), dtype=np.float64)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    return (x - lower) / (upper - lower)


def row_to_params(row: pd.Series, config: NNGradientConfig) -> dict[str, float]:
    """Convert a result row to full physical parameter dictionary."""
    params = {
        "Kb": float(config.Kb),
        "Mb": float(config.Mb),
        "hb": float(config.hb),
        "Mr": float(config.Mr),
    }

    for col in config.optimized_columns():
        params[col] = float(row[col])

    return params


def vector_to_params(
    x: Sequence[float],
    config: NNGradientConfig,
) -> dict[str, float]:
    """Convert optimized vector to full parameter dictionary."""
    Kr, hr, f0, f1, A, x_r_start = [float(v) for v in x]

    return {
        "Kb": float(config.Kb),
        "Kr": Kr,
        "Mb": float(config.Mb),
        "hb": float(config.hb),
        "hr": hr,
        "f0": f0,
        "f1": f1,
        "A": A,
        "x_r_start": x_r_start,
        "Mr": float(config.Mr),
    }


def extract_metric(
    metrics: dict[str, Any],
    key: str,
    default: float = np.nan,
) -> float:
    """Safely extract a scalar metric."""
    value = metrics.get(key, default)

    try:
        return float(value)
    except Exception:
        return float(default)


def get_max_abs_xr_from_metrics(metrics: dict[str, Any]) -> float:
    """Return max_abs_xr from simulator metrics with backward compatibility."""
    if "max_abs_xr" in metrics:
        return float(metrics["max_abs_xr"])

    if "max_xr" in metrics and "min_xr" in metrics:
        return float(
            max(
                abs(float(metrics["max_xr"])),
                abs(float(metrics["min_xr"])),
            )
        )

    if "max_xr" in metrics:
        return abs(float(metrics["max_xr"]))

    raise KeyError(
        "Could not find max_abs_xr, max_xr/min_xr, or max_xr in metrics."
    )


def get_constraint_violation_abs(
    metrics: dict[str, Any],
    max_abs_xr: float,
    config: NNGradientConfig,
) -> float:
    """Return absolute robot displacement violation."""
    if "constraint_violation_abs" in metrics:
        return float(metrics["constraint_violation_abs"])

    return float(max(0.0, max_abs_xr - config.robot_limit_true))


# =============================================================================
# LOADING NN SURROGATE
# =============================================================================

def load_nn_surrogate(
    config: NNGradientConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Load trained NN checkpoint and scalers."""
    print("Loading NN surrogate model...")

    model_path = NN_RESULTS_DIR / "nn_multioutput_model.pt"
    scaler_X_path = NN_RESULTS_DIR / "scaler_X.pkl"
    scaler_Y_path = NN_RESULTS_DIR / "scaler_Y.pkl"

    require_file(model_path)
    require_file(scaler_X_path)
    require_file(scaler_Y_path)

    checkpoint = torch_load_checkpoint(model_path, device=device)

    input_columns = list(checkpoint.get("input_columns", config.input_columns()))
    target_columns = list(checkpoint.get("target_columns", config.target_columns()))
    hidden_layers = list(checkpoint.get("hidden_layers", [32, 64, 32]))

    if input_columns != config.input_columns():
        raise ValueError(
            "NN input column order does not match the optimization script.\n"
            f"Checkpoint: {input_columns}\n"
            f"Expected:   {config.input_columns()}"
        )

    if "peak_y" not in target_columns or "max_abs_xr" not in target_columns:
        raise ValueError(
            "NN target columns must contain peak_y and max_abs_xr. "
            f"Found: {target_columns}"
        )

    ensemble_state_dicts = checkpoint.get("ensemble_state_dicts", None)

    if ensemble_state_dicts is not None and len(ensemble_state_dicts) > 0:
        model: list[MultiOutputMLP] | MultiOutputMLP = []

        for state_dict in ensemble_state_dicts:
            single_model = MultiOutputMLP(
                input_dim=len(input_columns),
                hidden_layers=hidden_layers,
                output_dim=len(target_columns),
            ).to(device)

            single_model.load_state_dict(state_dict)
            single_model.eval()
            model.append(single_model)

        model_is_ensemble = True

    else:
        model = MultiOutputMLP(
            input_dim=len(input_columns),
            hidden_layers=hidden_layers,
            output_dim=len(target_columns),
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model_is_ensemble = False

    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)

    scaler_tensors = {
        "X_mean": torch.tensor(scaler_X.mean_, dtype=torch.float32, device=device),
        "X_scale": torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device),
        "Y_mean": torch.tensor(scaler_Y.mean_, dtype=torch.float32, device=device),
        "Y_scale": torch.tensor(scaler_Y.scale_, dtype=torch.float32, device=device),
    }

    print("NN model loaded.")
    print(f"  Model path:       {model_path}")
    print(f"  Hidden layers:    {hidden_layers}")
    print(f"  Input columns:    {input_columns}")
    print(f"  Target columns:   {target_columns}")
    print(f"  Model type:       {'ensemble mean' if model_is_ensemble else 'best single seed'}")

    if model_is_ensemble:
        print(f"  Ensemble size:    {len(model)}")

    print(f"  Device:           {device}")
    print()

    return {
        "model": model,
        "model_is_ensemble": model_is_ensemble,
        "checkpoint": checkpoint,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "scaler_tensors": scaler_tensors,
        "input_columns": input_columns,
        "target_columns": target_columns,
        "peak_y_index": target_columns.index("peak_y"),
        "max_abs_xr_index": target_columns.index("max_abs_xr"),
    }


# =============================================================================
# DIFFERENTIABLE PARAMETERIZATION AND OBJECTIVE
# =============================================================================

def build_initial_z(
    y_target: float,
    config: NNGradientConfig,
    rng: np.random.Generator,
    n_random_starts: int | None = None,
    include_gp_de_start: bool | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """
    Build initial unconstrained variables z for batched multi-start Adam.

    z has shape [n_starts_total, 6]. It is transformed to bounded variables by:

        u = sigmoid(z)
        x = lower + u * (upper - lower)
    """
    n_random = config.n_random_starts if n_random_starts is None else int(n_random_starts)
    use_gp = config.include_gp_de_start if include_gp_de_start is None else bool(include_gp_de_start)

    u_random = rng.uniform(
        config.init_u_low,
        config.init_u_high,
        size=(n_random, 6),
    )

    z_list = [logit_np(u_random)]
    labels = ["random"] * n_random

    if use_gp and GP_DE_RESULTS_PATH.exists():
        try:
            gp_df = pd.read_csv(GP_DE_RESULTS_PATH)
            gp_row = gp_df[np.isclose(gp_df["target"].astype(float), float(y_target))]

            if not gp_row.empty:
                row = gp_row.iloc[0]
                x_gp = np.array(
                    [row[col] for col in config.optimized_columns()],
                    dtype=np.float64,
                )
                u_gp = physical_to_unit(x_gp, config)
                z_gp = logit_np(u_gp.reshape(1, -1), eps=1e-5)

                z_list.append(z_gp)
                labels.append("gp_de_start")

        except Exception as exc:
            print(f"Warning: could not load GP+DE start for target {y_target:.3f}: {exc}")

    z0 = np.vstack(z_list).astype(np.float32)

    return torch.tensor(z0, dtype=torch.float32), labels


def bounded_params_from_z(
    z: torch.Tensor,
    config: NNGradientConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert unconstrained z to physical optimized variables through sigmoid.

    Returns
    -------
    x_opt : torch.Tensor
        Physical variables [Kr, hr, f0, f1, A, x_r_start].
    u : torch.Tensor
        Unit variables in [0, 1], useful for sigmoid diagnostics.
    """
    device = z.device
    bounds = torch.tensor(config.bounds(), dtype=torch.float32, device=device)

    lower = bounds[:, 0]
    upper = bounds[:, 1]

    u = torch.sigmoid(z)
    x_opt = lower + u * (upper - lower)

    return x_opt, u


def build_full_feature_tensor(
    x_opt: torch.Tensor,
    config: NNGradientConfig,
) -> torch.Tensor:
    """Build full NN input tensor [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]."""
    batch = x_opt.shape[0]
    device = x_opt.device

    Kb = torch.full((batch,), float(config.Kb), dtype=torch.float32, device=device)
    Kr = x_opt[:, 0]

    Mb = torch.full((batch,), float(config.Mb), dtype=torch.float32, device=device)
    hb = torch.full((batch,), float(config.hb), dtype=torch.float32, device=device)

    hr = x_opt[:, 1]
    f0 = x_opt[:, 2]
    f1 = x_opt[:, 3]
    A = x_opt[:, 4]
    x_r_start = x_opt[:, 5]

    return torch.stack(
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start],
        dim=1,
    )


def nn_predict_from_xopt(
    x_opt: torch.Tensor,
    models: dict[str, Any],
    config: NNGradientConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiably predict [peak_y, max_abs_xr] from physical optimized variables.
    """
    X = build_full_feature_tensor(x_opt, config)

    scalers = models["scaler_tensors"]
    X_scaled = (X - scalers["X_mean"]) / scalers["X_scale"]

    nn_model = models["model"]

    if isinstance(nn_model, list):
        Y_scaled = torch.stack(
            [single_model(X_scaled) for single_model in nn_model],
            dim=0,
        ).mean(dim=0)
    else:
        Y_scaled = nn_model(X_scaled)

    Y = Y_scaled * scalers["Y_scale"] + scalers["Y_mean"]

    peak_y = Y[:, models["peak_y_index"]]
    max_abs_xr = Y[:, models["max_abs_xr_index"]]

    return peak_y, max_abs_xr, Y


def objective_terms_from_z(
    z: torch.Tensor,
    y_target: float,
    models: dict[str, Any],
    config: NNGradientConfig,
) -> dict[str, torch.Tensor]:
    """Compute objective and individual terms for all starts in the batch."""
    x_opt, u = bounded_params_from_z(z, config)
    peak_y_pred, max_abs_xr_pred, _ = nn_predict_from_xopt(x_opt, models, config)

    target_tensor = torch.tensor(
        float(y_target),
        dtype=torch.float32,
        device=z.device,
    )

    tracking_normalized = (peak_y_pred - target_tensor) / float(config.y_error_scale_m)

    violation_m = torch.relu(max_abs_xr_pred - float(config.robot_limit_opt))
    violation_normalized = violation_m / float(config.x_constraint_scale_m)

    tracking_cost = tracking_normalized**2
    constraint_cost = float(config.lambda_constraint) * violation_normalized**2
    total_cost = tracking_cost + constraint_cost

    return {
        "x_opt": x_opt,
        "u": u,
        "peak_y_pred": peak_y_pred,
        "max_abs_xr_pred": max_abs_xr_pred,
        "tracking_cost": tracking_cost,
        "constraint_cost": constraint_cost,
        "total_cost": total_cost,
        "predicted_violation_m": violation_m,
    }


# =============================================================================
# TRUE SIMULATOR VALIDATION
# =============================================================================

def simulate_candidate(
    params: dict[str, float],
    y_target: float,
    config: NNGradientConfig,
) -> dict[str, Any]:
    """Validate candidate parameters using the true dynamic simulator."""
    try:
        output = simulate_system(
            params,
            y_target=y_target,
            x_r_max=config.robot_limit_true,
            return_metrics=True,
        )
    except TypeError:
        output = simulate_system(params, return_metrics=True)

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
        f"Received type: {type(output)}."
    )


def validate_candidate(
    params: dict[str, float],
    y_target: float,
    config: NNGradientConfig,
    candidate_rank: int,
    start_row: pd.Series,
    run_label: str,
) -> dict[str, Any]:
    """Run true simulation and build validation record."""
    metrics = simulate_candidate(params, y_target, config)

    peak_y_true = extract_metric(metrics, "peak_y")
    max_abs_xr_true = get_max_abs_xr_from_metrics(metrics)

    constraint_violation_abs = get_constraint_violation_abs(
        metrics,
        max_abs_xr=max_abs_xr_true,
        config=config,
    )

    feasible_abs = bool(constraint_violation_abs <= config.feasibility_tolerance_m)

    target_error = abs(peak_y_true - y_target)
    reachability_gap = max(0.0, y_target - peak_y_true)
    residual_margin_m = config.robot_limit_true - max_abs_xr_true

    record: dict[str, Any] = {
        "method": "NN_Adam",
        "run_label": run_label,
        "target": float(y_target),
        "candidate_rank": int(candidate_rank),
        "source_start_id": int(start_row["start_id"]),
        "source_start_type": str(start_row["start_type"]),
        # Surrogate objective
        "objective_value": float(start_row["objective_value"]),
        "tracking_cost": float(start_row["tracking_cost"]),
        "constraint_cost": float(start_row["constraint_cost"]),
        # NN predictions
        "peak_y_pred": float(start_row["peak_y_pred"]),
        "max_abs_xr_pred": float(start_row["max_abs_xr_pred"]),
        "predicted_safety_violation_m": float(start_row["predicted_safety_violation_m"]),
        "predicted_safety_violation_mm": float(start_row["predicted_safety_violation_mm"]),
        # True simulator values
        "peak_y_true": float(peak_y_true),
        "max_abs_xr_true": float(max_abs_xr_true),
        "target_error_m": float(target_error),
        "target_error_mm": float(target_error * 1000.0),
        "reachability_gap_mm": float(reachability_gap * 1000.0),
        "residual_margin_m": float(residual_margin_m),
        "residual_margin_mm": float(residual_margin_m * 1000.0),
        "constraint_violation_abs_m": float(constraint_violation_abs),
        "constraint_violation_abs_mm": float(constraint_violation_abs * 1000.0),
        "feasible_abs": feasible_abs,
        "extra_reach": extract_metric(
            metrics,
            "extra_reach",
            peak_y_true - config.robot_limit_true,
        ),
        # NN prediction errors against simulator
        "nn_peak_y_error_vs_true_m": float(abs(start_row["peak_y_pred"] - peak_y_true)),
        "nn_peak_y_error_vs_true_mm": float(
            abs(start_row["peak_y_pred"] - peak_y_true) * 1000.0
        ),
        "nn_peak_y_error_signed_mm": float(
            (start_row["peak_y_pred"] - peak_y_true) * 1000.0
        ),
        "nn_max_abs_xr_error_vs_true_mm": float(
            abs(start_row["max_abs_xr_pred"] - max_abs_xr_true) * 1000.0
        ),
        "nn_max_abs_xr_error_signed_mm": float(
            (start_row["max_abs_xr_pred"] - max_abs_xr_true) * 1000.0
        ),
        # Extra simulator metrics
        "max_xr": extract_metric(metrics, "max_xr"),
        "min_xr": extract_metric(metrics, "min_xr"),
        "max_xb": extract_metric(metrics, "max_xb"),
        "min_xb": extract_metric(metrics, "min_xb"),
        "max_abs_xb": extract_metric(metrics, "max_abs_xb"),
        # Sigmoid-bound diagnostics
        "u_min": float(start_row["u_min"]),
        "u_max": float(start_row["u_max"]),
        "n_params_near_lower_bound": int(start_row["n_params_near_lower_bound"]),
        "n_params_near_upper_bound": int(start_row["n_params_near_upper_bound"]),
        "n_params_near_any_bound": int(start_row["n_params_near_any_bound"]),
    }

    for col in config.optimized_columns():
        record[col] = float(start_row[col])
        record[f"u_{col}"] = float(start_row[f"u_{col}"])

    record["Kb"] = float(config.Kb)
    record["Mb"] = float(config.Mb)
    record["hb"] = float(config.hb)
    record["Mr"] = float(config.Mr)

    return record


def select_best_validated(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Select best true-validated record.

    Priority:
    1. prefer true feasible solutions;
    2. among feasible, minimize true target error;
    3. if none feasible, minimize true violation, then target error.
    """
    if not records:
        raise ValueError("No validation records were provided.")

    def key(record: dict[str, Any]) -> tuple[int, float, float]:
        return (
            0 if bool(record["feasible_abs"]) else 1,
            0.0 if bool(record["feasible_abs"]) else float(record["constraint_violation_abs_m"]),
            float(record["target_error_m"]),
        )

    return dict(sorted(records, key=key)[0])


# =============================================================================
# OPTIMIZATION
# =============================================================================

def select_top_unique_start_indices(
    all_starts_df: pd.DataFrame,
    top_k: int,
    config: NNGradientConfig,
) -> list[int]:
    """Select top-k approximately unique starts by objective value."""
    sorted_df = all_starts_df.sort_values("objective_value")

    selected_indices: list[int] = []
    seen_keys = set()

    for idx, row in sorted_df.iterrows():
        key = tuple(round(float(row[col]), 8) for col in config.optimized_columns())

        if key in seen_keys:
            continue

        seen_keys.add(key)
        selected_indices.append(int(idx))

        if len(selected_indices) >= top_k:
            break

    if len(selected_indices) < top_k:
        for idx in sorted_df.index:
            if int(idx) not in selected_indices:
                selected_indices.append(int(idx))

            if len(selected_indices) >= top_k:
                break

    return selected_indices


def optimize_target_batched_adam(
    y_target: float,
    models: dict[str, Any],
    config: NNGradientConfig,
    device: torch.device,
    n_random_starts: int | None = None,
    adam_steps: int | None = None,
    learning_rate: float | None = None,
    top_k_validate: int | None = None,
    include_gp_de_start: bool | None = None,
    seed_offset: int = 0,
    run_label: str = "main",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimize one target with batched multi-start Adam and validate top candidates.
    """
    n_random = config.n_random_starts if n_random_starts is None else int(n_random_starts)
    steps = config.adam_steps if adam_steps is None else int(adam_steps)
    lr = config.learning_rate if learning_rate is None else float(learning_rate)
    k_validate = config.top_k_validate if top_k_validate is None else int(top_k_validate)

    rng_seed = int(config.random_seed + seed_offset + round(float(y_target) * 1000))
    rng = np.random.default_rng(rng_seed)

    z0_cpu, start_types = build_initial_z(
        y_target=y_target,
        config=config,
        rng=rng,
        n_random_starts=n_random,
        include_gp_de_start=include_gp_de_start,
    )

    z = z0_cpu.to(device).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    convergence_records: list[dict[str, Any]] = []
    start_time = time.time()

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)

        terms = objective_terms_from_z(
            z=z,
            y_target=y_target,
            models=models,
            config=config,
        )

        loss_per_start = terms["total_cost"]
        loss = loss_per_start.mean()

        if step < steps:
            loss.backward()
            optimizer.step()

        if step == 0 or step == steps or (step % config.log_every == 0):
            with torch.no_grad():
                loss_np = loss_per_start.detach().cpu().numpy()
                peak_np = terms["peak_y_pred"].detach().cpu().numpy()
                xr_np = terms["max_abs_xr_pred"].detach().cpu().numpy()
                best_idx = int(np.argmin(loss_np))

                convergence_records.append(
                    {
                        "run_label": run_label,
                        "target": float(y_target),
                        "step": int(step),
                        "best_objective": float(loss_np[best_idx]),
                        "median_objective": float(np.median(loss_np)),
                        "mean_objective": float(np.mean(loss_np)),
                        "best_peak_y_pred": float(peak_np[best_idx]),
                        "best_max_abs_xr_pred": float(xr_np[best_idx]),
                        "learning_rate": float(lr),
                        "n_random_starts": int(n_random),
                        "adam_steps": int(steps),
                    }
                )

    optimization_time_s = time.time() - start_time

    with torch.no_grad():
        terms = objective_terms_from_z(
            z=z,
            y_target=y_target,
            models=models,
            config=config,
        )

        x_opt_np = terms["x_opt"].detach().cpu().numpy()
        u_np = terms["u"].detach().cpu().numpy()
        peak_np = terms["peak_y_pred"].detach().cpu().numpy()
        xr_np = terms["max_abs_xr_pred"].detach().cpu().numpy()
        total_np = terms["total_cost"].detach().cpu().numpy()
        tracking_np = terms["tracking_cost"].detach().cpu().numpy()
        constraint_np = terms["constraint_cost"].detach().cpu().numpy()
        violation_np = terms["predicted_violation_m"].detach().cpu().numpy()

    all_start_records: list[dict[str, Any]] = []
    threshold = float(config.near_bound_threshold)

    for i in range(x_opt_np.shape[0]):
        near_lower = int(np.sum(u_np[i] < threshold))
        near_upper = int(np.sum(u_np[i] > 1.0 - threshold))

        record: dict[str, Any] = {
            "run_label": run_label,
            "target": float(y_target),
            "start_id": int(i),
            "start_type": start_types[i] if i < len(start_types) else "unknown",
            "objective_value": float(total_np[i]),
            "tracking_cost": float(tracking_np[i]),
            "constraint_cost": float(constraint_np[i]),
            "peak_y_pred": float(peak_np[i]),
            "max_abs_xr_pred": float(xr_np[i]),
            "target_error_pred_m": float(abs(peak_np[i] - y_target)),
            "target_error_pred_mm": float(abs(peak_np[i] - y_target) * 1000.0),
            "predicted_safety_violation_m": float(
                max(0.0, xr_np[i] - config.robot_limit_opt)
            ),
            "predicted_safety_violation_mm": float(
                max(0.0, xr_np[i] - config.robot_limit_opt) * 1000.0
            ),
            "predicted_violation_m": float(violation_np[i]),
            "u_min": float(np.min(u_np[i])),
            "u_max": float(np.max(u_np[i])),
            "n_params_near_lower_bound": near_lower,
            "n_params_near_upper_bound": near_upper,
            "n_params_near_any_bound": int(near_lower + near_upper),
            "learning_rate": float(lr),
            "adam_steps": int(steps),
            "n_random_starts": int(n_random),
            "optimization_time_s_all_starts": float(optimization_time_s),
        }

        for j, name in enumerate(config.optimized_columns()):
            record[name] = float(x_opt_np[i, j])
            record[f"u_{name}"] = float(u_np[i, j])

        all_start_records.append(record)

    all_starts_df = pd.DataFrame(all_start_records)
    convergence_df = pd.DataFrame(convergence_records)

    top_start_indices = select_top_unique_start_indices(
        all_starts_df=all_starts_df,
        top_k=k_validate,
        config=config,
    )

    validated_records: list[dict[str, Any]] = []

    for rank, idx in enumerate(top_start_indices, start=1):
        start_row = all_starts_df.loc[idx]
        params = row_to_params(start_row, config)

        validation = validate_candidate(
            params=params,
            y_target=y_target,
            config=config,
            candidate_rank=rank,
            start_row=start_row,
            run_label=run_label,
        )

        validated_records.append(validation)

    validated_df = pd.DataFrame(validated_records)
    best_record = select_best_validated(validated_records)

    pred_feasible = all_starts_df["max_abs_xr_pred"] <= config.robot_limit_opt
    pred_good_tracking_10mm = all_starts_df["target_error_pred_mm"] <= 10.0
    best_objective = float(all_starts_df["objective_value"].min())
    close_to_best = all_starts_df["objective_value"] <= best_objective * 1.05 + 1e-12

    best_record.update(
        {
            "run_label": run_label,
            "n_total_starts": int(len(all_starts_df)),
            "n_random_starts": int(n_random),
            "n_gp_de_starts": int((all_starts_df["start_type"] == "gp_de_start").sum()),
            "top_k_validate": int(k_validate),
            "adam_steps": int(steps),
            "learning_rate": float(lr),
            "optimization_time_s": float(optimization_time_s),
            "surrogate_function_evaluations": int(len(all_starts_df) * (steps + 1)),
            "predicted_feasible_start_rate_percent": float(pred_feasible.mean() * 100.0),
            "predicted_good_tracking_10mm_rate_percent": float(
                pred_good_tracking_10mm.mean() * 100.0
            ),
            "close_to_best_5pct_rate_percent": float(close_to_best.mean() * 100.0),
            "median_final_objective": float(all_starts_df["objective_value"].median()),
            "best_final_objective": best_objective,
        }
    )

    return best_record, all_starts_df, validated_df, convergence_df


# =============================================================================
# OPTIONAL SENSITIVITY ANALYSES
# =============================================================================

def run_learning_rate_sweep(
    models: dict[str, Any],
    base_config: NNGradientConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run lightweight learning-rate sensitivity on one target."""
    print("\n" + "=" * 80)
    print("OPTIONAL LEARNING-RATE SWEEP")
    print("=" * 80)
    print(f"Target: {base_config.lr_sweep_target:.3f} m")
    print(f"Values: {base_config.lr_sweep_values}")

    records = []

    for lr in base_config.lr_sweep_values:
        print(f"\nLearning rate = {lr:g}")

        cfg = replace(
            base_config,
            learning_rate=float(lr),
            n_random_starts=int(base_config.lr_sweep_n_starts),
            adam_steps=int(base_config.lr_sweep_steps),
            top_k_validate=int(base_config.lr_sweep_top_k_validate),
        )

        best, _, _, _ = optimize_target_batched_adam(
            y_target=cfg.lr_sweep_target,
            models=models,
            config=cfg,
            device=device,
            n_random_starts=cfg.lr_sweep_n_starts,
            adam_steps=cfg.lr_sweep_steps,
            learning_rate=lr,
            top_k_validate=cfg.lr_sweep_top_k_validate,
            include_gp_de_start=False,
            seed_offset=int(lr * 1_000_000),
            run_label="lr_sweep",
        )

        best["sweep_learning_rate"] = float(lr)
        records.append(best)

        print(
            f"  true_y={best['peak_y_true']:.4f} m | "
            f"err={best['target_error_mm']:.2f} mm | "
            f"true_xr_abs={best['max_abs_xr_true']:.4f} m | "
            f"feasible={best['feasible_abs']} | "
            f"time={best['optimization_time_s']:.3f} s"
        )

    df = pd.DataFrame(records)
    path = RESULTS_DIR / "nn_gradient_lr_sweep.csv"
    df.to_csv(path, index=False)

    print(f"Learning-rate sweep saved: {path}")

    return df


def run_nstarts_sweep(
    models: dict[str, Any],
    base_config: NNGradientConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run lightweight n-starts sensitivity on one target."""
    print("\n" + "=" * 80)
    print("OPTIONAL N_STARTS SWEEP")
    print("=" * 80)
    print(f"Target: {base_config.nstarts_sweep_target:.3f} m")
    print(f"Values: {base_config.nstarts_sweep_values}")

    records = []

    for n_starts in base_config.nstarts_sweep_values:
        print(f"\nn_random_starts = {n_starts}")

        cfg = replace(
            base_config,
            n_random_starts=int(n_starts),
            adam_steps=int(base_config.nstarts_sweep_steps),
            top_k_validate=int(base_config.nstarts_sweep_top_k_validate),
        )

        best, _, _, _ = optimize_target_batched_adam(
            y_target=cfg.nstarts_sweep_target,
            models=models,
            config=cfg,
            device=device,
            n_random_starts=n_starts,
            adam_steps=cfg.nstarts_sweep_steps,
            learning_rate=cfg.learning_rate,
            top_k_validate=cfg.nstarts_sweep_top_k_validate,
            include_gp_de_start=False,
            seed_offset=10_000 + int(n_starts),
            run_label="nstarts_sweep",
        )

        best["sweep_n_random_starts"] = int(n_starts)
        records.append(best)

        print(
            f"  true_y={best['peak_y_true']:.4f} m | "
            f"err={best['target_error_mm']:.2f} mm | "
            f"true_xr_abs={best['max_abs_xr_true']:.4f} m | "
            f"feasible={best['feasible_abs']} | "
            f"time={best['optimization_time_s']:.3f} s"
        )

    df = pd.DataFrame(records)
    path = RESULTS_DIR / "nn_gradient_nstarts_sweep.csv"
    df.to_csv(path, index=False)

    print(f"n_starts sweep saved: {path}")

    return df


# =============================================================================
# SUMMARY AND SAVE
# =============================================================================

def create_summary(
    results_df: pd.DataFrame,
    all_starts_df: pd.DataFrame,
    config: NNGradientConfig,
    total_wall_time_s: float,
) -> pd.DataFrame:
    """Create compact summary table."""
    return pd.DataFrame(
        [
            {
                "method": "NN_Adam",
                "n_targets": int(len(results_df)),
                "feasible_targets": int(results_df["feasible_abs"].sum()),
                "feasibility_rate_percent": float(results_df["feasible_abs"].mean() * 100.0),
                "mean_target_error_mm": float(results_df["target_error_mm"].mean()),
                "max_target_error_mm": float(results_df["target_error_mm"].max()),
                "mean_true_max_abs_xr_m": float(results_df["max_abs_xr_true"].mean()),
                "max_true_max_abs_xr_m": float(results_df["max_abs_xr_true"].max()),
                "mean_residual_margin_mm": float(results_df["residual_margin_mm"].mean()),
                "min_residual_margin_mm": float(results_df["residual_margin_mm"].min()),
                "mean_constraint_violation_abs_mm": float(
                    results_df["constraint_violation_abs_mm"].mean()
                ),
                "max_constraint_violation_abs_mm": float(
                    results_df["constraint_violation_abs_mm"].max()
                ),
                "mean_optimization_time_s": float(results_df["optimization_time_s"].mean()),
                "total_optimization_time_s": float(results_df["optimization_time_s"].sum()),
                "mean_surrogate_function_evaluations": float(
                    results_df["surrogate_function_evaluations"].mean()
                ),
                "total_surrogate_function_evaluations": int(
                    results_df["surrogate_function_evaluations"].sum()
                ),
                "online_true_simulator_calls": int(len(results_df) * config.top_k_validate),
                "offline_dataset_calls": 3000,
                "total_wall_time_s": float(total_wall_time_s),
                "mean_predicted_feasible_start_rate_percent": float(
                    results_df["predicted_feasible_start_rate_percent"].mean()
                ),
                "mean_good_tracking_10mm_start_rate_percent": float(
                    results_df["predicted_good_tracking_10mm_rate_percent"].mean()
                ),
                "mean_close_to_best_5pct_rate_percent": float(
                    results_df["close_to_best_5pct_rate_percent"].mean()
                ),
                "mean_params_near_bounds_best": float(
                    results_df["n_params_near_any_bound"].mean()
                ),
                "n_all_surrogate_starts": int(len(all_starts_df)),
            }
        ]
    )


def save_objective_settings(config: NNGradientConfig) -> None:
    """Save objective/settings metadata."""
    settings = asdict(config)

    settings.update(
        {
            "objective_formula": (
                "J = ((peak_y_pred - y_target)/y_error_scale_m)^2 + "
                "lambda_constraint * max(0, "
                "(max_abs_xr_pred - robot_limit_opt)/x_constraint_scale_m)^2"
            ),
            "method_note": (
                "The NN objective has no uncertainty term because the deterministic "
                "NN ensemble mean does not return GP-style predictive standard "
                "deviation. Do not compare absolute objective values with GP+DE; "
                "compare final true physical metrics."
            ),
        }
    )

    pd.DataFrame([settings]).to_csv(
        RESULTS_DIR / "nn_gradient_objective_settings.csv",
        index=False,
    )

    with (RESULTS_DIR / "nn_gradient_objective_settings.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(settings, file, indent=2)


def save_final_tables(
    results_df: pd.DataFrame,
    validated_df: pd.DataFrame,
    all_starts_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save final result tables."""
    paths = {
        "results": RESULTS_DIR / "nn_gradient_results.csv",
        "validated": RESULTS_DIR / "nn_gradient_validated_candidates.csv",
        "starts": RESULTS_DIR / "nn_gradient_all_starts.csv",
        "convergence": RESULTS_DIR / "nn_gradient_convergence.csv",
        "summary": RESULTS_DIR / "nn_gradient_summary.csv",
        "report": RESULTS_DIR / "nn_gradient_report_table.csv",
    }

    results_df.to_csv(paths["results"], index=False)
    validated_df.to_csv(paths["validated"], index=False)
    all_starts_df.to_csv(paths["starts"], index=False)
    convergence_df.to_csv(paths["convergence"], index=False)
    summary_df.to_csv(paths["summary"], index=False)

    report_cols = [
        "target",
        "peak_y_true",
        "peak_y_pred",
        "target_error_mm",
        "reachability_gap_mm",
        "residual_margin_mm",
        "nn_peak_y_error_signed_mm",
        "nn_peak_y_error_vs_true_mm",
        "max_abs_xr_true",
        "max_abs_xr_pred",
        "nn_max_abs_xr_error_signed_mm",
        "predicted_safety_violation_mm",
        "constraint_violation_abs_mm",
        "feasible_abs",
        "source_start_type",
        "n_total_starts",
        "predicted_feasible_start_rate_percent",
        "predicted_good_tracking_10mm_rate_percent",
        "close_to_best_5pct_rate_percent",
        "surrogate_function_evaluations",
        "optimization_time_s",
    ]

    available_report_cols = [col for col in report_cols if col in results_df.columns]
    results_df[available_report_cols].to_csv(paths["report"], index=False)

    return paths


# =============================================================================
# PLOTS
# =============================================================================

def plot_target_tracking(results_df: pd.DataFrame, config: NNGradientConfig) -> None:
    """Plot validated target tracking."""
    df = results_df.sort_values("target")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        df["target"],
        df["peak_y_true"],
        marker="o",
        linewidth=2,
        label="NN+Adam true validated",
    )

    ax.plot(
        [df["target"].min(), df["target"].max()],
        [df["target"].min(), df["target"].max()],
        linestyle="--",
        linewidth=2,
        label="Ideal tracking",
    )

    ax.axhline(
        config.robot_limit_true,
        linestyle=":",
        linewidth=2,
        label="Nominal robot reach",
    )

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("True peak_y [m]")
    ax.set_title("NN + Adam target tracking after true validation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "nn_gradient_target_tracking.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_target_error(results_df: pd.DataFrame) -> None:
    """Plot true target error."""
    df = results_df.sort_values("target")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(df["target"].astype(str), df["target_error_mm"], edgecolor="black")
    ax.axhline(10.0, linestyle="--", linewidth=2, label="10 mm tolerance")

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("True target error [mm]")
    ax.set_title("NN + Adam true target error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "nn_gradient_target_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_constraint_validation(
    results_df: pd.DataFrame,
    config: NNGradientConfig,
) -> None:
    """Plot predicted and true max_abs_xr after validation."""
    df = results_df.sort_values("target")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhspan(
        config.robot_limit_opt,
        config.robot_limit_true,
        alpha=0.15,
        label="Safety band",
    )

    ax.plot(
        df["target"],
        df["max_abs_xr_true"],
        marker="o",
        linewidth=2,
        label="True max_abs_xr",
    )

    ax.plot(
        df["target"],
        df["max_abs_xr_pred"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="NN predicted max_abs_xr",
    )

    ax.axhline(
        config.robot_limit_true,
        linestyle="--",
        linewidth=2,
        label="True physical limit",
    )

    ax.axhline(
        config.robot_limit_opt,
        linestyle=":",
        linewidth=2,
        label="Surrogate safety threshold",
    )

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("max_abs_xr [m]")
    ax.set_title("NN + Adam constraint validation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "nn_gradient_constraint_validation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_residual_margin(results_df: pd.DataFrame) -> None:
    """Plot residual margin after validation."""
    df = results_df.sort_values("target")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(df["target"].astype(str), df["residual_margin_mm"], edgecolor="black")
    ax.axhline(0.0, linestyle="--", linewidth=2, label="Constraint boundary")
    ax.axhline(5.0, linestyle=":", linewidth=2, label="5 mm reference margin")

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("Residual margin [mm]")
    ax.set_title("NN + Adam residual safety margin")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "nn_gradient_residual_margin.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_predicted_vs_true(results_df: pd.DataFrame) -> None:
    """Plot NN predictions against true simulator values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    y_true = results_df["peak_y_true"].to_numpy(dtype=float)
    y_pred = results_df["peak_y_pred"].to_numpy(dtype=float)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    axes[0].scatter(y_true, y_pred, s=70, edgecolor="black")
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

    for _, row in results_df.iterrows():
        axes[0].annotate(
            f"{row['target']:.2f}",
            (row["peak_y_true"], row["peak_y_pred"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    axes[0].set_xlabel("True peak_y [m]")
    axes[0].set_ylabel("NN predicted peak_y [m]")
    axes[0].set_title("peak_y prediction at optimized candidates")
    axes[0].grid(True, alpha=0.3)

    x_true = results_df["max_abs_xr_true"].to_numpy(dtype=float)
    x_pred = results_df["max_abs_xr_pred"].to_numpy(dtype=float)

    lo = min(x_true.min(), x_pred.min())
    hi = max(x_true.max(), x_pred.max())

    axes[1].scatter(x_true, x_pred, s=70, edgecolor="black")
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

    for _, row in results_df.iterrows():
        axes[1].annotate(
            f"{row['target']:.2f}",
            (row["max_abs_xr_true"], row["max_abs_xr_pred"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    axes[1].set_xlabel("True max_abs_xr [m]")
    axes[1].set_ylabel("NN predicted max_abs_xr [m]")
    axes[1].set_title("max_abs_xr prediction at optimized candidates")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("NN predictions vs true simulation at optimized candidates")

    path = FIGURES_DIR / "nn_gradient_predicted_vs_true.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_optimized_parameters(
    results_df: pd.DataFrame,
    config: NNGradientConfig,
) -> None:
    """Plot optimized controllable parameters normalized to their bounds."""
    df = results_df.sort_values("target")

    params = config.optimized_columns()
    bounds = config.bounds()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, param, bound in zip(axes, params, bounds):
        lower, upper = bound
        normalized = (df[param] - lower) / (upper - lower)

        ax.plot(df["target"], normalized, marker="o", linewidth=2)
        ax.set_title(param, fontweight="bold")
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

    fig.suptitle(
        "NN + Adam optimized controllable parameters",
        fontsize=16,
        fontweight="bold",
    )

    path = FIGURES_DIR / "nn_gradient_optimized_parameters.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_convergence(convergence_df: pd.DataFrame) -> None:
    """Plot Adam convergence."""
    if convergence_df.empty:
        return

    df = convergence_df[convergence_df["run_label"] == "main"].copy()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for target, df_target in df.groupby("target"):
        df_target = df_target.sort_values("step")

        ax.plot(
            df_target["step"],
            df_target["best_objective"],
            linewidth=2,
            label=f"target {target:.2f}",
        )

    ax.set_xlabel("Adam step")
    ax.set_ylabel("Best objective across starts")
    ax.set_title("NN + Adam convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / "nn_gradient_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_multistart_distribution(all_starts_df: pd.DataFrame) -> None:
    """Plot multi-start final objective and prediction distributions."""
    df = all_starts_df[all_starts_df["run_label"] == "main"].copy()

    if df.empty:
        return

    targets = sorted(df["target"].unique())

    obj_data = [
        df[np.isclose(df["target"], target)]["objective_value"].to_numpy(dtype=float)
        for target in targets
    ]

    peak_data = [
        df[np.isclose(df["target"], target)]["peak_y_pred"].to_numpy(dtype=float)
        for target in targets
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].boxplot(
        obj_data,
        tick_labels=[f"{target:.2f}" for target in targets],
        showfliers=False,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Target outreach [m]")
    axes[0].set_ylabel("Final objective")
    axes[0].set_title("Distribution of final objective across starts")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot(
        peak_data,
        tick_labels=[f"{target:.2f}" for target in targets],
        showfliers=False,
    )
    axes[1].set_xlabel("Target outreach [m]")
    axes[1].set_ylabel("Predicted peak_y [m]")
    axes[1].set_title("Distribution of predicted peak_y across starts")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("NN + Adam multi-start diagnostics")

    path = FIGURES_DIR / "nn_gradient_multistart_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_sigmoid_diagnostics(
    results_df: pd.DataFrame,
    all_starts_df: pd.DataFrame,
) -> None:
    """Plot sigmoid-bound saturation diagnostics."""
    df_best = results_df.sort_values("target")
    df_all = all_starts_df[all_starts_df["run_label"] == "main"].copy()

    if df_all.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(
        df_best["target"].astype(str),
        df_best["n_params_near_any_bound"],
        edgecolor="black",
    )
    axes[0].set_xlabel("Target outreach [m]")
    axes[0].set_ylabel("# optimized parameters near bounds")
    axes[0].set_title("Best validated solution: sigmoid-bound saturation")
    axes[0].grid(True, axis="y", alpha=0.3)

    targets = sorted(df_all["target"].unique())

    bound_data = [
        df_all[np.isclose(df_all["target"], target)]["n_params_near_any_bound"].to_numpy(dtype=float)
        for target in targets
    ]

    axes[1].boxplot(
        bound_data,
        tick_labels=[f"{target:.2f}" for target in targets],
        showfliers=False,
    )
    axes[1].set_xlabel("Target outreach [m]")
    axes[1].set_ylabel("# parameters near bounds")
    axes[1].set_title("All starts: sigmoid-bound saturation")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("NN + Adam sigmoid diagnostics")

    path = FIGURES_DIR / "nn_gradient_sigmoid_diagnostics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_lr_sweep(lr_df: pd.DataFrame, config: NNGradientConfig) -> None:
    """Plot learning-rate sweep."""
    if lr_df.empty or "sweep_learning_rate" not in lr_df.columns:
        return

    df = lr_df.sort_values("sweep_learning_rate")

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(
        df["sweep_learning_rate"],
        df["target_error_mm"],
        marker="o",
        linewidth=2,
        label="Target error",
    )
    ax1.set_xlabel("Learning rate")
    ax1.set_ylabel("Target error [mm]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        df["sweep_learning_rate"],
        df["max_abs_xr_true"],
        marker="s",
        linewidth=2,
        label="True max_abs_xr",
    )
    ax2.axhline(
        config.robot_limit_true,
        linestyle="--",
        linewidth=2,
        label="True limit",
    )
    ax2.axhline(
        config.robot_limit_opt,
        linestyle=":",
        linewidth=2,
        label="Safety threshold",
    )
    ax2.set_ylabel("True max_abs_xr [m]")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

    ax1.set_title(
        f"NN + Adam learning-rate sensitivity, target = {config.lr_sweep_target:.3f} m"
    )

    path = FIGURES_DIR / "nn_gradient_lr_sensitivity_target065.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_nstarts_sweep(ns_df: pd.DataFrame, config: NNGradientConfig) -> None:
    """Plot n-starts sweep."""
    if ns_df.empty or "sweep_n_random_starts" not in ns_df.columns:
        return

    df = ns_df.sort_values("sweep_n_random_starts")

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(
        df["sweep_n_random_starts"],
        df["target_error_mm"],
        marker="o",
        linewidth=2,
        label="Target error",
    )
    ax1.set_xlabel("Number of random starts")
    ax1.set_ylabel("Target error [mm]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        df["sweep_n_random_starts"],
        df["optimization_time_s"],
        marker="s",
        linewidth=2,
        label="Optimization time",
    )
    ax2.set_ylabel("Optimization time [s]")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

    ax1.set_title(
        f"NN + Adam n-starts sensitivity, target = {config.nstarts_sweep_target:.3f} m"
    )

    path = FIGURES_DIR / "nn_gradient_nstarts_sensitivity_target065.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# =============================================================================
# TIME RESPONSE PLOTS
# =============================================================================

def simulate_candidate_with_solution(
    params: dict[str, float],
    y_target: float,
    config: NNGradientConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run true simulator and return t, y, x_b, x_r, metrics."""
    output = None

    for kwargs in [
        {
            "y_target": y_target,
            "x_r_max": config.robot_limit_true,
            "return_metrics": True,
            "return_full": True,
        },
        {
            "y_target": y_target,
            "x_r_max": config.robot_limit_true,
            "return_metrics": True,
            "return_solution": True,
        },
        {
            "return_metrics": True,
            "return_full": True,
        },
        {
            "return_metrics": True,
            "return_solution": True,
        },
    ]:
        try:
            output = simulate_system(params, **kwargs)
            break
        except TypeError:
            continue

    if output is None or not isinstance(output, tuple):
        raise RuntimeError("Unexpected simulate_system output. Expected tuple with solution.")

    sol = None
    metrics = None
    peak_y = None

    for item in output:
        if hasattr(item, "t") and hasattr(item, "y"):
            sol = item
        elif isinstance(item, dict) and "peak_y" in item:
            metrics = dict(item)
        elif isinstance(item, (float, int, np.floating)):
            peak_y = float(item)

    if sol is None:
        raise RuntimeError("Could not extract ODE solution from simulate_system output.")

    t = np.asarray(sol.t)
    x_b = np.asarray(sol.y[2])
    x_r = np.asarray(sol.y[3])
    y = x_b + x_r

    if metrics is None:
        max_abs_xr = float(np.max(np.abs(x_r)))
        metrics = {
            "peak_y": float(np.max(y)) if peak_y is None else peak_y,
            "max_abs_xr": max_abs_xr,
            "max_xr": float(np.max(x_r)),
            "min_xr": float(np.min(x_r)),
            "max_xb": float(np.max(x_b)),
            "min_xb": float(np.min(x_b)),
            "max_abs_xb": float(np.max(np.abs(x_b))),
            "constraint_violation_abs": float(
                max(0.0, max_abs_xr - config.robot_limit_true)
            ),
        }

    if "max_abs_xr" not in metrics:
        metrics["max_abs_xr"] = get_max_abs_xr_from_metrics(metrics)

    if "peak_y" not in metrics:
        metrics["peak_y"] = float(np.max(y))

    return t, y, x_b, x_r, metrics


def plot_time_responses(
    results_df: pd.DataFrame,
    config: NNGradientConfig,
) -> None:
    """Create true time-response plots for best NN+Adam solution per target."""
    for _, row in results_df.sort_values("target").iterrows():
        target = float(row["target"])
        params = row_to_params(row, config)

        try:
            t, y, x_b, x_r, metrics = simulate_candidate_with_solution(
                params=params,
                y_target=target,
                config=config,
            )
        except Exception as exc:
            print(f"Time-response plot skipped for target {target:.3f}: {exc}")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

        axes[0].plot(t, y, linewidth=2.2, label="Total outreach y(t)")
        axes[0].axhline(
            target,
            linestyle=":",
            linewidth=2.2,
            label=f"Target {target:.3f} m",
        )
        axes[0].axhline(
            config.robot_limit_true,
            linestyle="--",
            linewidth=1.8,
            label="Nominal robot reach",
        )
        axes[0].set_ylabel("y(t) [m]")
        axes[0].set_title(
            f"NN + Adam true time response, target = {target:.3f} m\n"
            f"peak_y = {metrics['peak_y']:.3f} m, "
            f"error = {abs(metrics['peak_y'] - target) * 1000.0:.1f} mm, "
            f"max_abs_xr = {metrics['max_abs_xr']:.3f} m"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)

        axes[1].plot(t, x_r, linewidth=2.2, label="Robot displacement x_r(t)")
        axes[1].axhline(
            config.robot_limit_true,
            linestyle="--",
            linewidth=2,
            label="+ robot limit",
        )
        axes[1].axhline(
            -config.robot_limit_true,
            linestyle="--",
            linewidth=2,
            label="- robot limit",
        )
        axes[1].set_ylabel("x_r(t) [m]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

        axes[2].plot(t, x_b, linewidth=2.2, label="Base displacement x_b(t)")
        axes[2].axhline(
            0.0,
            linestyle=":",
            linewidth=1.5,
            label="Zero reference",
        )
        axes[2].set_ylabel("x_b(t) [m]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=9)

        target_tag = f"{int(round(target * 1000)):04d}"
        path = FIGURES_DIR / f"time_response_target{target_tag}.png"

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def generate_all_plots(
    results_df: pd.DataFrame,
    all_starts_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    config: NNGradientConfig,
    include_time_responses: bool = True,
) -> None:
    """Generate all NN + Adam figures."""
    print("\nGenerating NN + Adam plots...")

    plot_target_tracking(results_df, config)
    plot_target_error(results_df)
    plot_constraint_validation(results_df, config)
    plot_residual_margin(results_df)
    plot_predicted_vs_true(results_df)
    plot_optimized_parameters(results_df, config)
    plot_convergence(convergence_df)
    plot_multistart_distribution(all_starts_df)
    plot_sigmoid_diagnostics(results_df, all_starts_df)

    if include_time_responses:
        plot_time_responses(results_df, config)


# =============================================================================
# PRINTING
# =============================================================================

def print_settings(config: NNGradientConfig, device: torch.device) -> None:
    """Print main optimization settings."""
    print("=" * 80)
    print("NN + MULTI-START ADAM CONSTRAINT-AWARE OPTIMIZATION")
    print("=" * 80)
    print("Settings:")
    print(f"  Targets:                         {np.array(config.targets)}")
    print(f"  True robot limit:                {config.robot_limit_true:.3f} m")
    print(f"  Surrogate safety threshold:      {config.robot_limit_opt:.3f} m")
    print(f"  y error scale:                   {config.y_error_scale_m * 1000.0:.1f} mm")
    print(f"  x constraint scale:              {config.x_constraint_scale_m * 1000.0:.1f} mm")
    print(f"  lambda constraint:               {config.lambda_constraint:g}")
    print(f"  n random starts:                 {config.n_random_starts}")
    print(f"  include GP+DE start:             {config.include_gp_de_start}")
    print(f"  Adam steps / lr:                 {config.adam_steps} / {config.learning_rate:g}")
    print(f"  top_k_validate:                  {config.top_k_validate}")
    print(f"  Fixed params:                    {config.fixed_params()}")
    print(f"  Optimized params:                {config.optimized_columns()}")
    print(f"  Bounds:                          {config.bounds()}")
    print(f"  Device:                          {device}")
    print("=" * 80 + "\n")


def print_final_summary(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Print final compact summary."""
    print("\n" + "=" * 80)
    print("FINAL NN + ADAM SUMMARY")
    print("=" * 80)

    display_cols = [
        "target",
        "peak_y_true",
        "peak_y_pred",
        "target_error_mm",
        "reachability_gap_mm",
        "residual_margin_mm",
        "nn_peak_y_error_signed_mm",
        "max_abs_xr_true",
        "max_abs_xr_pred",
        "predicted_safety_violation_mm",
        "constraint_violation_abs_mm",
        "feasible_abs",
        "source_start_type",
        "surrogate_function_evaluations",
        "optimization_time_s",
    ]

    available_display_cols = [col for col in display_cols if col in results_df.columns]

    print(results_df[available_display_cols].to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("=" * 80)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_targets(raw: str) -> tuple[float, ...]:
    """Parse comma-separated outreach targets."""
    values = [item.strip() for item in raw.split(",") if item.strip()]

    if not values:
        raise ValueError("At least one target must be provided.")

    return tuple(float(value) for value in values)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NN + Adam constraint-aware inverse optimization."
    )

    parser.add_argument(
        "--targets",
        type=str,
        default="0.55,0.60,0.65,0.70,0.75",
        help="Comma-separated target outreach values.",
    )

    parser.add_argument(
        "--n_random_starts",
        type=int,
        default=NNGradientConfig.n_random_starts,
        help="Number of random starts.",
    )

    parser.add_argument(
        "--adam_steps",
        type=int,
        default=NNGradientConfig.adam_steps,
        help="Number of Adam steps.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=NNGradientConfig.learning_rate,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--top_k_validate",
        type=int,
        default=NNGradientConfig.top_k_validate,
        help="Number of surrogate-ranked candidates to validate with the true simulator.",
    )

    parser.add_argument(
        "--include_gp_de_start",
        action="store_true",
        help="Add GP+DE solution as one extra initialization when available.",
    )

    parser.add_argument(
        "--run_learning_rate_sweep",
        action="store_true",
        help="Run optional learning-rate sweep before final optimization.",
    )

    parser.add_argument(
        "--run_nstarts_sweep",
        action="store_true",
        help="Run optional n-starts sweep before final optimization.",
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip all plot generation.",
    )

    parser.add_argument(
        "--no_time_responses",
        action="store_true",
        help="Skip true-simulator time-response plots.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> NNGradientConfig:
    """Build configuration from command-line arguments."""
    return NNGradientConfig(
        targets=parse_targets(args.targets),
        n_random_starts=int(args.n_random_starts),
        adam_steps=int(args.adam_steps),
        learning_rate=float(args.learning_rate),
        top_k_validate=int(args.top_k_validate),
        include_gp_de_start=bool(args.include_gp_de_start),
        run_learning_rate_sweep=bool(args.run_learning_rate_sweep),
        run_nstarts_sweep=bool(args.run_nstarts_sweep),
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """Run the complete NN + Adam inverse optimization pipeline."""
    args = parse_args()
    config = build_config(args)

    ensure_output_dirs()

    device = get_device()
    print_settings(config, device)

    total_start_time = time.time()

    models = load_nn_surrogate(config, device)
    save_objective_settings(config)

    lr_sweep_df = pd.DataFrame()
    nstarts_sweep_df = pd.DataFrame()

    if config.run_learning_rate_sweep:
        lr_sweep_df = run_learning_rate_sweep(models, config, device)

        if not args.no_plots:
            plot_lr_sweep(lr_sweep_df, config)

    if config.run_nstarts_sweep:
        nstarts_sweep_df = run_nstarts_sweep(models, config, device)

        if not args.no_plots:
            plot_nstarts_sweep(nstarts_sweep_df, config)

    best_records: list[dict[str, Any]] = []
    all_start_frames: list[pd.DataFrame] = []
    validated_frames: list[pd.DataFrame] = []
    convergence_frames: list[pd.DataFrame] = []

    print("\n" + "=" * 80)
    print("MAIN NN + ADAM OPTIMIZATION")
    print("=" * 80)

    for target in config.targets:
        print("-" * 80)
        print(f"NN + Adam target: {target:.3f} m")
        print("-" * 80)

        best, all_starts, validated, convergence = optimize_target_batched_adam(
            y_target=float(target),
            models=models,
            config=config,
            device=device,
            run_label="main",
            seed_offset=0,
        )

        best_records.append(best)
        all_start_frames.append(all_starts)
        validated_frames.append(validated)
        convergence_frames.append(convergence)

        print(
            f"  Best true-validated solution: "
            f"true_y={best['peak_y_true']:.4f} m | "
            f"err={best['target_error_mm']:.2f} mm | "
            f"true_xr_abs={best['max_abs_xr_true']:.4f} m | "
            f"margin={best['residual_margin_mm']:.2f} mm | "
            f"violation={best['constraint_violation_abs_mm']:.2f} mm | "
            f"feasible={best['feasible_abs']} | "
            f"time={best['optimization_time_s']:.3f} s | "
            f"starts={best['n_total_starts']}"
        )

    results_df = (
        pd.DataFrame(best_records)
        .sort_values("target")
        .reset_index(drop=True)
    )

    all_starts_df = (
        pd.concat(all_start_frames, ignore_index=True)
        if all_start_frames
        else pd.DataFrame()
    )

    validated_df = (
        pd.concat(validated_frames, ignore_index=True)
        if validated_frames
        else pd.DataFrame()
    )

    convergence_df = (
        pd.concat(convergence_frames, ignore_index=True)
        if convergence_frames
        else pd.DataFrame()
    )

    total_wall_time_s = time.time() - total_start_time

    summary_df = create_summary(
        results_df=results_df,
        all_starts_df=all_starts_df,
        config=config,
        total_wall_time_s=total_wall_time_s,
    )

    saved_paths = save_final_tables(
        results_df=results_df,
        validated_df=validated_df,
        all_starts_df=all_starts_df,
        convergence_df=convergence_df,
        summary_df=summary_df,
    )

    if not args.no_plots:
        generate_all_plots(
            results_df=results_df,
            all_starts_df=all_starts_df,
            convergence_df=convergence_df,
            config=config,
            include_time_responses=not args.no_time_responses,
        )

    print_final_summary(results_df, summary_df)

    print("\nSaved files:")
    for path in saved_paths.values():
        print(f"  - {path}")

    print(f"  - {RESULTS_DIR / 'nn_gradient_objective_settings.csv'}")
    print(f"  - {RESULTS_DIR / 'nn_gradient_objective_settings.json'}")

    if config.run_learning_rate_sweep:
        print(f"  - {RESULTS_DIR / 'nn_gradient_lr_sweep.csv'}")

    if config.run_nstarts_sweep:
        print(f"  - {RESULTS_DIR / 'nn_gradient_nstarts_sweep.csv'}")

    if not args.no_plots:
        print(f"  - {FIGURES_DIR / 'nn_gradient_*.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()