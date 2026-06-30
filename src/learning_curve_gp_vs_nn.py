"""
learning_curve_gp_vs_nn_final.py
================================

Learning-curve comparison between the final selected surrogate configurations.

Compared models
---------------
1. GP pair:
   - GP_peak_y: Matérn 5/2, alpha = 1e-6
   - GP_max_abs_xr: Matérn 3/2, alpha = 1e-6

2. NN multi-output:
   - Architecture: [32, 64, 32]
   - Weighted loss: MSE(peak_y) + lambda * MSE(max_abs_xr)
   - Ensemble seeds: [42, 123, 2026]

Purpose
-------
This script compares sample efficiency. For each training size, both surrogate
families are trained using the same training subset and evaluated on the same
fixed test set.

This script is not the final model-training script. It is an analysis script
used for the report.

Run from project root:
    python src/learning_curve_gp_vs_nn_final.py

Quick debug:
    python src/learning_curve_gp_vs_nn_final.py --quick_debug

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "learning_curve"
FIGURES_DIR = PROJECT_ROOT / "figures" / "learning_curve"

DEFAULT_DATASET_PATH = DATA_DIR / "dataset_augmented.csv"


# =============================================================================
# GLOBAL SETTINGS
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

TARGET_COLUMNS = ["peak_y", "max_abs_xr"]

PEAK_COL = "peak_y"
MAX_ABS_XR_COL = "max_abs_xr"

TEST_SIZE = 0.20
RANDOM_STATE = 42

DEFAULT_TRAIN_SIZES = [200, 400, 800, 1200, 1600, 2000, 2400]
DEBUG_TRAIN_SIZES = [200, 400]

GP_ALPHA_PEAK_Y = 1e-6
GP_ALPHA_MAX_ABS_XR = 1e-6
GP_KERNEL_PEAK_Y = "matern52"
GP_KERNEL_MAX_ABS_XR = "matern32"
GP_LENGTH_SCALE_BOUNDS = (1e-2, 1e3)

DEFAULT_GP_N_RESTARTS = 1

NN_HIDDEN_LAYERS = [32, 64, 32]
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-5
NN_CONSTRAINT_LOSS_WEIGHT = 5.0
NN_BATCH_SIZE = 64
NN_MAX_EPOCHS = 1500
NN_PATIENCE = 120
NN_VAL_FRACTION_FROM_TRAIN_POOL = 0.15
NN_SEEDS = [42, 123, 2026]

DEBUG_NN_MAX_EPOCHS = 80
DEBUG_NN_PATIENCE = 15
DEBUG_NN_SEEDS = [42]

USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 40
SCHEDULER_MIN_LR = 1e-6

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495

NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520

HIGH_OUTREACH_THRESHOLD = 0.600


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LearningCurveConfig:
    """Configuration for the learning-curve experiment."""

    dataset_path: Path
    train_sizes: list[int]
    gp_n_restarts: int
    nn_max_epochs: int
    nn_patience: int
    nn_seeds: list[int]
    use_lr_scheduler: bool
    skip_plots: bool
    quick_debug: bool


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the previous pipeline step first."
        )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_train_sizes(raw: str) -> list[int]:
    """Parse comma-separated training sizes."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    train_sizes = [int(value) for value in values]

    if any(value <= 0 for value in train_sizes):
        raise ValueError("All training sizes must be positive.")

    return train_sizes


# =============================================================================
# DATASET
# =============================================================================

def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the final augmented dataset."""
    require_file(dataset_path)

    df = pd.read_csv(dataset_path, comment="#")

    required_cols = INPUT_COLUMNS + TARGET_COLUMNS
    require_columns(df, required_cols, "Dataset")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    Y = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    return X, Y, df


def print_dataset_summary(df: pd.DataFrame, dataset_path: Path) -> None:
    """Print compact dataset summary."""
    print("\n" + "=" * 70)
    print("LEARNING CURVE DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset path:              {dataset_path}")
    print(f"Samples:                   {len(df)}")
    print(f"Input dimensions:          {len(INPUT_COLUMNS)}")
    print(f"Targets:                   {TARGET_COLUMNS}")
    print(f"peak_y mean/std:           {df[PEAK_COL].mean():.6f} / {df[PEAK_COL].std():.6f} m")
    print(f"max_abs_xr mean/std:       {df[MAX_ABS_XR_COL].mean():.6f} / {df[MAX_ABS_XR_COL].std():.6f} m")

    if "feasible_abs" in df.columns:
        feasible_count = int(df["feasible_abs"].astype(bool).sum())
        print(f"Feasible_abs samples:      {feasible_count} ({100.0 * feasible_count / len(df):.1f}%)")

    if "dataset_type" in df.columns:
        print("Dataset types:")
        for name, count in df["dataset_type"].value_counts().items():
            print(f"  {str(name):22s}: {count}")

    print("=" * 70)


# =============================================================================
# METRICS
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute scalar regression metrics in meters and millimeters."""
    rmse_m = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_m = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse_m": rmse_m,
        "rmse_mm": rmse_m * 1000.0,
        "mae_m": mae_m,
        "mae_mm": mae_m * 1000.0,
        "r2": r2,
    }


def constraint_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    limit: float = ROBOT_LIMIT_TRUE,
) -> dict[str, float]:
    """Compute feasibility classification metrics for max_abs_xr."""
    true_feasible = y_true <= limit
    pred_feasible = y_pred <= limit

    accuracy = float(np.mean(true_feasible == pred_feasible))
    false_feasible = float(np.mean(pred_feasible & ~true_feasible))
    false_infeasible = float(np.mean(~pred_feasible & true_feasible))

    return {
        "constraint_accuracy_percent": accuracy * 100.0,
        "false_feasible_percent": false_feasible * 100.0,
        "false_infeasible_percent": false_infeasible * 100.0,
    }


def safety_margin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_limit: float = ROBOT_LIMIT_TRUE,
    opt_limit: float = ROBOT_LIMIT_OPT,
) -> dict[str, float]:
    """
    Compute safety-margin false-feasible metrics.

    Unsafe accepted means:
    - predicted safe under optimization margin: y_pred <= 0.495;
    - truly infeasible under physical limit: y_true > 0.500.
    """
    pred_safe_margin = y_pred <= opt_limit
    true_infeasible = y_true > true_limit

    unsafe_accepted = pred_safe_margin & true_infeasible
    n_pred_safe = int(pred_safe_margin.sum())

    return {
        "safety_margin_false_feasible_percent": float(np.mean(unsafe_accepted) * 100.0),
        "safety_margin_unsafe_given_pred_safe_percent": (
            float(unsafe_accepted.sum() / n_pred_safe * 100.0)
            if n_pred_safe > 0
            else np.nan
        ),
        "safety_margin_pred_safe_count": n_pred_safe,
        "safety_margin_unsafe_accepted_count": int(unsafe_accepted.sum()),
    }


def near_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low: float = NEAR_BOUNDARY_LOW,
    high: float = NEAR_BOUNDARY_HIGH,
) -> dict[str, float]:
    """Compute max_abs_xr metrics near the constraint boundary."""
    mask = (y_true >= low) & (y_true <= high)

    if not np.any(mask):
        return {
            "near_boundary_n": 0,
            "near_boundary_rmse_mm": np.nan,
            "near_boundary_mae_mm": np.nan,
            "near_boundary_accuracy_percent": np.nan,
            "near_boundary_false_feasible_percent": np.nan,
            "near_boundary_safety_margin_false_feasible_percent": np.nan,
        }

    y_true_nb = y_true[mask]
    y_pred_nb = y_pred[mask]

    reg = regression_metrics(y_true_nb, y_pred_nb)
    clf = constraint_metrics(y_true_nb, y_pred_nb, limit=ROBOT_LIMIT_TRUE)
    safety = safety_margin_metrics(y_true_nb, y_pred_nb)

    return {
        "near_boundary_n": int(np.sum(mask)),
        "near_boundary_rmse_mm": reg["rmse_mm"],
        "near_boundary_mae_mm": reg["mae_mm"],
        "near_boundary_accuracy_percent": clf["constraint_accuracy_percent"],
        "near_boundary_false_feasible_percent": clf["false_feasible_percent"],
        "near_boundary_safety_margin_false_feasible_percent": (
            safety["safety_margin_false_feasible_percent"]
        ),
    }


def high_outreach_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute peak_y metrics in the high-outreach region."""
    mask = y_true > HIGH_OUTREACH_THRESHOLD

    if not np.any(mask):
        return {
            "high_outreach_n": 0,
            "high_outreach_rmse_mm": np.nan,
            "high_outreach_mae_mm": np.nan,
        }

    reg = regression_metrics(y_true[mask], y_pred[mask])

    return {
        "high_outreach_n": int(np.sum(mask)),
        "high_outreach_rmse_mm": reg["rmse_mm"],
        "high_outreach_mae_mm": reg["mae_mm"],
    }


# =============================================================================
# GP MODELS
# =============================================================================

def create_gp(
    n_dims: int,
    target_name: str,
    gp_n_restarts: int,
) -> GaussianProcessRegressor:
    """Create a GP using the selected final kernel for each target."""
    if target_name == "peak_y":
        nu = 2.5
        alpha = GP_ALPHA_PEAK_Y
    elif target_name == "max_abs_xr":
        nu = 1.5
        alpha = GP_ALPHA_MAX_ABS_XR
    else:
        raise ValueError(f"Unknown target name: {target_name}")

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=nu,
        )
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=gp_n_restarts,
        random_state=RANDOM_STATE,
    )


def train_gp_pair(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    config: LearningCurveConfig,
) -> tuple[dict[str, float], np.ndarray]:
    """Train GP_peak_y and GP_max_abs_xr and evaluate on the fixed test set."""
    start_time = time.time()

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    predictions = []
    warnings_total = 0

    for target_idx, target_name in enumerate(TARGET_COLUMNS):
        y_train = Y_train[:, target_idx]

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        gp = create_gp(
            n_dims=X_train.shape[1],
            target_name=target_name,
            gp_n_restarts=config.gp_n_restarts,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(X_train_scaled, y_train_scaled)

        warnings_total += sum(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught_warnings
        )

        y_pred_scaled = gp.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        predictions.append(y_pred)

    elapsed_time = time.time() - start_time

    Y_pred = np.column_stack(predictions)

    peak_reg = regression_metrics(Y_test[:, 0], Y_pred[:, 0])
    max_reg = regression_metrics(Y_test[:, 1], Y_pred[:, 1])
    clf = constraint_metrics(Y_test[:, 1], Y_pred[:, 1])
    safety = safety_margin_metrics(Y_test[:, 1], Y_pred[:, 1])
    nb = near_boundary_metrics(Y_test[:, 1], Y_pred[:, 1])
    high = high_outreach_metrics(Y_test[:, 0], Y_pred[:, 0])

    metrics = {
        "model": "GP pair",
        "peak_y_rmse_mm": peak_reg["rmse_mm"],
        "peak_y_mae_mm": peak_reg["mae_mm"],
        "peak_y_r2": peak_reg["r2"],
        "max_abs_xr_rmse_mm": max_reg["rmse_mm"],
        "max_abs_xr_mae_mm": max_reg["mae_mm"],
        "max_abs_xr_r2": max_reg["r2"],
        **high,
        **clf,
        **safety,
        **nb,
        "training_time_s": elapsed_time,
        "warnings_total": warnings_total,
        "gp_peak_kernel": GP_KERNEL_PEAK_Y,
        "gp_constraint_kernel": GP_KERNEL_MAX_ABS_XR,
        "gp_n_restarts": config.gp_n_restarts,
    }

    return metrics, Y_pred


# =============================================================================
# NN MODEL
# =============================================================================

class MultiOutputNN(nn.Module):
    """Feed-forward multi-output neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int = len(TARGET_COLUMNS),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def weighted_multioutput_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    constraint_loss_weight: float = NN_CONSTRAINT_LOSS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted MSE on standardized outputs."""
    per_output_loss = torch.mean((prediction - target) ** 2, dim=0)
    total_loss = per_output_loss[0] + constraint_loss_weight * per_output_loss[1]

    return total_loss, per_output_loss


def train_one_nn(
    X_train_total: np.ndarray,
    Y_train_total: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    seed: int,
    config: LearningCurveConfig,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    """Train one NN with a train/validation split and evaluate on the fixed test set."""
    set_seed(seed)
    start_time = time.time()

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_total,
        Y_train_total,
        test_size=NN_VAL_FRACTION_FROM_TRAIN_POOL,
        random_state=seed,
        shuffle=True,
    )

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_val_scaled = scaler_Y.transform(Y_val)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(Y_train_scaled, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(NN_BATCH_SIZE, len(train_dataset)),
        shuffle=True,
    )

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    Y_val_tensor = torch.tensor(Y_val_scaled, dtype=torch.float32, device=device)

    model = MultiOutputNN(
        input_dim=X_train.shape[1],
        hidden_layers=NN_HIDDEN_LAYERS,
        output_dim=len(TARGET_COLUMNS),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_LR,
        weight_decay=NN_WEIGHT_DECAY,
    )

    scheduler = None

    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

    best_val_loss = np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history_rows = []

    for epoch in range(1, config.nn_max_epochs + 1):
        model.train()

        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(batch_x)
            loss, _ = weighted_multioutput_mse(
                prediction,
                batch_y,
                constraint_loss_weight=NN_CONSTRAINT_LOSS_WEIGHT,
            )

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()

        with torch.no_grad():
            val_prediction = model(X_val_tensor)
            val_loss_tensor, val_mse_per_output_tensor = weighted_multioutput_mse(
                val_prediction,
                Y_val_tensor,
                constraint_loss_weight=NN_CONSTRAINT_LOSS_WEIGHT,
            )

            val_loss = float(val_loss_tensor.detach().cpu())
            val_mse_per_output = val_mse_per_output_tensor.detach().cpu().numpy()

        train_loss = float(np.mean(train_losses))
        current_lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "epoch": epoch,
                "seed": seed,
                "learning_rate": current_lr,
                "constraint_loss_weight": NN_CONSTRAINT_LOSS_WEIGHT,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_peak_y_loss": float(val_mse_per_output[0]),
                "val_max_abs_xr_loss": float(val_mse_per_output[1]),
            }
        )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if patience_counter >= config.nn_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        Y_pred_scaled = model(X_test_tensor).detach().cpu().numpy()

    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    elapsed_time = time.time() - start_time

    peak_reg = regression_metrics(Y_test[:, 0], Y_pred[:, 0])
    max_reg = regression_metrics(Y_test[:, 1], Y_pred[:, 1])
    clf = constraint_metrics(Y_test[:, 1], Y_pred[:, 1])
    safety = safety_margin_metrics(Y_test[:, 1], Y_pred[:, 1])
    nb = near_boundary_metrics(Y_test[:, 1], Y_pred[:, 1])
    high = high_outreach_metrics(Y_test[:, 0], Y_pred[:, 0])

    metrics = {
        "model": "NN multi-output",
        "peak_y_rmse_mm": peak_reg["rmse_mm"],
        "peak_y_mae_mm": peak_reg["mae_mm"],
        "peak_y_r2": peak_reg["r2"],
        "max_abs_xr_rmse_mm": max_reg["rmse_mm"],
        "max_abs_xr_mae_mm": max_reg["mae_mm"],
        "max_abs_xr_r2": max_reg["r2"],
        **high,
        **clf,
        **safety,
        **nb,
        "training_time_s": elapsed_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "seed": seed,
        "nn_architecture": "-".join(map(str, NN_HIDDEN_LAYERS)),
        "nn_constraint_loss_weight": NN_CONSTRAINT_LOSS_WEIGHT,
    }

    return metrics, Y_pred, pd.DataFrame(history_rows)


def train_nn_multi_seed(
    X_train_total: np.ndarray,
    Y_train_total: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    config: LearningCurveConfig,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    """Train the NN over multiple seeds and aggregate predictions by ensemble mean."""
    seed_metrics = []
    seed_predictions = []
    histories = []

    for seed in config.nn_seeds:
        metrics, prediction, history = train_one_nn(
            X_train_total=X_train_total,
            Y_train_total=Y_train_total,
            X_test=X_test,
            Y_test=Y_test,
            seed=seed,
            config=config,
            device=device,
        )

        seed_metrics.append(metrics)
        seed_predictions.append(prediction)
        histories.append(history)

    metrics_df = pd.DataFrame(seed_metrics)

    Y_pred_mean = np.mean(seed_predictions, axis=0)

    peak_reg = regression_metrics(Y_test[:, 0], Y_pred_mean[:, 0])
    max_reg = regression_metrics(Y_test[:, 1], Y_pred_mean[:, 1])
    clf = constraint_metrics(Y_test[:, 1], Y_pred_mean[:, 1])
    safety = safety_margin_metrics(Y_test[:, 1], Y_pred_mean[:, 1])
    nb = near_boundary_metrics(Y_test[:, 1], Y_pred_mean[:, 1])
    high = high_outreach_metrics(Y_test[:, 0], Y_pred_mean[:, 0])

    metrics = {
        "model": "NN multi-output",
        "peak_y_rmse_mm": peak_reg["rmse_mm"],
        "peak_y_mae_mm": peak_reg["mae_mm"],
        "peak_y_r2": peak_reg["r2"],
        "max_abs_xr_rmse_mm": max_reg["rmse_mm"],
        "max_abs_xr_mae_mm": max_reg["mae_mm"],
        "max_abs_xr_r2": max_reg["r2"],
        **high,
        **clf,
        **safety,
        **nb,
        "training_time_s": float(metrics_df["training_time_s"].sum()),
        "best_epoch_mean": float(metrics_df["best_epoch"].mean()),
        "best_val_loss_mean": float(metrics_df["best_val_loss"].mean()),
        "n_seeds": len(config.nn_seeds),
        "nn_architecture": "-".join(map(str, NN_HIDDEN_LAYERS)),
        "nn_constraint_loss_weight": NN_CONSTRAINT_LOSS_WEIGHT,
    }

    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()

    return metrics, Y_pred_mean, history_df


# =============================================================================
# LEARNING CURVE
# =============================================================================

def run_learning_curve(
    X: np.ndarray,
    Y: np.ndarray,
    config: LearningCurveConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run GP-vs-NN learning curve using a fixed test set."""
    X_train_pool, X_test, Y_train_pool, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    rng = np.random.default_rng(RANDOM_STATE)
    train_pool_indices = rng.permutation(len(X_train_pool))

    valid_train_sizes = [
        n_train
        for n_train in config.train_sizes
        if n_train <= len(X_train_pool)
    ]

    if len(valid_train_sizes) == 0:
        raise ValueError("No valid training sizes. Check train_sizes and dataset size.")

    print("\n" + "=" * 70)
    print("GP VS NN LEARNING CURVE")
    print("=" * 70)
    print(f"Device:                    {device}")
    print(f"Fixed test samples:        {len(X_test)}")
    print(f"Train pool samples:        {len(X_train_pool)}")
    print(f"Training sizes:            {valid_train_sizes}")
    print(f"GP peak_y kernel:          {GP_KERNEL_PEAK_Y}")
    print(f"GP max_abs_xr kernel:      {GP_KERNEL_MAX_ABS_XR}")
    print(f"GP alpha peak_y:           {GP_ALPHA_PEAK_Y:.0e}")
    print(f"GP alpha max_abs_xr:       {GP_ALPHA_MAX_ABS_XR:.0e}")
    print(f"GP restarts:               {config.gp_n_restarts}")
    print(f"NN hidden layers:          {NN_HIDDEN_LAYERS}")
    print(f"NN lr / weight decay:      {NN_LR:.0e} / {NN_WEIGHT_DECAY:.0e}")
    print(f"NN constraint lambda:      {NN_CONSTRAINT_LOSS_WEIGHT:g}")
    print(f"NN scheduler:              {config.use_lr_scheduler}")
    print(f"NN seeds:                  {config.nn_seeds}")
    print("=" * 70)

    all_rows = []
    all_histories = []

    global_start = time.time()

    for n_train in valid_train_sizes:
        print("\n" + "#" * 70)
        print(f"TRAINING SIZE: {n_train}")
        print("#" * 70)

        selected = train_pool_indices[:n_train]
        X_train_n = X_train_pool[selected]
        Y_train_n = Y_train_pool[selected]

        print("\nTraining GP pair...")

        gp_metrics, _ = train_gp_pair(
            X_train=X_train_n,
            Y_train=Y_train_n,
            X_test=X_test,
            Y_test=Y_test,
            config=config,
        )

        gp_metrics["n_train"] = n_train
        all_rows.append(gp_metrics)

        print(
            f"  GP | peak_y RMSE {gp_metrics['peak_y_rmse_mm']:.2f} mm | "
            f"max_abs_xr RMSE {gp_metrics['max_abs_xr_rmse_mm']:.2f} mm | "
            f"false feasible {gp_metrics['false_feasible_percent']:.2f}% | "
            f"safety-margin false feasible "
            f"{gp_metrics['safety_margin_false_feasible_percent']:.2f}% | "
            f"time {gp_metrics['training_time_s']:.1f} s"
        )

        print("\nTraining NN multi-output...")

        nn_metrics, _, history_df = train_nn_multi_seed(
            X_train_total=X_train_n,
            Y_train_total=Y_train_n,
            X_test=X_test,
            Y_test=Y_test,
            config=config,
            device=device,
        )

        nn_metrics["n_train"] = n_train
        all_rows.append(nn_metrics)

        if len(history_df) > 0:
            history_df["n_train"] = n_train
            all_histories.append(history_df)

        print(
            f"  NN | peak_y RMSE {nn_metrics['peak_y_rmse_mm']:.2f} mm | "
            f"max_abs_xr RMSE {nn_metrics['max_abs_xr_rmse_mm']:.2f} mm | "
            f"false feasible {nn_metrics['false_feasible_percent']:.2f}% | "
            f"safety-margin false feasible "
            f"{nn_metrics['safety_margin_false_feasible_percent']:.2f}% | "
            f"time {nn_metrics['training_time_s']:.1f} s"
        )

        elapsed = time.time() - global_start
        print(f"\nElapsed total: {elapsed / 60.0:.1f} min")

        checkpoint_df = pd.DataFrame(all_rows)
        checkpoint_df.to_csv(
            RESULTS_DIR / "gp_vs_nn_learning_curve_checkpoint.csv",
            index=False,
        )

    results_df = pd.DataFrame(all_rows)

    if all_histories:
        histories_df = pd.concat(all_histories, ignore_index=True)
        histories_df.to_csv(
            RESULTS_DIR / "gp_vs_nn_learning_curve_nn_histories.csv",
            index=False,
        )

    test_df = pd.DataFrame(
        Y_test,
        columns=["true_peak_y", "true_max_abs_xr"],
    )
    test_df.to_csv(
        RESULTS_DIR / "gp_vs_nn_learning_curve_fixed_test_targets.csv",
        index=False,
    )

    return results_df


# =============================================================================
# REPORT TABLES
# =============================================================================

def make_report_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact report-friendly table."""
    cols = [
        "n_train",
        "model",
        "peak_y_rmse_mm",
        "peak_y_r2",
        "high_outreach_rmse_mm",
        "max_abs_xr_rmse_mm",
        "max_abs_xr_r2",
        "constraint_accuracy_percent",
        "false_feasible_percent",
        "safety_margin_false_feasible_percent",
        "near_boundary_rmse_mm",
        "near_boundary_false_feasible_percent",
        "near_boundary_safety_margin_false_feasible_percent",
        "training_time_s",
    ]

    report = results_df[cols].copy()

    rename = {
        "n_train": "Training samples",
        "model": "Model",
        "peak_y_rmse_mm": "peak_y RMSE [mm]",
        "peak_y_r2": "peak_y R2",
        "high_outreach_rmse_mm": "High-outreach RMSE [mm]",
        "max_abs_xr_rmse_mm": "max_abs_xr RMSE [mm]",
        "max_abs_xr_r2": "max_abs_xr R2",
        "constraint_accuracy_percent": "Constraint accuracy [%]",
        "false_feasible_percent": "False feasible [%]",
        "safety_margin_false_feasible_percent": "Safety-margin false feasible [%]",
        "near_boundary_rmse_mm": "Near-boundary RMSE [mm]",
        "near_boundary_false_feasible_percent": "Near-boundary false feasible [%]",
        "near_boundary_safety_margin_false_feasible_percent": "NB safety-margin false feasible [%]",
        "training_time_s": "Training time [s]",
    }

    report = report.rename(columns=rename)

    numeric_cols = report.select_dtypes(include=[np.number]).columns
    report[numeric_cols] = report[numeric_cols].round(3)

    return report


def save_table_markdown(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as markdown without requiring tabulate."""
    columns = list(df.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_line(
    results_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Generic line plot by training size for GP and NN."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name in ["GP pair", "NN multi-output"]:
        sub = results_df[results_df["model"] == model_name].sort_values("n_train")

        ax.plot(
            sub["n_train"],
            sub[metric_col],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_xlabel("Training samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def generate_plots(results_df: pd.DataFrame) -> None:
    """Generate all learning-curve figures."""
    print("\nGenerating learning-curve plots...")

    plot_metric_line(
        results_df,
        metric_col="peak_y_rmse_mm",
        ylabel="peak_y RMSE [mm]",
        title="Learning Curve: peak_y Prediction Error",
        filename="gp_vs_nn_peak_y_rmse.png",
    )

    plot_metric_line(
        results_df,
        metric_col="max_abs_xr_rmse_mm",
        ylabel="max_abs_xr RMSE [mm]",
        title="Learning Curve: Constraint Quantity Prediction Error",
        filename="gp_vs_nn_max_abs_xr_rmse.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, metric_col, title in [
        (axes[0], "peak_y_r2", "peak_y R²"),
        (axes[1], "max_abs_xr_r2", "max_abs_xr R²"),
    ]:
        for model_name in ["GP pair", "NN multi-output"]:
            sub = results_df[results_df["model"] == model_name].sort_values("n_train")

            ax.plot(
                sub["n_train"],
                sub[metric_col],
                marker="o",
                linewidth=2,
                label=model_name,
            )

        ax.set_xlabel("Training samples")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Learning Curve: Explained Variance")

    path = FIGURES_DIR / "gp_vs_nn_r2.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    plot_metric_line(
        results_df,
        metric_col="false_feasible_percent",
        ylabel="False feasible rate [%]",
        title="Learning Curve: Constraint Safety Error",
        filename="gp_vs_nn_constraint_safety.png",
    )

    plot_metric_line(
        results_df,
        metric_col="safety_margin_false_feasible_percent",
        ylabel="Safety-margin false feasible rate [%]",
        title="Learning Curve: Safety-Margin Constraint Error",
        filename="gp_vs_nn_safety_margin_false_feasible.png",
    )

    plot_metric_line(
        results_df,
        metric_col="training_time_s",
        ylabel="Training time [s]",
        title="Training Time vs Dataset Size",
        filename="gp_vs_nn_training_time.png",
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    plot_specs = [
        (axes[0, 0], "peak_y_rmse_mm", "peak_y RMSE [mm]", "Outreach prediction"),
        (axes[0, 1], "max_abs_xr_rmse_mm", "max_abs_xr RMSE [mm]", "Constraint prediction"),
        (
            axes[1, 0],
            "safety_margin_false_feasible_percent",
            "Safety-margin false feasible [%]",
            "Constraint safety",
        ),
        (axes[1, 1], "training_time_s", "Training time [s]", "Training cost"),
    ]

    for ax, metric_col, ylabel, title in plot_specs:
        for model_name in ["GP pair", "NN multi-output"]:
            sub = results_df[results_df["model"] == model_name].sort_values("n_train")

            ax.plot(
                sub["n_train"],
                sub[metric_col],
                marker="o",
                linewidth=2,
                label=model_name,
            )

        ax.set_xlabel("Training samples")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("GP vs NN Learning Curve Summary", fontsize=16, fontweight="bold")

    path = FIGURES_DIR / "gp_vs_nn_learning_curve_summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# =============================================================================
# INTERPRETATION
# =============================================================================

def save_interpretation(
    results_df: pd.DataFrame,
    config: LearningCurveConfig,
) -> Path:
    """Save short interpretation text."""
    lines = [
        "GP VS NN LEARNING CURVE INTERPRETATION",
        "=" * 70,
        "",
        f"Dataset: {config.dataset_path}",
        f"Fixed test fraction: {TEST_SIZE}",
        f"Training sizes: {sorted(results_df['n_train'].unique().tolist())}",
        "",
        "Model configurations:",
        f"  GP_peak_y: Matérn 5/2, alpha={GP_ALPHA_PEAK_Y:.0e}",
        f"  GP_max_abs_xr: Matérn 3/2, alpha={GP_ALPHA_MAX_ABS_XR:.0e}",
        f"  GP optimizer restarts in learning curve: {config.gp_n_restarts}",
        (
            f"  NN: hidden layers={NN_HIDDEN_LAYERS}, "
            f"lambda={NN_CONSTRAINT_LOSS_WEIGHT:g}, seeds={config.nn_seeds}"
        ),
        "",
    ]

    largest_n = int(results_df["n_train"].max())

    final_gp = results_df[
        (results_df["n_train"] == largest_n)
        & (results_df["model"] == "GP pair")
    ].iloc[0]

    final_nn = results_df[
        (results_df["n_train"] == largest_n)
        & (results_df["model"] == "NN multi-output")
    ].iloc[0]

    lines.extend(
        [
            f"Largest training size: {largest_n}",
            "",
            "At the largest training size:",
            (
                f"  GP peak_y RMSE: {final_gp['peak_y_rmse_mm']:.2f} mm | "
                f"NN peak_y RMSE: {final_nn['peak_y_rmse_mm']:.2f} mm"
            ),
            (
                f"  GP max_abs_xr RMSE: {final_gp['max_abs_xr_rmse_mm']:.2f} mm | "
                f"NN max_abs_xr RMSE: {final_nn['max_abs_xr_rmse_mm']:.2f} mm"
            ),
            (
                f"  GP false feasible: {final_gp['false_feasible_percent']:.2f}% | "
                f"NN false feasible: {final_nn['false_feasible_percent']:.2f}%"
            ),
            (
                f"  GP safety-margin false feasible: "
                f"{final_gp['safety_margin_false_feasible_percent']:.2f}% | "
                f"NN safety-margin false feasible: "
                f"{final_nn['safety_margin_false_feasible_percent']:.2f}%"
            ),
            "",
            "Interpretation guide:",
            (
                "  The main fair comparison is always performed at the same training "
                "size and on the same fixed test set."
            ),
            (
                "  If GP error saturates while NN error keeps decreasing, the GP is "
                "more sample-efficient for the current dataset, while the NN may "
                "benefit more from additional simulations."
            ),
            (
                "  Since GP_N_RESTARTS may be reduced here for computational "
                "feasibility, the learning curve should be interpreted as a "
                "controlled sample-efficiency analysis. The final model comparison "
                "should still be based on the fully trained final surrogates."
            ),
        ]
    )

    path = RESULTS_DIR / "gp_vs_nn_learning_curve_interpretation.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    return path


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GP-vs-NN learning-curve comparison."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset path. Default: {DEFAULT_DATASET_PATH}.",
    )

    parser.add_argument(
        "--train_sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TRAIN_SIZES),
        help="Comma-separated training sizes.",
    )

    parser.add_argument(
        "--gp_restarts",
        type=int,
        default=DEFAULT_GP_N_RESTARTS,
        help=f"GP optimizer restarts. Default: {DEFAULT_GP_N_RESTARTS}.",
    )

    parser.add_argument(
        "--quick_debug",
        action="store_true",
        help="Run a reduced configuration for quick testing.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LearningCurveConfig:
    """Build configuration from command-line arguments."""
    if args.quick_debug:
        return LearningCurveConfig(
            dataset_path=args.dataset,
            train_sizes=DEBUG_TRAIN_SIZES,
            gp_n_restarts=0,
            nn_max_epochs=DEBUG_NN_MAX_EPOCHS,
            nn_patience=DEBUG_NN_PATIENCE,
            nn_seeds=DEBUG_NN_SEEDS,
            use_lr_scheduler=False,
            skip_plots=args.skip_plots,
            quick_debug=True,
        )

    return LearningCurveConfig(
        dataset_path=args.dataset,
        train_sizes=parse_train_sizes(args.train_sizes),
        gp_n_restarts=args.gp_restarts,
        nn_max_epochs=NN_MAX_EPOCHS,
        nn_patience=NN_PATIENCE,
        nn_seeds=NN_SEEDS,
        use_lr_scheduler=USE_LR_SCHEDULER,
        skip_plots=args.skip_plots,
        quick_debug=False,
    )


def main() -> None:
    """Run the complete learning-curve analysis."""
    args = parse_args()
    config = build_config(args)

    ensure_dirs()

    print("\n" + "=" * 70)
    print("LEARNING CURVE: GP PAIR VS NN MULTI-OUTPUT")
    print("=" * 70)

    if config.quick_debug:
        print("Running in quick-debug mode.")

    device = get_device()

    X, Y, df = load_dataset(config.dataset_path)
    print_dataset_summary(df, config.dataset_path)

    results_df = run_learning_curve(
        X=X,
        Y=Y,
        config=config,
        device=device,
    )

    results_path = RESULTS_DIR / "gp_vs_nn_learning_curve.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    report_df = make_report_table(results_df)

    report_path = RESULTS_DIR / "gp_vs_nn_learning_curve_report_table.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Report table saved: {report_path}")

    md_path = RESULTS_DIR / "gp_vs_nn_learning_curve_report_table.md"
    save_table_markdown(report_df, md_path)
    print(f"Markdown report table saved: {md_path}")

    print("\nLEARNING CURVE REPORT TABLE")
    print(report_df.to_string(index=False))

    if not config.skip_plots:
        generate_plots(results_df)

    interpretation_path = save_interpretation(results_df, config)
    print(f"Interpretation saved: {interpretation_path}")

    print("\n" + "=" * 70)
    print("LEARNING CURVE COMPLETED")
    print("=" * 70)
    print("Generated files:")
    print(f"  - {results_path}")
    print(f"  - {report_path}")
    print(f"  - {md_path}")
    print(f"  - {interpretation_path}")

    if not config.skip_plots:
        print(f"  - {FIGURES_DIR / 'gp_vs_nn_*.png'}")

    print("\nNext step:")
    print("  Run the final surrogate model comparison.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()