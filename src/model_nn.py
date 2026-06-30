"""
model_nn.py
==============

Constraint-aware multi-output Neural Network surrogate model.

Supervised-learning task
------------------------
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] -> [peak_y, max_abs_xr]

Main features
-------------
1. Weighted multi-output loss:
       L = MSE(peak_y) + lambda * MSE(max_abs_xr)
2. Validation-based early stopping.
3. Optional learning-rate scheduler.
4. Hyperparameter tuning over architecture, learning rate and weight decay.
5. Constraint-loss lambda sweep.
6. Multi-seed ensemble prediction.
7. Constraint-specific safety metrics.

The primary model used for inverse optimization is the ensemble mean.

Date: 2026
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEFAULT_DATASET_PATH = DATA_DIR / "dataset_augmented.csv"

NN_RESULTS_DIR = RESULTS_DIR / "nn_v2"
NN_FIGURES_DIR = FIGURES_DIR / "nn_v2"


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

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN_POOL = 0.1875
RANDOM_STATE_SPLIT = 42

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495

NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520

HIGH_OUTREACH_THRESHOLD = 0.600
TOP_REACHABLE_THRESHOLD = 0.650

DEFAULT_CONSTRAINT_LOSS_WEIGHT = 2.0
DEFAULT_LAMBDA_SWEEP_VALUES = [1.0, 2.0, 3.0, 5.0]

USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 40
SCHEDULER_MIN_LR = 1e-6

INVERSE_TARGETS = [0.55, 0.60, 0.65, 0.70, 0.75]
TARGET_TOLERANCE = 0.025

DEFAULT_ARCHITECTURES = [[64, 64], [32, 64, 32], [128, 64]]
DEFAULT_LEARNING_RATES = [1e-3, 5e-4]
DEFAULT_WEIGHT_DECAYS = [1e-5, 1e-4]
DEFAULT_SEEDS = [42, 123, 2026]

DEFAULT_MAX_EPOCHS = 1500
DEFAULT_PATIENCE = 120
DEFAULT_BATCH_SIZE = 64

DEBUG_ARCHITECTURES = [[64, 64]]
DEBUG_LEARNING_RATES = [1e-3]
DEBUG_WEIGHT_DECAYS = [1e-5]
DEBUG_SEEDS = [42]
DEBUG_LAMBDA_SWEEP_VALUES = [1.0, 2.0]
DEBUG_MAX_EPOCHS = 80
DEBUG_PATIENCE = 15


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class NNConfig:
    """Configuration for NN training and tuning."""

    dataset_path: Path
    architectures: list[list[int]]
    learning_rates: list[float]
    weight_decays: list[float]
    seeds: list[int]
    lambda_sweep_values: list[float]
    default_constraint_loss_weight: float
    force_final_lambda: float | None
    max_epochs: int
    patience: int
    batch_size: int
    use_lr_scheduler: bool
    no_plots: bool
    quick_debug: bool


@dataclass
class TrainingResult:
    """Container for one trained NN model."""

    model_state: dict[str, torch.Tensor]
    history: pd.DataFrame
    best_val_loss: float
    best_val_peak_loss: float
    best_val_max_abs_xr_loss: float
    best_epoch: int
    training_time_s: float


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    NN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
    """Set seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def architecture_to_string(hidden_layers: list[int]) -> str:
    """Convert architecture list to a compact string."""
    return "-".join(str(layer) for layer in hidden_layers)


# =============================================================================
# DATA
# =============================================================================

def load_nn_dataset(
    filepath: Path = DEFAULT_DATASET_PATH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load dataset for multi-output NN training."""
    filepath = Path(filepath)
    require_file(filepath)

    df = pd.read_csv(filepath, comment="#")
    required_cols = INPUT_COLUMNS + TARGET_COLUMNS
    require_columns(df, required_cols, "Dataset")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with NaNs in required columns.")

    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    Y = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    print("\n" + "=" * 70)
    print("NN V2 DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset path:        {filepath}")
    print(f"Samples:             {len(df)}")
    print(f"Input dimensions:    {X.shape[1]}")
    print(f"Targets:             {TARGET_COLUMNS}")
    print(f"peak_y mean/std:     {df['peak_y'].mean():.6f} / {df['peak_y'].std():.6f} m")
    print(f"max_abs_xr mean/std: {df['max_abs_xr'].mean():.6f} / {df['max_abs_xr'].std():.6f} m")

    if "dataset_type" in df.columns:
        print("Dataset types:")
        for name, count in df["dataset_type"].value_counts().items():
            print(f"  {str(name):22s}: {count}")

    print("=" * 70)

    return X, Y, df


def prepare_splits(X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    """
    Create train, validation and test splits and standardize inputs/outputs.

    Overall split:
    - 65% train
    - 15% validation
    - 20% test
    """
    X_train_pool, X_test, Y_train_pool, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_SPLIT,
        shuffle=True,
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_pool,
        Y_train_pool,
        test_size=VAL_SIZE_FROM_TRAIN_POOL,
        random_state=RANDOM_STATE_SPLIT,
        shuffle=True,
    )

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_val_scaled = scaler_Y.transform(Y_val)
    Y_test_scaled = scaler_Y.transform(Y_test)

    print("\n" + "=" * 70)
    print("SPLIT AND SCALING")
    print("=" * 70)
    print(f"Train samples:       {len(X_train)}")
    print(f"Validation samples:  {len(X_val)}")
    print(f"Test samples:        {len(X_test)}")
    print(f"Validation fraction: {len(X_val) / len(X):.3f}")
    print("=" * 70)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "Y_train_scaled": Y_train_scaled,
        "Y_val_scaled": Y_val_scaled,
        "Y_test_scaled": Y_test_scaled,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
    }


def make_loader(
    X_scaled: np.ndarray,
    Y_scaled: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(Y_scaled, dtype=torch.float32),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# MODEL AND TRAINING
# =============================================================================

class MultiOutputMLP(nn.Module):
    """Fully connected multi-output MLP surrogate."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int = len(TARGET_COLUMNS),
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


def weighted_scaled_mse(
    per_output_loss: torch.Tensor,
    constraint_loss_weight: float,
) -> torch.Tensor:
    """Compute weighted multi-output loss on standardized outputs."""
    return per_output_loss[0] + float(constraint_loss_weight) * per_output_loss[1]


def evaluate_scaled_loss(
    model: nn.Module,
    X_scaled: np.ndarray,
    Y_scaled: np.ndarray,
    device: torch.device,
    constraint_loss_weight: float,
) -> tuple[float, float, float]:
    """Evaluate weighted validation loss on standardized outputs."""
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        prediction = model(X_tensor)
        per_output_loss = torch.mean((prediction - Y_tensor) ** 2, dim=0)
        total_loss = weighted_scaled_mse(per_output_loss, constraint_loss_weight)

    return (
        float(total_loss.cpu()),
        float(per_output_loss[0].cpu()),
        float(per_output_loss[1].cpu()),
    )


def train_once(
    splits: dict[str, Any],
    hidden_layers: list[int],
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    constraint_loss_weight: float,
    config: NNConfig,
    verbose: bool = False,
) -> TrainingResult:
    """Train one NN model for one architecture and one random seed."""
    set_seed(seed)

    X_train_scaled = splits["X_train_scaled"]
    Y_train_scaled = splits["Y_train_scaled"]
    X_val_scaled = splits["X_val_scaled"]
    Y_val_scaled = splits["Y_val_scaled"]

    model = MultiOutputMLP(
        input_dim=X_train_scaled.shape[1],
        hidden_layers=hidden_layers,
        output_dim=len(TARGET_COLUMNS),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
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

    loader = make_loader(
        X_scaled=X_train_scaled,
        Y_scaled=Y_train_scaled,
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_val_loss = np.inf
    best_peak_loss = np.inf
    best_xr_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    rows = []

    start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()

        train_losses = []
        peak_losses = []
        xr_losses = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(batch_x)
            per_output_loss = torch.mean((prediction - batch_y) ** 2, dim=0)
            loss = weighted_scaled_mse(per_output_loss, constraint_loss_weight)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            peak_losses.append(float(per_output_loss[0].detach().cpu()))
            xr_losses.append(float(per_output_loss[1].detach().cpu()))

        val_loss, val_peak_loss, val_xr_loss = evaluate_scaled_loss(
            model=model,
            X_scaled=X_val_scaled,
            Y_scaled=Y_val_scaled,
            device=device,
            constraint_loss_weight=constraint_loss_weight,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])

        rows.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "constraint_loss_weight": constraint_loss_weight,
                "train_loss": float(np.mean(train_losses)),
                "train_peak_y_loss": float(np.mean(peak_losses)),
                "train_max_abs_xr_loss": float(np.mean(xr_losses)),
                "val_loss": val_loss,
                "val_peak_y_loss": val_peak_loss,
                "val_max_abs_xr_loss": val_xr_loss,
            }
        )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_peak_loss = val_peak_loss
            best_xr_loss = val_xr_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if verbose and (epoch == 1 or epoch % 100 == 0):
            print(
                f"    epoch {epoch:4d} | val {val_loss:.5f} | "
                f"peak {val_peak_loss:.5f} | max_abs_xr {val_xr_loss:.5f} | "
                f"lr {current_lr:.2e}"
            )

        if stale_epochs >= config.patience:
            break

    return TrainingResult(
        model_state=best_state,
        history=pd.DataFrame(rows),
        best_val_loss=float(best_val_loss),
        best_val_peak_loss=float(best_peak_loss),
        best_val_max_abs_xr_loss=float(best_xr_loss),
        best_epoch=int(best_epoch),
        training_time_s=float(time.time() - start_time),
    )


def build_model_from_state(
    hidden_layers: list[int],
    model_state: dict[str, torch.Tensor],
    device: torch.device,
) -> MultiOutputMLP:
    """Rebuild a model and load a saved state dictionary."""
    model = MultiOutputMLP(
        input_dim=len(INPUT_COLUMNS),
        hidden_layers=hidden_layers,
        output_dim=len(TARGET_COLUMNS),
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    return model


# =============================================================================
# PREDICTION AND METRICS
# =============================================================================

def predict_physical_units(
    model: nn.Module | list[nn.Module],
    X_scaled: np.ndarray,
    scaler_Y: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    """Predict outputs in physical units."""
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        if isinstance(model, list):
            prediction_scaled = torch.stack(
                [single_model(X_tensor) for single_model in model],
                dim=0,
            ).mean(dim=0)
        else:
            prediction_scaled = model(X_tensor)

    return scaler_Y.inverse_transform(prediction_scaled.cpu().numpy())


def ensemble_prediction_stats_physical(
    models: list[nn.Module],
    X_scaled: np.ndarray,
    scaler_Y: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and standard deviation in physical units."""
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        prediction_scaled_all = torch.stack(
            [model(X_tensor).cpu() for model in models],
            dim=0,
        ).numpy()

    prediction_physical_all = np.stack(
        [
            scaler_Y.inverse_transform(prediction_scaled_all[i])
            for i in range(len(models))
        ],
        axis=0,
    )

    return prediction_physical_all.mean(axis=0), prediction_physical_all.std(axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics in meters and millimeters."""
    error = y_pred - y_true
    abs_error = np.abs(error)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "rmse_m": rmse,
        "rmse_mm": rmse * 1000.0,
        "mae_m": mae,
        "mae_mm": mae * 1000.0,
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error_m": float(np.mean(error)),
        "mean_error_mm": float(np.mean(error) * 1000.0),
        "max_abs_error_m": float(np.max(abs_error)),
        "max_abs_error_mm": float(np.max(abs_error) * 1000.0),
    }


def evaluate_constraint_classification(
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
    limit: float,
    label: str,
) -> dict[str, float]:
    """Evaluate feasible/infeasible classification for max_abs_xr <= limit."""
    true_feasible = y_true_xr <= limit
    pred_feasible = y_pred_xr <= limit

    cm = confusion_matrix(true_feasible, pred_feasible, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    return {
        f"{label}_limit_m": float(limit),
        f"{label}_classification_accuracy": float(np.mean(true_feasible == pred_feasible)),
        f"{label}_false_feasible_rate": float(np.mean(pred_feasible & ~true_feasible)),
        f"{label}_false_infeasible_rate": float(np.mean(~pred_feasible & true_feasible)),
        f"{label}_true_infeasible_pred_infeasible": int(tn),
        f"{label}_true_infeasible_pred_feasible": int(fp),
        f"{label}_true_feasible_pred_infeasible": int(fn),
        f"{label}_true_feasible_pred_feasible": int(tp),
    }


def evaluate_safety_margin_false_feasible(
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate unsafe accepted predictions.

    Unsafe accepted means:
    - predicted safe with optimization margin: pred_max_abs_xr <= 0.495;
    - actually infeasible under true physical limit: true_max_abs_xr > 0.500.
    """
    pred_safe_margin = y_pred_xr <= ROBOT_LIMIT_OPT
    true_infeasible = y_true_xr > ROBOT_LIMIT_TRUE
    unsafe_accepted = pred_safe_margin & true_infeasible

    n_pred_safe = int(pred_safe_margin.sum())

    return {
        "safety_margin_pred_safe_count": n_pred_safe,
        "safety_margin_unsafe_accepted_count": int(unsafe_accepted.sum()),
        "safety_margin_false_feasible_rate": float(np.mean(unsafe_accepted)),
        "safety_margin_unsafe_given_pred_safe_rate": (
            float(unsafe_accepted.sum() / n_pred_safe) if n_pred_safe > 0 else np.nan
        ),
    }


def evaluate_near_boundary(
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
) -> dict[str, float]:
    """Evaluate max_abs_xr prediction quality near the constraint boundary."""
    mask = (y_true_xr >= NEAR_BOUNDARY_LOW) & (y_true_xr <= NEAR_BOUNDARY_HIGH)

    if not np.any(mask):
        return {
            "near_boundary_low_m": NEAR_BOUNDARY_LOW,
            "near_boundary_high_m": NEAR_BOUNDARY_HIGH,
            "near_boundary_n_samples": 0,
            "near_boundary_rmse_mm": np.nan,
            "near_boundary_mae_mm": np.nan,
            "near_boundary_false_feasible_rate_true_limit": np.nan,
            "near_boundary_safety_margin_false_feasible_rate": np.nan,
            "near_boundary_safety_margin_unsafe_given_pred_safe_rate": np.nan,
        }

    reg = regression_metrics(y_true_xr[mask], y_pred_xr[mask])
    clf = evaluate_constraint_classification(
        y_true_xr[mask],
        y_pred_xr[mask],
        ROBOT_LIMIT_TRUE,
        "near_boundary_true_limit",
    )
    safety = evaluate_safety_margin_false_feasible(y_true_xr[mask], y_pred_xr[mask])

    return {
        "near_boundary_low_m": NEAR_BOUNDARY_LOW,
        "near_boundary_high_m": NEAR_BOUNDARY_HIGH,
        "near_boundary_n_samples": int(mask.sum()),
        "near_boundary_rmse_mm": reg["rmse_mm"],
        "near_boundary_mae_mm": reg["mae_mm"],
        "near_boundary_classification_accuracy_true_limit": (
            clf["near_boundary_true_limit_classification_accuracy"]
        ),
        "near_boundary_false_feasible_rate_true_limit": (
            clf["near_boundary_true_limit_false_feasible_rate"]
        ),
        "near_boundary_safety_margin_false_feasible_rate": (
            safety["safety_margin_false_feasible_rate"]
        ),
        "near_boundary_safety_margin_unsafe_given_pred_safe_rate": (
            safety["safety_margin_unsafe_given_pred_safe_rate"]
        ),
    }


def evaluate_inverse_targets(
    y_true_peak: np.ndarray,
    y_pred_peak: np.ndarray,
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
    targets: list[float] = INVERSE_TARGETS,
    tolerance: float = TARGET_TOLERANCE,
) -> pd.DataFrame:
    """Evaluate test accuracy in target-outreach bands."""
    rows = []

    for target in targets:
        mask = np.abs(y_true_peak - target) <= tolerance

        row: dict[str, float | int] = {
            "target_peak_y_m": float(target),
            "tolerance_m": float(tolerance),
            "n_samples": int(mask.sum()),
        }

        if np.any(mask):
            peak_reg = regression_metrics(y_true_peak[mask], y_pred_peak[mask])
            xr_reg = regression_metrics(y_true_xr[mask], y_pred_xr[mask])
            clf = evaluate_constraint_classification(
                y_true_xr[mask],
                y_pred_xr[mask],
                ROBOT_LIMIT_TRUE,
                "true_limit",
            )
            safety = evaluate_safety_margin_false_feasible(
                y_true_xr[mask],
                y_pred_xr[mask],
            )

            row.update(
                {
                    "peak_y_rmse_mm": peak_reg["rmse_mm"],
                    "peak_y_mae_mm": peak_reg["mae_mm"],
                    "peak_y_mean_error_mm": peak_reg["mean_error_mm"],
                    "max_abs_xr_rmse_mm": xr_reg["rmse_mm"],
                    "max_abs_xr_mae_mm": xr_reg["mae_mm"],
                    "max_abs_xr_mean_error_mm": xr_reg["mean_error_mm"],
                    "true_limit_accuracy": clf["true_limit_classification_accuracy"],
                    "true_limit_false_feasible_rate": clf["true_limit_false_feasible_rate"],
                    "safety_margin_false_feasible_rate": safety["safety_margin_false_feasible_rate"],
                    "safety_margin_unsafe_given_pred_safe_rate": (
                        safety["safety_margin_unsafe_given_pred_safe_rate"]
                    ),
                }
            )
        else:
            row.update(
                {
                    "peak_y_rmse_mm": np.nan,
                    "peak_y_mae_mm": np.nan,
                    "peak_y_mean_error_mm": np.nan,
                    "max_abs_xr_rmse_mm": np.nan,
                    "max_abs_xr_mae_mm": np.nan,
                    "max_abs_xr_mean_error_mm": np.nan,
                    "true_limit_accuracy": np.nan,
                    "true_limit_false_feasible_rate": np.nan,
                    "safety_margin_false_feasible_rate": np.nan,
                    "safety_margin_unsafe_given_pred_safe_rate": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_high_outreach_regions(
    y_true_peak: np.ndarray,
    y_pred_peak: np.ndarray,
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
) -> pd.DataFrame:
    """Evaluate model accuracy in high-outreach regions."""
    region_masks = {
        "high_outreach_peak_ge_0p60": y_true_peak >= HIGH_OUTREACH_THRESHOLD,
        "top_reachable_peak_ge_0p65": y_true_peak >= TOP_REACHABLE_THRESHOLD,
    }

    rows = []

    for region_name, mask in region_masks.items():
        row = {
            "region": region_name,
            "n_samples": int(mask.sum()),
        }

        if np.any(mask):
            peak_reg = regression_metrics(y_true_peak[mask], y_pred_peak[mask])
            xr_reg = regression_metrics(y_true_xr[mask], y_pred_xr[mask])
            safety = evaluate_safety_margin_false_feasible(
                y_true_xr[mask],
                y_pred_xr[mask],
            )

            row.update(
                {
                    "peak_y_rmse_mm": peak_reg["rmse_mm"],
                    "peak_y_mae_mm": peak_reg["mae_mm"],
                    "max_abs_xr_rmse_mm": xr_reg["rmse_mm"],
                    "max_abs_xr_mae_mm": xr_reg["mae_mm"],
                    "safety_margin_false_feasible_rate": (
                        safety["safety_margin_false_feasible_rate"]
                    ),
                    "safety_margin_unsafe_given_pred_safe_rate": (
                        safety["safety_margin_unsafe_given_pred_safe_rate"]
                    ),
                }
            )
        else:
            row.update(
                {
                    "peak_y_rmse_mm": np.nan,
                    "peak_y_mae_mm": np.nan,
                    "max_abs_xr_rmse_mm": np.nan,
                    "max_abs_xr_mae_mm": np.nan,
                    "safety_margin_false_feasible_rate": np.nan,
                    "safety_margin_unsafe_given_pred_safe_rate": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_model(
    model: nn.Module | list[nn.Module],
    splits: dict[str, Any],
    device: torch.device,
    model_label: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a single model or an ensemble on train, validation and test sets."""
    scaler_Y = splits["scaler_Y"]

    metrics: dict[str, float] = {"model_label": model_label}  # type: ignore[assignment]
    predictions_by_split = {}

    for split in ["train", "val", "test"]:
        X_scaled = splits[f"X_{split}_scaled"]
        Y_true = splits[f"Y_{split}"]

        Y_pred = predict_physical_units(
            model=model,
            X_scaled=X_scaled,
            scaler_Y=scaler_Y,
            device=device,
        )

        predictions_by_split[split] = (Y_true, Y_pred)

        for target_index, target_name in enumerate(TARGET_COLUMNS):
            reg = regression_metrics(Y_true[:, target_index], Y_pred[:, target_index])
            for key, value in reg.items():
                metrics[f"{split}_{target_name}_{key}"] = float(value)

    Y_test, Y_test_pred = predictions_by_split["test"]

    y_true_peak = Y_test[:, 0]
    y_pred_peak = Y_test_pred[:, 0]
    y_true_xr = Y_test[:, 1]
    y_pred_xr = Y_test_pred[:, 1]

    metrics.update(
        evaluate_constraint_classification(
            y_true_xr,
            y_pred_xr,
            ROBOT_LIMIT_TRUE,
            "true_limit",
        )
    )
    metrics.update(
        evaluate_constraint_classification(
            y_true_xr,
            y_pred_xr,
            ROBOT_LIMIT_OPT,
            "opt_limit",
        )
    )
    metrics.update(evaluate_safety_margin_false_feasible(y_true_xr, y_pred_xr))
    metrics.update(evaluate_near_boundary(y_true_xr, y_pred_xr))

    target_metrics_df = evaluate_inverse_targets(
        y_true_peak,
        y_pred_peak,
        y_true_xr,
        y_pred_xr,
    )

    high_region_metrics_df = evaluate_high_outreach_regions(
        y_true_peak,
        y_pred_peak,
        y_true_xr,
        y_pred_xr,
    )

    for _, row in target_metrics_df.iterrows():
        tag = f"target_{row['target_peak_y_m']:.2f}".replace(".", "p")
        metrics[f"{tag}_n_samples"] = int(row["n_samples"])
        metrics[f"{tag}_peak_y_rmse_mm"] = float(row["peak_y_rmse_mm"])
        metrics[f"{tag}_max_abs_xr_rmse_mm"] = float(row["max_abs_xr_rmse_mm"])
        metrics[f"{tag}_safety_margin_false_feasible_rate"] = float(
            row["safety_margin_false_feasible_rate"]
        )

    X_test = splits["X_test"]

    predictions_df = pd.DataFrame(X_test, columns=INPUT_COLUMNS)
    predictions_df["model_label"] = model_label

    predictions_df["true_peak_y"] = y_true_peak
    predictions_df["pred_peak_y"] = y_pred_peak
    predictions_df["error_peak_y_m"] = y_pred_peak - y_true_peak
    predictions_df["abs_error_peak_y_m"] = np.abs(y_pred_peak - y_true_peak)

    predictions_df["true_max_abs_xr"] = y_true_xr
    predictions_df["pred_max_abs_xr"] = y_pred_xr
    predictions_df["error_max_abs_xr_m"] = y_pred_xr - y_true_xr
    predictions_df["abs_error_max_abs_xr_m"] = np.abs(y_pred_xr - y_true_xr)

    predictions_df["true_feasible_abs"] = y_true_xr <= ROBOT_LIMIT_TRUE
    predictions_df["pred_feasible_abs_true_limit"] = y_pred_xr <= ROBOT_LIMIT_TRUE
    predictions_df["pred_safe_abs_opt_limit"] = y_pred_xr <= ROBOT_LIMIT_OPT
    predictions_df["false_feasible_abs_true_limit"] = (
        predictions_df["pred_feasible_abs_true_limit"]
        & ~predictions_df["true_feasible_abs"]
    )
    predictions_df["safety_margin_false_feasible"] = (
        predictions_df["pred_safe_abs_opt_limit"]
        & ~predictions_df["true_feasible_abs"]
    )
    predictions_df["near_boundary"] = (
        (y_true_xr >= NEAR_BOUNDARY_LOW)
        & (y_true_xr <= NEAR_BOUNDARY_HIGH)
    )
    predictions_df["high_outreach"] = y_true_peak >= HIGH_OUTREACH_THRESHOLD
    predictions_df["top_reachable_outreach"] = y_true_peak >= TOP_REACHABLE_THRESHOLD

    if isinstance(model, list) and len(model) > 1:
        _, prediction_std = ensemble_prediction_stats_physical(
            models=model,
            X_scaled=splits["X_test_scaled"],
            scaler_Y=scaler_Y,
            device=device,
        )

        predictions_df["ensemble_std_peak_y_m"] = prediction_std[:, 0]
        predictions_df["ensemble_std_max_abs_xr_m"] = prediction_std[:, 1]

    return metrics, predictions_df, target_metrics_df, high_region_metrics_df