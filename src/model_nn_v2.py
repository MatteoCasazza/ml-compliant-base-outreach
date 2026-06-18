"""
model_nn_v2.py
==============

Constraint-aware multi-output Neural Network surrogate model for the
ML-based excitation planning project.

Task:
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
        -> [peak_y, max_abs_xr]

Main improvements over the initial NN surrogate:
    1. Weighted multi-output loss:
           L = MSE(peak_y) + lambda * MSE(max_abs_xr)
    2. Learning-rate scheduler on validation loss.
    3. Multi-seed ensemble prediction.
    4. Lambda sweep for the constraint-loss weight.
    5. Constraint-specific safety metrics, including the unsafe accepted rate:
           pred_max_abs_xr <= 0.495 and true_max_abs_xr > 0.500
    6. Evaluation in inverse-optimization target bands and high-outreach regions.

Outputs are written to:
    results/nn_v2/
    figures/nn_v2/

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

DATASET_PATH = DATA_DIR / "dataset_augmented.csv"
NN_RESULTS_DIR = RESULTS_DIR / "nn_v2"
NN_FIGURES_DIR = FIGURES_DIR / "nn_v2"

for directory in [NN_RESULTS_DIR, NN_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

INPUT_COLUMNS = [
    "Kb", "Kr", "Mb", "hb", "hr", "f0", "f1", "A", "x_r_start",
]
TARGET_COLUMNS = ["peak_y", "max_abs_xr"]

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN_POOL = 0.1875  # 0.1875 * 0.80 = 0.15 overall validation
RANDOM_STATE_SPLIT = 42

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495
NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520
HIGH_OUTREACH_THRESHOLD = 0.600
TOP_REACHABLE_THRESHOLD = 0.650

# First tune architecture/lr/wd at this default lambda, then sweep lambda using
# the selected architecture/lr/wd. This keeps the experiment rigorous but not huge.
DEFAULT_CONSTRAINT_LOSS_WEIGHT = 2.0
LAMBDA_SWEEP_VALUES = [1.0, 2.0, 3.0, 5.0]
FORCE_FINAL_LAMBDA: float | None = None  # set e.g. 2.0 to override automatic lambda selection

USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 40
SCHEDULER_MIN_LR = 1e-6

INVERSE_TARGETS = [0.55, 0.60, 0.65, 0.70, 0.75]
TARGET_TOLERANCE = 0.025

ARCHITECTURES = [[64, 64], [32, 64, 32], [128, 64]]
LEARNING_RATES = [1e-3, 5e-4]
WEIGHT_DECAYS = [1e-5, 1e-4]
SEEDS = [42, 123, 2026]

MAX_EPOCHS = 1500
PATIENCE = 120
BATCH_SIZE = 64

QUICK_DEBUG = False
if QUICK_DEBUG:
    ARCHITECTURES = [[64, 64]]
    LEARNING_RATES = [1e-3]
    WEIGHT_DECAYS = [1e-5]
    SEEDS = [42]
    LAMBDA_SWEEP_VALUES = [1.0, 2.0]
    MAX_EPOCHS = 80
    PATIENCE = 15
    USE_LR_SCHEDULER = False


# =============================================================================
# REPRODUCIBILITY AND DEVICE
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA
# =============================================================================

def load_nn_dataset(filepath: str | Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    required_cols = INPUT_COLUMNS + TARGET_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

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
            print(f"  {name:22s}: {count}")
    print("=" * 70)
    return X, Y, df


def prepare_splits(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray | StandardScaler]:
    X_train_pool, X_test, Y_train_pool, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT, shuffle=True
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_pool, Y_train_pool,
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
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test,
        "X_train_scaled": X_train_scaled, "X_val_scaled": X_val_scaled, "X_test_scaled": X_test_scaled,
        "Y_train_scaled": Y_train_scaled, "Y_val_scaled": Y_val_scaled, "Y_test_scaled": Y_test_scaled,
        "scaler_X": scaler_X, "scaler_Y": scaler_Y,
    }


def make_loader(X_scaled: np.ndarray, Y_scaled: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(Y_scaled, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# MODEL AND TRAINING
# =============================================================================

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Iterable[int], output_dim: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(nn.ReLU())
            prev = int(hidden)
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TrainingResult:
    model_state: Dict[str, torch.Tensor]
    history: pd.DataFrame
    best_val_loss: float
    best_val_peak_loss: float
    best_val_max_abs_xr_loss: float
    best_epoch: int
    training_time_s: float


def architecture_to_string(hidden_layers: List[int]) -> str:
    return "-".join(str(x) for x in hidden_layers)


def weighted_scaled_mse(per_output_loss: torch.Tensor, constraint_loss_weight: float) -> torch.Tensor:
    return per_output_loss[0] + float(constraint_loss_weight) * per_output_loss[1]


def evaluate_scaled_loss(
    model: nn.Module,
    X_scaled: np.ndarray,
    Y_scaled: np.ndarray,
    device: torch.device,
    constraint_loss_weight: float,
) -> Tuple[float, float, float]:
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_tensor)
        per_output_loss = torch.mean((pred - Y_tensor) ** 2, dim=0)
        total = weighted_scaled_mse(per_output_loss, constraint_loss_weight)
    return float(total.cpu()), float(per_output_loss[0].cpu()), float(per_output_loss[1].cpu())


def train_once(
    splits: Dict[str, np.ndarray | StandardScaler],
    hidden_layers: List[int],
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    constraint_loss_weight: float,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    verbose: bool = False,
) -> TrainingResult:
    set_seed(seed)

    X_train_scaled = splits["X_train_scaled"]
    Y_train_scaled = splits["Y_train_scaled"]
    X_val_scaled = splits["X_val_scaled"]
    Y_val_scaled = splits["Y_val_scaled"]
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(Y_train_scaled, np.ndarray)
    assert isinstance(X_val_scaled, np.ndarray)
    assert isinstance(Y_val_scaled, np.ndarray)

    model = MultiOutputMLP(X_train_scaled.shape[1], hidden_layers, len(TARGET_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
        )

    loader = make_loader(X_train_scaled, Y_train_scaled, batch_size, shuffle=True)

    best_val = np.inf
    best_peak = np.inf
    best_xr = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    rows = []
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses, peak_losses, xr_losses = [], [], []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            per_output_loss = torch.mean((pred - yb) ** 2, dim=0)
            loss = weighted_scaled_mse(per_output_loss, constraint_loss_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            peak_losses.append(float(per_output_loss[0].detach().cpu()))
            xr_losses.append(float(per_output_loss[1].detach().cpu()))

        val_loss, val_peak, val_xr = evaluate_scaled_loss(
            model, X_val_scaled, Y_val_scaled, device, constraint_loss_weight
        )
        lr_now = float(optimizer.param_groups[0]["lr"])
        rows.append({
            "epoch": epoch,
            "learning_rate": lr_now,
            "constraint_loss_weight": constraint_loss_weight,
            "train_loss": float(np.mean(train_losses)),
            "train_peak_y_loss": float(np.mean(peak_losses)),
            "train_max_abs_xr_loss": float(np.mean(xr_losses)),
            "val_loss": val_loss,
            "val_peak_y_loss": val_peak,
            "val_max_abs_xr_loss": val_xr,
        })

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_peak = val_peak
            best_xr = val_xr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if verbose and (epoch == 1 or epoch % 100 == 0):
            print(
                f"    epoch {epoch:4d} | val {val_loss:.5f} | "
                f"peak {val_peak:.5f} | max_abs_xr {val_xr:.5f} | lr {lr_now:.2e}"
            )

        if stale >= patience:
            break

    return TrainingResult(
        model_state=best_state,
        history=pd.DataFrame(rows),
        best_val_loss=float(best_val),
        best_val_peak_loss=float(best_peak),
        best_val_max_abs_xr_loss=float(best_xr),
        best_epoch=int(best_epoch),
        training_time_s=float(time.time() - start),
    )


def build_model_from_state(hidden_layers: List[int], model_state: Dict[str, torch.Tensor], device: torch.device) -> MultiOutputMLP:
    model = MultiOutputMLP(len(INPUT_COLUMNS), hidden_layers, len(TARGET_COLUMNS)).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


# =============================================================================
# PREDICTION AND METRICS
# =============================================================================

def predict_physical_units(
    model: nn.Module | List[nn.Module],
    X_scaled: np.ndarray,
    scaler_Y: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        if isinstance(model, list):
            pred_scaled = torch.stack([m(X_tensor) for m in model], dim=0).mean(dim=0).cpu().numpy()
        else:
            pred_scaled = model(X_tensor).cpu().numpy()
    return scaler_Y.inverse_transform(pred_scaled)


def ensemble_prediction_stats_physical(
    models: List[nn.Module], X_scaled: np.ndarray, scaler_Y: StandardScaler, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_scaled_all = torch.stack([m(X_tensor).cpu() for m in models], dim=0).numpy()
    pred_physical_all = np.stack([scaler_Y.inverse_transform(pred_scaled_all[i]) for i in range(len(models))], axis=0)
    return pred_physical_all.mean(axis=0), pred_physical_all.std(axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "rmse_m": rmse,
        "rmse_mm": rmse * 1000.0,
        "mae_m": mae,
        "mae_mm": mae * 1000.0,
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error_m": float(np.mean(y_pred - y_true)),
        "mean_error_mm": float(np.mean(y_pred - y_true) * 1000.0),
        "max_abs_error_m": float(np.max(np.abs(y_pred - y_true))),
        "max_abs_error_mm": float(np.max(np.abs(y_pred - y_true)) * 1000.0),
    }


def evaluate_constraint_classification(y_true_xr: np.ndarray, y_pred_xr: np.ndarray, limit: float, label: str) -> Dict[str, float]:
    true_feasible = y_true_xr <= limit
    pred_feasible = y_pred_xr <= limit
    cm = confusion_matrix(true_feasible, pred_feasible, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    return {
        f"{label}_limit_m": float(limit),
        f"{label}_classification_accuracy": float(np.mean(true_feasible == pred_feasible)),
        f"{label}_false_feasible_rate": float(np.mean(pred_feasible & (~true_feasible))),
        f"{label}_false_infeasible_rate": float(np.mean((~pred_feasible) & true_feasible)),
        f"{label}_true_infeasible_pred_infeasible": int(tn),
        f"{label}_true_infeasible_pred_feasible": int(fp),
        f"{label}_true_feasible_pred_infeasible": int(fn),
        f"{label}_true_feasible_pred_feasible": int(tp),
    }


def evaluate_safety_margin_false_feasible(y_true_xr: np.ndarray, y_pred_xr: np.ndarray) -> Dict[str, float]:
    """
    Safety-relevant metric for surrogate-based optimization.

    Unsafe accepted means:
        predicted safe under optimization margin: pred_max_abs_xr <= 0.495
        but truly infeasible under physical limit: true_max_abs_xr > 0.500
    """
    pred_safe_margin = y_pred_xr <= ROBOT_LIMIT_OPT
    true_infeasible = y_true_xr > ROBOT_LIMIT_TRUE
    unsafe_accepted = pred_safe_margin & true_infeasible
    n_pred_safe = int(pred_safe_margin.sum())
    return {
        "safety_margin_pred_safe_count": n_pred_safe,
        "safety_margin_unsafe_accepted_count": int(unsafe_accepted.sum()),
        "safety_margin_false_feasible_rate": float(np.mean(unsafe_accepted)),
        "safety_margin_unsafe_given_pred_safe_rate": float(unsafe_accepted.sum() / n_pred_safe) if n_pred_safe > 0 else np.nan,
    }


def evaluate_near_boundary(y_true_xr: np.ndarray, y_pred_xr: np.ndarray) -> Dict[str, float]:
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
        }
    reg = regression_metrics(y_true_xr[mask], y_pred_xr[mask])
    clf = evaluate_constraint_classification(y_true_xr[mask], y_pred_xr[mask], ROBOT_LIMIT_TRUE, "near_boundary_true_limit")
    safety = evaluate_safety_margin_false_feasible(y_true_xr[mask], y_pred_xr[mask])
    return {
        "near_boundary_low_m": NEAR_BOUNDARY_LOW,
        "near_boundary_high_m": NEAR_BOUNDARY_HIGH,
        "near_boundary_n_samples": int(mask.sum()),
        "near_boundary_rmse_mm": reg["rmse_mm"],
        "near_boundary_mae_mm": reg["mae_mm"],
        "near_boundary_classification_accuracy_true_limit": clf["near_boundary_true_limit_classification_accuracy"],
        "near_boundary_false_feasible_rate_true_limit": clf["near_boundary_true_limit_false_feasible_rate"],
        "near_boundary_safety_margin_false_feasible_rate": safety["safety_margin_false_feasible_rate"],
        "near_boundary_safety_margin_unsafe_given_pred_safe_rate": safety["safety_margin_unsafe_given_pred_safe_rate"],
    }


def evaluate_inverse_targets(
    y_true_peak: np.ndarray,
    y_pred_peak: np.ndarray,
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
    targets: List[float] = INVERSE_TARGETS,
    tolerance: float = TARGET_TOLERANCE,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int]] = []
    for target in targets:
        mask = np.abs(y_true_peak - target) <= tolerance
        row: Dict[str, float | int] = {
            "target_peak_y_m": float(target),
            "tolerance_m": float(tolerance),
            "n_samples": int(mask.sum()),
        }
        if np.any(mask):
            peak_reg = regression_metrics(y_true_peak[mask], y_pred_peak[mask])
            xr_reg = regression_metrics(y_true_xr[mask], y_pred_xr[mask])
            clf = evaluate_constraint_classification(y_true_xr[mask], y_pred_xr[mask], ROBOT_LIMIT_TRUE, "true_limit")
            safety = evaluate_safety_margin_false_feasible(y_true_xr[mask], y_pred_xr[mask])
            row.update({
                "peak_y_rmse_mm": peak_reg["rmse_mm"],
                "peak_y_mae_mm": peak_reg["mae_mm"],
                "peak_y_mean_error_mm": peak_reg["mean_error_mm"],
                "max_abs_xr_rmse_mm": xr_reg["rmse_mm"],
                "max_abs_xr_mae_mm": xr_reg["mae_mm"],
                "max_abs_xr_mean_error_mm": xr_reg["mean_error_mm"],
                "true_limit_accuracy": clf["true_limit_classification_accuracy"],
                "true_limit_false_feasible_rate": clf["true_limit_false_feasible_rate"],
                "safety_margin_false_feasible_rate": safety["safety_margin_false_feasible_rate"],
                "safety_margin_unsafe_given_pred_safe_rate": safety["safety_margin_unsafe_given_pred_safe_rate"],
            })
        else:
            row.update({
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
            })
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_high_outreach_regions(
    y_true_peak: np.ndarray,
    y_pred_peak: np.ndarray,
    y_true_xr: np.ndarray,
    y_pred_xr: np.ndarray,
) -> pd.DataFrame:
    region_defs = {
        "high_outreach_peak_ge_0p60": y_true_peak >= HIGH_OUTREACH_THRESHOLD,
        "top_reachable_peak_ge_0p65": y_true_peak >= TOP_REACHABLE_THRESHOLD,
    }
    rows = []
    for name, mask in region_defs.items():
        row = {"region": name, "n_samples": int(mask.sum())}
        if np.any(mask):
            peak_reg = regression_metrics(y_true_peak[mask], y_pred_peak[mask])
            xr_reg = regression_metrics(y_true_xr[mask], y_pred_xr[mask])
            safety = evaluate_safety_margin_false_feasible(y_true_xr[mask], y_pred_xr[mask])
            row.update({
                "peak_y_rmse_mm": peak_reg["rmse_mm"],
                "peak_y_mae_mm": peak_reg["mae_mm"],
                "max_abs_xr_rmse_mm": xr_reg["rmse_mm"],
                "max_abs_xr_mae_mm": xr_reg["mae_mm"],
                "safety_margin_false_feasible_rate": safety["safety_margin_false_feasible_rate"],
                "safety_margin_unsafe_given_pred_safe_rate": safety["safety_margin_unsafe_given_pred_safe_rate"],
            })
        else:
            row.update({
                "peak_y_rmse_mm": np.nan,
                "peak_y_mae_mm": np.nan,
                "max_abs_xr_rmse_mm": np.nan,
                "max_abs_xr_mae_mm": np.nan,
                "safety_margin_false_feasible_rate": np.nan,
                "safety_margin_unsafe_given_pred_safe_rate": np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_model(
    model: nn.Module | List[nn.Module],
    splits: Dict[str, np.ndarray | StandardScaler],
    device: torch.device,
    model_label: str,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler_Y = splits["scaler_Y"]
    assert isinstance(scaler_Y, StandardScaler)

    metrics: Dict[str, float] = {"model_label": model_label}  # type: ignore[assignment]
    predictions_by_split = {}
    for split in ["train", "val", "test"]:
        X_scaled = splits[f"X_{split}_scaled"]
        Y_true = splits[f"Y_{split}"]
        assert isinstance(X_scaled, np.ndarray)
        assert isinstance(Y_true, np.ndarray)
        Y_pred = predict_physical_units(model, X_scaled, scaler_Y, device)
        predictions_by_split[split] = (Y_true, Y_pred)
        for j, target_name in enumerate(TARGET_COLUMNS):
            reg = regression_metrics(Y_true[:, j], Y_pred[:, j])
            for key, value in reg.items():
                metrics[f"{split}_{target_name}_{key}"] = float(value)

    Y_test, Y_test_pred = predictions_by_split["test"]
    y_true_peak = Y_test[:, 0]
    y_pred_peak = Y_test_pred[:, 0]
    y_true_xr = Y_test[:, 1]
    y_pred_xr = Y_test_pred[:, 1]

    metrics.update(evaluate_constraint_classification(y_true_xr, y_pred_xr, ROBOT_LIMIT_TRUE, "true_limit"))
    metrics.update(evaluate_constraint_classification(y_true_xr, y_pred_xr, ROBOT_LIMIT_OPT, "opt_limit"))
    metrics.update(evaluate_safety_margin_false_feasible(y_true_xr, y_pred_xr))
    metrics.update(evaluate_near_boundary(y_true_xr, y_pred_xr))

    target_metrics_df = evaluate_inverse_targets(y_true_peak, y_pred_peak, y_true_xr, y_pred_xr)
    high_region_metrics_df = evaluate_high_outreach_regions(y_true_peak, y_pred_peak, y_true_xr, y_pred_xr)

    for _, row in target_metrics_df.iterrows():
        tag = f"target_{row['target_peak_y_m']:.2f}".replace(".", "p")
        metrics[f"{tag}_n_samples"] = int(row["n_samples"])
        metrics[f"{tag}_peak_y_rmse_mm"] = float(row["peak_y_rmse_mm"])
        metrics[f"{tag}_max_abs_xr_rmse_mm"] = float(row["max_abs_xr_rmse_mm"])
        metrics[f"{tag}_safety_margin_false_feasible_rate"] = float(row["safety_margin_false_feasible_rate"])

    X_test = splits["X_test"]
    assert isinstance(X_test, np.ndarray)
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
    predictions_df["false_feasible_abs_true_limit"] = predictions_df["pred_feasible_abs_true_limit"] & (~predictions_df["true_feasible_abs"])
    predictions_df["safety_margin_false_feasible"] = predictions_df["pred_safe_abs_opt_limit"] & (~predictions_df["true_feasible_abs"])
    predictions_df["near_boundary"] = (y_true_xr >= NEAR_BOUNDARY_LOW) & (y_true_xr <= NEAR_BOUNDARY_HIGH)
    predictions_df["high_outreach"] = y_true_peak >= HIGH_OUTREACH_THRESHOLD
    predictions_df["top_reachable_outreach"] = y_true_peak >= TOP_REACHABLE_THRESHOLD

    if isinstance(model, list) and len(model) > 1:
        _, pred_std = ensemble_prediction_stats_physical(model, splits["X_test_scaled"], scaler_Y, device)  # type: ignore[arg-type]
        predictions_df["ensemble_std_peak_y_m"] = pred_std[:, 0]
        predictions_df["ensemble_std_max_abs_xr_m"] = pred_std[:, 1]

    return metrics, predictions_df, target_metrics_df, high_region_metrics_df


# =============================================================================
# TUNING AND LAMBDA SWEEP
# =============================================================================

def run_hyperparameter_tuning(splits: Dict[str, np.ndarray | StandardScaler], device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    print("\n" + "=" * 70)
    print("NN HYPERPARAMETER TUNING AT DEFAULT LAMBDA")
    print("=" * 70)
    print(f"Default constraint lambda: {DEFAULT_CONSTRAINT_LOSS_WEIGHT}")
    print(f"Architectures:             {ARCHITECTURES}")
    print(f"Learning rates:            {LEARNING_RATES}")
    print(f"Weight decays:             {WEIGHT_DECAYS}")
    print(f"Seeds:                     {SEEDS}")

    rows = []
    run_id = 0
    total = len(ARCHITECTURES) * len(LEARNING_RATES) * len(WEIGHT_DECAYS) * len(SEEDS)
    start = time.time()

    for hidden_layers in ARCHITECTURES:
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                for seed in SEEDS:
                    run_id += 1
                    arch = architecture_to_string(hidden_layers)
                    print(f"\nRun {run_id}/{total} | arch={arch} | lr={lr:.0e} | wd={wd:.0e} | seed={seed}")
                    result = train_once(
                        splits, hidden_layers, lr, wd, seed, device,
                        constraint_loss_weight=DEFAULT_CONSTRAINT_LOSS_WEIGHT,
                    )
                    rows.append({
                        "run_id": run_id,
                        "architecture": arch,
                        "hidden_layers": json.dumps(hidden_layers),
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "constraint_loss_weight": DEFAULT_CONSTRAINT_LOSS_WEIGHT,
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "best_val_loss": result.best_val_loss,
                        "best_val_peak_y_loss": result.best_val_peak_loss,
                        "best_val_max_abs_xr_loss": result.best_val_max_abs_xr_loss,
                        "training_time_s": result.training_time_s,
                    })
                    print(
                        f"  best epoch {result.best_epoch:4d} | val {result.best_val_loss:.6f} | "
                        f"peak {result.best_val_peak_loss:.6f} | xr {result.best_val_max_abs_xr_loss:.6f}"
                    )
                    pd.DataFrame(rows).to_csv(NN_RESULTS_DIR / "tuning_results_checkpoint.csv", index=False)

    tuning_df = pd.DataFrame(rows)
    summary_df = (
        tuning_df.groupby(["architecture", "hidden_layers", "learning_rate", "weight_decay"])
        .agg(
            val_loss_mean=("best_val_loss", "mean"),
            val_loss_std=("best_val_loss", "std"),
            val_peak_y_loss_mean=("best_val_peak_y_loss", "mean"),
            val_max_abs_xr_loss_mean=("best_val_max_abs_xr_loss", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            training_time_s_mean=("training_time_s", "mean"),
            training_time_s_total=("training_time_s", "sum"),
            n_runs=("run_id", "count"),
        )
        .reset_index()
        .sort_values("val_loss_mean")
        .reset_index(drop=True)
    )
    tuning_df.to_csv(NN_RESULTS_DIR / "tuning_results.csv", index=False)
    summary_df.to_csv(NN_RESULTS_DIR / "tuning_summary.csv", index=False)

    best = summary_df.iloc[0]
    best_config = {
        "architecture": best["architecture"],
        "hidden_layers": json.loads(best["hidden_layers"]),
        "learning_rate": float(best["learning_rate"]),
        "weight_decay": float(best["weight_decay"]),
        "default_constraint_loss_weight": DEFAULT_CONSTRAINT_LOSS_WEIGHT,
        "val_loss_mean_default_lambda": float(best["val_loss_mean"]),
        "val_loss_std_default_lambda": float(best["val_loss_std"]),
    }

    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print(summary_df.head(10).to_string(index=False))
    print("\nSelected config:")
    print(best_config)
    print(f"Total tuning time: {(time.time() - start) / 60:.1f} min")
    print("=" * 70)
    return tuning_df, summary_df, best_config


def evaluate_validation_for_lambda(
    model: nn.Module | List[nn.Module], splits: Dict[str, np.ndarray | StandardScaler], device: torch.device
) -> Dict[str, float]:
    scaler_Y = splits["scaler_Y"]
    assert isinstance(scaler_Y, StandardScaler)
    X_val_scaled = splits["X_val_scaled"]
    Y_val = splits["Y_val"]
    assert isinstance(X_val_scaled, np.ndarray)
    assert isinstance(Y_val, np.ndarray)
    Y_pred = predict_physical_units(model, X_val_scaled, scaler_Y, device)
    y_true_peak, y_true_xr = Y_val[:, 0], Y_val[:, 1]
    y_pred_peak, y_pred_xr = Y_pred[:, 0], Y_pred[:, 1]
    peak_reg = regression_metrics(y_true_peak, y_pred_peak)
    xr_reg = regression_metrics(y_true_xr, y_pred_xr)
    clf = evaluate_constraint_classification(y_true_xr, y_pred_xr, ROBOT_LIMIT_TRUE, "val_true_limit")
    near = evaluate_near_boundary(y_true_xr, y_pred_xr)
    safety = evaluate_safety_margin_false_feasible(y_true_xr, y_pred_xr)
    return {
        "val_peak_y_rmse_mm": peak_reg["rmse_mm"],
        "val_peak_y_mae_mm": peak_reg["mae_mm"],
        "val_max_abs_xr_rmse_mm": xr_reg["rmse_mm"],
        "val_max_abs_xr_mae_mm": xr_reg["mae_mm"],
        "val_true_limit_false_feasible_rate": clf["val_true_limit_false_feasible_rate"],
        "val_near_boundary_false_feasible_rate_true_limit": near["near_boundary_false_feasible_rate_true_limit"],
        "val_safety_margin_false_feasible_rate": safety["safety_margin_false_feasible_rate"],
        "val_safety_margin_unsafe_given_pred_safe_rate": safety["safety_margin_unsafe_given_pred_safe_rate"],
    }


def run_lambda_sweep(
    splits: Dict[str, np.ndarray | StandardScaler], best_config: Dict[str, object], device: torch.device
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    hidden_layers = list(best_config["hidden_layers"])
    lr = float(best_config["learning_rate"])
    wd = float(best_config["weight_decay"])

    print("\n" + "=" * 70)
    print("CONSTRAINT LOSS WEIGHT LAMBDA SWEEP")
    print("=" * 70)
    print(f"Fixed architecture: {architecture_to_string(hidden_layers)}")
    print(f"Fixed lr/wd:        {lr:.0e} / {wd:.0e}")
    print(f"Lambda values:      {LAMBDA_SWEEP_VALUES}")

    rows = []
    for lam in LAMBDA_SWEEP_VALUES:
        seed_models = []
        for seed in SEEDS:
            print(f"\nLambda {lam:g} | seed={seed}")
            result = train_once(
                splits, hidden_layers, lr, wd, seed, device,
                constraint_loss_weight=float(lam),
            )
            model = build_model_from_state(hidden_layers, result.model_state, device)
            seed_models.append(model)
            val_metrics = evaluate_validation_for_lambda(model, splits, device)
            row = {
                "constraint_loss_weight": float(lam),
                "seed": int(seed),
                "best_epoch": result.best_epoch,
                "best_val_weighted_loss": result.best_val_loss,
                "best_val_peak_y_loss_scaled": result.best_val_peak_loss,
                "best_val_max_abs_xr_loss_scaled": result.best_val_max_abs_xr_loss,
                "training_time_s": result.training_time_s,
            }
            row.update(val_metrics)
            rows.append(row)
            print(
                f"  val peak RMSE={val_metrics['val_peak_y_rmse_mm']:.2f} mm | "
                f"xr RMSE={val_metrics['val_max_abs_xr_rmse_mm']:.2f} mm | "
                f"safety false feasible={val_metrics['val_safety_margin_false_feasible_rate']*100:.2f}%"
            )

        # Also evaluate ensemble validation metrics for this lambda.
        ensemble_metrics = evaluate_validation_for_lambda(seed_models, splits, device)
        ensemble_row = {
            "constraint_loss_weight": float(lam),
            "seed": "ensemble_mean",
            "best_epoch": np.nan,
            "best_val_weighted_loss": np.nan,
            "best_val_peak_y_loss_scaled": np.nan,
            "best_val_max_abs_xr_loss_scaled": np.nan,
            "training_time_s": np.nan,
        }
        ensemble_row.update(ensemble_metrics)
        rows.append(ensemble_row)
        print(
            f"  ensemble val peak RMSE={ensemble_metrics['val_peak_y_rmse_mm']:.2f} mm | "
            f"xr RMSE={ensemble_metrics['val_max_abs_xr_rmse_mm']:.2f} mm | "
            f"safety false feasible={ensemble_metrics['val_safety_margin_false_feasible_rate']*100:.2f}%"
        )

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(NN_RESULTS_DIR / "lambda_sweep_results.csv", index=False)

    # Selection uses ensemble metrics only, because final surrogate is the ensemble.
    ensemble_df = sweep_df[sweep_df["seed"] == "ensemble_mean"].copy()
    summary_df = ensemble_df[[
        "constraint_loss_weight",
        "val_peak_y_rmse_mm",
        "val_max_abs_xr_rmse_mm",
        "val_true_limit_false_feasible_rate",
        "val_near_boundary_false_feasible_rate_true_limit",
        "val_safety_margin_false_feasible_rate",
        "val_safety_margin_unsafe_given_pred_safe_rate",
    ]].copy()

    # Rank by safety first, then constraint RMSE, then peak_y RMSE.
    summary_df = summary_df.sort_values(
        [
            "val_safety_margin_false_feasible_rate",
            "val_near_boundary_false_feasible_rate_true_limit",
            "val_true_limit_false_feasible_rate",
            "val_max_abs_xr_rmse_mm",
            "val_peak_y_rmse_mm",
        ],
        ascending=True,
    ).reset_index(drop=True)
    summary_df["selection_rank"] = np.arange(1, len(summary_df) + 1)
    summary_df.to_csv(NN_RESULTS_DIR / "lambda_sweep_summary.csv", index=False)

    selected_lambda = float(summary_df.iloc[0]["constraint_loss_weight"])
    if FORCE_FINAL_LAMBDA is not None:
        selected_lambda = float(FORCE_FINAL_LAMBDA)

    print("\n" + "=" * 70)
    print("LAMBDA SWEEP SUMMARY - ENSEMBLE VALIDATION METRICS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSelected lambda: {selected_lambda:g}")
    print("=" * 70)
    return sweep_df, summary_df, selected_lambda


# =============================================================================
# FINAL TRAINING, PLOTS, SAVE
# =============================================================================

def train_final_ensemble(
    splits: Dict[str, np.ndarray | StandardScaler], best_config: Dict[str, object], selected_lambda: float, device: torch.device
) -> Tuple[MultiOutputMLP, TrainingResult, int, List[MultiOutputMLP], pd.DataFrame]:
    hidden_layers = list(best_config["hidden_layers"])
    lr = float(best_config["learning_rate"])
    wd = float(best_config["weight_decay"])

    print("\n" + "=" * 70)
    print("FINAL NN V2 TRAINING - ENSEMBLE")
    print("=" * 70)
    print(f"Architecture: {architecture_to_string(hidden_layers)}")
    print(f"Learning rate: {lr:.0e}")
    print(f"Weight decay:  {wd:.0e}")
    print(f"Selected λ:    {selected_lambda:g}")
    print(f"Seeds:         {SEEDS}")

    best_result = None
    best_seed = SEEDS[0]
    ensemble_models = []
    seed_rows = []

    for seed in SEEDS:
        print(f"\nFinal training seed={seed}")
        result = train_once(
            splits, hidden_layers, lr, wd, seed, device,
            constraint_loss_weight=selected_lambda,
            verbose=True,
        )
        model = build_model_from_state(hidden_layers, result.model_state, device)
        ensemble_models.append(model)
        seed_rows.append({
            "seed": seed,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "best_val_peak_y_loss": result.best_val_peak_loss,
            "best_val_max_abs_xr_loss": result.best_val_max_abs_xr_loss,
            "training_time_s": result.training_time_s,
        })
        if best_result is None or result.best_val_loss < best_result.best_val_loss:
            best_result = result
            best_seed = seed

    assert best_result is not None
    best_model = build_model_from_state(hidden_layers, best_result.model_state, device)
    seed_results_df = pd.DataFrame(seed_rows)
    print(f"\nBest single seed: {best_seed}")
    print(f"Ensemble size:    {len(ensemble_models)}")
    return best_model, best_result, best_seed, ensemble_models, seed_results_df


def print_metrics(metrics: Dict[str, float], title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"peak_y RMSE:      {metrics['test_peak_y_rmse_mm']:.2f} mm | R2={metrics['test_peak_y_r2']:.4f}")
    print(f"max_abs_xr RMSE:  {metrics['test_max_abs_xr_rmse_mm']:.2f} mm | R2={metrics['test_max_abs_xr_r2']:.4f}")
    print(f"true-limit false feasible:        {metrics['true_limit_false_feasible_rate']*100:.2f}%")
    print(f"safety-margin false feasible:     {metrics['safety_margin_false_feasible_rate']*100:.2f}%")
    if not np.isnan(metrics.get("safety_margin_unsafe_given_pred_safe_rate", np.nan)):
        print(f"unsafe accepted / pred safe:      {metrics['safety_margin_unsafe_given_pred_safe_rate']*100:.2f}%")
    print(f"near-boundary RMSE max_abs_xr:     {metrics['near_boundary_rmse_mm']:.2f} mm")
    print(f"near-boundary safety false feas.:  {metrics['near_boundary_safety_margin_false_feasible_rate']*100:.2f}%")
    print("=" * 70)


def plot_training_curve(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["epoch"], history["train_loss"], label="Train weighted loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation weighted loss")
    ax.plot(history["epoch"], history["val_peak_y_loss"], linestyle="--", label="Val peak_y MSE")
    ax.plot(history["epoch"], history["val_max_abs_xr_loss"], linestyle=":", label="Val max_abs_xr MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss on standardized outputs")
    ax.set_title("NN v2 training curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = NN_FIGURES_DIR / "nn_v2_training_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_parity(predictions_df: pd.DataFrame) -> None:
    specs = [
        ("peak_y", "true_peak_y", "pred_peak_y", "Peak outreach [m]"),
        ("max_abs_xr", "true_max_abs_xr", "pred_max_abs_xr", "Maximum absolute robot displacement [m]"),
    ]
    for name, true_col, pred_col, label in specs:
        y_true = predictions_df[true_col].to_numpy()
        y_pred = predictions_df[pred_col].to_numpy()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.65, s=45, edgecolors="black", linewidth=0.5)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2, label="Identity")
        if name == "max_abs_xr":
            ax.axhline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="True limit")
            ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2)
            ax.axhline(ROBOT_LIMIT_OPT, linestyle="-.", linewidth=1.5, label="Optimization margin")
            ax.axvline(ROBOT_LIMIT_OPT, linestyle="-.", linewidth=1.5)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"NN v2 parity plot: {name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path = NN_FIGURES_DIR / f"nn_v2_parity_{name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_lambda_sweep(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.sort_values("constraint_loss_weight")
    x = plot_df["constraint_loss_weight"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, plot_df["val_peak_y_rmse_mm"], marker="o", label="peak_y RMSE")
    ax.plot(x, plot_df["val_max_abs_xr_rmse_mm"], marker="o", label="max_abs_xr RMSE")
    ax.set_xlabel("Constraint loss weight λ")
    ax.set_ylabel("Validation RMSE [mm]")
    ax.set_title("NN v2 lambda sweep: regression trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = NN_FIGURES_DIR / "nn_v2_lambda_sweep_rmse.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, plot_df["val_safety_margin_false_feasible_rate"] * 100.0, marker="o", label="Safety-margin false feasible")
    ax.plot(x, plot_df["val_near_boundary_false_feasible_rate_true_limit"] * 100.0, marker="o", label="Near-boundary false feasible")
    ax.set_xlabel("Constraint loss weight λ")
    ax.set_ylabel("Rate [%]")
    ax.set_title("NN v2 lambda sweep: safety metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = NN_FIGURES_DIR / "nn_v2_lambda_sweep_safety.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved lambda sweep plots in {NN_FIGURES_DIR}")


def plot_confusion(predictions_df: pd.DataFrame) -> None:
    y_true = predictions_df["true_max_abs_xr"].to_numpy()
    y_pred = predictions_df["pred_max_abs_xr"].to_numpy()
    limits = [("True limit 0.500 m", ROBOT_LIMIT_TRUE), ("Optimization margin 0.495 m", ROBOT_LIMIT_OPT)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, limit) in zip(axes, limits):
        cm = confusion_matrix(y_true <= limit, y_pred <= limit, labels=[False, True])
        ax.imshow(cm)
        ax.set_title(title)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred infeasible", "Pred feasible"])
        ax.set_yticklabels(["True infeasible", "True feasible"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    plt.suptitle("NN v2 constraint classification")
    plt.tight_layout()
    path = NN_FIGURES_DIR / "nn_v2_constraint_classification.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def save_outputs(
    best_model: MultiOutputMLP,
    ensemble_models: List[MultiOutputMLP],
    seed_results_df: pd.DataFrame,
    best_result: TrainingResult,
    best_config: Dict[str, object],
    selected_lambda: float,
    best_seed: int,
    splits: Dict[str, np.ndarray | StandardScaler],
    ensemble_metrics: Dict[str, float],
    ensemble_predictions_df: pd.DataFrame,
    ensemble_target_metrics_df: pd.DataFrame,
    ensemble_high_region_metrics_df: pd.DataFrame,
    best_seed_metrics: Dict[str, float],
    best_seed_predictions_df: pd.DataFrame,
    best_seed_target_metrics_df: pd.DataFrame,
    best_seed_high_region_metrics_df: pd.DataFrame,
    device: torch.device,
) -> None:
    scaler_X = splits["scaler_X"]
    scaler_Y = splits["scaler_Y"]
    assert isinstance(scaler_X, StandardScaler)
    assert isinstance(scaler_Y, StandardScaler)

    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "ensemble_state_dicts": [m.state_dict() for m in ensemble_models],
        "ensemble_seeds": SEEDS,
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "hidden_layers": best_config["hidden_layers"],
        "learning_rate": best_config["learning_rate"],
        "weight_decay": best_config["weight_decay"],
        "constraint_loss_weight": selected_lambda,
        "default_constraint_loss_weight": DEFAULT_CONSTRAINT_LOSS_WEIGHT,
        "lambda_sweep_values": LAMBDA_SWEEP_VALUES,
        "use_lr_scheduler": USE_LR_SCHEDULER,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "best_seed": best_seed,
        "best_epoch": best_result.best_epoch,
        "device_used": str(device),
    }
    model_path = NN_RESULTS_DIR / "nn_multioutput_model.pt"
    torch.save(checkpoint, model_path)
    joblib.dump(scaler_X, NN_RESULTS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_Y, NN_RESULTS_DIR / "scaler_Y.pkl")

    common_meta = {
        "target_columns": ",".join(TARGET_COLUMNS),
        "input_columns": ",".join(INPUT_COLUMNS),
        "architecture": architecture_to_string(list(best_config["hidden_layers"])),
        "learning_rate": float(best_config["learning_rate"]),
        "weight_decay": float(best_config["weight_decay"]),
        "constraint_loss_weight": float(selected_lambda),
        "use_lr_scheduler": USE_LR_SCHEDULER,
        "best_seed": int(best_seed),
        "best_epoch": int(best_result.best_epoch),
        "best_val_loss": float(best_result.best_val_loss),
        "best_val_peak_y_loss": float(best_result.best_val_peak_loss),
        "best_val_max_abs_xr_loss": float(best_result.best_val_max_abs_xr_loss),
        "ensemble_n_models": len(ensemble_models),
    }

    em = dict(ensemble_metrics); em.update(common_meta); em["model"] = "NN_v2_multioutput_ensemble"
    bm = dict(best_seed_metrics); bm.update(common_meta); bm["model"] = "NN_v2_multioutput_best_seed"
    pd.DataFrame([em]).to_csv(NN_RESULTS_DIR / "metrics.csv", index=False)
    pd.DataFrame([bm]).to_csv(NN_RESULTS_DIR / "metrics_best_seed.csv", index=False)

    ensemble_predictions_df.to_csv(NN_RESULTS_DIR / "test_predictions.csv", index=False)
    best_seed_predictions_df.to_csv(NN_RESULTS_DIR / "test_predictions_best_seed.csv", index=False)
    ensemble_target_metrics_df.to_csv(NN_RESULTS_DIR / "target_metrics.csv", index=False)
    best_seed_target_metrics_df.to_csv(NN_RESULTS_DIR / "target_metrics_best_seed.csv", index=False)
    ensemble_high_region_metrics_df.to_csv(NN_RESULTS_DIR / "high_outreach_region_metrics.csv", index=False)
    best_seed_high_region_metrics_df.to_csv(NN_RESULTS_DIR / "high_outreach_region_metrics_best_seed.csv", index=False)
    seed_results_df.to_csv(NN_RESULTS_DIR / "final_seed_results.csv", index=False)
    best_result.history.to_csv(NN_RESULTS_DIR / "training_history.csv", index=False)

    info = NN_RESULTS_DIR / "model_info.txt"
    with open(info, "w", encoding="utf-8") as f:
        f.write("NN v2 multi-output ensemble surrogate\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Results dir: {NN_RESULTS_DIR}\n")
        f.write(f"Inputs: {INPUT_COLUMNS}\n")
        f.write(f"Targets: {TARGET_COLUMNS}\n")
        f.write(f"Architecture: {architecture_to_string(list(best_config['hidden_layers']))}\n")
        f.write(f"Learning rate: {float(best_config['learning_rate']):.6g}\n")
        f.write(f"Weight decay: {float(best_config['weight_decay']):.6g}\n")
        f.write(f"Selected lambda: {selected_lambda:.3f}\n")
        f.write(f"Ensemble seeds: {SEEDS}\n")
        f.write(f"Ensemble models: {len(ensemble_models)}\n")
        f.write("\nPrimary test metrics: ensemble\n")
        f.write(f"peak_y RMSE: {ensemble_metrics['test_peak_y_rmse_mm']:.3f} mm\n")
        f.write(f"max_abs_xr RMSE: {ensemble_metrics['test_max_abs_xr_rmse_mm']:.3f} mm\n")
        f.write(f"true-limit false feasible: {ensemble_metrics['true_limit_false_feasible_rate'] * 100:.3f}%\n")
        f.write(f"safety-margin false feasible: {ensemble_metrics['safety_margin_false_feasible_rate'] * 100:.3f}%\n")
        f.write("\nThe primary model for inverse optimization is the ensemble mean.\n")

    print(f"✓ Saved model checkpoint: {model_path}")
    print(f"✓ Saved scalers and metrics in: {NN_RESULTS_DIR}")
    print(f"✓ Saved model info: {info}")


def load_nn_ensemble(
    model_path: str | Path = NN_RESULTS_DIR / "nn_multioutput_model.pt",
    scaler_dir: str | Path = NN_RESULTS_DIR,
    device: torch.device | None = None,
) -> Tuple[List[MultiOutputMLP], StandardScaler, StandardScaler]:
    if device is None:
        device = get_device()
    model_path = Path(model_path)
    scaler_dir = Path(scaler_dir)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state_dicts = checkpoint.get("ensemble_state_dicts")
    if state_dicts is None:
        raise ValueError("Checkpoint does not contain ensemble_state_dicts.")
    models = []
    for state in state_dicts:
        model = MultiOutputMLP(
            input_dim=len(checkpoint["input_columns"]),
            hidden_layers=checkpoint["hidden_layers"],
            output_dim=len(checkpoint["target_columns"]),
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    scaler_X = joblib.load(scaler_dir / "scaler_X.pkl")
    scaler_Y = joblib.load(scaler_dir / "scaler_Y.pkl")
    return models, scaler_X, scaler_Y


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("CONSTRAINT-AWARE MULTI-OUTPUT NN SURROGATE V2")
    print("=" * 70)
    device = get_device()
    print(f"Using device: {device}")

    X, Y, _ = load_nn_dataset(DATASET_PATH)
    splits = prepare_splits(X, Y)

    _, _, best_config = run_hyperparameter_tuning(splits, device)
    _, lambda_summary_df, selected_lambda = run_lambda_sweep(splits, best_config, device)
    plot_lambda_sweep(lambda_summary_df)

    best_model, best_result, best_seed, ensemble_models, seed_results_df = train_final_ensemble(
        splits, best_config, selected_lambda, device
    )

    best_seed_metrics, best_seed_predictions_df, best_seed_target_metrics_df, best_seed_high_region_metrics_df = evaluate_model(
        best_model, splits, device, "best_seed"
    )
    ensemble_metrics, ensemble_predictions_df, ensemble_target_metrics_df, ensemble_high_region_metrics_df = evaluate_model(
        ensemble_models, splits, device, "ensemble"
    )

    print_metrics(best_seed_metrics, "FINAL NN V2 TEST METRICS - BEST SINGLE SEED")
    print_metrics(ensemble_metrics, "FINAL NN V2 TEST METRICS - ENSEMBLE")

    plot_training_curve(best_result.history)
    plot_parity(ensemble_predictions_df)
    plot_confusion(ensemble_predictions_df)

    save_outputs(
        best_model=best_model,
        ensemble_models=ensemble_models,
        seed_results_df=seed_results_df,
        best_result=best_result,
        best_config=best_config,
        selected_lambda=selected_lambda,
        best_seed=best_seed,
        splits=splits,
        ensemble_metrics=ensemble_metrics,
        ensemble_predictions_df=ensemble_predictions_df,
        ensemble_target_metrics_df=ensemble_target_metrics_df,
        ensemble_high_region_metrics_df=ensemble_high_region_metrics_df,
        best_seed_metrics=best_seed_metrics,
        best_seed_predictions_df=best_seed_predictions_df,
        best_seed_target_metrics_df=best_seed_target_metrics_df,
        best_seed_high_region_metrics_df=best_seed_high_region_metrics_df,
        device=device,
    )

    print("\n" + "=" * 70)
    print("NN V2 TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model:      {NN_RESULTS_DIR / 'nn_multioutput_model.pt'}")
    print(f"Metrics:    {NN_RESULTS_DIR / 'metrics.csv'}")
    print(f"Lambda:     {NN_RESULTS_DIR / 'lambda_sweep_summary.csv'}")
    print(f"Plots:      {NN_FIGURES_DIR / 'nn_v2_*.png'}")
    print("Primary surrogate for inverse optimization: ensemble mean.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
