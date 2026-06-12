"""
model_nn.py
===========

Multi-output Neural Network surrogate model for the ML-based excitation
planning project.

Main supervised-learning task:
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
        -> [peak_y, max_abs_xr]

Why multi-output?
-----------------
The same dynamic simulation produces both the performance quantity peak_y and
constraint quantity max_abs_xr. A shared neural representation can learn common
features and provides a differentiable surrogate for gradient-based inverse
optimization.

Workflow
--------
1. Load data/dataset_augmented.csv
2. Use the same 80/20 test split convention used by the GP scripts
3. Split the training pool into train/validation sets for early stopping
4. Standardize inputs and both outputs
5. Tune a small grid of MLP architectures, learning rates and weight decay
   values with multiple random seeds
6. Select the best configuration by mean validation loss across seeds
7. Retrain the selected configuration across the same seeds and keep the best
   validation run
8. Evaluate on the held-out test set
9. Save model, scalers, tuning results, metrics, predictions and plots

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
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
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

NN_RESULTS_DIR = RESULTS_DIR / "nn"
NN_FIGURES_DIR = FIGURES_DIR / "nn"

for directory in [NN_RESULTS_DIR, NN_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


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
VAL_SIZE_FROM_TRAIN_POOL = 0.1875  # 0.1875 * 0.80 = 0.15 overall validation
RANDOM_STATE_SPLIT = 42

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495
NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520
HIGH_OUTREACH_THRESHOLD = 0.600

# Normal mode: robust but still manageable.
ARCHITECTURES = [
    [64, 64],
    [32, 64, 32],
    [128, 64],
]
LEARNING_RATES = [1e-3, 5e-4]
WEIGHT_DECAYS = [1e-5, 1e-4]
SEEDS = [42, 123, 2026]

MAX_EPOCHS = 1500
PATIENCE = 120
BATCH_SIZE = 64

# Set QUICK_DEBUG = True only to check that the script works quickly.
QUICK_DEBUG = False

if QUICK_DEBUG:
    ARCHITECTURES = [[64, 64]]
    LEARNING_RATES = [1e-3]
    WEIGHT_DECAYS = [1e-5]
    SEEDS = [42]
    MAX_EPOCHS = 80
    PATIENCE = 15


# =============================================================================
# REPRODUCIBILITY AND DEVICE
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode improves reproducibility, at a possible speed cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_nn_dataset(filepath: str | Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load dataset for multi-output NN training.

    Outputs
    -------
    X : ndarray, shape (n_samples, 9)
        Input parameters.
    Y : ndarray, shape (n_samples, 2)
        Targets [peak_y, max_abs_xr].
    df : DataFrame
        Full cleaned dataset.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    required_cols = INPUT_COLUMNS + TARGET_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df)

    if after < before:
        print(f"Dropped {before - after} rows with NaNs in required columns.")

    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    Y = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    print("\n" + "=" * 70)
    print("NN DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset path:              {filepath}")
    print(f"Samples:                   {len(df)}")
    print(f"Input dimensions:          {X.shape[1]}")
    print(f"Targets:                   {TARGET_COLUMNS}")
    print(f"peak_y mean/std:           {df['peak_y'].mean():.6f} / {df['peak_y'].std():.6f} m")
    print(f"max_abs_xr mean/std:       {df['max_abs_xr'].mean():.6f} / {df['max_abs_xr'].std():.6f} m")

    if "constraint_violation_abs" in df.columns:
        feasible_abs = df["constraint_violation_abs"] <= 0.002
        print(f"Feasible_abs samples:      {int(feasible_abs.sum())} ({100 * feasible_abs.mean():.1f}%)")

    if "dataset_type" in df.columns:
        print("Dataset types:")
        for name, count in df["dataset_type"].value_counts().items():
            print(f"  {name:22s}: {count}")

    print("=" * 70)

    return X, Y, df


def prepare_splits(
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, np.ndarray | StandardScaler]:
    """
    Create train/validation/test splits and standardize inputs/outputs.

    The held-out test set uses the same convention as the GP scripts:
        train_test_split(..., test_size=0.2, random_state=42)

    The validation set is carved only from the training pool, so no test data is
    used for model selection or early stopping.
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
    print(f"Train samples:             {len(X_train)}")
    print(f"Validation samples:        {len(X_val)}")
    print(f"Test samples:              {len(X_test)}")
    print(f"Test fraction:             {TEST_SIZE:.2f}")
    print(f"Validation fraction total: {len(X_val) / len(X):.3f}")
    print("\nInput scaler check:")
    print(f"  Mean first 3 features:   {X_train_scaled.mean(axis=0)[:3]}")
    print(f"  Std first 3 features:    {X_train_scaled.std(axis=0)[:3]}")
    print("\nOutput scaler check:")
    print(f"  Mean targets:            {Y_train_scaled.mean(axis=0)}")
    print(f"  Std targets:             {Y_train_scaled.std(axis=0)}")
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
    """
    Build a PyTorch DataLoader from scaled numpy arrays.
    """
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# MODEL
# =============================================================================

class MultiOutputMLP(nn.Module):
    """
    Simple fully connected multi-output neural surrogate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int = 2,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

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


# =============================================================================
# TRAINING
# =============================================================================

def evaluate_scaled_loss(
    model: nn.Module,
    X_scaled: np.ndarray,
    Y_scaled: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Compute total and per-output MSE loss on scaled data.
    """
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(X_tensor)
        per_output_loss = torch.mean((pred - Y_tensor) ** 2, dim=0)
        total_loss = torch.mean((pred - Y_tensor) ** 2)

    return (
        float(total_loss.cpu().item()),
        float(per_output_loss[0].cpu().item()),
        float(per_output_loss[1].cpu().item()),
    )


def train_once(
    splits: Dict[str, np.ndarray | StandardScaler],
    hidden_layers: List[int],
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    verbose: bool = False,
) -> TrainingResult:
    """
    Train one NN model with early stopping.
    """
    set_seed(seed)

    X_train_scaled = splits["X_train_scaled"]
    Y_train_scaled = splits["Y_train_scaled"]
    X_val_scaled = splits["X_val_scaled"]
    Y_val_scaled = splits["Y_val_scaled"]

    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(Y_train_scaled, np.ndarray)
    assert isinstance(X_val_scaled, np.ndarray)
    assert isinstance(Y_val_scaled, np.ndarray)

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

    train_loader = make_loader(
        X_train_scaled,
        Y_train_scaled,
        batch_size=batch_size,
        shuffle=True,
    )

    mse_loss = nn.MSELoss()

    best_val_loss = np.inf
    best_val_peak_loss = np.inf
    best_val_max_abs_xr_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history_rows = []
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_peak_losses = []
        epoch_constraint_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)

            loss = mse_loss(pred, yb)
            per_output_loss = torch.mean((pred - yb) ** 2, dim=0)

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_peak_losses.append(float(per_output_loss[0].detach().cpu().item()))
            epoch_constraint_losses.append(float(per_output_loss[1].detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses))
        train_peak_loss = float(np.mean(epoch_peak_losses))
        train_max_abs_xr_loss = float(np.mean(epoch_constraint_losses))

        val_loss, val_peak_loss, val_max_abs_xr_loss = evaluate_scaled_loss(
            model,
            X_val_scaled,
            Y_val_scaled,
            device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_peak_y_loss": train_peak_loss,
                "train_max_abs_xr_loss": train_max_abs_xr_loss,
                "val_loss": val_loss,
                "val_peak_y_loss": val_peak_loss,
                "val_max_abs_xr_loss": val_max_abs_xr_loss,
            }
        )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_val_peak_loss = val_peak_loss
            best_val_max_abs_xr_loss = val_max_abs_xr_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch == 1 or epoch % 100 == 0):
            print(
                f"    epoch {epoch:4d} | train {train_loss:.5f} | "
                f"val {val_loss:.5f} | val peak {val_peak_loss:.5f} | "
                f"val max_abs_xr {val_max_abs_xr_loss:.5f}"
            )

        if epochs_without_improvement >= patience:
            break

    training_time_s = time.time() - start_time

    return TrainingResult(
        model_state=best_state,
        history=pd.DataFrame(history_rows),
        best_val_loss=float(best_val_loss),
        best_val_peak_loss=float(best_val_peak_loss),
        best_val_max_abs_xr_loss=float(best_val_max_abs_xr_loss),
        best_epoch=int(best_epoch),
        training_time_s=float(training_time_s),
    )


def architecture_to_string(hidden_layers: List[int]) -> str:
    return "-".join(str(x) for x in hidden_layers)


def run_tuning(
    splits: Dict[str, np.ndarray | StandardScaler],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Run tuning grid with multiple seeds.

    Selection rule:
    - Aggregate validation loss across seeds for each configuration.
    - Select the configuration with the lowest mean validation loss.
    """
    print("\n" + "=" * 70)
    print("NN HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Device:          {device}")
    print(f"Architectures:   {ARCHITECTURES}")
    print(f"Learning rates:  {LEARNING_RATES}")
    print(f"Weight decays:   {WEIGHT_DECAYS}")
    print(f"Seeds:           {SEEDS}")
    print(f"Max epochs:      {MAX_EPOCHS}")
    print(f"Patience:        {PATIENCE}")
    print(f"Batch size:      {BATCH_SIZE}")

    total_runs = len(ARCHITECTURES) * len(LEARNING_RATES) * len(WEIGHT_DECAYS) * len(SEEDS)
    print(f"Total runs:      {total_runs}")
    print("=" * 70)

    rows = []
    run_id = 0
    tuning_start = time.time()

    for hidden_layers in ARCHITECTURES:
        for learning_rate in LEARNING_RATES:
            for weight_decay in WEIGHT_DECAYS:
                for seed in SEEDS:
                    run_id += 1
                    arch_str = architecture_to_string(hidden_layers)
                    print(
                        f"\nRun {run_id}/{total_runs} | arch={arch_str} | "
                        f"lr={learning_rate:.0e} | wd={weight_decay:.0e} | seed={seed}"
                    )

                    result = train_once(
                        splits=splits,
                        hidden_layers=hidden_layers,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        seed=seed,
                        device=device,
                        max_epochs=MAX_EPOCHS,
                        patience=PATIENCE,
                        batch_size=BATCH_SIZE,
                        verbose=False,
                    )

                    rows.append(
                        {
                            "run_id": run_id,
                            "architecture": arch_str,
                            "hidden_layers": json.dumps(hidden_layers),
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "seed": seed,
                            "best_epoch": result.best_epoch,
                            "best_val_loss": result.best_val_loss,
                            "best_val_peak_y_loss": result.best_val_peak_loss,
                            "best_val_max_abs_xr_loss": result.best_val_max_abs_xr_loss,
                            "training_time_s": result.training_time_s,
                        }
                    )

                    print(
                        f"  best epoch {result.best_epoch:4d} | "
                        f"val loss {result.best_val_loss:.6f} | "
                        f"peak {result.best_val_peak_loss:.6f} | "
                        f"max_abs_xr {result.best_val_max_abs_xr_loss:.6f} | "
                        f"time {result.training_time_s:.1f} s"
                    )

                    pd.DataFrame(rows).to_csv(
                        NN_RESULTS_DIR / "tuning_results_checkpoint.csv",
                        index=False,
                    )

    tuning_df = pd.DataFrame(rows)

    summary_df = (
        tuning_df
        .groupby(["architecture", "hidden_layers", "learning_rate", "weight_decay"])
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
        .sort_values("val_loss_mean", ascending=True)
        .reset_index(drop=True)
    )

    tuning_df.to_csv(NN_RESULTS_DIR / "tuning_results.csv", index=False)
    summary_df.to_csv(NN_RESULTS_DIR / "tuning_summary.csv", index=False)

    best_row = summary_df.iloc[0]
    best_config = {
        "architecture": best_row["architecture"],
        "hidden_layers": json.loads(best_row["hidden_layers"]),
        "learning_rate": float(best_row["learning_rate"]),
        "weight_decay": float(best_row["weight_decay"]),
        "val_loss_mean": float(best_row["val_loss_mean"]),
        "val_loss_std": float(best_row["val_loss_std"]),
    }

    elapsed = time.time() - tuning_start

    print("\n" + "=" * 70)
    print("NN TUNING SUMMARY")
    print("=" * 70)
    print(summary_df.head(10).to_string(index=False))
    print("\nSelected configuration by mean validation loss:")
    print(f"  Architecture:     {best_config['architecture']}")
    print(f"  Learning rate:    {best_config['learning_rate']:.0e}")
    print(f"  Weight decay:     {best_config['weight_decay']:.0e}")
    print(f"  Val loss mean:    {best_config['val_loss_mean']:.6f}")
    print(f"  Val loss std:     {best_config['val_loss_std']:.6f}")
    print(f"Total tuning time:  {elapsed / 60:.1f} min")
    print("=" * 70)

    return tuning_df, summary_df, best_config


def train_final_model(
    splits: Dict[str, np.ndarray | StandardScaler],
    best_config: Dict[str, object],
    device: torch.device,
) -> Tuple[MultiOutputMLP, TrainingResult, int]:
    """
    Retrain the selected configuration across the predefined seeds and keep the
    run with the lowest validation loss.
    """
    hidden_layers = list(best_config["hidden_layers"])
    learning_rate = float(best_config["learning_rate"])
    weight_decay = float(best_config["weight_decay"])

    print("\n" + "=" * 70)
    print("FINAL NN TRAINING")
    print("=" * 70)
    print(f"Architecture:     {architecture_to_string(hidden_layers)}")
    print(f"Learning rate:    {learning_rate:.0e}")
    print(f"Weight decay:     {weight_decay:.0e}")
    print(f"Seeds:            {SEEDS}")
    print("=" * 70)

    best_result: TrainingResult | None = None
    best_seed = SEEDS[0]

    for seed in SEEDS:
        print(f"\nFinal training with seed={seed}")
        result = train_once(
            splits=splits,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            verbose=True,
        )

        print(
            f"  best epoch {result.best_epoch} | "
            f"val loss {result.best_val_loss:.6f} | "
            f"time {result.training_time_s:.1f} s"
        )

        if best_result is None or result.best_val_loss < best_result.best_val_loss:
            best_result = result
            best_seed = seed

    assert best_result is not None

    input_dim = len(INPUT_COLUMNS)
    output_dim = len(TARGET_COLUMNS)
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
    ).to(device)
    model.load_state_dict(best_result.model_state)

    print("\nSelected final seed:")
    print(f"  Seed:          {best_seed}")
    print(f"  Best epoch:    {best_result.best_epoch}")
    print(f"  Val loss:      {best_result.best_val_loss:.6f}")

    return model, best_result, best_seed


# =============================================================================
# EVALUATION
# =============================================================================

def predict_physical_units(
    model: nn.Module,
    X_scaled: np.ndarray,
    scaler_Y: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    """
    Predict in original physical units.
    """
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()

    pred = scaler_Y.inverse_transform(pred_scaled)
    return pred


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics in physical units.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse_m": rmse,
        "rmse_mm": rmse * 1000.0,
        "mae_m": mae,
        "mae_mm": mae * 1000.0,
        "r2": r2,
        "mean_error_m": float(np.mean(y_pred - y_true)),
        "max_abs_error_m": float(np.max(np.abs(y_pred - y_true))),
        "max_abs_error_mm": float(np.max(np.abs(y_pred - y_true)) * 1000.0),
    }


def evaluate_constraint_classification(
    y_true_max_abs_xr: np.ndarray,
    y_pred_max_abs_xr: np.ndarray,
    limit: float,
    label: str,
) -> Dict[str, float]:
    """
    Evaluate feasibility classification for max_abs_xr <= limit.
    """
    true_feasible = y_true_max_abs_xr <= limit
    pred_feasible = y_pred_max_abs_xr <= limit

    accuracy = float(np.mean(true_feasible == pred_feasible))
    false_feasible = float(np.mean((pred_feasible == True) & (true_feasible == False)))
    false_infeasible = float(np.mean((pred_feasible == False) & (true_feasible == True)))

    cm = confusion_matrix(
        true_feasible,
        pred_feasible,
        labels=[False, True],
    )

    # Rows: true [infeasible, feasible], cols: pred [infeasible, feasible]
    tn, fp, fn, tp = cm.ravel()

    return {
        f"{label}_limit_m": limit,
        f"{label}_classification_accuracy": accuracy,
        f"{label}_false_feasible_rate": false_feasible,
        f"{label}_false_infeasible_rate": false_infeasible,
        f"{label}_true_infeasible_pred_infeasible": int(tn),
        f"{label}_true_infeasible_pred_feasible": int(fp),
        f"{label}_true_feasible_pred_infeasible": int(fn),
        f"{label}_true_feasible_pred_feasible": int(tp),
    }


def evaluate_near_boundary(
    y_true_max_abs_xr: np.ndarray,
    y_pred_max_abs_xr: np.ndarray,
    low: float = NEAR_BOUNDARY_LOW,
    high: float = NEAR_BOUNDARY_HIGH,
) -> Dict[str, float]:
    """
    Evaluate max_abs_xr prediction near the constraint boundary.
    """
    mask = (y_true_max_abs_xr >= low) & (y_true_max_abs_xr <= high)

    if not np.any(mask):
        return {
            "near_boundary_low_m": low,
            "near_boundary_high_m": high,
            "near_boundary_n_samples": 0,
            "near_boundary_rmse_mm": np.nan,
            "near_boundary_mae_mm": np.nan,
            "near_boundary_classification_accuracy_true_limit": np.nan,
            "near_boundary_false_feasible_rate_true_limit": np.nan,
        }

    y_true_nb = y_true_max_abs_xr[mask]
    y_pred_nb = y_pred_max_abs_xr[mask]

    reg = regression_metrics(y_true_nb, y_pred_nb)
    clf = evaluate_constraint_classification(
        y_true_nb,
        y_pred_nb,
        limit=ROBOT_LIMIT_TRUE,
        label="near_boundary_true_limit",
    )

    return {
        "near_boundary_low_m": low,
        "near_boundary_high_m": high,
        "near_boundary_n_samples": int(mask.sum()),
        "near_boundary_rmse_mm": reg["rmse_mm"],
        "near_boundary_mae_mm": reg["mae_mm"],
        "near_boundary_classification_accuracy_true_limit": clf[
            "near_boundary_true_limit_classification_accuracy"
        ],
        "near_boundary_false_feasible_rate_true_limit": clf[
            "near_boundary_true_limit_false_feasible_rate"
        ],
    }


def evaluate_final_model(
    model: nn.Module,
    splits: Dict[str, np.ndarray | StandardScaler],
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate NN on train, validation and test sets.
    """
    scaler_Y = splits["scaler_Y"]
    assert isinstance(scaler_Y, StandardScaler)

    Y_train = splits["Y_train"]
    Y_val = splits["Y_val"]
    Y_test = splits["Y_test"]
    X_train_scaled = splits["X_train_scaled"]
    X_val_scaled = splits["X_val_scaled"]
    X_test_scaled = splits["X_test_scaled"]

    assert isinstance(Y_train, np.ndarray)
    assert isinstance(Y_val, np.ndarray)
    assert isinstance(Y_test, np.ndarray)
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_val_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)

    Y_train_pred = predict_physical_units(model, X_train_scaled, scaler_Y, device)
    Y_val_pred = predict_physical_units(model, X_val_scaled, scaler_Y, device)
    Y_test_pred = predict_physical_units(model, X_test_scaled, scaler_Y, device)

    metrics: Dict[str, float] = {}

    split_data = {
        "train": (Y_train, Y_train_pred),
        "val": (Y_val, Y_val_pred),
        "test": (Y_test, Y_test_pred),
    }

    for split_name, (Y_true, Y_pred) in split_data.items():
        for target_idx, target_name in enumerate(TARGET_COLUMNS):
            reg = regression_metrics(Y_true[:, target_idx], Y_pred[:, target_idx])
            for key, value in reg.items():
                metrics[f"{split_name}_{target_name}_{key}"] = value

    # Constraint-specific metrics on test set.
    y_true_max_abs = Y_test[:, 1]
    y_pred_max_abs = Y_test_pred[:, 1]

    metrics.update(
        evaluate_constraint_classification(
            y_true_max_abs,
            y_pred_max_abs,
            limit=ROBOT_LIMIT_TRUE,
            label="true_limit",
        )
    )

    metrics.update(
        evaluate_constraint_classification(
            y_true_max_abs,
            y_pred_max_abs,
            limit=ROBOT_LIMIT_OPT,
            label="opt_limit",
        )
    )

    metrics.update(evaluate_near_boundary(y_true_max_abs, y_pred_max_abs))

    high_mask = Y_test[:, 0] > HIGH_OUTREACH_THRESHOLD
    metrics["test_high_outreach_n_samples"] = int(high_mask.sum())
    if np.any(high_mask):
        high_reg = regression_metrics(Y_test[high_mask, 0], Y_test_pred[high_mask, 0])
        for key, value in high_reg.items():
            metrics[f"test_high_outreach_peak_y_{key}"] = value

    # Save test predictions with original input parameters.
    X_test = splits["X_test"]
    assert isinstance(X_test, np.ndarray)

    predictions_df = pd.DataFrame(X_test, columns=INPUT_COLUMNS)
    predictions_df["true_peak_y"] = Y_test[:, 0]
    predictions_df["pred_peak_y"] = Y_test_pred[:, 0]
    predictions_df["error_peak_y_m"] = Y_test_pred[:, 0] - Y_test[:, 0]
    predictions_df["abs_error_peak_y_m"] = np.abs(Y_test_pred[:, 0] - Y_test[:, 0])

    predictions_df["true_max_abs_xr"] = Y_test[:, 1]
    predictions_df["pred_max_abs_xr"] = Y_test_pred[:, 1]
    predictions_df["error_max_abs_xr_m"] = Y_test_pred[:, 1] - Y_test[:, 1]
    predictions_df["abs_error_max_abs_xr_m"] = np.abs(Y_test_pred[:, 1] - Y_test[:, 1])

    predictions_df["true_feasible_abs"] = Y_test[:, 1] <= ROBOT_LIMIT_TRUE
    predictions_df["pred_feasible_abs"] = Y_test_pred[:, 1] <= ROBOT_LIMIT_TRUE
    predictions_df["false_feasible_abs"] = (
        predictions_df["pred_feasible_abs"] & (~predictions_df["true_feasible_abs"])
    )
    predictions_df["near_boundary"] = (
        (Y_test[:, 1] >= NEAR_BOUNDARY_LOW) & (Y_test[:, 1] <= NEAR_BOUNDARY_HIGH)
    )
    predictions_df["high_outreach"] = Y_test[:, 0] > HIGH_OUTREACH_THRESHOLD

    return metrics, predictions_df


def print_final_metrics(metrics: Dict[str, float]) -> None:
    """
    Print compact final NN metrics.
    """
    print("\n" + "=" * 70)
    print("FINAL NN TEST METRICS")
    print("=" * 70)

    print("\n--- peak_y ---")
    print(f"  Test RMSE: {metrics['test_peak_y_rmse_m']:.6f} m ({metrics['test_peak_y_rmse_mm']:.2f} mm)")
    print(f"  Test MAE:  {metrics['test_peak_y_mae_m']:.6f} m ({metrics['test_peak_y_mae_mm']:.2f} mm)")
    print(f"  Test R²:   {metrics['test_peak_y_r2']:.6f}")

    if "test_high_outreach_peak_y_rmse_mm" in metrics:
        print("\n--- peak_y high-outreach region ---")
        print(f"  Samples:   {metrics['test_high_outreach_n_samples']}")
        print(f"  RMSE:      {metrics['test_high_outreach_peak_y_rmse_mm']:.2f} mm")
        print(f"  MAE:       {metrics['test_high_outreach_peak_y_mae_mm']:.2f} mm")

    print("\n--- max_abs_xr ---")
    print(f"  Test RMSE: {metrics['test_max_abs_xr_rmse_m']:.6f} m ({metrics['test_max_abs_xr_rmse_mm']:.2f} mm)")
    print(f"  Test MAE:  {metrics['test_max_abs_xr_mae_m']:.6f} m ({metrics['test_max_abs_xr_mae_mm']:.2f} mm)")
    print(f"  Test R²:   {metrics['test_max_abs_xr_r2']:.6f}")

    print("\n--- absolute constraint classification, true limit ---")
    print(f"  Accuracy:          {metrics['true_limit_classification_accuracy'] * 100:.2f}%")
    print(f"  False feasible:    {metrics['true_limit_false_feasible_rate'] * 100:.2f}%")
    print(f"  False infeasible:  {metrics['true_limit_false_infeasible_rate'] * 100:.2f}%")

    print("\n--- near-boundary region ---")
    print(
        f"  Samples:           {metrics['near_boundary_n_samples']} "
        f"[{NEAR_BOUNDARY_LOW:.3f}, {NEAR_BOUNDARY_HIGH:.3f}] m"
    )
    print(f"  RMSE:              {metrics['near_boundary_rmse_mm']:.2f} mm")
    print(f"  MAE:               {metrics['near_boundary_mae_mm']:.2f} mm")
    print(f"  Accuracy:          {metrics['near_boundary_classification_accuracy_true_limit'] * 100:.2f}%")
    print(f"  False feasible:    {metrics['near_boundary_false_feasible_rate_true_limit'] * 100:.2f}%")
    print("=" * 70)


# =============================================================================
# PLOTS
# =============================================================================

def plot_training_curve(history: pd.DataFrame, save_dir: str | Path = NN_FIGURES_DIR) -> None:
    """
    Plot total and per-output train/validation losses.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history["epoch"], history["train_loss"], label="Train total loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation total loss")
    ax.plot(history["epoch"], history["val_peak_y_loss"], linestyle="--", label="Validation peak_y loss")
    ax.plot(history["epoch"], history["val_max_abs_xr_loss"], linestyle=":", label="Validation max_abs_xr loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on standardized outputs")
    ax.set_title("NN Training Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    path = save_dir / "nn_training_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_parity_and_residuals(predictions_df: pd.DataFrame, save_dir: str | Path = NN_FIGURES_DIR) -> None:
    """
    Create parity and residual plots for both outputs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("peak_y", "true_peak_y", "pred_peak_y", "Peak outreach $peak_y$ [m]"),
        ("max_abs_xr", "true_max_abs_xr", "pred_max_abs_xr", "Maximum absolute robot displacement $max|x_r|$ [m]"),
    ]

    for name, true_col, pred_col, label in plot_specs:
        y_true = predictions_df[true_col].to_numpy()
        y_pred = predictions_df[pred_col].to_numpy()
        residuals_mm = (y_pred - y_true) * 1000.0

        # Parity plot.
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.65, s=45, edgecolors="black", linewidth=0.5)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, label="Identity")

        if name == "max_abs_xr":
            ax.axhline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
            ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2)

        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"NN parity plot: {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = save_dir / f"nn_parity_{name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")

        # Residual plot.
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_pred, residuals_mm, alpha=0.65, s=45, edgecolors="black", linewidth=0.5)
        ax.axhline(0.0, linestyle="--", linewidth=2)

        if name == "max_abs_xr":
            ax.axvline(ROBOT_LIMIT_TRUE, linestyle=":", linewidth=2, label="Robot limit")
            ax.legend()

        ax.set_xlabel(f"Predicted {label}")
        ax.set_ylabel("Residual [mm]")
        ax.set_title(f"NN residuals: {name}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = save_dir / f"nn_residuals_{name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_constraint_confusion(predictions_df: pd.DataFrame, save_dir: str | Path = NN_FIGURES_DIR) -> None:
    """
    Plot confusion matrices for true and safety limits.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions_df["true_max_abs_xr"].to_numpy()
    y_pred = predictions_df["pred_max_abs_xr"].to_numpy()

    limits = [
        ("True limit 0.500 m", ROBOT_LIMIT_TRUE),
        ("Safety limit 0.495 m", ROBOT_LIMIT_OPT),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (title, limit) in zip(axes, limits):
        true_feasible = y_true <= limit
        pred_feasible = y_pred <= limit

        cm = confusion_matrix(
            true_feasible,
            pred_feasible,
            labels=[False, True],
        )

        ax.imshow(cm)
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred infeasible", "Pred feasible"])
        ax.set_yticklabels(["True infeasible", "True feasible"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    plt.suptitle("NN absolute constraint classification")
    plt.tight_layout()

    path = save_dir / "nn_constraint_classification.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def generate_plots(history: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    print("\nGenerating NN plots...")
    plot_training_curve(history, save_dir=NN_FIGURES_DIR)
    plot_parity_and_residuals(predictions_df, save_dir=NN_FIGURES_DIR)
    plot_constraint_confusion(predictions_df, save_dir=NN_FIGURES_DIR)


# =============================================================================
# SAVE
# =============================================================================

def save_outputs(
    model: MultiOutputMLP,
    best_result: TrainingResult,
    best_config: Dict[str, object],
    best_seed: int,
    splits: Dict[str, np.ndarray | StandardScaler],
    metrics: Dict[str, float],
    predictions_df: pd.DataFrame,
    device: torch.device,
) -> None:
    """
    Save model, scalers, metrics, predictions and metadata.
    """
    scaler_X = splits["scaler_X"]
    scaler_Y = splits["scaler_Y"]

    assert isinstance(scaler_X, StandardScaler)
    assert isinstance(scaler_Y, StandardScaler)

    model_path = NN_RESULTS_DIR / "nn_multioutput_model.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "hidden_layers": best_config["hidden_layers"],
        "learning_rate": best_config["learning_rate"],
        "weight_decay": best_config["weight_decay"],
        "best_seed": best_seed,
        "best_epoch": best_result.best_epoch,
        "device_used": str(device),
    }

    torch.save(checkpoint, model_path)
    print(f"✓ Model saved: {model_path}")

    joblib.dump(scaler_X, NN_RESULTS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_Y, NN_RESULTS_DIR / "scaler_Y.pkl")
    print(f"✓ Scalers saved in: {NN_RESULTS_DIR}")

    metrics_with_meta = dict(metrics)
    metrics_with_meta.update(
        {
            "model": "NN_multioutput",
            "target_columns": ",".join(TARGET_COLUMNS),
            "input_columns": ",".join(INPUT_COLUMNS),
            "architecture": architecture_to_string(list(best_config["hidden_layers"])),
            "learning_rate": float(best_config["learning_rate"]),
            "weight_decay": float(best_config["weight_decay"]),
            "best_seed": int(best_seed),
            "best_epoch": int(best_result.best_epoch),
            "best_val_loss": float(best_result.best_val_loss),
            "best_val_peak_y_loss": float(best_result.best_val_peak_loss),
            "best_val_max_abs_xr_loss": float(best_result.best_val_max_abs_xr_loss),
            "training_time_s": float(best_result.training_time_s),
        }
    )

    pd.DataFrame([metrics_with_meta]).to_csv(NN_RESULTS_DIR / "metrics.csv", index=False)
    print(f"✓ Metrics saved: {NN_RESULTS_DIR / 'metrics.csv'}")

    predictions_df.to_csv(NN_RESULTS_DIR / "test_predictions.csv", index=False)
    print(f"✓ Test predictions saved: {NN_RESULTS_DIR / 'test_predictions.csv'}")

    best_result.history.to_csv(NN_RESULTS_DIR / "training_history.csv", index=False)
    print(f"✓ Training history saved: {NN_RESULTS_DIR / 'training_history.csv'}")

    info_path = NN_RESULTS_DIR / "model_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Multi-output Neural Network Surrogate\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Inputs: {INPUT_COLUMNS}\n")
        f.write(f"Targets: {TARGET_COLUMNS}\n")
        f.write(f"Architecture: {architecture_to_string(list(best_config['hidden_layers']))}\n")
        f.write(f"Learning rate: {float(best_config['learning_rate']):.6g}\n")
        f.write(f"Weight decay: {float(best_config['weight_decay']):.6g}\n")
        f.write(f"Best seed: {best_seed}\n")
        f.write(f"Best epoch: {best_result.best_epoch}\n")
        f.write(f"Best validation loss: {best_result.best_val_loss:.8f}\n")
        f.write("\nTest metrics:\n")
        f.write(f"peak_y RMSE: {metrics['test_peak_y_rmse_mm']:.3f} mm\n")
        f.write(f"peak_y MAE:  {metrics['test_peak_y_mae_mm']:.3f} mm\n")
        f.write(f"peak_y R2:   {metrics['test_peak_y_r2']:.6f}\n")
        f.write(f"max_abs_xr RMSE: {metrics['test_max_abs_xr_rmse_mm']:.3f} mm\n")
        f.write(f"max_abs_xr MAE:  {metrics['test_max_abs_xr_mae_mm']:.3f} mm\n")
        f.write(f"max_abs_xr R2:   {metrics['test_max_abs_xr_r2']:.6f}\n")
        f.write("\nConstraint classification, true limit:\n")
        f.write(f"Accuracy:       {metrics['true_limit_classification_accuracy'] * 100:.2f}%\n")
        f.write(f"False feasible: {metrics['true_limit_false_feasible_rate'] * 100:.2f}%\n")
        f.write("\nNear-boundary:\n")
        f.write(f"Samples: {metrics['near_boundary_n_samples']}\n")
        f.write(f"RMSE:    {metrics['near_boundary_rmse_mm']:.3f} mm\n")

    print(f"✓ Model info saved: {info_path}")


def load_nn_model(
    model_path: str | Path = NN_RESULTS_DIR / "nn_multioutput_model.pt",
    scaler_dir: str | Path = NN_RESULTS_DIR,
    device: torch.device | None = None,
) -> Tuple[MultiOutputMLP, StandardScaler, StandardScaler]:
    """
    Load trained NN model and scalers.
    """
    if device is None:
        device = get_device()

    model_path = Path(model_path)
    scaler_dir = Path(scaler_dir)

    checkpoint = torch.load(model_path, map_location=device)

    model = MultiOutputMLP(
        input_dim=len(checkpoint["input_columns"]),
        hidden_layers=checkpoint["hidden_layers"],
        output_dim=len(checkpoint["target_columns"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_X = joblib.load(scaler_dir / "scaler_X.pkl")
    scaler_Y = joblib.load(scaler_dir / "scaler_Y.pkl")

    return model, scaler_X, scaler_Y


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("MULTI-OUTPUT NEURAL NETWORK SURROGATE")
    print("=" * 70)

    device = get_device()
    print(f"Using device: {device}")

    X, Y, df = load_nn_dataset(DATASET_PATH)
    splits = prepare_splits(X, Y)

    tuning_df, tuning_summary_df, best_config = run_tuning(splits, device)

    model, best_result, best_seed = train_final_model(
        splits=splits,
        best_config=best_config,
        device=device,
    )

    metrics, predictions_df = evaluate_final_model(
        model=model,
        splits=splits,
        device=device,
    )

    print_final_metrics(metrics)

    generate_plots(best_result.history, predictions_df)

    save_outputs(
        model=model,
        best_result=best_result,
        best_config=best_config,
        best_seed=best_seed,
        splits=splits,
        metrics=metrics,
        predictions_df=predictions_df,
        device=device,
    )

    print("\n" + "=" * 70)
    print("NN TRAINING COMPLETED")
    print("=" * 70)
    print("Generated files:")
    print(f"  {NN_RESULTS_DIR / 'nn_multioutput_model.pt'}")
    print(f"  {NN_RESULTS_DIR / 'scaler_X.pkl'}")
    print(f"  {NN_RESULTS_DIR / 'scaler_Y.pkl'}")
    print(f"  {NN_RESULTS_DIR / 'metrics.csv'}")
    print(f"  {NN_RESULTS_DIR / 'tuning_results.csv'}")
    print(f"  {NN_RESULTS_DIR / 'tuning_summary.csv'}")
    print(f"  {NN_RESULTS_DIR / 'test_predictions.csv'}")
    print(f"  {NN_RESULTS_DIR / 'training_history.csv'}")
    print(f"  {NN_FIGURES_DIR / 'nn_*.png'}")
    print("\nNext step: compare NN_multioutput against GP_peak_y + GP_max_abs_xr.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
