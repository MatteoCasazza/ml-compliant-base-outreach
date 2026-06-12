"""
learning_curve_gp_vs_nn.py
==========================

Learning-curve comparison between:
    1. GP pair: GP_peak_y + GP_max_abs_xr
    2. NN multi-output: [peak_y, max_abs_xr]

Goal
----
Evaluate how surrogate accuracy changes as the number of training samples
increases. The test set is fixed for all experiments, so GP and NN are compared
fairly at the same training-set sizes.

Mappings
--------
Inputs:
    [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]

Targets:
    peak_y
    max_abs_xr

Outputs
-------
results/learning_curve/
    gp_vs_nn_learning_curve.csv
    gp_vs_nn_learning_curve_report_table.csv

figures/learning_curve/
    gp_vs_nn_peak_y_rmse.png
    gp_vs_nn_max_abs_xr_rmse.png
    gp_vs_nn_r2.png
    gp_vs_nn_constraint_safety.png
    gp_vs_nn_training_time.png
    gp_vs_nn_learning_curve_summary.png

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "learning_curve"
FIGURES_DIR = PROJECT_ROOT / "figures" / "learning_curve"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SETTINGS
# ============================================================================

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

# With the current 1600-sample dataset, the 80% train pool contains 1280 samples.
# Values larger than the available train pool are automatically skipped.
TRAIN_SIZES = [200, 400, 800, 1200, 1280]

# GP settings. Keep n_restarts small for learning curves because many GPs are fitted.
GP_ALPHA = 1e-6
GP_N_RESTARTS = 1
GP_LENGTH_SCALE_BOUNDS = (1e-2, 1e3)

# NN settings: fixed configuration selected from the previous tuning.
NN_HIDDEN_LAYERS = [128, 64]
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4
NN_BATCH_SIZE = 64
NN_MAX_EPOCHS = 1500
NN_PATIENCE = 120
NN_VAL_FRACTION_FROM_TRAIN_POOL = 0.15
NN_SEEDS = [42]
# For a more robust but slower final curve, use:
# NN_SEEDS = [42, 123, 2026]

ROBOT_LIMIT_TRUE = 0.500
ROBOT_LIMIT_OPT = 0.495
NEAR_BOUNDARY_LOW = 0.480
NEAR_BOUNDARY_HIGH = 0.520
HIGH_OUTREACH_THRESHOLD = 0.600

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load final augmented dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = INPUT_COLUMNS + TARGET_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    X = df[INPUT_COLUMNS].values.astype(np.float64)
    Y = df[TARGET_COLUMNS].values.astype(np.float64)

    return X, Y, df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print compact dataset summary."""
    print("\n" + "=" * 70)
    print("LEARNING CURVE DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset path:              {DATA_PATH}")
    print(f"Samples:                   {len(df)}")
    print(f"Input dimensions:          {len(INPUT_COLUMNS)}")
    print(f"Targets:                   {TARGET_COLUMNS}")
    print(f"peak_y mean/std:           {df[PEAK_COL].mean():.6f} / {df[PEAK_COL].std():.6f} m")
    print(f"max_abs_xr mean/std:       {df[MAX_ABS_XR_COL].mean():.6f} / {df[MAX_ABS_XR_COL].std():.6f} m")

    if "feasible_abs" in df.columns:
        feasible_count = int(df["feasible_abs"].sum())
        print(f"Feasible_abs samples:      {feasible_count} ({100 * feasible_count / len(df):.1f}%)")

    if "dataset_type" in df.columns:
        print("Dataset types:")
        for name, count in df["dataset_type"].value_counts().items():
            print(f"  {name:22s}: {count}")

    print("=" * 70)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
) -> Dict[str, float]:
    """Compute feasibility classification metrics for max_abs_xr."""
    true_feasible = y_true <= limit
    pred_feasible = y_pred <= limit

    accuracy = float(np.mean(true_feasible == pred_feasible))
    false_feasible = float(np.mean((pred_feasible == True) & (true_feasible == False)))
    false_infeasible = float(np.mean((pred_feasible == False) & (true_feasible == True)))

    return {
        "constraint_accuracy_percent": accuracy * 100.0,
        "false_feasible_percent": false_feasible * 100.0,
        "false_infeasible_percent": false_infeasible * 100.0,
    }


def near_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low: float = NEAR_BOUNDARY_LOW,
    high: float = NEAR_BOUNDARY_HIGH,
) -> Dict[str, float]:
    """Compute max_abs_xr metrics near the constraint boundary."""
    mask = (y_true >= low) & (y_true <= high)

    if not np.any(mask):
        return {
            "near_boundary_n": 0,
            "near_boundary_rmse_mm": np.nan,
            "near_boundary_mae_mm": np.nan,
            "near_boundary_accuracy_percent": np.nan,
            "near_boundary_false_feasible_percent": np.nan,
        }

    y_true_nb = y_true[mask]
    y_pred_nb = y_pred[mask]

    reg = regression_metrics(y_true_nb, y_pred_nb)
    clf = constraint_metrics(y_true_nb, y_pred_nb, limit=ROBOT_LIMIT_TRUE)

    return {
        "near_boundary_n": int(np.sum(mask)),
        "near_boundary_rmse_mm": reg["rmse_mm"],
        "near_boundary_mae_mm": reg["mae_mm"],
        "near_boundary_accuracy_percent": clf["constraint_accuracy_percent"],
        "near_boundary_false_feasible_percent": clf["false_feasible_percent"],
    }


def high_outreach_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


# ============================================================================
# GP MODELS
# ============================================================================

def create_gp(n_dims: int) -> GaussianProcessRegressor:
    """Create a Matern 5/2 GP with ARD length-scales."""
    kernel = (
        C(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=GP_ALPHA,
        normalize_y=False,
        n_restarts_optimizer=GP_N_RESTARTS,
        random_state=RANDOM_STATE,
    )


def train_gp_pair(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
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

        gp = create_gp(n_dims=X_train.shape[1])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(X_train_scaled, y_train_scaled)

        warnings_total += sum(
            issubclass(w.category, ConvergenceWarning)
            for w in caught_warnings
        )

        y_pred_scaled = gp.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        predictions.append(y_pred)

    elapsed = time.time() - start_time

    Y_pred = np.column_stack(predictions)

    peak_reg = regression_metrics(Y_test[:, 0], Y_pred[:, 0])
    max_reg = regression_metrics(Y_test[:, 1], Y_pred[:, 1])
    clf = constraint_metrics(Y_test[:, 1], Y_pred[:, 1])
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
        **nb,
        "training_time_s": elapsed,
        "warnings_total": warnings_total,
    }

    return metrics, Y_pred


# ============================================================================
# NN MODEL
# ============================================================================

class MultiOutputNN(nn.Module):
    """Simple feed-forward multi-output neural network."""

    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int = 2):
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_nn(
    X_train_total: np.ndarray,
    Y_train_total: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray, pd.DataFrame]:
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

    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(Y_train_scaled, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(NN_BATCH_SIZE, len(train_ds)),
        shuffle=True,
    )

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=DEVICE)
    Y_val_t = torch.tensor(Y_val_scaled, dtype=torch.float32, device=DEVICE)

    model = MultiOutputNN(
        input_dim=X_train.shape[1],
        hidden_layers=NN_HIDDEN_LAYERS,
        output_dim=2,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_LR,
        weight_decay=NN_WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history_rows = []

    for epoch in range(1, NN_MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
            val_mse_per_output = torch.mean((val_pred - Y_val_t) ** 2, dim=0).cpu().numpy()

        train_loss = float(np.mean(train_losses))

        history_rows.append({
            "epoch": epoch,
            "seed": seed,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_peak_y_loss": float(val_mse_per_output[0]),
            "val_max_abs_xr_loss": float(val_mse_per_output[1]),
        })

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= NN_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
        Y_pred_scaled = model(X_test_t).cpu().numpy()

    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    elapsed = time.time() - start_time

    peak_reg = regression_metrics(Y_test[:, 0], Y_pred[:, 0])
    max_reg = regression_metrics(Y_test[:, 1], Y_pred[:, 1])
    clf = constraint_metrics(Y_test[:, 1], Y_pred[:, 1])
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
        **nb,
        "training_time_s": elapsed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "seed": seed,
    }

    return metrics, Y_pred, pd.DataFrame(history_rows)


def train_nn_multi_seed(
    X_train_total: np.ndarray,
    Y_train_total: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, pd.DataFrame]:
    """Train NN over one or more seeds and aggregate metrics."""
    seed_metrics = []
    seed_predictions = []
    histories = []

    for seed in NN_SEEDS:
        metrics, pred, history = train_one_nn(
            X_train_total=X_train_total,
            Y_train_total=Y_train_total,
            X_test=X_test,
            Y_test=Y_test,
            seed=seed,
        )
        seed_metrics.append(metrics)
        seed_predictions.append(pred)
        histories.append(history)

    metrics_df = pd.DataFrame(seed_metrics)

    # Average predictions over seeds. With NN_SEEDS=[42], this is just the single model.
    Y_pred_mean = np.mean(seed_predictions, axis=0)

    # Recompute metrics on averaged predictions.
    peak_reg = regression_metrics(Y_test[:, 0], Y_pred_mean[:, 0])
    max_reg = regression_metrics(Y_test[:, 1], Y_pred_mean[:, 1])
    clf = constraint_metrics(Y_test[:, 1], Y_pred_mean[:, 1])
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
        **nb,
        "training_time_s": float(metrics_df["training_time_s"].sum()),
        "best_epoch_mean": float(metrics_df["best_epoch"].mean()),
        "best_val_loss_mean": float(metrics_df["best_val_loss"].mean()),
        "n_seeds": len(NN_SEEDS),
    }

    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()

    return metrics, Y_pred_mean, history_df


# ============================================================================
# LEARNING CURVE
# ============================================================================

def run_learning_curve(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
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

    valid_train_sizes = [n for n in TRAIN_SIZES if n <= len(X_train_pool)]
    if len(valid_train_sizes) == 0:
        raise ValueError("No valid training sizes. Check TRAIN_SIZES and dataset size.")

    print("\n" + "=" * 70)
    print("GP VS NN LEARNING CURVE")
    print("=" * 70)
    print(f"Device:                    {DEVICE}")
    print(f"Fixed test samples:        {len(X_test)}")
    print(f"Train pool samples:        {len(X_train_pool)}")
    print(f"Training sizes:            {valid_train_sizes}")
    print(f"GP restarts:               {GP_N_RESTARTS}")
    print(f"NN hidden layers:          {NN_HIDDEN_LAYERS}")
    print(f"NN lr / weight decay:      {NN_LR} / {NN_WEIGHT_DECAY}")
    print(f"NN seeds:                  {NN_SEEDS}")
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

        # GP pair
        print("\nTraining GP pair...")
        gp_metrics, _ = train_gp_pair(
            X_train=X_train_n,
            Y_train=Y_train_n,
            X_test=X_test,
            Y_test=Y_test,
        )
        gp_metrics["n_train"] = n_train
        all_rows.append(gp_metrics)

        print(
            f"  GP | peak_y RMSE {gp_metrics['peak_y_rmse_mm']:.2f} mm | "
            f"max_abs_xr RMSE {gp_metrics['max_abs_xr_rmse_mm']:.2f} mm | "
            f"false feasible {gp_metrics['false_feasible_percent']:.2f}% | "
            f"time {gp_metrics['training_time_s']:.1f} s"
        )

        # NN multi-output
        print("\nTraining NN multi-output...")
        nn_metrics, _, history_df = train_nn_multi_seed(
            X_train_total=X_train_n,
            Y_train_total=Y_train_n,
            X_test=X_test,
            Y_test=Y_test,
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
            f"time {nn_metrics['training_time_s']:.1f} s"
        )

        elapsed = time.time() - global_start
        print(f"\nElapsed total: {elapsed / 60:.1f} min")

        # Checkpoint after each training size.
        checkpoint_df = pd.DataFrame(all_rows)
        checkpoint_df.to_csv(RESULTS_DIR / "gp_vs_nn_learning_curve_checkpoint.csv", index=False)

    results_df = pd.DataFrame(all_rows)

    if all_histories:
        histories_df = pd.concat(all_histories, ignore_index=True)
        histories_df.to_csv(RESULTS_DIR / "gp_vs_nn_learning_curve_nn_histories.csv", index=False)

    # Save fixed test-set metadata.
    test_df = pd.DataFrame(Y_test, columns=["true_peak_y", "true_max_abs_xr"])
    test_df.to_csv(RESULTS_DIR / "gp_vs_nn_learning_curve_fixed_test_targets.csv", index=False)

    return results_df


# ============================================================================
# REPORT TABLES
# ============================================================================

def make_report_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact report-friendly table."""
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
        "near_boundary_rmse_mm",
        "near_boundary_false_feasible_percent",
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
        "near_boundary_rmse_mm": "Near-boundary RMSE [mm]",
        "near_boundary_false_feasible_percent": "Near-boundary false feasible [%]",
        "training_time_s": "Training time [s]",
    }

    report = report.rename(columns=rename)

    numeric_cols = report.select_dtypes(include=[np.number]).columns
    report[numeric_cols] = report[numeric_cols].round(3)

    return report


# ============================================================================
# PLOTS
# ============================================================================

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

    plt.tight_layout()

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def generate_plots(results_df: pd.DataFrame) -> None:
    """Generate all learning curve figures."""
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

    # R2 figure with two axes.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, metric_col, title in [
        (axes[0], "peak_y_r2", "peak_y R²"),
        (axes[1], "max_abs_xr_r2", "max_abs_xr R²"),
    ]:
        for model_name in ["GP pair", "NN multi-output"]:
            sub = results_df[results_df["model"] == model_name].sort_values("n_train")
            ax.plot(sub["n_train"], sub[metric_col], marker="o", linewidth=2, label=model_name)

        ax.set_xlabel("Training samples")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Learning Curve: Explained Variance")
    plt.tight_layout()

    path = FIGURES_DIR / "gp_vs_nn_r2.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    plot_metric_line(
        results_df,
        metric_col="false_feasible_percent",
        ylabel="False feasible rate [%]",
        title="Learning Curve: Constraint Safety Error",
        filename="gp_vs_nn_constraint_safety.png",
    )

    plot_metric_line(
        results_df,
        metric_col="training_time_s",
        ylabel="Training time [s]",
        title="Training Time vs Dataset Size",
        filename="gp_vs_nn_training_time.png",
    )

    # Summary 2x2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    plot_specs = [
        (axes[0, 0], "peak_y_rmse_mm", "peak_y RMSE [mm]", "Outreach prediction"),
        (axes[0, 1], "max_abs_xr_rmse_mm", "max_abs_xr RMSE [mm]", "Constraint prediction"),
        (axes[1, 0], "false_feasible_percent", "False feasible [%]", "Constraint safety"),
        (axes[1, 1], "training_time_s", "Training time [s]", "Training cost"),
    ]

    for ax, metric_col, ylabel, title in plot_specs:
        for model_name in ["GP pair", "NN multi-output"]:
            sub = results_df[results_df["model"] == model_name].sort_values("n_train")
            ax.plot(sub["n_train"], sub[metric_col], marker="o", linewidth=2, label=model_name)

        ax.set_xlabel("Training samples")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("GP vs NN Learning Curve Summary", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / "gp_vs_nn_learning_curve_summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# ============================================================================
# INTERPRETATION
# ============================================================================

def save_interpretation(results_df: pd.DataFrame) -> Path:
    """Save short interpretation text."""
    lines = []
    lines.append("GP VS NN LEARNING CURVE INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: {DATA_PATH}")
    lines.append(f"Fixed test fraction: {TEST_SIZE}")
    lines.append(f"Training sizes: {sorted(results_df['n_train'].unique().tolist())}")
    lines.append("")

    largest_n = int(results_df["n_train"].max())
    final_gp = results_df[(results_df["n_train"] == largest_n) & (results_df["model"] == "GP pair")].iloc[0]
    final_nn = results_df[(results_df["n_train"] == largest_n) & (results_df["model"] == "NN multi-output")].iloc[0]

    lines.append(f"Largest training size: {largest_n}")
    lines.append("")
    lines.append("At the largest training size:")
    lines.append(
        f"  GP peak_y RMSE: {final_gp['peak_y_rmse_mm']:.2f} mm | "
        f"NN peak_y RMSE: {final_nn['peak_y_rmse_mm']:.2f} mm"
    )
    lines.append(
        f"  GP max_abs_xr RMSE: {final_gp['max_abs_xr_rmse_mm']:.2f} mm | "
        f"NN max_abs_xr RMSE: {final_nn['max_abs_xr_rmse_mm']:.2f} mm"
    )
    lines.append(
        f"  GP false feasible: {final_gp['false_feasible_percent']:.2f}% | "
        f"NN false feasible: {final_nn['false_feasible_percent']:.2f}%"
    )
    lines.append("")

    lines.append("Interpretation guide:")
    lines.append(
        "  If GP error saturates while NN error keeps decreasing, the GP is more "
        "sample-efficient for the current dataset, while the NN may benefit more "
        "from additional simulations."
    )
    lines.append(
        "  The main fair comparison is always performed at the same training size. "
        "A comparison such as GP with 1600 samples vs NN with 3000 samples can be "
        "reported only as a practical engineering trade-off, not as a fair model comparison."
    )

    path = RESULTS_DIR / "gp_vs_nn_learning_curve_interpretation.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("LEARNING CURVE: GP PAIR VS NN MULTI-OUTPUT")
    print("=" * 70)

    X, Y, df = load_dataset()
    print_dataset_summary(df)

    results_df = run_learning_curve(X, Y)

    results_path = RESULTS_DIR / "gp_vs_nn_learning_curve.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    report_df = make_report_table(results_df)
    report_path = RESULTS_DIR / "gp_vs_nn_learning_curve_report_table.csv"
    report_df.to_csv(report_path, index=False)
    print(f"✓ Report table saved: {report_path}")

    md_path = RESULTS_DIR / "gp_vs_nn_learning_curve_report_table.md"
    try:
        markdown_text = report_df.to_markdown(index=False)
    except ImportError:
        markdown_text = report_df.to_string(index=False)

    md_path.write_text(markdown_text, encoding="utf-8")
    
    print(f"✓ Markdown report table saved: {md_path}")

    print("\nLEARNING CURVE REPORT TABLE")
    print(report_df.to_string(index=False))

    generate_plots(results_df)

    interpretation_path = save_interpretation(results_df)
    print(f"✓ Interpretation saved: {interpretation_path}")

    print("\n" + "=" * 70)
    print("LEARNING CURVE COMPLETED")
    print("=" * 70)
    print("Generated files:")
    print(f"  {results_path}")
    print(f"  {report_path}")
    print(f"  {md_path}")
    print(f"  {interpretation_path}")
    print(f"  {FIGURES_DIR / 'gp_vs_nn_peak_y_rmse.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_nn_max_abs_xr_rmse.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_nn_r2.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_nn_constraint_safety.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_nn_training_time.png'}")
    print(f"  {FIGURES_DIR / 'gp_vs_nn_learning_curve_summary.png'}")
    print("\nNext step: decide whether to keep the current dataset size or generate the final 3000-sample dataset.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
