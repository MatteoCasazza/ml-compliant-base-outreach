"""
dataset.py
==========

Dataset generation for ML surrogate training using Latin Hypercube Sampling.

Workflow
--------
1. Define the parameter ranges.
2. Sample the parameter space using Latin Hypercube Sampling.
3. Simulate the dynamic system for each sampled point.
4. Save the dataset to CSV.
5. Generate summary statistics and diagnostic plots.

Main supervised-learning targets
--------------------------------
peak_y      : maximum total outreach reached during the simulation.
max_abs_xr  : maximum absolute robot relative displacement, used for constraints.

The generated dataset supports:
- Gaussian Process surrogate modeling;
- Neural Network surrogate modeling;
- constraint-aware inverse optimization;
- sensitivity and robustness analysis.


Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import qmc

from dynamics import simulate_system


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset"
FIGURES_DIR = PROJECT_ROOT / "figures" / "dataset"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_N_SAMPLES = 2000
DEFAULT_T_SIM = 60.0
DEFAULT_DT = 0.001
DEFAULT_X_R_MAX = 0.500
DEFAULT_N_JOBS = -1
DEFAULT_SEED = 42
HIGH_OUTREACH_THRESHOLD = 0.600

DATASET_FILENAME = "dataset_outreach.csv"
SUMMARY_FILENAME = "dataset_summary_uniform.csv"


# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ParameterRanges:
    """
    Parameter ranges used to generate the simulation dataset.

    Input vector:
        X = [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]

    where:
    - Kb, Mb, hb describe the passive compliant base;
    - Kr, hr describe the robot impedance behavior;
    - f0, f1, A, x_r_start describe the chirp excitation.
    """

    # Passive base parameters
    Kb_min: float = 100.0
    Kb_max: float = 1500.0

    Mb_min: float = 10.0
    Mb_max: float = 80.0

    hb_min: float = 0.05
    hb_max: float = 0.25

    # Robot impedance parameters
    Kr_min: float = 500.0
    Kr_max: float = 5000.0

    hr_min: float = 0.10
    hr_max: float = 0.80

    # Chirp excitation parameters
    f0_min: float = 0.05
    f0_max: float = 0.50

    f1_min: float = 1.00
    f1_max: float = 8.00

    A_min: float = 0.03
    A_max: float = 0.12

    x_r_start_min: float = 0.35
    x_r_start_max: float = 0.45

    # Fixed robot mass used by dynamics.py when not explicitly passed.
    Mr: float = 10.0

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds in PARAM_NAMES order."""
        lower_bounds = np.array(
            [
                self.Kb_min,
                self.Kr_min,
                self.Mb_min,
                self.hb_min,
                self.hr_min,
                self.f0_min,
                self.f1_min,
                self.A_min,
                self.x_r_start_min,
            ],
            dtype=float,
        )

        upper_bounds = np.array(
            [
                self.Kb_max,
                self.Kr_max,
                self.Mb_max,
                self.hb_max,
                self.hr_max,
                self.f0_max,
                self.f1_max,
                self.A_max,
                self.x_r_start_max,
            ],
            dtype=float,
        )

        return lower_bounds, upper_bounds

    @staticmethod
    def get_param_names() -> list[str]:
        """Return parameter names in dataset input order."""
        return ["Kb", "Kr", "Mb", "hb", "hr", "f0", "f1", "A", "x_r_start"]


PARAM_NAMES = ParameterRanges.get_param_names()

METRIC_COLUMNS = [
    "peak_y",
    "t_peak",
    "final_y",
    "max_xr",
    "min_xr",
    "max_abs_xr",
    "max_xb",
    "min_xb",
    "max_abs_xb",
    "extra_reach",
    "constraint_violation",
    "constraint_violation_abs",
    "feasible",
    "feasible_abs",
]

BOOLEAN_COLUMNS = ["feasible", "feasible_abs"]


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def clean_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert feasibility columns to boolean dtype after CSV round-trips."""
    out = df.copy()

    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"])
            else:
                out[col] = out[col].astype(bool)

    return out


# =============================================================================
# LATIN HYPERCUBE SAMPLING
# =============================================================================

def generate_lhs_samples(
    n_samples: int,
    param_ranges: ParameterRanges,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate parameter samples using Latin Hypercube Sampling.

    Latin Hypercube Sampling is used to obtain a space-filling design over the
    selected parameter ranges.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    lower_bounds, upper_bounds = param_ranges.get_bounds()
    n_dimensions = len(lower_bounds)

    print(f"Generating {n_samples} LHS samples in a {n_dimensions}D space...")

    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    samples_unit = sampler.random(n=n_samples)
    samples_physical = qmc.scale(samples_unit, lower_bounds, upper_bounds)

    print(f"Samples generated: shape {samples_physical.shape}")

    return samples_physical


# =============================================================================
# DATASET SIMULATION
# =============================================================================

def nan_metrics() -> dict[str, Any]:
    """Return NaN/default metrics for a failed simulation."""
    metrics = {col: np.nan for col in METRIC_COLUMNS}
    metrics["feasible"] = False
    metrics["feasible_abs"] = False
    return metrics


def simulate_parameter_set(
    params_array: np.ndarray,
    idx: int,
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    x_r_max: float = DEFAULT_X_R_MAX,
) -> tuple[int, dict[str, Any]]:
    """
    Simulate the system for one parameter set.

    This function is called by joblib, so it must remain top-level.
    """
    try:
        _, metrics = simulate_system(
            params_array,
            T_sim=T_sim,
            dt=dt,
            return_metrics=True,
            x_r_max=x_r_max,
        )

        row_metrics = nan_metrics()

        for col in METRIC_COLUMNS:
            if col in metrics:
                row_metrics[col] = metrics[col]

        return idx, row_metrics

    except Exception as exc:
        print(f"Simulation #{idx} failed: {exc}")
        return idx, nan_metrics()


def generate_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    param_ranges: ParameterRanges | None = None,
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    x_r_max: float = DEFAULT_X_R_MAX,
    n_jobs: int = DEFAULT_N_JOBS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate the complete uniform simulation dataset.

    The output DataFrame contains:
    - input parameters;
    - physical simulation metrics;
    - dataset_type = "uniform".
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if T_sim <= 0:
        raise ValueError("T_sim must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if x_r_max <= 0:
        raise ValueError("x_r_max must be positive.")

    if param_ranges is None:
        param_ranges = ParameterRanges()

    print("=" * 70)
    print("UNIFORM DATASET GENERATION")
    print("=" * 70)
    print(f"Number of samples:  {n_samples}")
    print(f"Simulation time:    {T_sim} s")
    print(f"Time step:          {dt} s")
    print(f"Robot limit:        {x_r_max} m")
    print(f"Parallel jobs:      {n_jobs}")
    print(f"Random seed:        {seed}")
    print("=" * 70)

    samples = generate_lhs_samples(
        n_samples=n_samples,
        param_ranges=param_ranges,
        seed=seed,
    )

    print(f"\nRunning {n_samples} simulations...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(simulate_parameter_set)(
            samples[i],
            i,
            T_sim,
            dt,
            x_r_max,
        )
        for i in range(n_samples)
    )

    elapsed_time = time.time() - start_time

    df = pd.DataFrame(samples, columns=PARAM_NAMES)

    for col in METRIC_COLUMNS:
        if col in BOOLEAN_COLUMNS:
            df[col] = False
        else:
            df[col] = np.nan

    df["dataset_type"] = "uniform"

    for idx, metrics in results:
        for col in METRIC_COLUMNS:
            if col in BOOLEAN_COLUMNS:
                df.loc[idx, col] = bool(metrics.get(col, False))
            else:
                df.loc[idx, col] = metrics.get(col, np.nan)

    n_failed = int(df["peak_y"].isna().sum())

    if n_failed > 0:
        print(f"\nWarning: {n_failed}/{n_samples} simulations failed and will be removed.")
        df = df.dropna(subset=["peak_y"]).reset_index(drop=True)

    df = clean_boolean_columns(df)

    print(f"\nDataset generated in {elapsed_time:.1f} s")
    print(f"Valid samples: {len(df)}/{n_samples}")
    print(f"Average time/sample: {elapsed_time / n_samples:.3f} s")

    return df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_dataset_summary(
    df: pd.DataFrame,
    x_r_max: float = DEFAULT_X_R_MAX,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> pd.DataFrame:
    """Compute compact dataset statistics for reporting."""
    require_columns(df, ["peak_y", "extra_reach"], "Dataset")

    if "feasible_abs" in df.columns:
        feasible_abs = df[df["feasible_abs"].astype(bool)].copy()
    else:
        feasible_abs = df.iloc[0:0].copy()

    extra_abs = feasible_abs[feasible_abs["extra_reach"] > 0.0]
    high_abs = feasible_abs[feasible_abs["peak_y"] > high_outreach_threshold]

    summary: dict[str, Any] = {
        "samples": int(len(df)),
        "peak_y_mean_m": float(df["peak_y"].mean()),
        "peak_y_std_m": float(df["peak_y"].std()),
        "peak_y_min_m": float(df["peak_y"].min()),
        "peak_y_max_m": float(df["peak_y"].max()),
        "feasible_abs_samples": int(len(feasible_abs)),
        "feasible_abs_rate_percent": (
            float(100.0 * len(feasible_abs) / len(df)) if len(df) else np.nan
        ),
        "feasible_abs_extra_reach_samples": int(len(extra_abs)),
        "feasible_abs_high_outreach_samples": int(len(high_abs)),
        "max_feasible_abs_peak_y_m": (
            float(feasible_abs["peak_y"].max()) if len(feasible_abs) else np.nan
        ),
        "max_feasible_abs_extra_reach_m": (
            float(extra_abs["extra_reach"].max()) if len(extra_abs) else np.nan
        ),
        "x_r_max_m": float(x_r_max),
        "high_outreach_threshold_m": float(high_outreach_threshold),
    }

    if "constraint_violation_abs" in df.columns:
        summary["violation_abs_rate_percent"] = float(
            100.0 * (df["constraint_violation_abs"] > 0.0).mean()
        )
    else:
        summary["violation_abs_rate_percent"] = np.nan

    # Kept for backward compatibility with older one-sided constraint analyses.
    if "feasible" in df.columns:
        feasible = df[df["feasible"].astype(bool)]
        summary["feasible_samples_one_sided"] = int(len(feasible))

    if "constraint_violation" in df.columns:
        summary["violation_rate_one_sided_percent"] = float(
            100.0 * (df["constraint_violation"] > 0.0).mean()
        )

    return pd.DataFrame([summary])


def print_dataset_summary(df: pd.DataFrame, x_r_max: float = DEFAULT_X_R_MAX) -> None:
    """Print main dataset statistics to the terminal."""
    summary = compute_dataset_summary(df, x_r_max=x_r_max).iloc[0]

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Samples:                          {int(summary['samples'])}")
    print(f"Peak_y mean:                      {summary['peak_y_mean_m']:.6f} m")
    print(f"Peak_y max:                       {summary['peak_y_max_m']:.6f} m")
    print(f"Feasible_abs samples:             {int(summary['feasible_abs_samples'])}")
    print(f"Feasible_abs rate:                {summary['feasible_abs_rate_percent']:.2f}%")
    print(f"Feasible_abs extra-reach cases:   {int(summary['feasible_abs_extra_reach_samples'])}")
    print(f"Feasible_abs high-outreach cases: {int(summary['feasible_abs_high_outreach_samples'])}")
    print(f"Max feasible_abs peak_y:          {summary['max_feasible_abs_peak_y_m']:.6f} m")
    print(f"Max feasible_abs extra_reach:     {summary['max_feasible_abs_extra_reach_m']:.6f} m")
    print(f"Violation_abs rate:               {summary['violation_abs_rate_percent']:.2f}%")
    print("=" * 70)


# =============================================================================
# SAVE AND LOAD
# =============================================================================

def save_dataset(
    df: pd.DataFrame,
    filepath: Path | str = DATA_DIR / DATASET_FILENAME,
    param_ranges: ParameterRanges | None = None,
    x_r_max: float = DEFAULT_X_R_MAX,
) -> pd.DataFrame:
    """Save the dataset to CSV with a metadata header."""
    require_columns(df, PARAM_NAMES + ["peak_y"], "Dataset")

    if param_ranges is None:
        param_ranges = ParameterRanges()

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    y_valid = df["peak_y"].dropna().to_numpy(dtype=float)

    with filepath.open("w", encoding="utf-8") as file:
        file.write("# Dataset Outreach - ML Compliant Base Project\n")
        file.write(f"# Samples: {len(df)}\n")
        file.write("# Sampling: Latin Hypercube Sampling\n")
        file.write(f"# Robot maximum relative displacement x_r_max: {x_r_max:.6f} m\n")
        file.write("# Main targets: peak_y, max_abs_xr\n")
        file.write("# Main feasibility metric: feasible_abs based on |x_r(t)| <= x_r_max\n")
        file.write("# Additional metrics: t_peak, final_y, max_xb, extra_reach, constraint violations\n")

        if len(y_valid) > 0:
            file.write("# Peak_y statistics:\n")
            file.write(f"#   Mean:   {y_valid.mean():.6f} m\n")
            file.write(f"#   Std:    {y_valid.std():.6f} m\n")
            file.write(f"#   Min:    {y_valid.min():.6f} m\n")
            file.write(f"#   Max:    {y_valid.max():.6f} m\n")
            file.write(f"#   Median: {np.median(y_valid):.6f} m\n")

        file.write("# Parameter ranges:\n")
        lower_bounds, upper_bounds = param_ranges.get_bounds()

        for i, name in enumerate(PARAM_NAMES):
            file.write(f"#   {name:12s}: [{lower_bounds[i]:.6f}, {upper_bounds[i]:.6f}]\n")

        file.write("#\n")

    df.to_csv(filepath, index=False, mode="a")

    print(f"\nDataset saved: {filepath}")
    print(f"Shape: {df.shape}")

    return df


def save_dataset_summary(
    df: pd.DataFrame,
    filepath: Path | str = RESULTS_DIR / SUMMARY_FILENAME,
    x_r_max: float = DEFAULT_X_R_MAX,
) -> pd.DataFrame:
    """Save compact dataset statistics for report tables."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    summary_df = compute_dataset_summary(df, x_r_max=x_r_max)
    summary_df.to_csv(filepath, index=False)

    print(f"Dataset summary saved: {filepath}")

    return summary_df


def load_dataset(
    filepath: Path | str = DATA_DIR / DATASET_FILENAME,
) -> tuple[np.ndarray, np.ndarray]:
    """Load input matrix X and main target y = peak_y."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath, comment="#")
    require_columns(df, PARAM_NAMES + ["peak_y"], "Dataset")

    X = df[PARAM_NAMES].to_numpy(dtype=float)
    y = df["peak_y"].to_numpy(dtype=float)

    print(f"Dataset loaded: {filepath}")
    print(f"Samples: {len(y)}")

    return X, y


def load_dataset_dataframe(
    filepath: Path | str = DATA_DIR / DATASET_FILENAME,
) -> pd.DataFrame:
    """Load the full dataset as a DataFrame."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath, comment="#")
    df = clean_boolean_columns(df)

    print(f"Full dataset loaded: {filepath}")
    print(f"Shape: {df.shape}")

    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_peak_y_distribution(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot the distribution of peak_y."""
    y = df["peak_y"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(y.mean(), linestyle="--", linewidth=2, label=f"Mean = {y.mean():.3f} m")
    ax.axvline(np.median(y), linestyle="--", linewidth=2, label=f"Median = {np.median(y):.3f} m")
    ax.axvline(DEFAULT_X_R_MAX, linestyle=":", linewidth=2, label="Nominal reach = 0.5 m")

    ax.set_xlabel("Peak outreach [m]")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Peak outreach distribution (n={len(y)})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = save_dir / "dataset_peak_y_distribution.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_feasibility_map(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot peak_y against max_abs_xr to visualize the feasible region."""
    if "max_abs_xr" not in df.columns:
        return

    feasible_abs = (
        df["feasible_abs"].astype(bool)
        if "feasible_abs" in df.columns
        else df["max_abs_xr"] <= DEFAULT_X_R_MAX
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df.loc[~feasible_abs, "max_abs_xr"],
        df.loc[~feasible_abs, "peak_y"],
        alpha=0.5,
        s=35,
        label="Infeasible",
    )

    ax.scatter(
        df.loc[feasible_abs, "max_abs_xr"],
        df.loc[feasible_abs, "peak_y"],
        alpha=0.7,
        s=35,
        label="Feasible",
    )

    ax.axvline(DEFAULT_X_R_MAX, linestyle="--", linewidth=2, label=r"$|x_r|$ limit = 0.5 m")
    ax.axhline(DEFAULT_X_R_MAX, linestyle=":", linewidth=2, label="Nominal reach = 0.5 m")
    ax.axhline(HIGH_OUTREACH_THRESHOLD, linestyle="-.", linewidth=2, label="High outreach = 0.60 m")

    ax.set_xlabel(r"Maximum absolute robot displacement $|x_r|$ [m]")
    ax.set_ylabel("Peak outreach [m]")
    ax.set_title("Feasibility map: peak outreach vs robot displacement", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = save_dir / "dataset_peak_y_vs_max_abs_xr.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_correlation_matrix(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot the correlation matrix of parameters and numerical metrics."""
    numeric_metric_cols = [
        col
        for col in METRIC_COLUMNS
        if col in df.columns and col not in BOOLEAN_COLUMNS
    ]

    corr_cols = [col for col in PARAM_NAMES + numeric_metric_cols if col in df.columns]

    if len(corr_cols) < 2:
        return

    corr = df[corr_cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title(
        "Correlation matrix: parameters and simulation metrics",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    filepath = save_dir / "dataset_correlations.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_constraint_violation(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot the distribution of absolute constraint violation."""
    if "constraint_violation_abs" not in df.columns:
        return

    violation = df["constraint_violation_abs"].to_numpy(dtype=float)
    violation_rate = 100.0 * np.mean(violation > 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(violation, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0.0, linestyle="--", linewidth=2, label="No violation")

    ax.set_xlabel("Absolute constraint violation [m]")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Robot limit violation distribution ({violation_rate:.1f}% violating samples)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = save_dir / "dataset_constraint_violation_abs.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_key_parameters(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot selected input parameters against peak_y."""
    key_params = [param for param in ["A", "Kr", "f1", "Kb"] if param in df.columns]

    if not key_params:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for i, param_name in enumerate(key_params):
        ax = axes[i]

        scatter = ax.scatter(
            df[param_name],
            df["peak_y"],
            c=df["peak_y"],
            cmap="viridis",
            alpha=0.6,
            s=45,
            edgecolors="black",
            linewidth=0.4,
        )

        ax.set_xlabel(param_name, fontweight="bold")
        ax.set_ylabel("Peak outreach [m]")
        ax.set_title(f"{param_name} vs peak outreach")
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Peak outreach [m]")

    for j in range(len(key_params), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Key parameters vs peak outreach", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    filepath = save_dir / "dataset_key_parameters.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


def plot_dataset_stats(
    df: pd.DataFrame,
    save_dir: Path | str = FIGURES_DIR,
) -> None:
    """
    Generate statistical plots for the dataset.

    Generated figures:
    - dataset_peak_y_distribution.png
    - dataset_peak_y_vs_max_abs_xr.png
    - dataset_correlations.png
    - dataset_constraint_violation_abs.png
    - dataset_key_parameters.png
    """
    require_columns(df, ["peak_y"], "Dataset")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_peak_y_distribution(df, save_dir)
    plot_feasibility_map(df, save_dir)
    plot_correlation_matrix(df, save_dir)
    plot_constraint_violation(df, save_dir)
    plot_key_parameters(df, save_dir)

    print(f"\nDataset plots completed in: {save_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the uniform simulation dataset using Latin Hypercube Sampling."
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of LHS samples. Default: {DEFAULT_N_SAMPLES}.",
    )

    parser.add_argument(
        "--T_sim",
        type=float,
        default=DEFAULT_T_SIM,
        help=f"Simulation duration in seconds. Default: {DEFAULT_T_SIM}.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"Time step for saved simulation output. Default: {DEFAULT_DT}.",
    )

    parser.add_argument(
        "--x_r_max",
        type=float,
        default=DEFAULT_X_R_MAX,
        help=f"Robot relative-displacement limit. Default: {DEFAULT_X_R_MAX}.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel jobs. Use -1 for all cores. Default: {DEFAULT_N_JOBS}.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}.",
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip dataset diagnostic plots.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate, save, and validate the uniform dataset."""
    args = parse_args()
    ensure_dirs()

    print("\n" + "=" * 70)
    print("RUN: src/dataset.py")
    print("=" * 70 + "\n")

    param_ranges = ParameterRanges()

    df_dataset = generate_dataset(
        n_samples=args.n_samples,
        param_ranges=param_ranges,
        T_sim=args.T_sim,
        dt=args.dt,
        x_r_max=args.x_r_max,
        n_jobs=args.n_jobs,
        seed=args.seed,
        verbose=True,
    )

    print_dataset_summary(df_dataset, x_r_max=args.x_r_max)

    dataset_path = DATA_DIR / DATASET_FILENAME
    summary_path = RESULTS_DIR / SUMMARY_FILENAME

    save_dataset(
        df_dataset,
        filepath=dataset_path,
        param_ranges=param_ranges,
        x_r_max=args.x_r_max,
    )

    save_dataset_summary(
        df_dataset,
        filepath=summary_path,
        x_r_max=args.x_r_max,
    )

    if not args.no_plots:
        print("\nGenerating dataset plots...")
        plot_dataset_stats(df_dataset, save_dir=FIGURES_DIR)

    print("\nTesting dataset reload...")
    X_loaded, y_loaded = load_dataset(dataset_path)

    if not np.allclose(df_dataset[PARAM_NAMES].to_numpy(dtype=float), X_loaded):
        raise RuntimeError("Reload check failed: input matrix mismatch.")

    if not np.allclose(df_dataset["peak_y"].to_numpy(dtype=float), y_loaded):
        raise RuntimeError("Reload check failed: peak_y mismatch.")

    print("Reload OK: data are identical.")

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Generated files:")
    print(f"  - {dataset_path}")
    print(f"  - {summary_path}")

    if not args.no_plots:
        print(f"  - {FIGURES_DIR}/dataset_*.png")

    print("\nNext step:")
    print("  Run src/augment_high_outreach.py if you want to create the augmented dataset.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()