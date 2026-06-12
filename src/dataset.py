"""
dataset.py
==========
Dataset generation for ML training using Latin Hypercube Sampling.

Workflow:
1. Define the parameter ranges.
2. Sample the parameter space using Latin Hypercube Sampling.
3. Simulate the dynamic system for each sampled point.
4. Save the dataset to CSV.
5. Generate basic statistics and plots.

This version is prepared for the updated constraint-aware pipeline.

Main supervised-learning targets:
- peak_y: maximum total outreach reached during the simulation
- max_xr / max_abs_xr: robot relative displacement metrics for constraints

Additional physical metrics are saved to support:
- Gaussian Process surrogate modeling
- Neural Network forward surrogate modeling
- constraint-aware inverse optimization
- sensitivity and robustness analysis

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import qmc

from dynamics import simulate_system


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset"
FIGURES_DIR = PROJECT_ROOT / "figures" / "dataset"


# ============================================================================
# PARAMETER CONFIGURATION
# ============================================================================

@dataclass
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

    hr_min: float = 0.1
    hr_max: float = 0.8

    # Chirp excitation parameters
    f0_min: float = 0.05
    f0_max: float = 0.5

    f1_min: float = 1.0
    f1_max: float = 8.0

    A_min: float = 0.03
    A_max: float = 0.12

    x_r_start_min: float = 0.35
    x_r_start_max: float = 0.45

    # Fixed robot mass used by dynamics.py if not passed explicitly
    Mr: float = 10.0

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds as arrays in PARAM_NAMES order."""
        lb = np.array([
            self.Kb_min,
            self.Kr_min,
            self.Mb_min,
            self.hb_min,
            self.hr_min,
            self.f0_min,
            self.f1_min,
            self.A_min,
            self.x_r_start_min,
        ], dtype=float)

        ub = np.array([
            self.Kb_max,
            self.Kr_max,
            self.Mb_max,
            self.hb_max,
            self.hr_max,
            self.f0_max,
            self.f1_max,
            self.A_max,
            self.x_r_start_max,
        ], dtype=float)

        return lb, ub

    @staticmethod
    def get_param_names() -> list[str]:
        """Return parameter names in the same order used by the dataset."""
        return ["Kb", "Kr", "Mb", "hb", "hr", "f0", "f1", "A", "x_r_start"]


PARAM_NAMES = ParameterRanges.get_param_names()

# Metrics expected from the updated dynamics.py.
# feasible_abs and constraint_violation_abs are the main constraint indicators
# for the new pipeline because they correspond to |x_r(t)| <= x_r_max.
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


# ============================================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================================

def generate_lhs_samples(
    n_samples: int,
    param_ranges: ParameterRanges,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate parameter samples using Latin Hypercube Sampling.

    LHS is used to obtain a space-filling design in the parameter space.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    lb, ub = param_ranges.get_bounds()
    n_dims = len(lb)

    print(f"Generating {n_samples} LHS samples in a {n_dims}D space...")

    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples_normalized = sampler.random(n=n_samples)
    samples = qmc.scale(samples_normalized, lb, ub)

    print(f"✓ Samples generated: shape {samples.shape}")
    return samples


# ============================================================================
# DATASET SIMULATION
# ============================================================================

def _nan_metrics() -> dict:
    """Return a dictionary with NaN/default values for failed simulations."""
    values = {col: np.nan for col in METRIC_COLUMNS}
    values["feasible"] = False
    values["feasible_abs"] = False
    return values


def simulate_parameter_set(
    params_array: np.ndarray,
    idx: int,
    T_sim: float = 60.0,
    dt: float = 0.001,
    x_r_max: float = 0.5,
) -> Tuple[int, dict]:
    """
    Simulate the system for one parameter set.

    This function is called in parallel by joblib and must remain top-level.
    """
    try:
        _, metrics = simulate_system(
            params_array,
            T_sim=T_sim,
            dt=dt,
            return_metrics=True,
            x_r_max=x_r_max,
        )

        row_metrics = _nan_metrics()
        for col in METRIC_COLUMNS:
            if col in metrics:
                row_metrics[col] = metrics[col]

        return idx, row_metrics

    except Exception as exc:
        print(f"⚠️  Simulation #{idx} failed: {exc}")
        return idx, _nan_metrics()


def generate_dataset(
    n_samples: int = 1000,
    param_ranges: Optional[ParameterRanges] = None,
    T_sim: float = 60.0,
    dt: float = 0.001,
    x_r_max: float = 0.5,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate the complete uniform simulation dataset.

    Output DataFrame contains:
    - input parameters;
    - physical metrics;
    - dataset_type = "uniform".
    """
    if param_ranges is None:
        param_ranges = ParameterRanges()

    print("=" * 70)
    print("UNIFORM DATASET GENERATION")
    print("=" * 70)
    print(f"Number of samples:  {n_samples}")
    print(f"Simulation time:    {T_sim} s")
    print(f"Time step:          {dt} s")
    print(f"Robot max position: {x_r_max} m")
    print(f"Parallel jobs:      {n_jobs}")
    print("=" * 70)

    X = generate_lhs_samples(n_samples, param_ranges, seed=seed)

    print(f"\nRunning {n_samples} simulations in parallel...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(simulate_parameter_set)(X[i], i, T_sim, dt, x_r_max)
        for i in range(n_samples)
    )

    elapsed = time.time() - start_time

    df = pd.DataFrame(X, columns=PARAM_NAMES)

    metrics_columns = [
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

        "dataset_type",
    ]
    
    # Initialize metric columns with appropriate dtypes
    for col in metrics_columns:
        if col in ["feasible", "feasible_abs"]:
            df[col] = False
        elif col == "dataset_type":
            df[col] = "uniform"
        else:
            df[col] = np.nan

    # Fill metric values
    for idx, metrics in results:
        for col in metrics_columns:
            if col == "dataset_type":
                df.loc[idx, col] = "uniform"
            elif col in ["feasible", "feasible_abs"]:
                df.loc[idx, col] = bool(metrics.get(col, False))
            else:
                df.loc[idx, col] = metrics.get(col, np.nan)

    df["dataset_type"] = "uniform"

    n_failed = df["peak_y"].isna().sum()
    if n_failed > 0:
        print(f"\n⚠️  {n_failed}/{n_samples} simulations failed and will be removed.")
        df = df.dropna(subset=["peak_y"]).reset_index(drop=True)

    # Ensure boolean columns are clean after CSV round-trips.
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].astype(bool)
    if "feasible_abs" in df.columns:
        df["feasible_abs"] = df["feasible_abs"].astype(bool)

    print(f"\n✓ Dataset generated in {elapsed:.1f} s")
    print(f"  Valid samples: {len(df)}/{n_samples}")
    print(f"  Average time/sample: {elapsed / n_samples:.3f} s")

    return df


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_dataset_summary(
    df: pd.DataFrame,
    x_r_max: float = 0.5,
    high_outreach_threshold: float = 0.60,
) -> pd.DataFrame:
    """Compute compact dataset statistics for reporting."""
    feasible_abs = df[df["feasible_abs"]] if "feasible_abs" in df else df.iloc[0:0]
    extra_abs = feasible_abs[feasible_abs["extra_reach"] > 0.0]
    high_abs = feasible_abs[feasible_abs["peak_y"] > high_outreach_threshold]

    summary = {
        "samples": len(df),
        "peak_y_mean_m": df["peak_y"].mean(),
        "peak_y_std_m": df["peak_y"].std(),
        "peak_y_min_m": df["peak_y"].min(),
        "peak_y_max_m": df["peak_y"].max(),
        "feasible_abs_samples": len(feasible_abs),
        "feasible_abs_rate_percent": 100.0 * len(feasible_abs) / len(df) if len(df) else np.nan,
        "feasible_abs_extra_reach_samples": len(extra_abs),
        "feasible_abs_high_outreach_samples": len(high_abs),
        "max_feasible_abs_peak_y_m": feasible_abs["peak_y"].max() if len(feasible_abs) else np.nan,
        "max_feasible_abs_extra_reach_m": extra_abs["extra_reach"].max() if len(extra_abs) else np.nan,
        "violation_abs_rate_percent": 100.0 * (df["constraint_violation_abs"] > 0.0).mean()
        if "constraint_violation_abs" in df else np.nan,
        "x_r_max_m": x_r_max,
        "high_outreach_threshold_m": high_outreach_threshold,
    }

    # Keep old one-sided constraint statistics too, for backward compatibility.
    if "feasible" in df.columns:
        feasible = df[df["feasible"]]
        summary["feasible_samples_one_sided"] = len(feasible)
        summary["violation_rate_one_sided_percent"] = 100.0 * (df["constraint_violation"] > 0.0).mean()

    return pd.DataFrame([summary])


def print_dataset_summary(df: pd.DataFrame, x_r_max: float = 0.5) -> None:
    """Print main dataset statistics to terminal."""
    summary = compute_dataset_summary(df, x_r_max=x_r_max).iloc[0]

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Samples:                         {int(summary['samples'])}")
    print(f"Peak_y mean:                     {summary['peak_y_mean_m']:.6f} m")
    print(f"Peak_y max:                      {summary['peak_y_max_m']:.6f} m")
    print(f"Feasible_abs samples:            {int(summary['feasible_abs_samples'])}")
    print(f"Feasible_abs rate:               {summary['feasible_abs_rate_percent']:.2f}%")
    print(f"Feasible_abs extra-reach cases:  {int(summary['feasible_abs_extra_reach_samples'])}")
    print(f"Feasible_abs high-outreach >0.60:{int(summary['feasible_abs_high_outreach_samples'])}")
    print(f"Max feasible_abs peak_y:         {summary['max_feasible_abs_peak_y_m']:.6f} m")
    print(f"Max feasible_abs extra_reach:    {summary['max_feasible_abs_extra_reach_m']:.6f} m")
    print(f"Violation_abs rate:              {summary['violation_abs_rate_percent']:.2f}%")
    print("=" * 70)


# ============================================================================
# SAVE AND LOAD
# ============================================================================

def save_dataset(
    df: pd.DataFrame,
    filepath: Path | str = DATA_DIR / "dataset_outreach.csv",
    param_ranges: Optional[ParameterRanges] = None,
    x_r_max: float = 0.5,
) -> pd.DataFrame:
    """Save the dataset to CSV with commented metadata header."""
    if param_ranges is None:
        param_ranges = ParameterRanges()

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    y_valid = df["peak_y"].dropna().values

    with filepath.open("w", encoding="utf-8") as f:
        f.write("# Dataset Outreach - ML Compliant Base Project\n")
        f.write(f"# Samples: {len(df)}\n")
        f.write("# Sampling: Latin Hypercube\n")
        f.write(f"# Robot maximum relative position x_r_max: {x_r_max:.6f} m\n")
        f.write("# Main targets: peak_y, max_xr, max_abs_xr\n")
        f.write("# Main feasibility metric: feasible_abs based on |x_r(t)| <= x_r_max\n")
        f.write("# Additional metrics: t_peak, final_y, max_xb, extra_reach, constraint_violation_abs\n")
        f.write("# Peak y statistics:\n")
        f.write(f"#   Mean:   {y_valid.mean():.6f} m\n")
        f.write(f"#   Std:    {y_valid.std():.6f} m\n")
        f.write(f"#   Min:    {y_valid.min():.6f} m\n")
        f.write(f"#   Max:    {y_valid.max():.6f} m\n")
        f.write(f"#   Median: {np.median(y_valid):.6f} m\n")
        f.write("# Parameter ranges:\n")

        lb, ub = param_ranges.get_bounds()
        for i, name in enumerate(PARAM_NAMES):
            f.write(f"#   {name:12s}: [{lb[i]:.6f}, {ub[i]:.6f}]\n")

        f.write("#\n")

    df.to_csv(filepath, index=False, mode="a")

    print(f"\n✓ Dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    return df


def save_dataset_summary(
    df: pd.DataFrame,
    filepath: Path | str = RESULTS_DIR / "dataset_summary.csv",
    x_r_max: float = 0.5,
) -> pd.DataFrame:
    """Save compact dataset statistics for report tables."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    summary_df = compute_dataset_summary(df, x_r_max=x_r_max)
    summary_df.to_csv(filepath, index=False)

    print(f"✓ Dataset summary saved: {filepath}")
    return summary_df


def load_dataset(filepath: Path | str = DATA_DIR / "dataset_outreach.csv") -> Tuple[np.ndarray, np.ndarray]:
    """Load input matrix X and main target y = peak_y."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath, comment="#")

    X = df[PARAM_NAMES].values
    y = df["peak_y"].values

    print(f"✓ Dataset loaded: {filepath}")
    print(f"  Samples: {len(y)}")
    return X, y


def load_dataset_dataframe(filepath: Path | str = DATA_DIR / "dataset_outreach.csv") -> pd.DataFrame:
    """Load the full dataset as a DataFrame."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath, comment="#")
    print(f"✓ Full dataset loaded: {filepath}")
    print(f"  Shape: {df.shape}")
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

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
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y = df["peak_y"].values

    # Plot 1: peak_y distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(y.mean(), linestyle="--", linewidth=2, label=f"Mean = {y.mean():.3f} m")
    ax.axvline(np.median(y), linestyle="--", linewidth=2, label=f"Median = {np.median(y):.3f} m")
    ax.axvline(0.5, linestyle=":", linewidth=2, label="Nominal reach = 0.5 m")
    ax.set_xlabel("Peak Outreach [m]", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Peak Outreach Distribution (n={len(y)})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filepath = save_dir / "dataset_peak_y_distribution.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filepath}")

    # Plot 2: physical feasibility map peak_y vs max_abs_xr
    if "max_abs_xr" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        feasible_abs = df["feasible_abs"].astype(bool) if "feasible_abs" in df else df["max_abs_xr"] <= 0.5

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
            label="Feasible_abs",
        )
        ax.axvline(0.5, linestyle="--", linewidth=2, label="|x_r| limit = 0.5 m")
        ax.axhline(0.5, linestyle=":", linewidth=2, label="Nominal reach = 0.5 m")
        ax.axhline(0.60, linestyle="-.", linewidth=2, label="High outreach = 0.60 m")
        ax.set_xlabel("Maximum Absolute Robot Displacement |x_r| [m]")
        ax.set_ylabel("Peak Outreach [m]")
        ax.set_title("Feasibility Map: Peak Outreach vs Robot Displacement", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        filepath = save_dir / "dataset_peak_y_vs_max_abs_xr.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {filepath}")

    # Plot 3: correlation heatmap
    corr_cols = PARAM_NAMES + [col for col in METRIC_COLUMNS if col in df.columns and df[col].dtype != bool]
    corr_cols = [c for c in corr_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df[corr_cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix: Parameters and Simulation Metrics", fontsize=14, fontweight="bold", pad=20)
    filepath = save_dir / "dataset_correlations.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filepath}")

    # Plot 4: absolute constraint violation distribution
    if "constraint_violation_abs" in df.columns:
        violation = df["constraint_violation_abs"].values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(violation, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0.0, linestyle="--", linewidth=2, label="No violation")
        violation_rate = 100.0 * np.mean(violation > 0.0)
        ax.set_xlabel("Absolute Constraint Violation [m]")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Robot Limit Violation Distribution ({violation_rate:.1f}% violating samples)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        filepath = save_dir / "dataset_constraint_violation_abs.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {filepath}")

    # Plot 5: key parameters vs peak_y
    key_params = ["A", "Kr", "f1", "Kb"]
    key_params = [p for p in key_params if p in df.columns]

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
        ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Peak Outreach [m]", fontsize=12)
        ax.set_title(f"{param_name} vs Peak Outreach", fontsize=13)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Peak Outreach [m]", fontsize=10)

    for j in range(len(key_params), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Key Parameters vs Peak Outreach", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    filepath = save_dir / "dataset_key_parameters.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filepath}")

    print(f"\n✓ Dataset plots completed in: {save_dir}/")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST/RUN: src/dataset.py")
    print("=" * 70 + "\n")

    # For a quick test, use 50-100 samples.
    # For the final uniform dataset, use 1000 samples or more.
    N_SAMPLES = 2000
    T_SIM = 60.0
    DT = 0.001
    X_R_MAX = 0.5
    N_JOBS = -1
    SEED = 42

    param_ranges = ParameterRanges()

    df_dataset = generate_dataset(
        n_samples=N_SAMPLES,
        param_ranges=param_ranges,
        T_sim=T_SIM,
        dt=DT,
        x_r_max=X_R_MAX,
        n_jobs=N_JOBS,
        seed=SEED,
        verbose=True,
    )

    print_dataset_summary(df_dataset, x_r_max=X_R_MAX)

    save_dataset(
        df_dataset,
        filepath=DATA_DIR / "dataset_outreach.csv",
        param_ranges=param_ranges,
        x_r_max=X_R_MAX,
    )

    save_dataset_summary(
        df_dataset,
        filepath=RESULTS_DIR / "dataset_summary_uniform.csv",
        x_r_max=X_R_MAX,
    )

    print("\nGenerating plots...")
    plot_dataset_stats(df_dataset, save_dir=FIGURES_DIR)

    print("\nTesting dataset reload...")
    X_loaded, y_loaded = load_dataset(DATA_DIR / "dataset_outreach.csv")

    assert np.allclose(df_dataset[PARAM_NAMES].values, X_loaded), "❌ Error: X mismatch!"
    assert np.allclose(df_dataset["peak_y"].values, y_loaded), "❌ Error: y mismatch!"
    print("✓ Reload OK: data are identical")

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated files:")
    print(f"  - {DATA_DIR / 'dataset_outreach.csv'}")
    print(f"  - {RESULTS_DIR / 'dataset_summary_uniform.csv'}")
    print(f"  - {FIGURES_DIR}/dataset_*.png")
    print("\nNext step:")
    print("  Run src/augment_high_outreach.py after it has been updated.")
    print("=" * 70 + "\n")
