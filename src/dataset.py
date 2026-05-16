"""
dataset.py
==========
Dataset generation for ML training using Latin Hypercube Sampling.

Workflow:
1. Define the parameter ranges
2. Sample the parameter space using Latin Hypercube Sampling
3. Simulate the dynamic system for each sampled point
4. Save the dataset to CSV
5. Generate basic statistics and plots

The main supervised-learning target is:
- peak_y: maximum total outreach reached during the simulation

Additional physical metrics are also saved:
- max_xr
- max_xb
- extra_reach
- constraint_violation

Author: MatteoCasazza
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc
from joblib import Parallel, delayed
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import time

from dynamics import simulate_system


# ============================================================================
# PARAMETER CONFIGURATION
# ============================================================================

@dataclass
class ParameterRanges:
    """
    Parameter ranges used to generate the simulation dataset.

    The input vector is:
        X = [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]

    where:
    - Kb, Mb, hb describe the passive compliant base
    - Kr, hr describe the robot impedance behavior
    - f0, f1, A, x_r_start describe the chirp excitation
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

    # Fixed robot mass
    Mr: float = 10.0            # Robot mass [kg]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return lower and upper bounds as arrays.

        Outputs
        -------
        lb : ndarray, shape (9,)
            Lower bounds.
        ub : ndarray, shape (9,)
            Upper bounds.
        """
        lb = np.array([
            self.Kb_min,
            self.Kr_min,
            self.Mb_min,
            self.hb_min,
            self.hr_min,
            self.f0_min,
            self.f1_min,
            self.A_min,
            self.x_r_start_min
        ])

        ub = np.array([
            self.Kb_max,
            self.Kr_max,
            self.Mb_max,
            self.hb_max,
            self.hr_max,
            self.f0_max,
            self.f1_max,
            self.A_max,
            self.x_r_start_max
        ])

        return lb, ub

    @staticmethod
    def get_param_names() -> list:
        """
        Return parameter names in the same order used by the dataset.
        """
        return ['Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start']


# ============================================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================================

def generate_lhs_samples(
    n_samples: int,
    param_ranges: ParameterRanges,
    seed: int = 42
) -> np.ndarray:
    """
    Generate parameter samples using Latin Hypercube Sampling.

    Inputs
    ------
    n_samples : int
        Number of samples to generate.
    param_ranges : ParameterRanges
        Parameter range configuration.
    seed : int
        Random seed for reproducibility.

    Output
    ------
    samples : ndarray, shape (n_samples, 9)
        Sampled parameter matrix.

    Notes
    -----
    LHS is used to obtain a better space-filling design than pure random
    sampling, especially with a limited number of simulations.
    """
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

def simulate_parameter_set(
    params_array: np.ndarray,
    idx: int,
    T_sim: float = 60.0,
    dt: float = 0.001,
    x_r_max: float = 0.5
) -> Tuple[int, dict]:
    """
    Simulate the system for one parameter set.

    Inputs
    ------
    params_array : ndarray, shape (9,)
        Parameter vector:
            [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
    idx : int
        Sample index.
    T_sim : float
        Simulation duration [s].
    dt : float
        Time step used for storing the solution [s].
    x_r_max : float
        Maximum admissible robot relative position [m].

    Output
    ------
    idx : int
        Sample index.
    metrics : dict
        Simulation metrics returned by dynamics.compute_metrics.
        If the simulation fails, all values are set to NaN.

    Notes
    -----
    This function is called in parallel by joblib.
    It must remain top-level for serialization.
    """
    try:
        _, metrics = simulate_system(
            params_array,
            T_sim=T_sim,
            dt=dt,
            return_metrics=True,
            x_r_max=x_r_max
        )
        return idx, metrics

    except Exception as e:
        print(f"⚠️  Simulation #{idx} failed: {e}")
        return idx, {
            'peak_y': np.nan,
            'max_xr': np.nan,
            'max_xb': np.nan,
            'extra_reach': np.nan,
            'constraint_violation': np.nan
        }


def generate_dataset(
    n_samples: int = 1000,
    param_ranges: Optional[ParameterRanges] = None,
    T_sim: float = 60.0,
    dt: float = 0.001,
    x_r_max: float = 0.5,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate the complete simulation dataset.

    Inputs
    ------
    n_samples : int
        Number of parameter samples.
    param_ranges : ParameterRanges, optional
        Parameter range configuration. If None, default ranges are used.
    T_sim : float
        Simulation duration [s].
    dt : float
        Time step used for storing the solution [s].
    x_r_max : float
        Maximum admissible robot relative position [m].
    n_jobs : int
        Number of parallel jobs. Use -1 to use all available cores.
    seed : int
        Random seed.
    verbose : bool
        If True, print joblib progress.

    Output
    ------
    df : DataFrame
        Complete dataset containing input parameters and output metrics.
    """
    if param_ranges is None:
        param_ranges = ParameterRanges()

    print("=" * 70)
    print("DATASET GENERATION")
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

    param_names = ParameterRanges.get_param_names()
    df = pd.DataFrame(X, columns=param_names)

    metrics_columns = [
        'peak_y',
        'max_xr',
        'max_xb',
        'extra_reach',
        'constraint_violation'
    ]

    for col in metrics_columns:
        df[col] = np.nan

    for idx, metrics in results:
        for col in metrics_columns:
            df.loc[idx, col] = metrics.get(col, np.nan)

    n_failed = df['peak_y'].isna().sum()
    if n_failed > 0:
        print(f"\n⚠️  {n_failed}/{n_samples} simulations failed and will be removed.")
        df = df.dropna(subset=['peak_y']).reset_index(drop=True)

    print(f"\n✓ Dataset generated in {elapsed:.1f} s")
    print(f"  Valid samples: {len(df)}/{n_samples}")
    print(f"  Average time/sample: {elapsed/n_samples:.3f} s")

    return df


# ============================================================================
# SAVE AND LOAD
# ============================================================================

def save_dataset(
    df: pd.DataFrame,
    filepath: str = 'data/dataset_outreach.csv',
    param_ranges: Optional[ParameterRanges] = None,
    x_r_max: float = 0.5
) -> pd.DataFrame:
    """
    Save the dataset to a CSV file with metadata.

    Inputs
    ------
    df : DataFrame
        Dataset containing input parameters and output metrics.
    filepath : str
        Output CSV path.
    param_ranges : ParameterRanges, optional
        Parameter ranges used to write metadata.
    x_r_max : float
        Maximum admissible robot relative position [m].

    Output
    ------
    df : DataFrame
        Saved dataset.
    """
    if param_ranges is None:
        param_ranges = ParameterRanges()

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    y = df['peak_y'].values
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]

    with open(filepath, 'w') as f:
        f.write("# Dataset Outreach - ML Compliant Base Project\n")
        f.write(f"# Samples: {len(df)}\n")
        f.write("# Sampling: Latin Hypercube\n")
        f.write(f"# Robot maximum relative position x_r_max: {x_r_max:.6f} m\n")
        f.write("# Main supervised-learning target: peak_y\n")
        f.write("# Additional metrics: max_xr, max_xb, extra_reach, constraint_violation\n")
        f.write("# Peak y statistics:\n")
        f.write(f"#   Mean:   {y_valid.mean():.6f} m\n")
        f.write(f"#   Std:    {y_valid.std():.6f} m\n")
        f.write(f"#   Min:    {y_valid.min():.6f} m\n")
        f.write(f"#   Max:    {y_valid.max():.6f} m\n")
        f.write(f"#   Median: {np.median(y_valid):.6f} m\n")
        f.write("# Parameter ranges:\n")

        lb, ub = param_ranges.get_bounds()
        for i, name in enumerate(ParameterRanges.get_param_names()):
            f.write(f"#   {name:12s}: [{lb[i]:.6f}, {ub[i]:.6f}]\n")

        f.write("#\n")

    df.to_csv(filepath, index=False, mode='a')

    print(f"\n✓ Dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")

    return df


def load_dataset(filepath: str = 'data/dataset_outreach.csv') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from CSV.

    Inputs
    ------
    filepath : str
        CSV file path.

    Outputs
    -------
    X : ndarray, shape (n_samples, 9)
        Input parameter matrix.
    y : ndarray, shape (n_samples,)
        Main target vector: peak_y.
    """
    df = pd.read_csv(filepath, comment='#')

    param_names = ParameterRanges.get_param_names()
    X = df[param_names].values
    y = df['peak_y'].values

    print(f"✓ Dataset loaded: {filepath}")
    print(f"  Samples: {len(y)}")

    return X, y


def load_dataset_dataframe(filepath: str = 'data/dataset_outreach.csv') -> pd.DataFrame:
    """
    Load the full dataset as a DataFrame.

    This is useful when additional metrics are needed, for example:
    - max_xr
    - extra_reach
    - constraint_violation
    """
    df = pd.read_csv(filepath, comment='#')
    print(f"✓ Full dataset loaded: {filepath}")
    print(f"  Shape: {df.shape}")
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_dataset_stats(
    df: pd.DataFrame,
    save_dir: str = 'figures'
) -> None:
    """
    Generate statistical plots for the dataset.

    Inputs
    ------
    df : DataFrame
        Dataset containing input parameters and output metrics.
    save_dir : str
        Directory where figures are saved.

    Generated figures
    -----------------
    - dataset_distribution.png
    - dataset_correlations.png
    - dataset_scatter.png
    - dataset_constraint_violation.png
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    param_names = ParameterRanges.get_param_names()
    y = df['peak_y'].values

    # Plot 1: peak_y distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(y, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(y.mean(), linestyle='--', linewidth=2,
               label=f'Mean = {y.mean():.3f} m')
    ax.axvline(np.median(y), linestyle='--', linewidth=2,
               label=f'Median = {np.median(y):.3f} m')

    ax.set_xlabel('Peak Outreach [m]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Peak Outreach Distribution (n={len(y)})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    filepath = f'{save_dir}/dataset_distribution.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

    # Plot 2: correlation heatmap
    corr_cols = param_names + [
        'peak_y',
        'max_xr',
        'max_xb',
        'extra_reach',
        'constraint_violation'
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(13, 11))

    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    ax.set_title('Correlation Matrix: Parameters and Simulation Metrics',
                 fontsize=14, fontweight='bold', pad=20)

    filepath = f'{save_dir}/dataset_correlations.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

    # Plot 3: key parameters vs peak_y
    key_params = ['A', 'Kr', 'f1', 'Kb']
    key_indices = [param_names.index(p) for p in key_params]

    X = df[param_names].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for i, (param_idx, param_name) in enumerate(zip(key_indices, key_params)):
        ax = axes[i]

        scatter = ax.scatter(
            X[:, param_idx],
            y,
            c=y,
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Outreach [m]', fontsize=12)
        ax.set_title(f'{param_name} vs Peak Outreach', fontsize=13)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Peak Outreach [m]', fontsize=10)

    plt.suptitle('Key Parameters vs Peak Outreach',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    filepath = f'{save_dir}/dataset_scatter.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

    # Plot 4: constraint violation distribution
    if 'constraint_violation' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        violation = df['constraint_violation'].values
        ax.hist(violation, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0.0, linestyle='--', linewidth=2, label='No violation')

        violation_rate = 100.0 * np.mean(violation > 0.0)

        ax.set_xlabel('Constraint Violation [m]', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Robot Limit Violation Distribution '
                     f'({violation_rate:.1f}% violating samples)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        filepath = f'{save_dir}/dataset_constraint_violation.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()

    print(f"\n✓ Dataset plots completed in: {save_dir}/")


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST: src/dataset.py")
    print("=" * 70 + "\n")

    # For a quick test, use 50-100 samples.
    # For the final dataset, use 400-500 samples for the GP surrogate.
    N_SAMPLES = 1000

    X_R_MAX = 0.5

    param_ranges = ParameterRanges()

    df = generate_dataset(
        n_samples=N_SAMPLES,
        param_ranges=param_ranges,
        T_sim=60.0,
        dt=0.001,
        x_r_max=X_R_MAX,
        n_jobs=-1,
        seed=42,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Number of samples: {len(df)}")
    print(f"Shape:             {df.shape}")

    print("\nPeak outreach [m]:")
    print(f"  Mean:   {df['peak_y'].mean():.6f}")
    print(f"  Std:    {df['peak_y'].std():.6f}")
    print(f"  Min:    {df['peak_y'].min():.6f}")
    print(f"  Max:    {df['peak_y'].max():.6f}")
    print(f"  Median: {np.median(df['peak_y']):.6f}")
    print(f"  Q1:     {np.percentile(df['peak_y'], 25):.6f}")
    print(f"  Q3:     {np.percentile(df['peak_y'], 75):.6f}")

    print("\nConstraint violation:")
    violation_rate = 100.0 * np.mean(df['constraint_violation'] > 0.0)
    print(f"  Mean violation: {df['constraint_violation'].mean():.6f} m")
    print(f"  Max violation:  {df['constraint_violation'].max():.6f} m")
    print(f"  Violation rate: {violation_rate:.1f}%")

    df = save_dataset(
        df,
        filepath='data/dataset_outreach.csv',
        param_ranges=param_ranges,
        x_r_max=X_R_MAX
    )

    print("\nGenerating plots...")
    plot_dataset_stats(df, save_dir='figures')

    print("\nTesting dataset reload...")
    X_loaded, y_loaded = load_dataset('data/dataset_outreach.csv')

    param_names = ParameterRanges.get_param_names()
    assert np.allclose(df[param_names].values, X_loaded), "❌ Error: X mismatch!"
    assert np.allclose(df['peak_y'].values, y_loaded), "❌ Error: y mismatch!"
    print("✓ Reload OK: data are identical")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check: data/dataset_outreach.csv")
    print("  2. Check: figures/dataset_*.png")
    print("  3. Re-train the GP with: src/models.py")
    print("=" * 70 + "\n")