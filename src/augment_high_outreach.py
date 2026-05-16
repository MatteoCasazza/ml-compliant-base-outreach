"""
augment_high_outreach.py
========================
Generate additional targeted simulations in the high-outreach region.

The goal is to enrich the dataset with samples that are more likely to produce
large outreach values, especially peak_y > 0.60 m, while still keeping the robot
relative motion close to the nominal limit.

This script:
1. Generates targeted LHS samples.
2. Runs simulations.
3. Saves the new high-outreach dataset.
4. Merges it with the original dataset.
5. Saves the augmented dataset.

Author: MatteoCasazza
Date: 2026
"""

import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import qmc
from joblib import Parallel, delayed

from dynamics import simulate_system
from dataset import ParameterRanges


# ============================================================================
# TARGETED PARAMETER RANGES
# ============================================================================

@dataclass
class HighOutreachRanges:
    """
    Targeted parameter ranges for high-outreach augmentation.

    These ranges are based on physical insight and preliminary targeted search:
    - low-to-medium base stiffness;
    - low base mass;
    - low base damping;
    - medium robot damping;
    - relatively high chirp amplitude;
    - initial robot position below the nominal limit.
    """

    # Passive base parameters
    Kb_min: float = 700.0
    Kb_max: float = 1500.0

    Mb_min: float = 10.0
    Mb_max: float = 25.0

    hb_min: float = 0.05
    hb_max: float = 0.12

    # Robot impedance parameters
    Kr_min: float = 1500.0
    Kr_max: float = 5000.0

    hr_min: float = 0.10
    hr_max: float = 0.45

    # Chirp excitation parameters
    f0_min: float = 0.10
    f0_max: float = 0.45

    f1_min: float = 1.0
    f1_max: float = 4.0

    A_min: float = 0.09
    A_max: float = 0.12

    x_r_start_min: float = 0.35
    x_r_start_max: float = 0.40

    @classmethod
    def get_bounds(cls):
        """
        Return bounds in the same parameter order used by the main dataset.

        Parameter order:
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
        """
        r = cls()
        return np.array([
            [r.Kb_min, r.Kb_max],
            [r.Kr_min, r.Kr_max],
            [r.Mb_min, r.Mb_max],
            [r.hb_min, r.hb_max],
            [r.hr_min, r.hr_max],
            [r.f0_min, r.f0_max],
            [r.f1_min, r.f1_max],
            [r.A_min, r.A_max],
            [r.x_r_start_min, r.x_r_start_max],
        ])


# ============================================================================
# SAMPLING AND SIMULATION
# ============================================================================

def generate_lhs_samples(
    n_samples: int,
    random_state: int = 123
) -> np.ndarray:
    """
    Generate Latin Hypercube samples in the targeted high-outreach region.
    """
    bounds = HighOutreachRanges.get_bounds()
    n_dims = bounds.shape[0]

    sampler = qmc.LatinHypercube(d=n_dims, seed=random_state)
    unit_samples = sampler.random(n=n_samples)

    lower = bounds[:, 0]
    upper = bounds[:, 1]

    samples = qmc.scale(unit_samples, lower, upper)

    return samples


def simulate_parameter_set(
    params_array: np.ndarray,
    T_sim: float = 60.0,
    dt: float = 0.001,
    x_r_max: float = 0.5
) -> dict:
    """
    Simulate one parameter set and return inputs plus physical metrics.
    """
    param_names = ParameterRanges.get_param_names()

    try:
        _, metrics = simulate_system(
            params_array,
            T_sim=T_sim,
            dt=dt,
            return_metrics=True,
            x_r_max=x_r_max
        )

        row = {name: value for name, value in zip(param_names, params_array)}

        # Keep the same columns as the main dataset.
        row.update({
            'peak_y': metrics['peak_y'],
            'max_xr': metrics['max_xr'],
            'max_xb': metrics['max_xb'],
            'extra_reach': metrics['extra_reach'],
            'constraint_violation': metrics['constraint_violation']
        })

        return row

    except Exception as e:
        row = {name: value for name, value in zip(param_names, params_array)}
        row.update({
            'peak_y': np.nan,
            'max_xr': np.nan,
            'max_xb': np.nan,
            'extra_reach': np.nan,
            'constraint_violation': np.nan,
            'error': str(e)
        })
        return row


def generate_high_outreach_dataset(
    n_samples: int = 300,
    T_sim: float = 60.0,
    dt: float = 0.001,
    x_r_max: float = 0.5,
    n_jobs: int = -1,
    random_state: int = 123
) -> pd.DataFrame:
    """
    Generate targeted high-outreach simulations.
    """
    print("\n" + "=" * 70)
    print("HIGH-OUTREACH DATASET AUGMENTATION")
    print("=" * 70)
    print(f"Number of targeted samples: {n_samples}")
    print(f"Simulation time:            {T_sim} s")
    print(f"Time step:                  {dt} s")
    print(f"Robot max position:         {x_r_max} m")
    print(f"Parallel jobs:              {n_jobs}")
    print("=" * 70)

    print("\nGenerating targeted LHS samples...")
    samples = generate_lhs_samples(n_samples, random_state=random_state)
    print(f"✓ Samples generated: shape {samples.shape}")

    print("\nRunning simulations...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(simulate_parameter_set)(
            sample,
            T_sim=T_sim,
            dt=dt,
            x_r_max=x_r_max
        )
        for sample in samples
    )

    elapsed = time.time() - start_time

    df = pd.DataFrame(results)
    df = df.dropna(subset=['peak_y']).reset_index(drop=True)

    print(f"\n✓ Targeted dataset generated in {elapsed:.1f} s")
    print(f"  Valid simulations: {len(df)}/{n_samples}")
    print(f"  Average time/sample: {elapsed / n_samples:.3f} s")

    return df


# ============================================================================
# MERGE AND REPORT
# ============================================================================

def print_dataset_stats(
    df: pd.DataFrame,
    name: str,
    tolerance: float = 0.002
) -> None:
    """
    Print compact dataset statistics.
    """
    feasible = df[df['constraint_violation'] <= tolerance]
    good = feasible[feasible['extra_reach'] > 0.0]
    high = feasible[feasible['peak_y'] > 0.60]

    print("\n" + "=" * 70)
    print(f"{name.upper()} STATISTICS")
    print("=" * 70)
    print(f"Samples:                    {len(df)}")
    print(f"Peak_y mean:                 {df['peak_y'].mean():.6f} m")
    print(f"Peak_y max:                  {df['peak_y'].max():.6f} m")
    print(f"Feasible samples:            {len(feasible)}")
    print(f"Feasible extra-reach cases:  {len(good)}")
    print(f"Feasible high-outreach >0.60:{len(high)}")
    print(f"Max feasible peak_y:         {feasible['peak_y'].max():.6f} m")
    if len(good) > 0:
        print(f"Max feasible extra_reach:    {good['extra_reach'].max():.6f} m")
    print(f"Violation rate:              {100 * (df['constraint_violation'] > tolerance).mean():.1f}%")
    print("=" * 70)


def merge_datasets(
    base_path: str = 'data/dataset_outreach.csv',
    augmentation_path: str = 'data/dataset_high_outreach.csv',
    output_path: str = 'data/dataset_augmented.csv'
) -> pd.DataFrame:
    """
    Merge original and augmented datasets.

    Notes
    -----
    The original dataset may contain commented header lines. Therefore,
    it is loaded with comment='#'.
    """
    base_df = pd.read_csv(base_path, comment='#')
    aug_df = pd.read_csv(augmentation_path, comment='#')

    common_columns = [col for col in base_df.columns if col in aug_df.columns]

    base_df = base_df[common_columns]
    aug_df = aug_df[common_columns]

    merged_df = pd.concat([base_df, aug_df], ignore_index=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    return merged_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    N_AUGMENT = 600
    T_SIM = 60.0
    DT = 0.001
    X_R_MAX = 0.5
    TOLERANCE = 0.002
    N_JOBS = -1
    RANDOM_STATE = 123

    Path('data').mkdir(exist_ok=True)

    # 1. Generate targeted dataset
    df_aug = generate_high_outreach_dataset(
        n_samples=N_AUGMENT,
        T_sim=T_SIM,
        dt=DT,
        x_r_max=X_R_MAX,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    augmentation_path = 'data/dataset_high_outreach.csv'
    df_aug.to_csv(augmentation_path, index=False)
    print(f"\n✓ Targeted dataset saved: {augmentation_path}")

    print_dataset_stats(
        df_aug,
        name='Targeted augmentation',
        tolerance=TOLERANCE
    )

    # 2. Merge with original dataset
    output_path = 'data/dataset_augmented.csv'
    merged_df = merge_datasets(
        base_path='data/dataset_outreach.csv',
        augmentation_path=augmentation_path,
        output_path=output_path
    )

    print(f"\n✓ Augmented dataset saved: {output_path}")

    print_dataset_stats(
        merged_df,
        name='Final augmented dataset',
        tolerance=TOLERANCE
    )

    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETED")
    print("=" * 70)
    print("Generated files:")
    print("  - data/dataset_high_outreach.csv")
    print("  - data/dataset_augmented.csv")
    print("\nNext step:")
    print("  Update models.py to use data/dataset_augmented.csv")
    print("=" * 70 + "\n")