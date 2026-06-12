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
import matplotlib.pyplot as plt

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
            "peak_y": metrics.get("peak_y", np.nan),
            "t_peak": metrics.get("t_peak", np.nan),
            "final_y": metrics.get("final_y", np.nan),

            "max_xr": metrics.get("max_xr", np.nan),
            "min_xr": metrics.get("min_xr", np.nan),
            "max_abs_xr": metrics.get("max_abs_xr", np.nan),

            "max_xb": metrics.get("max_xb", np.nan),
            "min_xb": metrics.get("min_xb", np.nan),
            "max_abs_xb": metrics.get("max_abs_xb", np.nan),

            "extra_reach": metrics.get("extra_reach", np.nan),

            "constraint_violation": metrics.get("constraint_violation", np.nan),
            "constraint_violation_abs": metrics.get("constraint_violation_abs", np.nan),

            "feasible": bool(metrics.get("feasible", False)),
            "feasible_abs": bool(metrics.get("feasible_abs", False)),

            "dataset_type": "targeted_augmented",
        })

        return row

    except Exception as e:
        row = {name: value for name, value in zip(param_names, params_array)}
        row.update({
            "peak_y": np.nan,
            "t_peak": np.nan,
            "final_y": np.nan,

            "max_xr": np.nan,
            "min_xr": np.nan,
            "max_abs_xr": np.nan,

            "max_xb": np.nan,
            "min_xb": np.nan,
            "max_abs_xb": np.nan,

            "extra_reach": np.nan,

            "constraint_violation": np.nan,
            "constraint_violation_abs": np.nan,

            "feasible": False,
            "feasible_abs": False,

            "dataset_type": "targeted_augmented",
            "error": str(e),
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
    feasible = df[df["constraint_violation_abs"] <= tolerance]
    good = feasible[feasible["extra_reach"] > 0.0]
    high = feasible[feasible["peak_y"] > 0.60]

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
    print(
        f"Violation abs rate:         "
        f"{100 * (df['constraint_violation_abs'] > tolerance).mean():.1f}%"
    )
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

    missing_in_aug = [col for col in base_df.columns if col not in aug_df.columns]
    missing_in_base = [col for col in aug_df.columns if col not in base_df.columns]

    if missing_in_aug:
        print(f"Warning: columns missing in augmentation dataset: {missing_in_aug}")

    if missing_in_base:
        print(f"Warning: columns missing in base dataset: {missing_in_base}")

    base_df = base_df[common_columns]
    aug_df = aug_df[common_columns]

    merged_df = pd.concat([base_df, aug_df], ignore_index=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    return merged_df


def save_augmentation_summary(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: str = "results/augmentation/dataset_augmentation_summary.csv",
    tolerance: float = 0.002
) -> pd.DataFrame:
    """
    Save before/after augmentation statistics.
    """

    def summarize(df, name):
        feasible = df[df["constraint_violation_abs"] <= tolerance]
        feasible_extra = feasible[feasible["extra_reach"] > 0.0]
        feasible_high = feasible[feasible["peak_y"] > 0.60]

        return {
            "dataset": name,
            "samples": len(df),
            "peak_y_mean_m": df["peak_y"].mean(),
            "peak_y_max_m": df["peak_y"].max(),
            "feasible_abs_samples": len(feasible),
            "feasible_abs_extra_reach": len(feasible_extra),
            "feasible_abs_high_outreach": len(feasible_high),
            "max_feasible_abs_peak_y_m": feasible["peak_y"].max() if len(feasible) > 0 else np.nan,
            "max_feasible_abs_extra_reach_m": feasible_extra["extra_reach"].max() if len(feasible_extra) > 0 else np.nan,
            "violation_abs_rate_percent": 100.0 * (df["constraint_violation_abs"] > tolerance).mean(),
        }

    summary = pd.DataFrame([
        summarize(base_df, "uniform"),
        summarize(aug_df, "targeted_augmented"),
        summarize(merged_df, "augmented"),
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\n✓ Augmentation summary saved: {output_path}")

    return summary


def plot_augmentation_analysis(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    save_dir: str = "figures/augmentation",
    x_r_max: float = 0.5,
    high_outreach_threshold: float = 0.60,
    tolerance: float = 0.002
) -> None:
    """
    Generate plots to justify the targeted augmentation strategy.

    The plots compare the uniform dataset, the targeted augmented dataset,
    and the final merged dataset.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Plot 1: peak_y distribution before/after augmentation
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        base_df["peak_y"],
        bins=30,
        alpha=0.6,
        label="Uniform dataset",
        edgecolor="black"
    )

    plt.hist(
        merged_df["peak_y"],
        bins=30,
        alpha=0.5,
        label="Augmented dataset",
        edgecolor="black"
    )

    plt.axvline(x_r_max, linestyle="--", linewidth=2, label="Nominal reach")
    plt.axvline(high_outreach_threshold, linestyle=":", linewidth=2, label="High-outreach threshold")

    plt.xlabel("Peak outreach $peak_y$ [m]")
    plt.ylabel("Frequency")
    plt.title("Peak Outreach Distribution Before and After Augmentation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = Path(save_dir) / "augmentation_peak_y_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # ------------------------------------------------------------
    # Plot 2: peak_y vs max_abs_xr feasibility map
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 7))

    plt.scatter(
        base_df["max_abs_xr"],
        base_df["peak_y"],
        alpha=0.45,
        s=35,
        label="Uniform dataset"
    )

    plt.scatter(
        aug_df["max_abs_xr"],
        aug_df["peak_y"],
        alpha=0.65,
        s=35,
        label="Targeted augmented samples"
    )

    plt.axvline(x_r_max, linestyle="--", linewidth=2, label="$|x_r|$ limit")
    plt.axhline(x_r_max, linestyle=":", linewidth=2, label="Nominal reach")
    plt.axhline(high_outreach_threshold, linestyle="-.", linewidth=2, label="High-outreach threshold")

    plt.xlabel("Maximum absolute robot displacement $max|x_r|$ [m]")
    plt.ylabel("Peak outreach $peak_y$ [m]")
    plt.title("Feasible High-Outreach Region Before and After Augmentation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = Path(save_dir) / "augmentation_feasible_high_outreach_region.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # ------------------------------------------------------------
    # Plot 3: feasible high-outreach sample counts
    # ------------------------------------------------------------
    def count_feasible_high(df):
        feasible = df[df["constraint_violation_abs"] <= tolerance]
        high = feasible[feasible["peak_y"] > high_outreach_threshold]
        return len(high)

    counts = pd.DataFrame({
        "dataset": ["Uniform", "Targeted", "Augmented"],
        "feasible_high_outreach": [
            count_feasible_high(base_df),
            count_feasible_high(aug_df),
            count_feasible_high(merged_df),
        ]
    })

    plt.figure(figsize=(8, 6))
    plt.bar(
        counts["dataset"],
        counts["feasible_high_outreach"],
        edgecolor="black"
    )

    plt.ylabel("Number of feasible high-outreach samples")
    plt.title("Feasible High-Outreach Samples Before and After Augmentation")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = Path(save_dir) / "augmentation_feasible_high_outreach_counts.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

    # Save the count table too
    counts_path = Path(save_dir) / "augmentation_feasible_high_outreach_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"✓ Saved: {counts_path}")

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

    base_df = pd.read_csv("data/dataset_outreach.csv", comment="#")

    summary = save_augmentation_summary(
        base_df=base_df,
        aug_df=df_aug,
        merged_df=merged_df,
        output_path="results/augmentation/dataset_augmentation_summary.csv",
        tolerance=TOLERANCE
    )

    print("\nAugmentation summary:")
    print(summary.to_string(index=False))

    print("\nGenerating augmentation analysis plots...")

    plot_augmentation_analysis(
        base_df=base_df,
        aug_df=df_aug,
        merged_df=merged_df,
        save_dir="figures/augmentation",
        x_r_max=X_R_MAX,
        high_outreach_threshold=0.60,
        tolerance=TOLERANCE
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