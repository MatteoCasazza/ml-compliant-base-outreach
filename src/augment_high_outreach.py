"""
augment_high_outreach.py
========================

Generate additional targeted simulations in the high-outreach region.

Purpose
-------
The uniform dataset may contain relatively few samples with high feasible
outreach. This script enriches the training data by sampling a physically
motivated region that is more likely to produce:

    peak_y > 0.60 m

while keeping the robot relative displacement close to the admissible limit.

Workflow
--------
1. Generate targeted Latin Hypercube samples.
2. Run the true dynamic simulator.
3. Save the targeted high-outreach dataset.
4. Merge the targeted dataset with the original uniform dataset.
5. Save the augmented dataset.
6. Generate summary tables and diagnostic plots.

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
from joblib import Parallel, delayed
from scipy.stats import qmc

from dataset import ParameterRanges
from dynamics import simulate_system


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "augmentation"
FIGURES_DIR = PROJECT_ROOT / "figures" / "augmentation"

BASE_DATASET_PATH = DATA_DIR / "dataset_outreach.csv"
TARGETED_DATASET_PATH = DATA_DIR / "dataset_high_outreach.csv"
AUGMENTED_DATASET_PATH = DATA_DIR / "dataset_augmented.csv"
SUMMARY_PATH = RESULTS_DIR / "dataset_augmentation_summary.csv"


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_N_AUGMENT = 1000
DEFAULT_T_SIM = 60.0
DEFAULT_DT = 0.001
DEFAULT_X_R_MAX = 0.500
DEFAULT_TOLERANCE = 0.002
DEFAULT_N_JOBS = -1
DEFAULT_RANDOM_STATE = 123
HIGH_OUTREACH_THRESHOLD = 0.600

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
# TARGETED PARAMETER RANGES
# =============================================================================

@dataclass(frozen=True)
class HighOutreachRanges:
    """
    Targeted parameter ranges for high-outreach dataset augmentation.

    These ranges are based on physical insight and preliminary targeted search:
    - low-to-medium base stiffness;
    - low base mass;
    - low base damping;
    - moderate robot damping;
    - relatively high chirp amplitude;
    - initial robot position below the nominal robot limit.
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

    f1_min: float = 1.00
    f1_max: float = 4.00

    A_min: float = 0.09
    A_max: float = 0.12

    x_r_start_min: float = 0.35
    x_r_start_max: float = 0.40

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


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    """Raise a clear error if a DataFrame is missing required columns."""
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


def load_dataset_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataset CSV, allowing commented metadata headers."""
    require_file(path)
    df = pd.read_csv(path, comment="#")
    df = clean_boolean_columns(df)

    return df


# =============================================================================
# SAMPLING
# =============================================================================

def generate_lhs_samples(
    n_samples: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    ranges: HighOutreachRanges | None = None,
) -> np.ndarray:
    """Generate Latin Hypercube samples in the targeted high-outreach region."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if ranges is None:
        ranges = HighOutreachRanges()

    lower_bounds, upper_bounds = ranges.get_bounds()
    n_dimensions = len(lower_bounds)

    sampler = qmc.LatinHypercube(d=n_dimensions, seed=random_state)
    unit_samples = sampler.random(n=n_samples)
    samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return samples


# =============================================================================
# SIMULATION
# =============================================================================

def failed_row(params_array: np.ndarray, error_message: str) -> dict[str, Any]:
    """Return a row with NaN/default metrics for a failed simulation."""
    row = dict(zip(PARAM_NAMES, params_array))

    for col in METRIC_COLUMNS:
        row[col] = False if col in BOOLEAN_COLUMNS else np.nan

    row["dataset_type"] = "targeted_augmented"
    row["error"] = error_message

    return row


def simulate_parameter_set(
    params_array: np.ndarray,
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    x_r_max: float = DEFAULT_X_R_MAX,
) -> dict[str, Any]:
    """Simulate one parameter set and return inputs plus physical metrics."""
    try:
        _, metrics = simulate_system(
            params_array,
            T_sim=T_sim,
            dt=dt,
            return_metrics=True,
            x_r_max=x_r_max,
        )

        row = dict(zip(PARAM_NAMES, params_array))

        for col in METRIC_COLUMNS:
            if col in BOOLEAN_COLUMNS:
                row[col] = bool(metrics.get(col, False))
            else:
                row[col] = metrics.get(col, np.nan)

        row["dataset_type"] = "targeted_augmented"

        return row

    except Exception as exc:
        return failed_row(params_array, str(exc))


def generate_high_outreach_dataset(
    n_samples: int = DEFAULT_N_AUGMENT,
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    x_r_max: float = DEFAULT_X_R_MAX,
    n_jobs: int = DEFAULT_N_JOBS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Generate the targeted high-outreach simulation dataset."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if T_sim <= 0:
        raise ValueError("T_sim must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if x_r_max <= 0:
        raise ValueError("x_r_max must be positive.")

    print("\n" + "=" * 70)
    print("HIGH-OUTREACH DATASET AUGMENTATION")
    print("=" * 70)
    print(f"Number of targeted samples: {n_samples}")
    print(f"Simulation time:            {T_sim} s")
    print(f"Time step:                  {dt} s")
    print(f"Robot limit:                {x_r_max} m")
    print(f"Parallel jobs:              {n_jobs}")
    print(f"Random state:               {random_state}")
    print("=" * 70)

    print("\nGenerating targeted LHS samples...")
    samples = generate_lhs_samples(n_samples=n_samples, random_state=random_state)
    print(f"Samples generated: shape {samples.shape}")

    print("\nRunning simulations...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(simulate_parameter_set)(
            sample,
            T_sim=T_sim,
            dt=dt,
            x_r_max=x_r_max,
        )
        for sample in samples
    )

    elapsed_time = time.time() - start_time

    df = pd.DataFrame(results)
    n_failed = int(df["peak_y"].isna().sum()) if "peak_y" in df.columns else len(df)

    if n_failed > 0:
        print(f"\nWarning: {n_failed}/{n_samples} simulations failed and will be removed.")
        df = df.dropna(subset=["peak_y"]).reset_index(drop=True)

    df = clean_boolean_columns(df)

    print(f"\nTargeted dataset generated in {elapsed_time:.1f} s")
    print(f"Valid simulations: {len(df)}/{n_samples}")
    print(f"Average time/sample: {elapsed_time / n_samples:.3f} s")

    return df


# =============================================================================
# SUMMARY
# =============================================================================

def compute_dataset_stats(
    df: pd.DataFrame,
    name: str,
    tolerance: float = DEFAULT_TOLERANCE,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> dict[str, Any]:
    """Compute compact dataset statistics."""
    require_columns(
        df,
        ["peak_y", "extra_reach", "constraint_violation_abs"],
        name,
    )

    feasible = df[df["constraint_violation_abs"] <= tolerance]
    feasible_extra = feasible[feasible["extra_reach"] > 0.0]
    feasible_high = feasible[feasible["peak_y"] > high_outreach_threshold]

    return {
        "dataset": name,
        "samples": int(len(df)),
        "peak_y_mean_m": float(df["peak_y"].mean()),
        "peak_y_max_m": float(df["peak_y"].max()),
        "feasible_abs_samples": int(len(feasible)),
        "feasible_abs_extra_reach": int(len(feasible_extra)),
        "feasible_abs_high_outreach": int(len(feasible_high)),
        "max_feasible_abs_peak_y_m": (
            float(feasible["peak_y"].max()) if len(feasible) else np.nan
        ),
        "max_feasible_abs_extra_reach_m": (
            float(feasible_extra["extra_reach"].max()) if len(feasible_extra) else np.nan
        ),
        "violation_abs_rate_percent": float(
            100.0 * (df["constraint_violation_abs"] > tolerance).mean()
        ),
        "tolerance_m": float(tolerance),
        "high_outreach_threshold_m": float(high_outreach_threshold),
    }


def print_dataset_stats(
    df: pd.DataFrame,
    name: str,
    tolerance: float = DEFAULT_TOLERANCE,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> None:
    """Print compact dataset statistics."""
    stats = compute_dataset_stats(
        df=df,
        name=name,
        tolerance=tolerance,
        high_outreach_threshold=high_outreach_threshold,
    )

    print("\n" + "=" * 70)
    print(f"{name.upper()} STATISTICS")
    print("=" * 70)
    print(f"Samples:                     {stats['samples']}")
    print(f"Peak_y mean:                  {stats['peak_y_mean_m']:.6f} m")
    print(f"Peak_y max:                   {stats['peak_y_max_m']:.6f} m")
    print(f"Feasible_abs samples:         {stats['feasible_abs_samples']}")
    print(f"Feasible_abs extra-reach:     {stats['feasible_abs_extra_reach']}")
    print(f"Feasible_abs high-outreach:   {stats['feasible_abs_high_outreach']}")
    print(f"Max feasible_abs peak_y:      {stats['max_feasible_abs_peak_y_m']:.6f} m")
    print(f"Max feasible_abs extra_reach: {stats['max_feasible_abs_extra_reach_m']:.6f} m")
    print(f"Violation_abs rate:           {stats['violation_abs_rate_percent']:.1f}%")
    print("=" * 70)


def save_augmentation_summary(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: Path = SUMMARY_PATH,
    tolerance: float = DEFAULT_TOLERANCE,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> pd.DataFrame:
    """Save before/after augmentation statistics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            compute_dataset_stats(
                base_df,
                name="uniform",
                tolerance=tolerance,
                high_outreach_threshold=high_outreach_threshold,
            ),
            compute_dataset_stats(
                aug_df,
                name="targeted_augmented",
                tolerance=tolerance,
                high_outreach_threshold=high_outreach_threshold,
            ),
            compute_dataset_stats(
                merged_df,
                name="augmented",
                tolerance=tolerance,
                high_outreach_threshold=high_outreach_threshold,
            ),
        ]
    )

    summary_df.to_csv(output_path, index=False)

    print(f"\nAugmentation summary saved: {output_path}")

    return summary_df


# =============================================================================
# MERGE
# =============================================================================

def merge_datasets(
    base_path: Path = BASE_DATASET_PATH,
    augmentation_path: Path = TARGETED_DATASET_PATH,
    output_path: Path = AUGMENTED_DATASET_PATH,
) -> pd.DataFrame:
    """
    Merge the original uniform dataset and the targeted augmentation dataset.

    The output keeps the column order of the base dataset. Extra columns from
    the augmentation dataset are not included in the final training dataset.
    """
    base_path = Path(base_path)
    augmentation_path = Path(augmentation_path)
    output_path = Path(output_path)

    base_df = load_dataset_dataframe(base_path)
    aug_df = load_dataset_dataframe(augmentation_path)

    require_columns(base_df, PARAM_NAMES + ["peak_y"], "Base dataset")
    require_columns(aug_df, PARAM_NAMES + ["peak_y"], "Augmentation dataset")

    base_columns = list(base_df.columns)

    missing_in_aug = [col for col in base_columns if col not in aug_df.columns]
    if missing_in_aug:
        raise KeyError(
            "The augmentation dataset cannot be merged because it is missing "
            f"base columns: {missing_in_aug}"
        )

    aug_df = aug_df[base_columns].copy()
    base_df = base_df[base_columns].copy()

    merged_df = pd.concat([base_df, aug_df], ignore_index=True)
    merged_df = clean_boolean_columns(merged_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"\nAugmented dataset saved: {output_path}")
    print(f"Shape: {merged_df.shape}")

    return merged_df


# =============================================================================
# PLOTS
# =============================================================================

def plot_peak_y_distribution(
    base_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    save_dir: Path,
    x_r_max: float = DEFAULT_X_R_MAX,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> None:
    """Plot peak_y distribution before and after augmentation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        merged_df["peak_y"],
        bins=30,
        alpha=0.5,
        label="Augmented dataset",
        edgecolor="black",
    )

    ax.hist(
        base_df["peak_y"],
        bins=30,
        alpha=0.6,
        label="Uniform dataset",
        edgecolor="black",
    )

    ax.axvline(x_r_max, linestyle="--", linewidth=2, label="Nominal reach")
    ax.axvline(
        high_outreach_threshold,
        linestyle=":",
        linewidth=2,
        label="High-outreach threshold",
    )

    ax.set_xlabel(r"Peak outreach $y_{\mathrm{peak}}$ [m]")
    ax.set_ylabel("Frequency")
    ax.set_title("Peak outreach distribution before and after augmentation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = save_dir / "augmentation_peak_y_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_feasible_region(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    save_dir: Path,
    x_r_max: float = DEFAULT_X_R_MAX,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> None:
    """Plot the high-outreach feasible region before and after augmentation."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        base_df["max_abs_xr"],
        base_df["peak_y"],
        alpha=0.45,
        s=35,
        label="Uniform dataset",
    )

    ax.scatter(
        aug_df["max_abs_xr"],
        aug_df["peak_y"],
        alpha=0.65,
        s=35,
        label="Targeted augmented samples",
    )

    ax.axvline(x_r_max, linestyle="--", linewidth=2, label=r"$|x_r|$ limit")
    ax.axhline(x_r_max, linestyle=":", linewidth=2, label="Nominal reach")
    ax.axhline(
        high_outreach_threshold,
        linestyle="-.",
        linewidth=2,
        label="High-outreach threshold",
    )

    ax.set_xlabel(r"Maximum absolute robot displacement $|x_r|_{\max}$ [m]")
    ax.set_ylabel(r"Peak outreach $y_{\mathrm{peak}}$ [m]")
    ax.set_title("Feasible high-outreach region before and after augmentation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = save_dir / "augmentation_feasible_high_outreach_region.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def count_feasible_high_outreach(
    df: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> int:
    """Count feasible high-outreach samples."""
    feasible = df[df["constraint_violation_abs"] <= tolerance]
    high = feasible[feasible["peak_y"] > high_outreach_threshold]

    return int(len(high))


def plot_feasible_high_outreach_counts(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    save_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
) -> None:
    """Plot feasible high-outreach sample counts before and after augmentation."""
    counts = pd.DataFrame(
        {
            "dataset": ["Uniform", "Targeted", "Augmented"],
            "feasible_high_outreach": [
                count_feasible_high_outreach(base_df, tolerance, high_outreach_threshold),
                count_feasible_high_outreach(aug_df, tolerance, high_outreach_threshold),
                count_feasible_high_outreach(merged_df, tolerance, high_outreach_threshold),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(counts["dataset"], counts["feasible_high_outreach"], edgecolor="black")

    ax.set_ylabel("Number of feasible high-outreach samples")
    ax.set_title("Feasible high-outreach samples before and after augmentation")
    ax.grid(True, axis="y", alpha=0.3)

    path = save_dir / "augmentation_feasible_high_outreach_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    counts_path = save_dir / "augmentation_feasible_high_outreach_counts.csv"
    counts.to_csv(counts_path, index=False)

    print(f"Saved: {path}")
    print(f"Saved: {counts_path}")


def plot_augmentation_analysis(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    save_dir: Path = FIGURES_DIR,
    x_r_max: float = DEFAULT_X_R_MAX,
    high_outreach_threshold: float = HIGH_OUTREACH_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """
    Generate plots that justify the targeted augmentation strategy.

    The plots compare:
    - the original uniform dataset;
    - the targeted augmentation samples;
    - the final merged augmented dataset.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    require_columns(base_df, ["peak_y", "max_abs_xr", "constraint_violation_abs"], "Base dataset")
    require_columns(aug_df, ["peak_y", "max_abs_xr", "constraint_violation_abs"], "Augmentation dataset")
    require_columns(merged_df, ["peak_y", "max_abs_xr", "constraint_violation_abs"], "Merged dataset")

    plot_peak_y_distribution(
        base_df=base_df,
        merged_df=merged_df,
        save_dir=save_dir,
        x_r_max=x_r_max,
        high_outreach_threshold=high_outreach_threshold,
    )

    plot_feasible_region(
        base_df=base_df,
        aug_df=aug_df,
        save_dir=save_dir,
        x_r_max=x_r_max,
        high_outreach_threshold=high_outreach_threshold,
    )

    plot_feasible_high_outreach_counts(
        base_df=base_df,
        aug_df=aug_df,
        merged_df=merged_df,
        save_dir=save_dir,
        tolerance=tolerance,
        high_outreach_threshold=high_outreach_threshold,
    )

    print(f"\nAugmentation plots completed in: {save_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate targeted high-outreach samples and merge them with the uniform dataset."
    )

    parser.add_argument(
        "--n_augment",
        type=int,
        default=DEFAULT_N_AUGMENT,
        help=f"Number of targeted augmentation samples. Default: {DEFAULT_N_AUGMENT}.",
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
        help=f"Simulation time step. Default: {DEFAULT_DT}.",
    )

    parser.add_argument(
        "--x_r_max",
        type=float,
        default=DEFAULT_X_R_MAX,
        help=f"Robot relative-displacement limit. Default: {DEFAULT_X_R_MAX}.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Feasibility tolerance in meters. Default: {DEFAULT_TOLERANCE}.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel jobs. Use -1 for all cores. Default: {DEFAULT_N_JOBS}.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for targeted LHS sampling. Default: {DEFAULT_RANDOM_STATE}.",
    )

    parser.add_argument(
        "--base_path",
        type=Path,
        default=BASE_DATASET_PATH,
        help=f"Path to the base uniform dataset. Default: {BASE_DATASET_PATH}.",
    )

    parser.add_argument(
        "--targeted_path",
        type=Path,
        default=TARGETED_DATASET_PATH,
        help=f"Path where the targeted dataset is saved. Default: {TARGETED_DATASET_PATH}.",
    )

    parser.add_argument(
        "--augmented_path",
        type=Path,
        default=AUGMENTED_DATASET_PATH,
        help=f"Path where the merged augmented dataset is saved. Default: {AUGMENTED_DATASET_PATH}.",
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip augmentation diagnostic plots.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate the targeted dataset, merge it, and save diagnostics."""
    args = parse_args()
    ensure_dirs()

    require_file(args.base_path)

    df_aug = generate_high_outreach_dataset(
        n_samples=args.n_augment,
        T_sim=args.T_sim,
        dt=args.dt,
        x_r_max=args.x_r_max,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    args.targeted_path.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(args.targeted_path, index=False)
    print(f"\nTargeted dataset saved: {args.targeted_path}")

    print_dataset_stats(
        df=df_aug,
        name="Targeted augmentation",
        tolerance=args.tolerance,
        high_outreach_threshold=HIGH_OUTREACH_THRESHOLD,
    )

    merged_df = merge_datasets(
        base_path=args.base_path,
        augmentation_path=args.targeted_path,
        output_path=args.augmented_path,
    )

    base_df = load_dataset_dataframe(args.base_path)

    summary_df = save_augmentation_summary(
        base_df=base_df,
        aug_df=df_aug,
        merged_df=merged_df,
        output_path=SUMMARY_PATH,
        tolerance=args.tolerance,
        high_outreach_threshold=HIGH_OUTREACH_THRESHOLD,
    )

    print("\nAugmentation summary:")
    print(summary_df.to_string(index=False))

    if not args.no_plots:
        print("\nGenerating augmentation analysis plots...")
        plot_augmentation_analysis(
            base_df=base_df,
            aug_df=df_aug,
            merged_df=merged_df,
            save_dir=FIGURES_DIR,
            x_r_max=args.x_r_max,
            high_outreach_threshold=HIGH_OUTREACH_THRESHOLD,
            tolerance=args.tolerance,
        )

    print_dataset_stats(
        df=merged_df,
        name="Final augmented dataset",
        tolerance=args.tolerance,
        high_outreach_threshold=HIGH_OUTREACH_THRESHOLD,
    )

    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Generated files:")
    print(f"  - {args.targeted_path}")
    print(f"  - {args.augmented_path}")
    print(f"  - {SUMMARY_PATH}")

    if not args.no_plots:
        print(f"  - {FIGURES_DIR}/augmentation_*.png")

    print("\nNext step:")
    print("  Train the surrogate models using data/dataset_augmented.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()