"""
monte_carlo_robustness.py
=========================

Monte Carlo robustness analysis around the final optimized inverse-design
solutions.

This script is intended to be run after:

    - physical_validation.py
    - sensitivity_oat.py

It does not run a new optimization. It takes the selected final NN+Adam
solutions and perturbs multiple physical/model parameters simultaneously using
multiplicative Gaussian uncertainty. Each perturbed candidate is evaluated with
the true dynamic simulator.

Default setup
-------------
Reference method:
    NN+Adam

Targets:
    0.65 m and 0.75 m

Uncertain physical/model parameters:
    Kb, Mb, hb, Mr

Noise levels:
    5% and 10% Gaussian multiplicative standard deviation

Samples:
    500 per target per noise level

Rationale
---------
OAT sensitivity answers:

    Which single parameter is most influential locally?

Monte Carlo robustness answers:

    What happens when multiple uncertain physical parameters vary
    simultaneously?

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamics import simulate_system


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "monte_carlo_robustness"
FIGURES_DIR = PROJECT_ROOT / "figures" / "monte_carlo_robustness"

DEFAULT_PHYSICAL_VALIDATION_SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "physical_validation"
    / "physical_validation_selected_candidates.csv"
)

DEFAULT_NN_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "optimization_nn_gradient"
    / "nn_gradient_results.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

REFERENCE_METHOD = "NN+Adam"
REFERENCE_METHOD_ALIASES = ("NN+Adam", "NN+Adam safe", "NN_Adam")

DEFAULT_TARGETS_TO_ANALYZE = (0.65, 0.75)
DEFAULT_NOISE_LEVELS = (0.05, 0.10)
DEFAULT_N_SAMPLES = 500
DEFAULT_RANDOM_SEED = 42

ROBOT_LIMIT_TRUE = 0.500
FEASIBILITY_TOLERANCE_M = 1e-9
SUCCESS_TOLERANCE_MM = 10.0

# Gaussian noise is clipped to avoid rare extreme samples dominating the result.
# With sigma=10%, CLIP_SIGMA=3 keeps each uncertain parameter within about ±30%.
CLIP_SIGMA = 3.0

PARAM_COLUMNS = [
    "Kb",
    "Kr",
    "Mb",
    "hb",
    "hr",
    "f0",
    "f1",
    "A",
    "x_r_start",
    "Mr",
]

FIXED_DEFAULTS = {
    "Kb": 1000.0,
    "Mb": 20.0,
    "hb": 0.10,
    "Mr": 10.0,
}

UNCERTAIN_PARAMS = ["Kb", "Mb", "hb", "Mr"]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for Monte Carlo robustness analysis."""

    physical_validation_selected_path: Path = DEFAULT_PHYSICAL_VALIDATION_SELECTED_PATH
    nn_results_path: Path = DEFAULT_NN_RESULTS_PATH

    targets_to_analyze: tuple[float, ...] = DEFAULT_TARGETS_TO_ANALYZE
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS
    n_samples: int = DEFAULT_N_SAMPLES
    random_seed: int = DEFAULT_RANDOM_SEED

    reference_method: str = REFERENCE_METHOD
    skip_plots: bool = False
    skip_correlations: bool = False


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def target_tag(target: float) -> str:
    """Return target tag in millimeters."""
    return f"target{int(round(float(target) * 1000.0)):04d}"


def noise_tag(noise_level: float) -> str:
    """Return noise-level tag."""
    return f"noise{int(round(float(noise_level) * 100.0)):02d}"


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""
    values = [item.strip() for item in raw.split(",") if item.strip()]

    if not values:
        raise ValueError("At least one value must be provided.")

    return tuple(float(value) for value in values)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run the previous pipeline step first or check the path."
        )


def as_bool(value: Any) -> bool:
    """Convert common boolean-like values to bool."""
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value) > 0.5

    return bool(value)


def complete_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed physical parameters if only optimized columns are stored."""
    out = df.copy()

    for col, value in FIXED_DEFAULTS.items():
        if col not in out.columns:
            out[col] = float(value)

    return out


def assert_has_parameters(df: pd.DataFrame, label: str) -> None:
    """Check that all simulator parameters are available."""
    missing = [col for col in PARAM_COLUMNS if col not in df.columns]

    if missing:
        raise KeyError(
            f"{label} does not contain all simulator parameters. Missing: {missing}\n"
            f"Required columns: {PARAM_COLUMNS}"
        )


def row_to_params(row: pd.Series) -> dict[str, float]:
    """Convert one row to a simulator parameter dictionary."""
    return {col: float(row[col]) for col in PARAM_COLUMNS}


def normalize_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old/new method labels to the clean reference method label."""
    if df.empty or "method" not in df.columns:
        return df

    out = df.copy()
    out["method"] = out["method"].astype(str)

    alias_mask = out["method"].isin(REFERENCE_METHOD_ALIASES)
    out.loc[alias_mask, "method"] = REFERENCE_METHOD

    return out


def selection_sort_key(row: pd.Series) -> tuple[int, float, float]:
    """
    Select best physical candidate consistently.

    Priority:
        feasible first, then target error;
        if infeasible, minimize violation before target error.
    """
    feasible = as_bool(row.get("feasible_abs", False))

    target_error_m = row.get("target_error_m", np.nan)

    if not np.isfinite(float(target_error_m)):
        target_error_m = float(row.get("target_error_mm", np.inf)) / 1000.0

    violation_m = row.get("constraint_violation_abs_m", np.nan)

    if not np.isfinite(float(violation_m)):
        violation_m = float(row.get("constraint_violation_abs_mm", np.inf)) / 1000.0

    return (
        0 if feasible else 1,
        0.0 if feasible else float(violation_m),
        float(target_error_m),
    )


def binomial_ci95_halfwidth_percent(successes: int, n_samples: int) -> float:
    """Approximate 95% binomial confidence half-width in percentage points."""
    if n_samples <= 0:
        return float("nan")

    p_hat = float(successes) / float(n_samples)
    half_width = 1.96 * np.sqrt(p_hat * (1.0 - p_hat) / float(n_samples))

    return float(half_width * 100.0)


# =============================================================================
# LOAD REFERENCE SOLUTIONS
# =============================================================================

def load_reference_solutions(config: MonteCarloConfig) -> pd.DataFrame:
    """Load one final selected reference solution per target."""
    if config.physical_validation_selected_path.exists():
        df = pd.read_csv(config.physical_validation_selected_path)
        source = config.physical_validation_selected_path
    else:
        require_file(config.nn_results_path)
        df = pd.read_csv(config.nn_results_path)

        if "method" not in df.columns:
            df["method"] = config.reference_method

        source = config.nn_results_path

    df = complete_parameter_columns(df)
    df = normalize_method_labels(df)

    assert_has_parameters(df, str(source))

    if "method" in df.columns:
        df = df[df["method"].astype(str) == config.reference_method].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for reference method {config.reference_method!r} in {source}."
        )

    rows = []

    for target in config.targets_to_analyze:
        group = df[np.isclose(df["target"].astype(float), float(target))].copy()

        if group.empty:
            raise ValueError(f"No reference solution found for target={target:.3f}.")

        sorted_idx = sorted(group.index, key=lambda idx: selection_sort_key(group.loc[idx]))

        best = group.loc[sorted_idx[0]].copy()
        best["target"] = float(target)
        best["method"] = config.reference_method

        rows.append(best)

    selected = (
        pd.DataFrame(rows)
        .sort_values("target")
        .reset_index(drop=True)
    )

    path = RESULTS_DIR / "monte_carlo_reference_solutions.csv"
    selected.to_csv(path, index=False)

    print(f"Saved reference solutions: {path}")

    return selected


# =============================================================================
# SIMULATOR HELPERS
# =============================================================================

def extract_metrics_from_simulator_output(output: Any) -> dict[str, Any]:
    """Extract metrics dictionary from simulate_system output."""
    if isinstance(output, dict):
        return dict(output)

    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict) and "peak_y" in item:
                return dict(item)

        if len(output) >= 2 and isinstance(output[1], dict):
            return dict(output[1])

    raise TypeError(
        "Could not extract metrics dictionary from simulate_system output. "
        f"Received type: {type(output)}."
    )


def complete_metrics(metrics: dict[str, Any], target: float) -> dict[str, Any]:
    """Complete simulator metrics with target error, feasibility and success."""
    completed = dict(metrics)

    if "peak_y" not in completed:
        raise KeyError("Simulator metrics do not contain 'peak_y'.")

    if "max_abs_xr" not in completed:
        if "max_xr" in completed and "min_xr" in completed:
            completed["max_abs_xr"] = max(
                abs(float(completed["max_xr"])),
                abs(float(completed["min_xr"])),
            )
        elif "max_xr" in completed:
            completed["max_abs_xr"] = abs(float(completed["max_xr"]))
        else:
            raise KeyError(
                "Simulator metrics do not contain 'max_abs_xr' or usable max_xr/min_xr."
            )

    if "max_abs_xb" not in completed:
        if "max_xb" in completed and "min_xb" in completed:
            completed["max_abs_xb"] = max(
                abs(float(completed["max_xb"])),
                abs(float(completed["min_xb"])),
            )
        elif "max_xb" in completed:
            completed["max_abs_xb"] = abs(float(completed["max_xb"]))
        else:
            completed["max_abs_xb"] = np.nan

    peak_y = float(completed["peak_y"])
    max_abs_xr = float(completed["max_abs_xr"])

    completed["target_error_m"] = abs(peak_y - float(target))
    completed["target_error_mm"] = completed["target_error_m"] * 1000.0

    completed["residual_margin_m"] = ROBOT_LIMIT_TRUE - max_abs_xr
    completed["residual_margin_mm"] = completed["residual_margin_m"] * 1000.0

    completed["constraint_violation_abs_m"] = max(
        0.0,
        max_abs_xr - ROBOT_LIMIT_TRUE,
    )
    completed["constraint_violation_abs_mm"] = (
        completed["constraint_violation_abs_m"] * 1000.0
    )

    completed["feasible_abs"] = (
        completed["constraint_violation_abs_m"] <= FEASIBILITY_TOLERANCE_M
    )
    completed["success_10mm_abs"] = (
        bool(completed["feasible_abs"])
        and completed["target_error_mm"] <= SUCCESS_TOLERANCE_MM
    )
    completed["extra_reach"] = peak_y - ROBOT_LIMIT_TRUE

    return completed


def simulate_candidate(params: dict[str, float], target: float) -> dict[str, Any]:
    """Run true simulator and return completed scalar metrics."""
    try:
        output = simulate_system(
            params,
            y_target=float(target),
            x_r_max=ROBOT_LIMIT_TRUE,
            return_metrics=True,
        )
    except TypeError:
        try:
            output = simulate_system(params, return_metrics=True)
        except TypeError:
            output = simulate_system(params)

    metrics = extract_metrics_from_simulator_output(output)

    return complete_metrics(metrics, target=float(target))


# =============================================================================
# MONTE CARLO CORE
# =============================================================================

def perturb_physical_params(
    nominal_params: dict[str, float],
    rng: np.random.Generator,
    noise_level: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply simultaneous multiplicative Gaussian perturbations."""
    perturbed = dict(nominal_params)
    epsilons: dict[str, float] = {}

    for name in UNCERTAIN_PARAMS:
        epsilon = float(rng.normal(loc=0.0, scale=float(noise_level)))
        epsilon = float(
            np.clip(
                epsilon,
                -CLIP_SIGMA * float(noise_level),
                CLIP_SIGMA * float(noise_level),
            )
        )

        epsilons[name] = epsilon

        value = float(nominal_params[name]) * (1.0 + epsilon)
        value = max(value, 1e-12)

        perturbed[name] = value

    return perturbed, epsilons


def build_nominal_record(
    nominal_params: dict[str, float],
    nominal_metrics: dict[str, Any],
    target: float,
    config: MonteCarloConfig,
) -> dict[str, Any]:
    """Build nominal metrics record."""
    return {
        "method": config.reference_method,
        "target": float(target),
        **{f"nominal_{param}": nominal_params[param] for param in PARAM_COLUMNS},
        "nominal_peak_y": float(nominal_metrics["peak_y"]),
        "nominal_target_error_mm": float(nominal_metrics["target_error_mm"]),
        "nominal_max_abs_xr": float(nominal_metrics["max_abs_xr"]),
        "nominal_residual_margin_mm": float(nominal_metrics["residual_margin_mm"]),
        "nominal_feasible_abs": bool(nominal_metrics["feasible_abs"]),
    }


def build_mc_sample_record(
    target: float,
    noise_level: float,
    sample_idx: int,
    nominal_params: dict[str, float],
    perturbed_params: dict[str, float],
    epsilons: dict[str, float],
    nominal_metrics: dict[str, Any],
    metrics: dict[str, Any],
    config: MonteCarloConfig,
) -> dict[str, Any]:
    """Build one Monte Carlo sample row."""
    row = {
        "method": config.reference_method,
        "target": float(target),
        "noise_level": float(noise_level),
        "noise_std_percent": float(noise_level * 100.0),
        "sample_index": int(sample_idx),
        "peak_y": float(metrics["peak_y"]),
        "delta_peak_y_mm": (
            float(metrics["peak_y"]) - float(nominal_metrics["peak_y"])
        ) * 1000.0,
        "target_error_mm": float(metrics["target_error_mm"]),
        "max_abs_xr": float(metrics["max_abs_xr"]),
        "delta_max_abs_xr_mm": (
            float(metrics["max_abs_xr"]) - float(nominal_metrics["max_abs_xr"])
        ) * 1000.0,
        "residual_margin_mm": float(metrics["residual_margin_mm"]),
        "constraint_violation_mm": float(metrics["constraint_violation_abs_mm"]),
        "feasible_abs": bool(metrics["feasible_abs"]),
        "success_10mm_abs": bool(metrics["success_10mm_abs"]),
        "extra_reach": float(metrics["extra_reach"]),
        "max_abs_xb": float(metrics.get("max_abs_xb", np.nan)),
        "nominal_peak_y": float(nominal_metrics["peak_y"]),
        "nominal_target_error_mm": float(nominal_metrics["target_error_mm"]),
        "nominal_max_abs_xr": float(nominal_metrics["max_abs_xr"]),
        "nominal_residual_margin_mm": float(nominal_metrics["residual_margin_mm"]),
    }

    for param_name in PARAM_COLUMNS:
        row[param_name] = float(perturbed_params[param_name])

    for name in UNCERTAIN_PARAMS:
        row[f"epsilon_{name}"] = float(epsilons[name])
        row[f"epsilon_{name}_percent"] = float(epsilons[name] * 100.0)
        row[f"delta_{name}"] = float(perturbed_params[name] - nominal_params[name])
        row[f"delta_{name}_percent"] = float(epsilons[name] * 100.0)

    return row


def summarize_monte_carlo(
    results_df: pd.DataFrame,
    config: MonteCarloConfig,
) -> pd.DataFrame:
    """Create Monte Carlo robustness summary table."""
    summary_rows = []

    for (target, noise_level), group in results_df.groupby(["target", "noise_level"]):
        peak = group["peak_y"].to_numpy(dtype=float)
        err = group["target_error_mm"].to_numpy(dtype=float)
        xr = group["max_abs_xr"].to_numpy(dtype=float)
        margin = group["residual_margin_mm"].to_numpy(dtype=float)
        violation = group["constraint_violation_mm"].to_numpy(dtype=float)

        delta_peak = group["delta_peak_y_mm"].to_numpy(dtype=float)
        abs_delta_peak = np.abs(delta_peak)

        delta_xr = group["delta_max_abs_xr_mm"].to_numpy(dtype=float)
        abs_delta_xr = np.abs(delta_xr)

        feasible = group["feasible_abs"].astype(bool).to_numpy()
        success = group["success_10mm_abs"].astype(bool).to_numpy()

        tracking_success = err <= SUCCESS_TOLERANCE_MM
        stability_10 = abs_delta_peak <= 10.0
        stability_20 = abs_delta_peak <= 20.0

        n_group = int(len(group))

        worst_margin_idx = group["residual_margin_mm"].astype(float).idxmin()
        worst_error_idx = group["target_error_mm"].astype(float).idxmax()

        summary_rows.append(
            {
                "method": config.reference_method,
                "target": float(target),
                "noise_level": float(noise_level),
                "noise_std_percent": float(noise_level * 100.0),
                "n_samples": n_group,
                "feasibility_rate_percent": float(np.mean(feasible) * 100.0),
                "feasibility_rate_ci95_halfwidth_percent": binomial_ci95_halfwidth_percent(
                    int(np.sum(feasible)),
                    n_group,
                ),
                "success_rate_10mm_percent": float(np.mean(success) * 100.0),
                "success_rate_10mm_ci95_halfwidth_percent": binomial_ci95_halfwidth_percent(
                    int(np.sum(success)),
                    n_group,
                ),
                "tracking_success_rate_10mm_percent": float(
                    np.mean(tracking_success) * 100.0
                ),
                "tracking_success_rate_10mm_ci95_halfwidth_percent": (
                    binomial_ci95_halfwidth_percent(
                        int(np.sum(tracking_success)),
                        n_group,
                    )
                ),
                "stability_rate_10mm_percent": float(np.mean(stability_10) * 100.0),
                "stability_rate_10mm_ci95_halfwidth_percent": (
                    binomial_ci95_halfwidth_percent(
                        int(np.sum(stability_10)),
                        n_group,
                    )
                ),
                "stability_rate_20mm_percent": float(np.mean(stability_20) * 100.0),
                "stability_rate_20mm_ci95_halfwidth_percent": (
                    binomial_ci95_halfwidth_percent(
                        int(np.sum(stability_20)),
                        n_group,
                    )
                ),
                "mean_abs_delta_peak_y_mm": float(np.mean(abs_delta_peak)),
                "p95_abs_delta_peak_y_mm": float(np.percentile(abs_delta_peak, 95)),
                "max_abs_delta_peak_y_mm": float(np.max(abs_delta_peak)),
                "mean_abs_delta_max_abs_xr_mm": float(np.mean(abs_delta_xr)),
                "p95_abs_delta_max_abs_xr_mm": float(np.percentile(abs_delta_xr, 95)),
                "max_abs_delta_max_abs_xr_mm": float(np.max(abs_delta_xr)),
                "mean_peak_y": float(np.mean(peak)),
                "std_peak_y_mm": float(np.std(peak) * 1000.0),
                "p05_peak_y": float(np.percentile(peak, 5)),
                "p95_peak_y": float(np.percentile(peak, 95)),
                "mean_target_error_mm": float(np.mean(err)),
                "median_target_error_mm": float(np.median(err)),
                "p95_target_error_mm": float(np.percentile(err, 95)),
                "max_target_error_mm": float(np.max(err)),
                "mean_max_abs_xr": float(np.mean(xr)),
                "max_max_abs_xr": float(np.max(xr)),
                "mean_residual_margin_mm": float(np.mean(margin)),
                "p05_residual_margin_mm": float(np.percentile(margin, 5)),
                "min_residual_margin_mm": float(np.min(margin)),
                "mean_constraint_violation_mm": float(np.mean(violation)),
                "max_constraint_violation_mm": float(np.max(violation)),
                "n_infeasible": int(np.sum(~feasible)),
                "n_success_10mm": int(np.sum(success)),
                "worst_margin_sample_index": int(group.loc[worst_margin_idx, "sample_index"]),
                "worst_margin_mm": float(group.loc[worst_margin_idx, "residual_margin_mm"]),
                "worst_margin_peak_y": float(group.loc[worst_margin_idx, "peak_y"]),
                "worst_margin_max_abs_xr": float(group.loc[worst_margin_idx, "max_abs_xr"]),
                "worst_margin_target_error_mm": float(
                    group.loc[worst_margin_idx, "target_error_mm"]
                ),
                "worst_error_sample_index": int(group.loc[worst_error_idx, "sample_index"]),
                "worst_error_mm": float(group.loc[worst_error_idx, "target_error_mm"]),
                "worst_error_peak_y": float(group.loc[worst_error_idx, "peak_y"]),
                "worst_error_max_abs_xr": float(group.loc[worst_error_idx, "max_abs_xr"]),
            }
        )

    return (
        pd.DataFrame(summary_rows)
        .sort_values(["target", "noise_level"])
        .reset_index(drop=True)
    )


def run_monte_carlo(
    reference_df: pd.DataFrame,
    config: MonteCarloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete Monte Carlo robustness analysis."""
    all_rows = []
    nominal_rows = []

    print("\n" + "=" * 80)
    print("MONTE CARLO ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print(f"Reference method:  {config.reference_method}")
    print(f"Targets:           {config.targets_to_analyze}")
    print(f"Uncertain params:  {UNCERTAIN_PARAMS}")
    print(f"Noise levels:      {[int(100 * level) for level in config.noise_levels]}% Gaussian std")
    print(f"Samples:           {config.n_samples} per target per noise level")
    print(f"Success criterion: feasible and target error <= {SUCCESS_TOLERANCE_MM:.1f} mm")
    print("=" * 80)

    for _, ref_row in reference_df.iterrows():
        target = float(ref_row["target"])
        nominal_params = row_to_params(ref_row)
        nominal_metrics = simulate_candidate(nominal_params, target)

        print(
            f"\nNominal {config.reference_method}, target={target:.2f}: "
            f"peak_y={nominal_metrics['peak_y']:.4f} m | "
            f"err={nominal_metrics['target_error_mm']:.2f} mm | "
            f"max_abs_xr={nominal_metrics['max_abs_xr']:.4f} m | "
            f"margin={nominal_metrics['residual_margin_mm']:.2f} mm | "
            f"feasible={nominal_metrics['feasible_abs']}"
        )

        nominal_rows.append(
            build_nominal_record(
                nominal_params=nominal_params,
                nominal_metrics=nominal_metrics,
                target=target,
                config=config,
            )
        )

        for noise_level in config.noise_levels:
            rng_seed = (
                int(config.random_seed)
                + int(round(target * 1000.0))
                + int(round(float(noise_level) * 10000.0))
            )

            rng = np.random.default_rng(rng_seed)

            print(
                f"  Running MC: target={target:.2f}, "
                f"noise={noise_level * 100:.0f}% "
                f"({config.n_samples} samples)"
            )

            for sample_idx in range(config.n_samples):
                perturbed_params, epsilons = perturb_physical_params(
                    nominal_params=nominal_params,
                    rng=rng,
                    noise_level=float(noise_level),
                )

                metrics = simulate_candidate(perturbed_params, target)

                all_rows.append(
                    build_mc_sample_record(
                        target=target,
                        noise_level=float(noise_level),
                        sample_idx=sample_idx,
                        nominal_params=nominal_params,
                        perturbed_params=perturbed_params,
                        epsilons=epsilons,
                        nominal_metrics=nominal_metrics,
                        metrics=metrics,
                        config=config,
                    )
                )

    results_df = pd.DataFrame(all_rows)
    nominal_df = pd.DataFrame(nominal_rows)
    summary_df = summarize_monte_carlo(results_df, config)

    return results_df, summary_df, nominal_df


# =============================================================================
# CORRELATION POST-PROCESSING
# =============================================================================

def compute_uncertainty_correlations(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations between sampled uncertainties and outputs."""
    output_cols = [
        "peak_y",
        "delta_peak_y_mm",
        "target_error_mm",
        "max_abs_xr",
        "delta_max_abs_xr_mm",
        "residual_margin_mm",
        "constraint_violation_mm",
    ]

    rows = []

    for (target, noise_level), group in results_df.groupby(["target", "noise_level"]):
        for param in UNCERTAIN_PARAMS:
            eps_col = f"epsilon_{param}"

            if eps_col not in group.columns:
                continue

            eps = group[eps_col].to_numpy(dtype=float)

            for output in output_cols:
                if output not in group.columns:
                    continue

                values = group[output].to_numpy(dtype=float)
                valid = np.isfinite(eps) & np.isfinite(values)

                if (
                    np.sum(valid) < 3
                    or np.std(eps[valid]) <= 0.0
                    or np.std(values[valid]) <= 0.0
                ):
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(eps[valid], values[valid])[0, 1])

                rows.append(
                    {
                        "target": float(target),
                        "noise_level": float(noise_level),
                        "noise_std_percent": float(noise_level * 100.0),
                        "uncertain_parameter": param,
                        "output_metric": output,
                        "pearson_correlation": corr,
                        "abs_pearson_correlation": (
                            abs(corr) if np.isfinite(corr) else np.nan
                        ),
                    }
                )

    corr_df = pd.DataFrame(rows)

    if not corr_df.empty:
        corr_df = (
            corr_df.sort_values(
                [
                    "target",
                    "noise_level",
                    "output_metric",
                    "abs_pearson_correlation",
                ],
                ascending=[True, True, True, False],
            )
            .reset_index(drop=True)
        )

    return corr_df


# =============================================================================
# PLOTS
# =============================================================================

def save_distribution_figure(
    group: pd.DataFrame,
    target: float,
    noise_level: float,
) -> None:
    """Save distribution figure for one target/noise case."""
    tag = f"{target_tag(target)}_{noise_tag(noise_level)}"
    save_path = FIGURES_DIR / f"monte_carlo_distributions_{tag}.png"

    peak = group["peak_y"].to_numpy(dtype=float)
    err = group["target_error_mm"].to_numpy(dtype=float)
    margin = group["residual_margin_mm"].to_numpy(dtype=float)
    xr = group["max_abs_xr"].to_numpy(dtype=float)

    nominal_peak = float(group["nominal_peak_y"].iloc[0])
    nominal_margin = float(group["nominal_residual_margin_mm"].iloc[0])
    nominal_xr = float(group["nominal_max_abs_xr"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].hist(peak, bins=32, edgecolor="black", alpha=0.78)
    axes[0].axvline(target, linestyle="--", linewidth=2.0, label="Target")
    axes[0].axvline(nominal_peak, linestyle="-.", linewidth=2.0, label="Nominal")
    axes[0].set_xlabel("Peak outreach [m]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Peak outreach distribution")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=9)

    axes[1].hist(err, bins=32, edgecolor="black", alpha=0.78)
    axes[1].axvline(
        SUCCESS_TOLERANCE_MM,
        linestyle="--",
        linewidth=2.0,
        label="10 mm tolerance",
    )
    axes[1].set_xlabel("Target error [mm]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Target error distribution")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=9)

    axes[2].hist(xr, bins=32, edgecolor="black", alpha=0.78)
    axes[2].axvline(
        ROBOT_LIMIT_TRUE,
        linestyle="--",
        linewidth=2.0,
        label="True limit = 0.500 m",
    )
    axes[2].axvline(nominal_xr, linestyle="-.", linewidth=2.0, label="Nominal")
    axes[2].set_xlabel("max_abs_xr [m]")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Robot displacement distribution")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].legend(fontsize=9)

    axes[3].hist(margin, bins=32, edgecolor="black", alpha=0.78)
    axes[3].axvline(
        0.0,
        linestyle="--",
        linewidth=2.0,
        label="Feasibility boundary",
    )
    axes[3].axvline(nominal_margin, linestyle="-.", linewidth=2.0, label="Nominal")

    if np.min(margin) < 0.0:
        axes[3].axvspan(
            np.min(margin),
            0.0,
            alpha=0.15,
            label="Infeasible region",
        )

    axes[3].set_xlabel("Residual margin [mm]")
    axes[3].set_ylabel("Count")
    axes[3].set_title("Constraint margin distribution")
    axes[3].grid(True, alpha=0.3, axis="y")
    axes[3].legend(fontsize=9)

    feasible_rate = group["feasible_abs"].mean() * 100.0
    success_rate = group["success_10mm_abs"].mean() * 100.0

    fig.suptitle(
        f"Monte Carlo robustness: target={target:.2f} m, "
        f"noise std={noise_level * 100:.0f}%\n"
        f"feasible={feasible_rate:.1f}%, "
        f"success <=10 mm={success_rate:.1f}%",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


def save_peak_vs_xr_figure(
    group: pd.DataFrame,
    target: float,
    noise_level: float,
) -> None:
    """Save peak outreach versus robot displacement scatter plot."""
    tag = f"{target_tag(target)}_{noise_tag(noise_level)}"
    save_path = FIGURES_DIR / f"monte_carlo_peak_vs_xr_{tag}.png"

    feasible = group["feasible_abs"].astype(bool).to_numpy()
    peak = group["peak_y"].to_numpy(dtype=float)
    xr = group["max_abs_xr"].to_numpy(dtype=float)
    err = group["target_error_mm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 7))

    if np.any(feasible):
        ax.scatter(
            xr[feasible],
            peak[feasible],
            s=28,
            alpha=0.65,
            label="Feasible samples",
        )

    if np.any(~feasible):
        ax.scatter(
            xr[~feasible],
            peak[~feasible],
            s=36,
            marker="x",
            alpha=0.85,
            label="Infeasible samples",
        )

    ax.axvline(ROBOT_LIMIT_TRUE, linestyle="--", linewidth=2.0, label="x_r limit")
    ax.axhline(float(target), linestyle=":", linewidth=2.0, label="Target")

    ax.axhline(
        float(target) - SUCCESS_TOLERANCE_MM / 1000.0,
        linestyle="-.",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.axhline(
        float(target) + SUCCESS_TOLERANCE_MM / 1000.0,
        linestyle="-.",
        linewidth=1.2,
        alpha=0.8,
        label="±10 mm band",
    )

    ax.set_xlabel("max_abs_xr [m]")
    ax.set_ylabel("Peak outreach [m]")
    ax.set_title(
        f"Peak outreach vs robot displacement\n"
        f"target={target:.2f} m, noise std={noise_level * 100:.0f}%, "
        f"mean error={np.mean(err):.1f} mm"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


def save_summary_bar_plots(summary_df: pd.DataFrame) -> None:
    """Save compact bar plots from Monte Carlo summary."""
    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df["case"] = plot_df.apply(
        lambda row: f"{row['target']:.2f} m\n{row['noise_std_percent']:.0f}%",
        axis=1,
    )

    def bar_plot(
        value_col: str,
        ylabel: str,
        title: str,
        save_name: str,
        hline: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 6))

        ax.bar(plot_df["case"], plot_df[value_col], edgecolor="black")

        if hline is not None:
            ax.axhline(hline, linestyle="--", linewidth=2.0)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        path = FIGURES_DIR / save_name

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")

    bar_plot(
        "feasibility_rate_percent",
        "Feasibility rate [%]",
        "Monte Carlo feasibility rate",
        "monte_carlo_feasibility_summary.png",
        hline=100.0,
    )

    bar_plot(
        "success_rate_10mm_percent",
        "Safe tracking success rate [%]",
        "Monte Carlo safe tracking success: feasible and error <= 10 mm",
        "monte_carlo_success_summary.png",
        hline=100.0,
    )

    bar_plot(
        "stability_rate_20mm_percent",
        "Stability rate [%]",
        "Monte Carlo stability relative to nominal peak: |delta peak_y| <= 20 mm",
        "monte_carlo_stability_summary.png",
        hline=100.0,
    )

    bar_plot(
        "mean_target_error_mm",
        "Mean target error [mm]",
        "Monte Carlo mean target error",
        "monte_carlo_error_summary.png",
        hline=SUCCESS_TOLERANCE_MM,
    )

    bar_plot(
        "p05_residual_margin_mm",
        "5th percentile residual margin [mm]",
        "Monte Carlo lower-tail residual margin",
        "monte_carlo_margin_summary.png",
        hline=0.0,
    )


def save_correlation_heatmaps(corr_df: pd.DataFrame) -> None:
    """Save compact heatmaps for uncertainty-output correlations."""
    if corr_df.empty:
        return

    key_outputs = [
        "peak_y",
        "target_error_mm",
        "max_abs_xr",
        "residual_margin_mm",
    ]

    for (target, noise_level), group in corr_df.groupby(["target", "noise_level"]):
        plot_df = group[group["output_metric"].isin(key_outputs)].copy()

        if plot_df.empty:
            continue

        pivot = (
            plot_df.pivot(
                index="uncertain_parameter",
                columns="output_metric",
                values="pearson_correlation",
            )
            .reindex(index=UNCERTAIN_PARAMS, columns=key_outputs)
        )

        fig, ax = plt.subplots(figsize=(9.5, 5.2))

        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
        )

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_title(
            f"Monte Carlo uncertainty-output correlations\n"
            f"target={float(target):.2f} m, "
            f"noise std={float(noise_level) * 100:.0f}%"
        )

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]

                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Pearson correlation")

        path = (
            FIGURES_DIR
            / f"monte_carlo_correlation_heatmap_"
            / f"{target_tag(float(target))}_{noise_tag(float(noise_level))}.png"
        )

        # Avoid accidental nested path construction.
        path = FIGURES_DIR / (
            f"monte_carlo_correlation_heatmap_"
            f"{target_tag(float(target))}_{noise_tag(float(noise_level))}.png"
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def generate_plots(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    corr_df: pd.DataFrame | None,
    config: MonteCarloConfig,
) -> None:
    """Generate all Monte Carlo robustness figures."""
    print("\nGenerating Monte Carlo plots...")

    for (target, noise_level), group in results_df.groupby(["target", "noise_level"]):
        save_distribution_figure(group, float(target), float(noise_level))
        save_peak_vs_xr_figure(group, float(target), float(noise_level))

    save_summary_bar_plots(summary_df)

    if corr_df is not None and not config.skip_correlations:
        save_correlation_heatmaps(corr_df)


# =============================================================================
# OUTPUT AND PRINTING
# =============================================================================

def save_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    nominal_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save Monte Carlo output tables."""
    paths = {
        "results": RESULTS_DIR / "monte_carlo_results.csv",
        "summary": RESULTS_DIR / "monte_carlo_summary.csv",
        "nominal": RESULTS_DIR / "monte_carlo_nominal_metrics.csv",
        "correlations": RESULTS_DIR / "monte_carlo_uncertainty_correlations.csv",
    }

    results_df.to_csv(paths["results"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    nominal_df.to_csv(paths["nominal"], index=False)
    corr_df.to_csv(paths["correlations"], index=False)

    return paths


def print_summary(summary_df: pd.DataFrame) -> None:
    """Print compact Monte Carlo summary."""
    print("\n" + "=" * 80)
    print("MONTE CARLO ROBUSTNESS SUMMARY")
    print("=" * 80)

    display_cols = [
        "target",
        "noise_std_percent",
        "n_samples",
        "feasibility_rate_percent",
        "feasibility_rate_ci95_halfwidth_percent",
        "tracking_success_rate_10mm_percent",
        "success_rate_10mm_percent",
        "stability_rate_10mm_percent",
        "stability_rate_20mm_percent",
        "mean_abs_delta_peak_y_mm",
        "p95_abs_delta_peak_y_mm",
        "mean_target_error_mm",
        "p95_target_error_mm",
        "mean_residual_margin_mm",
        "p05_residual_margin_mm",
        "min_residual_margin_mm",
        "max_constraint_violation_mm",
    ]

    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo robustness analysis around final NN+Adam solutions."
    )

    parser.add_argument(
        "--physical_validation_selected",
        type=Path,
        default=DEFAULT_PHYSICAL_VALIDATION_SELECTED_PATH,
        help=(
            "Selected physical-validation candidates CSV. "
            f"Default: {DEFAULT_PHYSICAL_VALIDATION_SELECTED_PATH}."
        ),
    )

    parser.add_argument(
        "--nn_results",
        type=Path,
        default=DEFAULT_NN_RESULTS_PATH,
        help=f"Fallback NN+Adam results CSV. Default: {DEFAULT_NN_RESULTS_PATH}.",
    )

    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(str(target) for target in DEFAULT_TARGETS_TO_ANALYZE),
        help="Comma-separated targets to analyze.",
    )

    parser.add_argument(
        "--noise_levels",
        type=str,
        default=",".join(str(level) for level in DEFAULT_NOISE_LEVELS),
        help="Comma-separated Gaussian multiplicative noise standard deviations.",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per target/noise case. Default: {DEFAULT_N_SAMPLES}.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Base random seed. Default: {DEFAULT_RANDOM_SEED}.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    parser.add_argument(
        "--skip_correlations",
        action="store_true",
        help="Skip uncertainty correlation computation and heatmaps.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MonteCarloConfig:
    """Build configuration from command-line arguments."""
    if args.n_samples <= 0:
        raise ValueError("--n_samples must be positive.")

    noise_levels = parse_float_tuple(args.noise_levels)

    if any(level <= 0.0 for level in noise_levels):
        raise ValueError("All noise levels must be positive.")

    return MonteCarloConfig(
        physical_validation_selected_path=args.physical_validation_selected,
        nn_results_path=args.nn_results,
        targets_to_analyze=parse_float_tuple(args.targets),
        noise_levels=noise_levels,
        n_samples=int(args.n_samples),
        random_seed=int(args.random_seed),
        reference_method=REFERENCE_METHOD,
        skip_plots=bool(args.skip_plots),
        skip_correlations=bool(args.skip_correlations),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete Monte Carlo robustness analysis."""
    args = parse_args()
    config = build_config(args)

    ensure_dirs()

    print("\n" + "=" * 80)
    print("MONTE CARLO ROBUSTNESS OF FINAL OPTIMIZATION SOLUTIONS")
    print("=" * 80)
    print(f"Reference method:       {config.reference_method}")
    print(f"Targets to analyze:     {config.targets_to_analyze}")
    print(f"Noise levels:           {config.noise_levels}")
    print(f"Samples per case:       {config.n_samples}")
    print(f"True robot limit:       ±{ROBOT_LIMIT_TRUE:.3f} m")
    print(f"Selected candidates:    {config.physical_validation_selected_path}")
    print(f"Fallback NN results:    {config.nn_results_path}")
    print("=" * 80)

    reference_df = load_reference_solutions(config)
    results_df, summary_df, nominal_df = run_monte_carlo(reference_df, config)

    if config.skip_correlations:
        corr_df = pd.DataFrame()
    else:
        corr_df = compute_uncertainty_correlations(results_df)

    saved_paths = save_results(
        results_df=results_df,
        summary_df=summary_df,
        nominal_df=nominal_df,
        corr_df=corr_df,
    )

    print("\nSaved files:")
    for path in saved_paths.values():
        print(f"  {path}")

    if not config.skip_plots:
        generate_plots(
            results_df=results_df,
            summary_df=summary_df,
            corr_df=corr_df,
            config=config,
        )

    print_summary(summary_df)

    if not config.skip_plots:
        print(f"\nSaved figures in: {FIGURES_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()