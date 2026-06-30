"""
final_optimization_comparison.py
================================

Final comparison of inverse-optimization strategies:

    1. GP + Differential Evolution
    2. NN ensemble + multi-start Adam
    3. Bayesian Optimization on the true simulator
    4. Random Search baseline

The script reads the final result CSV files, standardizes the column names,
creates comparison tables, and generates report-ready figures.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GP_DIR = PROJECT_ROOT / "results" / "optimization_gp_de"
DEFAULT_NN_DIR = PROJECT_ROOT / "results" / "optimization_nn_gradient"
DEFAULT_BO_DIR = PROJECT_ROOT / "results" / "optimization_bo_final"

RESULTS_DIR = PROJECT_ROOT / "results" / "final_optimization_comparison"
FIGURES_DIR = PROJECT_ROOT / "figures" / "final_optimization_comparison"


# =============================================================================
# SETTINGS
# =============================================================================

TRUE_ROBOT_LIMIT_M = 0.500
OFFLINE_DATASET_CALLS_SURROGATE = 3000

TARGETS = np.array([0.55, 0.60, 0.65, 0.70, 0.75], dtype=float)
REACHABLE_TARGET_MAX = 0.65

METHOD_ORDER = [
    "GP+DE",
    "NN+Adam",
    "BO mean",
    "Random Search mean",
]

METHOD_MARKERS = {
    "GP+DE": "o",
    "NN+Adam": "s",
    "BO mean": "^",
    "Random Search mean": "D",
}

OPTIMIZED_PARAMS = ["Kr", "hr", "f0", "f1", "A", "x_r_start"]

PARAM_BOUNDS = {
    "Kr": (1500.0, 5000.0),
    "hr": (0.10, 0.45),
    "f0": (0.10, 0.45),
    "f1": (1.00, 4.00),
    "A": (0.09, 0.12),
    "x_r_start": (0.35, 0.40),
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for the final optimization comparison."""

    gp_dir: Path = DEFAULT_GP_DIR
    nn_dir: Path = DEFAULT_NN_DIR
    bo_dir: Path = DEFAULT_BO_DIR

    skip_plots: bool = False
    require_random_search: bool = False


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output folders."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path, required: bool = False) -> pd.DataFrame | None:
    """Read a CSV if available."""
    if path.exists():
        return pd.read_csv(path)

    if required:
        raise FileNotFoundError(f"Required CSV file not found: {path}")

    print(f"Optional file not found: {path}")

    return None


def first_existing(paths: Sequence[Path], required: bool = False) -> Path | None:
    """Return the first existing path from a list."""
    for path in paths:
        if path.exists():
            return path

    if required:
        joined = "\n".join(str(path) for path in paths)
        raise FileNotFoundError(f"None of these required files were found:\n{joined}")

    return None


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first matching column name from candidates, case-insensitive."""
    if df is None or df.empty:
        return None

    exact = {column: column for column in df.columns}

    for candidate in candidates:
        if candidate in exact:
            return candidate

    lower_map = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def to_bool_series(series: pd.Series) -> pd.Series:
    """Convert common boolean-like values to booleans."""
    if series.dtype == bool:
        return series

    if np.issubdtype(series.dtype, np.number):
        return series.astype(float) > 0.5

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def infer_target_error_mm(
    df: pd.DataFrame,
    target_col: str,
    peak_col: str,
) -> pd.Series:
    """Infer absolute target error in millimeters."""
    return (safe_numeric(df[peak_col]) - safe_numeric(df[target_col])).abs() * 1000.0


def infer_residual_margin_mm(
    df: pd.DataFrame,
    xr_col: str,
) -> pd.Series:
    """Infer residual robot-stroke margin in millimeters."""
    return (TRUE_ROBOT_LIMIT_M - safe_numeric(df[xr_col])) * 1000.0


def infer_violation_mm(
    df: pd.DataFrame,
    xr_col: str,
) -> pd.Series:
    """Infer absolute constraint violation in millimeters."""
    return np.maximum(
        0.0,
        (safe_numeric(df[xr_col]) - TRUE_ROBOT_LIMIT_M) * 1000.0,
    )


def count_by_target(
    df: pd.DataFrame | None,
    target_col_candidates: Sequence[str],
) -> dict[float, int]:
    """Count rows by target value."""
    if df is None or df.empty:
        return {}

    target_col = find_col(df, target_col_candidates)

    if target_col is None:
        return {}

    grouped = df.groupby(safe_numeric(df[target_col]).round(6)).size()

    return {float(key): int(value) for key, value in grouped.items()}


def match_target_counts(
    targets: pd.Series,
    counts: dict[float, int],
    default: int,
) -> list[int]:
    """Map target values to row counts."""
    output = []

    for target in targets:
        key = round(float(target), 6)
        output.append(int(counts.get(key, default)))

    return output


def method_sort_key(method: str) -> int:
    """Return ordering index for a method label."""
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return 999


def sort_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a table by method order and target."""
    if df.empty or "method" not in df.columns:
        return df

    df = df.copy()
    df["_method_order"] = df["method"].map(method_sort_key)

    sort_cols = ["_method_order"]

    if "target" in df.columns:
        sort_cols.append("target")

    df = df.sort_values(sort_cols).drop(columns="_method_order")

    return df.reset_index(drop=True)


def save_table_markdown(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a markdown table without external dependencies."""
    columns = list(df.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def normalize_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw method labels from BO final output files."""
    if df.empty:
        return df

    df = df.copy()

    if "method" not in df.columns:
        df["method"] = "BO"

    df["method"] = df["method"].astype(str)

    if "run_label" in df.columns:
        labels = df["run_label"].astype(str)

        random_mask = labels.str.contains("Random|RandomSearch|RS", case=False, regex=True)
        bo_mask = labels.str.contains("BO", case=False, regex=True)

        df.loc[random_mask, "method"] = "RandomSearch"
        df.loc[bo_mask & ~random_mask, "method"] = "BO"

    return df


# =============================================================================
# STANDARDIZATION
# =============================================================================

def standardize_direct_result(
    df: pd.DataFrame,
    method: str,
    offline_calls: int,
    online_calls_by_target: dict[float, int] | None = None,
    default_online_calls_per_target: int = 1,
) -> pd.DataFrame:
    """
    Standardize GP+DE and NN+Adam result files.

    These methods usually have one final validated row per target.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    target_col = find_col(df, ["target", "y_target", "target_peak_y_m"])
    peak_col = find_col(df, ["peak_y_true", "true_peak_y", "peak_y", "best_peak_y"])
    err_col = find_col(df, ["target_error_mm", "mean_target_error_mm", "error_mm"])
    xr_col = find_col(df, ["max_abs_xr_true", "true_max_abs_xr", "max_abs_xr"])
    viol_col = find_col(df, ["constraint_violation_abs_mm", "constraint_violation_mm", "violation_mm"])
    feas_col = find_col(df, ["feasible_abs", "feasible", "is_feasible"])
    time_col = find_col(df, ["optimization_time_s", "time_s", "runtime_s", "wall_time_s"])
    surrogate_col = find_col(df, ["surrogate_function_evaluations", "surrogate_evaluations", "n_surrogate_eval"])

    required = {
        "target": target_col,
        "peak_y_true": peak_col,
        "max_abs_xr_true": xr_col,
    }

    missing = [name for name, column in required.items() if column is None]

    if missing:
        raise ValueError(
            f"Cannot standardize {method}: missing columns {missing}. "
            f"Existing columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["method"] = method
    out["target"] = safe_numeric(df[target_col]).reset_index(drop=True)
    out["peak_y_true"] = safe_numeric(df[peak_col]).reset_index(drop=True)

    if err_col is not None:
        out["target_error_mm"] = safe_numeric(df[err_col]).reset_index(drop=True)
    else:
        out["target_error_mm"] = infer_target_error_mm(df, target_col, peak_col).reset_index(drop=True)

    out["max_abs_xr_true"] = safe_numeric(df[xr_col]).reset_index(drop=True)
    out["residual_margin_mm"] = infer_residual_margin_mm(df, xr_col).reset_index(drop=True)

    if viol_col is not None:
        out["constraint_violation_abs_mm"] = safe_numeric(df[viol_col]).reset_index(drop=True)
    else:
        out["constraint_violation_abs_mm"] = infer_violation_mm(df, xr_col).reset_index(drop=True)

    if feas_col is not None:
        out["feasible_abs"] = to_bool_series(df[feas_col]).reset_index(drop=True)
    else:
        out["feasible_abs"] = out["constraint_violation_abs_mm"] <= 1e-6

    if time_col is not None:
        out["optimization_time_s"] = safe_numeric(df[time_col]).reset_index(drop=True)
    else:
        out["optimization_time_s"] = np.nan

    if surrogate_col is not None:
        out["surrogate_function_evaluations"] = safe_numeric(df[surrogate_col]).reset_index(drop=True)
    else:
        out["surrogate_function_evaluations"] = np.nan

    if online_calls_by_target is not None:
        out["online_true_simulator_calls"] = match_target_counts(
            out["target"],
            online_calls_by_target,
            default_online_calls_per_target,
        )
    else:
        out["online_true_simulator_calls"] = int(default_online_calls_per_target)

    out["offline_dataset_calls"] = int(offline_calls)
    out["total_simulator_calls_equivalent"] = (
        out["offline_dataset_calls"] + out["online_true_simulator_calls"]
    )

    for pred_col in ["peak_y_pred", "max_abs_xr_pred", "peak_y_std", "max_abs_xr_std"]:
        source = find_col(df, [pred_col])

        if source is not None:
            out[pred_col] = safe_numeric(df[source]).reset_index(drop=True)

    for param in OPTIMIZED_PARAMS:
        source = find_col(df, [param])

        if source is not None:
            out[param] = safe_numeric(df[source]).reset_index(drop=True)

    return out


def aggregate_seed_results(
    df: pd.DataFrame,
    method: str,
    offline_calls: int,
    online_calls_per_seed: int,
) -> pd.DataFrame:
    """
    Aggregate BO or Random Search seed-level final results to target-level means.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_method_labels(df)

    target_col = find_col(df, ["target", "y_target", "target_peak_y_m"])
    peak_col = find_col(df, ["peak_y_true", "true_peak_y", "peak_y", "best_peak_y"])
    err_col = find_col(df, ["target_error_mm", "error_mm", "best_error_mm"])
    xr_col = find_col(df, ["max_abs_xr_true", "true_max_abs_xr", "max_abs_xr"])
    viol_col = find_col(df, ["constraint_violation_abs_mm", "constraint_violation_mm", "violation_mm"])
    feas_col = find_col(df, ["feasible_abs", "feasible", "is_feasible"])
    time_col = find_col(df, ["optimization_time_s", "time_s", "runtime_s", "wall_time_s"])
    calls_col = find_col(df, ["n_true_simulator_evaluations", "online_true_simulator_calls", "true_simulator_calls", "budget", "total_calls"])

    required = {
        "target": target_col,
        "peak_y_true": peak_col,
        "max_abs_xr_true": xr_col,
    }

    missing = [name for name, column in required.items() if column is None]

    if missing:
        raise ValueError(
            f"Cannot aggregate {method}: missing columns {missing}. "
            f"Existing columns: {list(df.columns)}"
        )

    work = pd.DataFrame()
    work["target"] = safe_numeric(df[target_col])
    work["peak_y_true"] = safe_numeric(df[peak_col])

    if err_col is not None:
        work["target_error_mm"] = safe_numeric(df[err_col])
    else:
        work["target_error_mm"] = (work["peak_y_true"] - work["target"]).abs() * 1000.0

    work["max_abs_xr_true"] = safe_numeric(df[xr_col])
    work["residual_margin_mm"] = (TRUE_ROBOT_LIMIT_M - work["max_abs_xr_true"]) * 1000.0

    if viol_col is not None:
        work["constraint_violation_abs_mm"] = safe_numeric(df[viol_col])
    else:
        work["constraint_violation_abs_mm"] = np.maximum(0.0, -work["residual_margin_mm"])

    if feas_col is not None:
        work["feasible_abs"] = to_bool_series(df[feas_col])
    else:
        work["feasible_abs"] = work["constraint_violation_abs_mm"] <= 1e-6

    if time_col is not None:
        work["optimization_time_s"] = safe_numeric(df[time_col])
    else:
        work["optimization_time_s"] = np.nan

    if calls_col is not None:
        work["calls_per_seed"] = safe_numeric(df[calls_col])
    else:
        work["calls_per_seed"] = float(online_calls_per_seed)

    for param in OPTIMIZED_PARAMS:
        source = find_col(df, [param, f"best_{param}"])

        if source is not None:
            work[param] = safe_numeric(df[source])

    rows: list[dict[str, Any]] = []

    for _, group in work.groupby(work["target"].round(6), as_index=False):
        row: dict[str, Any] = {
            "method": method,
            "target": float(group["target"].iloc[0]),
            "peak_y_true": float(group["peak_y_true"].mean()),
            "peak_y_true_std": float(group["peak_y_true"].std(ddof=0)) if len(group) > 1 else 0.0,
            "target_error_mm": float(group["target_error_mm"].mean()),
            "target_error_mm_std": float(group["target_error_mm"].std(ddof=0)) if len(group) > 1 else 0.0,
            "target_error_mm_min": float(group["target_error_mm"].min()),
            "target_error_mm_max": float(group["target_error_mm"].max()),
            "max_abs_xr_true": float(group["max_abs_xr_true"].mean()),
            "max_abs_xr_true_std": float(group["max_abs_xr_true"].std(ddof=0)) if len(group) > 1 else 0.0,
            "residual_margin_mm": float(group["residual_margin_mm"].mean()),
            "residual_margin_mm_min_seed": float(group["residual_margin_mm"].min()),
            "constraint_violation_abs_mm": float(group["constraint_violation_abs_mm"].mean()),
            "feasible_abs": bool(group["feasible_abs"].all()),
            "feasibility_rate_seed_percent": float(group["feasible_abs"].mean() * 100.0),
            "optimization_time_s": (
                float(group["optimization_time_s"].mean())
                if group["optimization_time_s"].notna().any()
                else np.nan
            ),
            "online_true_simulator_calls": int(group["calls_per_seed"].sum()),
            "offline_dataset_calls": int(offline_calls),
            "surrogate_function_evaluations": np.nan,
        }

        row["total_simulator_calls_equivalent"] = (
            row["online_true_simulator_calls"] + row["offline_dataset_calls"]
        )

        for param in OPTIMIZED_PARAMS:
            if param in group.columns:
                row[param] = float(group[param].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def standardize_summary_by_target(
    df: pd.DataFrame,
    method: str,
    offline_calls: int,
    fallback_online_calls_per_target: int,
) -> pd.DataFrame:
    """Standardize BO summary-by-target tables when seed-level results are missing."""
    if df is None or df.empty:
        return pd.DataFrame()

    target_col = find_col(df, ["target", "y_target", "target_peak_y_m"])
    peak_col = find_col(df, ["mean_peak_y_true", "mean_peak_y", "peak_y_true"])
    err_col = find_col(df, ["mean_target_error_mm", "target_error_mm", "mean_error_mm"])
    xr_col = find_col(df, ["mean_max_abs_xr_true", "mean_max_abs_xr", "max_abs_xr_true"])
    margin_col = find_col(df, ["mean_residual_margin_mm", "residual_margin_mm"])
    min_margin_col = find_col(df, ["min_residual_margin_mm", "residual_margin_min_mm"])
    feas_col = find_col(df, ["feasibility_rate_percent", "feasible_rate_percent", "feasible_abs"])
    calls_col = find_col(df, ["total_true_simulator_calls", "online_true_simulator_calls", "true_simulator_calls"])
    time_col = find_col(df, ["mean_optimization_time_s", "optimization_time_s", "time_s"])

    if target_col is None or peak_col is None or err_col is None or xr_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["method"] = method
    out["target"] = safe_numeric(df[target_col]).reset_index(drop=True)
    out["peak_y_true"] = safe_numeric(df[peak_col]).reset_index(drop=True)
    out["target_error_mm"] = safe_numeric(df[err_col]).reset_index(drop=True)
    out["max_abs_xr_true"] = safe_numeric(df[xr_col]).reset_index(drop=True)

    if margin_col is not None:
        out["residual_margin_mm"] = safe_numeric(df[margin_col]).reset_index(drop=True)
    else:
        out["residual_margin_mm"] = (TRUE_ROBOT_LIMIT_M - out["max_abs_xr_true"]) * 1000.0

    if min_margin_col is not None:
        out["residual_margin_mm_min_seed"] = safe_numeric(df[min_margin_col]).reset_index(drop=True)

    out["constraint_violation_abs_mm"] = np.maximum(0.0, -out["residual_margin_mm"])

    if feas_col is not None and df[feas_col].dtype == bool:
        out["feasible_abs"] = to_bool_series(df[feas_col]).reset_index(drop=True)
    elif feas_col is not None:
        values = safe_numeric(df[feas_col])
        out["feasible_abs"] = values >= 99.999
        out["feasibility_rate_seed_percent"] = values.reset_index(drop=True)
    else:
        out["feasible_abs"] = out["constraint_violation_abs_mm"] <= 1e-6

    if calls_col is not None:
        out["online_true_simulator_calls"] = (
            safe_numeric(df[calls_col])
            .fillna(fallback_online_calls_per_target)
            .astype(int)
            .reset_index(drop=True)
        )
    else:
        out["online_true_simulator_calls"] = int(fallback_online_calls_per_target)

    out["offline_dataset_calls"] = int(offline_calls)
    out["total_simulator_calls_equivalent"] = (
        out["offline_dataset_calls"] + out["online_true_simulator_calls"]
    )

    if time_col is not None:
        out["optimization_time_s"] = safe_numeric(df[time_col]).reset_index(drop=True)
    else:
        out["optimization_time_s"] = np.nan

    out["surrogate_function_evaluations"] = np.nan

    return out


# =============================================================================
# LOADERS
# =============================================================================

def load_gp_de(config: ComparisonConfig) -> pd.DataFrame:
    """Load final GP+DE results."""
    print("Loading GP+DE results...")

    results = read_csv_if_exists(config.gp_dir / "gp_de_results.csv", required=True)
    validated = read_csv_if_exists(
        config.gp_dir / "gp_de_all_validated_candidates.csv",
        required=False,
    )

    counts = count_by_target(validated, ["target"])

    return standardize_direct_result(
        results,
        method="GP+DE",
        offline_calls=OFFLINE_DATASET_CALLS_SURROGATE,
        online_calls_by_target=counts,
        default_online_calls_per_target=25,
    )


def load_nn_adam(config: ComparisonConfig) -> pd.DataFrame:
    """Load final NN+Adam results."""
    print("Loading NN+Adam results...")

    results_path = first_existing(
        [
            config.nn_dir / "nn_gradient_results.csv",
            config.nn_dir / "nn_adam_results.csv",
        ],
        required=True,
    )

    results = read_csv_if_exists(results_path, required=True)

    validated_path = first_existing(
        [
            config.nn_dir / "nn_gradient_validated_candidates.csv",
            config.nn_dir / "nn_adam_validated_candidates.csv",
        ],
        required=False,
    )

    validated = (
        read_csv_if_exists(validated_path, required=False)
        if validated_path is not None
        else None
    )

    counts = count_by_target(validated, ["target"])

    return standardize_direct_result(
        results,
        method="NN+Adam",
        offline_calls=OFFLINE_DATASET_CALLS_SURROGATE,
        online_calls_by_target=counts,
        default_online_calls_per_target=50,
    )


def filter_final_method(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Filter BO final all-results table by method."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_method_labels(df)

    if "method" not in df.columns:
        return pd.DataFrame()

    return df[df["method"].astype(str) == method_name].copy()


def load_bo_final(config: ComparisonConfig) -> pd.DataFrame:
    """Load final Bayesian Optimization results."""
    print("Loading BO final results...")

    bo_only_path = first_existing(
        [
            config.bo_dir / "bo_final_results_by_seed.csv",
            config.bo_dir / "bo_final_results.csv",
        ],
        required=False,
    )

    if bo_only_path is not None:
        by_seed = read_csv_if_exists(bo_only_path, required=False)

        if by_seed is not None and not by_seed.empty:
            return aggregate_seed_results(
                by_seed,
                method="BO mean",
                offline_calls=0,
                online_calls_per_seed=200,
            )

    all_results_path = first_existing(
        [
            config.bo_dir / "bo_final_all_results_by_seed.csv",
            config.bo_dir / "bo_final_all_results.csv",
        ],
        required=False,
    )

    if all_results_path is not None:
        all_results = read_csv_if_exists(all_results_path, required=False)
        bo_results = filter_final_method(all_results, "BO")

        if not bo_results.empty:
            return aggregate_seed_results(
                bo_results,
                method="BO mean",
                offline_calls=0,
                online_calls_per_seed=200,
            )

    summary_path = config.bo_dir / "bo_final_summary_by_target.csv"
    summary = read_csv_if_exists(summary_path, required=True)

    if "method" in summary.columns:
        summary = summary[summary["method"].astype(str) == "BO"].copy()

    out = standardize_summary_by_target(
        summary,
        method="BO mean",
        offline_calls=0,
        fallback_online_calls_per_target=400,
    )

    if out.empty:
        raise ValueError(
            f"Could not standardize BO final summary file. "
            f"Columns: {list(summary.columns)}"
        )

    return out


def load_random_search(config: ComparisonConfig) -> pd.DataFrame:
    """Load final Random Search baseline results."""
    print("Loading Random Search final results...")

    random_path = first_existing(
        [
            config.bo_dir / "random_search_results.csv",
            config.bo_dir / "random_search_final_results_by_seed.csv",
        ],
        required=False,
    )

    if random_path is not None:
        by_seed = read_csv_if_exists(random_path, required=False)

        if by_seed is not None and not by_seed.empty:
            return aggregate_seed_results(
                by_seed,
                method="Random Search mean",
                offline_calls=0,
                online_calls_per_seed=200,
            )

    all_results_path = first_existing(
        [
            config.bo_dir / "bo_final_all_results_by_seed.csv",
            config.bo_dir / "bo_final_all_results.csv",
        ],
        required=config.require_random_search,
    )

    if all_results_path is None:
        print("Random Search results not found; skipping baseline.")
        return pd.DataFrame()

    all_results = read_csv_if_exists(all_results_path, required=config.require_random_search)

    random_results = filter_final_method(all_results, "RandomSearch")

    if random_results.empty:
        if config.require_random_search:
            raise FileNotFoundError("Random Search rows were not found in final BO outputs.")

        print("Random Search rows not found; skipping baseline.")
        return pd.DataFrame()

    return aggregate_seed_results(
        random_results,
        method="Random Search mean",
        offline_calls=0,
        online_calls_per_seed=200,
    )


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def create_by_target_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """Create standardized by-target comparison table."""
    keep_cols = [
        "method",
        "target",
        "peak_y_true",
        "peak_y_true_std",
        "target_error_mm",
        "target_error_mm_std",
        "target_error_mm_min",
        "target_error_mm_max",
        "max_abs_xr_true",
        "max_abs_xr_true_std",
        "residual_margin_mm",
        "residual_margin_mm_min_seed",
        "constraint_violation_abs_mm",
        "feasible_abs",
        "feasibility_rate_seed_percent",
        "optimization_time_s",
        "online_true_simulator_calls",
        "offline_dataset_calls",
        "total_simulator_calls_equivalent",
        "surrogate_function_evaluations",
        "peak_y_pred",
        "max_abs_xr_pred",
        "peak_y_std",
        "max_abs_xr_std",
        *OPTIMIZED_PARAMS,
    ]

    all_df = all_df.copy()

    for col in keep_cols:
        if col not in all_df.columns:
            all_df[col] = np.nan

    out = all_df[keep_cols].copy()
    out = out[out["method"].notna()].copy()

    return sort_methods(out)


def create_reachable_vs_high_summary(by_target: pd.DataFrame) -> pd.DataFrame:
    """Compare reachable targets and saturated high targets."""
    rows: list[dict[str, Any]] = []

    for method, df_method in by_target.groupby("method", sort=False):
        reachable = df_method[df_method["target"] <= REACHABLE_TARGET_MAX]
        high = df_method[df_method["target"] > REACHABLE_TARGET_MAX]

        rows.append(
            {
                "method": method,
                "reachable_targets": "0.55-0.65",
                "high_targets": "0.70-0.75",
                "mean_error_reachable_targets_mm": float(reachable["target_error_mm"].mean()),
                "max_error_reachable_targets_mm": float(reachable["target_error_mm"].max()),
                "mean_error_high_targets_mm": float(high["target_error_mm"].mean()),
                "max_error_high_targets_mm": float(high["target_error_mm"].max()),
            }
        )

    return sort_methods(pd.DataFrame(rows))


def create_summary_table(by_target: pd.DataFrame) -> pd.DataFrame:
    """Create method-level summary table."""
    reachable_high = create_reachable_vs_high_summary(by_target)
    rows: list[dict[str, Any]] = []

    for method, df_method in by_target.groupby("method", sort=False):
        row: dict[str, Any] = {
            "method": method,
            "n_targets": int(len(df_method)),
            "feasible_targets": int(df_method["feasible_abs"].sum()),
            "feasibility_rate_percent": float(df_method["feasible_abs"].mean() * 100.0),
            "mean_target_error_mm": float(df_method["target_error_mm"].mean()),
            "median_target_error_mm": float(df_method["target_error_mm"].median()),
            "max_target_error_mm": float(df_method["target_error_mm"].max()),
            "mean_true_max_abs_xr_m": float(df_method["max_abs_xr_true"].mean()),
            "max_true_max_abs_xr_m": float(df_method["max_abs_xr_true"].max()),
            "mean_residual_margin_mm": float(df_method["residual_margin_mm"].mean()),
            "min_residual_margin_mm": float(df_method["residual_margin_mm"].min()),
            "mean_constraint_violation_abs_mm": float(df_method["constraint_violation_abs_mm"].mean()),
            "max_constraint_violation_abs_mm": float(df_method["constraint_violation_abs_mm"].max()),
            "offline_simulator_calls": int(df_method["offline_dataset_calls"].max()),
            "online_simulator_calls_total": int(df_method["online_true_simulator_calls"].sum()),
            "online_simulator_calls_per_target_mean": float(df_method["online_true_simulator_calls"].mean()),
            "total_simulator_calls_equivalent": int(
                df_method["offline_dataset_calls"].max()
                + df_method["online_true_simulator_calls"].sum()
            ),
            "total_optimization_time_s": float(df_method["optimization_time_s"].sum(skipna=True)),
            "mean_optimization_time_s_per_target": float(df_method["optimization_time_s"].mean(skipna=True)),
            "mean_surrogate_function_evaluations": float(
                df_method["surrogate_function_evaluations"].mean(skipna=True)
            ),
            "total_surrogate_function_evaluations": float(
                df_method["surrogate_function_evaluations"].sum(skipna=True)
            ),
        }

        rh = reachable_high[reachable_high["method"] == method]

        if not rh.empty:
            row.update(rh.iloc[0].drop(labels="method").to_dict())

        rows.append(row)

    return sort_methods(pd.DataFrame(rows))


def create_cost_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Create simulator-call and runtime summary."""
    cols = [
        "method",
        "offline_simulator_calls",
        "online_simulator_calls_total",
        "online_simulator_calls_per_target_mean",
        "total_simulator_calls_equivalent",
        "total_optimization_time_s",
        "mean_optimization_time_s_per_target",
        "total_surrogate_function_evaluations",
    ]

    return summary[cols].copy()


def create_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create compact report-friendly method comparison table."""
    cols = [
        "method",
        "feasibility_rate_percent",
        "mean_target_error_mm",
        "mean_error_reachable_targets_mm",
        "mean_error_high_targets_mm",
        "max_target_error_mm",
        "min_residual_margin_mm",
        "online_simulator_calls_total",
        "offline_simulator_calls",
        "total_simulator_calls_equivalent",
    ]

    report = summary[cols].copy()

    report = report.rename(
        columns={
            "method": "Method",
            "feasibility_rate_percent": "Feasible [%]",
            "mean_target_error_mm": "Mean error [mm]",
            "mean_error_reachable_targets_mm": "Reachable error [mm]",
            "mean_error_high_targets_mm": "High-target error [mm]",
            "max_target_error_mm": "Max error [mm]",
            "min_residual_margin_mm": "Min margin [mm]",
            "online_simulator_calls_total": "Online calls",
            "offline_simulator_calls": "Offline calls",
            "total_simulator_calls_equivalent": "Total calls",
        }
    )

    numeric_cols = report.select_dtypes(include=[np.number]).columns
    report[numeric_cols] = report[numeric_cols].round(3)

    return report


# =============================================================================
# PLOTS
# =============================================================================

def plot_target_tracking(by_target: pd.DataFrame) -> None:
    """Plot true peak outreach versus target."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHOD_ORDER:
        df = by_target[by_target["method"] == method].sort_values("target")

        if df.empty:
            continue

        ax.plot(
            df["target"],
            df["peak_y_true"],
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2,
            label=method,
        )

    ax.plot(TARGETS, TARGETS, linestyle=":", linewidth=2, label="Ideal tracking")
    ax.axhline(TRUE_ROBOT_LIMIT_M, linestyle="--", linewidth=1.5, label="Nominal robot reach")

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("True peak_y [m]")
    ax.set_title("Final inverse-optimization target tracking")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    path = FIGURES_DIR / "01_final_target_tracking.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_grouped_bars(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
    hline_label: str | None = None,
) -> None:
    """Plot grouped bar chart by target and method."""
    pivot = df.pivot_table(
        index="target",
        columns="method",
        values=value_col,
        aggfunc="first",
    )

    pivot = pivot[[method for method in METHOD_ORDER if method in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot.plot(kind="bar", ax=ax, edgecolor="black", width=0.82)

    if hline is not None:
        ax.axhline(
            hline,
            linestyle="--",
            linewidth=2,
            label=hline_label or f"{hline:g}",
        )

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticklabels([f"{float(target):.2f}" for target in pivot.index], rotation=0)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_target_error(by_target: pd.DataFrame) -> None:
    """Plot target error by method and target."""
    plot_grouped_bars(
        by_target,
        value_col="target_error_mm",
        ylabel="True target error [mm]",
        title="Final target error by method and target",
        path=FIGURES_DIR / "02_final_target_error_by_target.png",
        hline=10.0,
        hline_label="10 mm reference",
    )


def plot_constraint_validation(by_target: pd.DataFrame) -> None:
    """Plot true max_abs_xr by method and target."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHOD_ORDER:
        df = by_target[by_target["method"] == method].sort_values("target")

        if df.empty:
            continue

        ax.plot(
            df["target"],
            df["max_abs_xr_true"],
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2,
            label=method,
        )

    ax.axhspan(0.495, 0.500, alpha=0.15, label="Nominal safety band")
    ax.axhline(TRUE_ROBOT_LIMIT_M, linestyle="--", linewidth=2, label="True limit 0.500 m")
    ax.axhline(0.495, linestyle=":", linewidth=2, label="Surrogate safety reference 0.495 m")

    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("True max_abs_xr [m]")
    ax.set_title("Constraint validation on true simulator")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    path = FIGURES_DIR / "03_final_constraint_validation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_residual_margin(by_target: pd.DataFrame) -> None:
    """Plot residual margin by method and target."""
    plot_grouped_bars(
        by_target,
        value_col="residual_margin_mm",
        ylabel="Residual margin [mm]",
        title="Residual margin to true robot limit",
        path=FIGURES_DIR / "04_final_residual_margin.png",
        hline=0.0,
        hline_label="Feasibility boundary",
    )


def plot_reachable_vs_high(reachable_high: pd.DataFrame) -> None:
    """Plot reachable-target and high-target errors."""
    if reachable_high.empty:
        return

    plot_df = reachable_high.set_index("method")[
        [
            "mean_error_reachable_targets_mm",
            "mean_error_high_targets_mm",
        ]
    ]

    plot_df = plot_df.reindex([method for method in METHOD_ORDER if method in plot_df.index])

    plot_df = plot_df.rename(
        columns={
            "mean_error_reachable_targets_mm": "Reachable targets 0.55-0.65",
            "mean_error_high_targets_mm": "High targets 0.70-0.75",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    plot_df.plot(kind="bar", ax=ax, edgecolor="black")

    ax.set_xlabel("Method")
    ax.set_ylabel("Mean target error [mm]")
    ax.set_title("Tracking error in reachable and saturated target regions")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
    ax.legend(fontsize=8)

    path = FIGURES_DIR / "05_final_reachable_vs_high_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_runtime(summary: pd.DataFrame) -> None:
    """Plot runtime comparison."""
    if summary.empty:
        return

    df = summary.set_index("method").reindex(
        [method for method in METHOD_ORDER if method in summary["method"].values]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(df.index, df["total_optimization_time_s"], edgecolor="black")
    ax.set_yscale("log")

    ax.set_xlabel("Method")
    ax.set_ylabel("Total optimization wall time [s], log scale")
    ax.set_title("Optimization runtime comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    path = FIGURES_DIR / "06_final_runtime_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_simulator_calls(summary: pd.DataFrame) -> None:
    """Plot offline and online simulator-call cost."""
    if summary.empty:
        return

    df = summary.set_index("method").reindex(
        [method for method in METHOD_ORDER if method in summary["method"].values]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        df.index,
        df["offline_simulator_calls"],
        edgecolor="black",
        label="Offline dataset calls",
    )

    ax.bar(
        df.index,
        df["online_simulator_calls_total"],
        bottom=df["offline_simulator_calls"],
        edgecolor="black",
        label="Online optimization / validation calls",
    )

    ax.set_xlabel("Method")
    ax.set_ylabel("Simulator calls")
    ax.set_title("Offline and online simulator-call cost")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)

    path = FIGURES_DIR / "07_final_simulator_calls_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_error_vs_cost(summary: pd.DataFrame) -> None:
    """Plot mean target error versus total simulator-call equivalent."""
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in summary.iterrows():
        method = row["method"]

        ax.scatter(
            row["total_simulator_calls_equivalent"],
            row["mean_target_error_mm"],
            s=110,
            marker=METHOD_MARKERS.get(method, "o"),
            edgecolor="black",
            label=method,
        )

        ax.annotate(
            method,
            (
                row["total_simulator_calls_equivalent"],
                row["mean_target_error_mm"],
            ),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )

    ax.set_xlabel(
        "Total simulator calls equivalent\n"
        "(offline dataset + online validation / optimization)"
    )
    ax.set_ylabel("Mean target error [mm]")
    ax.set_title("Accuracy-cost trade-off")
    ax.grid(True, alpha=0.3)

    if summary["total_simulator_calls_equivalent"].min() > 0:
        ax.set_xscale("log")

    path = FIGURES_DIR / "08_final_error_vs_cost_pareto.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_parameter_comparison_target065(by_target: pd.DataFrame) -> None:
    """Plot normalized optimized parameters for target 0.65 m."""
    df = by_target[np.isclose(by_target["target"], 0.65)].copy()

    available_params = [
        param
        for param in OPTIMIZED_PARAMS
        if param in df.columns and df[param].notna().any()
    ]

    if df.empty or not available_params:
        print("Skipping parameter comparison at target 0.65: no parameter columns available.")
        return

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        for param in available_params:
            if pd.isna(row[param]):
                continue

            low, high = PARAM_BOUNDS[param]

            rows.append(
                {
                    "method": row["method"],
                    "parameter": param,
                    "normalized_value": (float(row[param]) - low) / (high - low),
                }
            )

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        print("Skipping parameter comparison at target 0.65: empty normalized table.")
        return

    pivot = plot_df.pivot_table(
        index="parameter",
        columns="method",
        values="normalized_value",
        aggfunc="first",
    )

    pivot = pivot[[method for method in METHOD_ORDER if method in pivot.columns]]
    pivot = pivot.reindex([param for param in OPTIMIZED_PARAMS if param in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot.plot(kind="bar", ax=ax, edgecolor="black")

    ax.set_xlabel("Optimized parameter")
    ax.set_ylabel("Normalized value within bounds")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Optimized parameter comparison at target 0.65 m")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(fontsize=8)

    path = FIGURES_DIR / "09_final_parameter_comparison_target065.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def generate_plots(
    by_target: pd.DataFrame,
    summary: pd.DataFrame,
    reachable_high: pd.DataFrame,
) -> None:
    """Generate all final comparison figures."""
    print("\nGenerating final comparison plots...")

    plot_target_tracking(by_target)
    plot_target_error(by_target)
    plot_constraint_validation(by_target)
    plot_residual_margin(by_target)
    plot_reachable_vs_high(reachable_high)
    plot_runtime(summary)
    plot_simulator_calls(summary)
    plot_error_vs_cost(summary)
    plot_parameter_comparison_target065(by_target)


# =============================================================================
# OUTPUT
# =============================================================================

def save_tables(
    by_target: pd.DataFrame,
    summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    reachable_high: pd.DataFrame,
    report_table: pd.DataFrame,
) -> dict[str, Path]:
    """Save final comparison tables."""
    paths = {
        "by_target": RESULTS_DIR / "final_comparison_by_target.csv",
        "summary": RESULTS_DIR / "final_comparison_summary.csv",
        "cost": RESULTS_DIR / "final_cost_summary.csv",
        "reachable_high": RESULTS_DIR / "final_reachable_vs_high_summary.csv",
        "report_table": RESULTS_DIR / "final_comparison_report_table.csv",
        "report_table_md": RESULTS_DIR / "final_comparison_report_table.md",
    }

    by_target.to_csv(paths["by_target"], index=False)
    summary.to_csv(paths["summary"], index=False)
    cost_summary.to_csv(paths["cost"], index=False)
    reachable_high.to_csv(paths["reachable_high"], index=False)
    report_table.to_csv(paths["report_table"], index=False)
    save_table_markdown(report_table, paths["report_table_md"])

    return paths


def print_loaded_table(by_target: pd.DataFrame) -> None:
    """Print compact by-target table."""
    display_cols = [
        "method",
        "target",
        "peak_y_true",
        "target_error_mm",
        "max_abs_xr_true",
        "residual_margin_mm",
        "feasible_abs",
        "online_true_simulator_calls",
        "offline_dataset_calls",
    ]

    available_cols = [col for col in display_cols if col in by_target.columns]

    print("\nFinal comparison by target:")
    print(by_target[available_cols].to_string(index=False))


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print compact method-level summary."""
    display_cols = [
        "method",
        "feasibility_rate_percent",
        "mean_target_error_mm",
        "mean_error_reachable_targets_mm",
        "mean_error_high_targets_mm",
        "max_target_error_mm",
        "min_residual_margin_mm",
        "online_simulator_calls_total",
        "offline_simulator_calls",
        "total_simulator_calls_equivalent",
    ]

    available_cols = [col for col in display_cols if col in summary.columns]

    print("\nFinal comparison summary:")
    print(summary[available_cols].to_string(index=False))


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create final comparison tables and figures for all optimization methods."
    )

    parser.add_argument(
        "--gp_dir",
        type=Path,
        default=DEFAULT_GP_DIR,
        help=f"GP+DE results directory. Default: {DEFAULT_GP_DIR}.",
    )

    parser.add_argument(
        "--nn_dir",
        type=Path,
        default=DEFAULT_NN_DIR,
        help=f"NN+Adam results directory. Default: {DEFAULT_NN_DIR}.",
    )

    parser.add_argument(
        "--bo_dir",
        type=Path,
        default=DEFAULT_BO_DIR,
        help=f"BO final results directory. Default: {DEFAULT_BO_DIR}.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )

    parser.add_argument(
        "--require_random_search",
        action="store_true",
        help="Raise an error if Random Search final results are missing.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ComparisonConfig:
    """Build comparison config from CLI arguments."""
    return ComparisonConfig(
        gp_dir=args.gp_dir,
        nn_dir=args.nn_dir,
        bo_dir=args.bo_dir,
        skip_plots=bool(args.skip_plots),
        require_random_search=bool(args.require_random_search),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete final optimization comparison."""
    args = parse_args()
    config = build_config(args)

    ensure_dirs()

    print("=" * 80)
    print("FINAL OPTIMIZATION METHOD COMPARISON")
    print("=" * 80)
    print(f"GP+DE directory:      {config.gp_dir}")
    print(f"NN+Adam directory:    {config.nn_dir}")
    print(f"BO directory:         {config.bo_dir}")
    print(f"Results directory:    {RESULTS_DIR}")
    print(f"Figures directory:    {FIGURES_DIR}")
    print("=" * 80)

    tables: list[pd.DataFrame] = []

    loaders = [
        load_gp_de,
        load_nn_adam,
        load_bo_final,
        load_random_search,
    ]

    for loader in loaders:
        try:
            table = loader(config)

            if table is not None and not table.empty:
                tables.append(table)

        except Exception as exc:
            print(f"Could not load method with {loader.__name__}: {exc}")

    if not tables:
        raise RuntimeError("No comparison tables could be loaded.")

    all_df = pd.concat(tables, ignore_index=True, sort=False)

    by_target = create_by_target_table(all_df)
    reachable_high = create_reachable_vs_high_summary(by_target)
    summary = create_summary_table(by_target)
    cost_summary = create_cost_summary(summary)
    report_table = create_report_table(summary)

    saved_paths = save_tables(
        by_target=by_target,
        summary=summary,
        cost_summary=cost_summary,
        reachable_high=reachable_high,
        report_table=report_table,
    )

    print_loaded_table(by_target)
    print_summary_table(summary)

    if not config.skip_plots:
        generate_plots(by_target, summary, reachable_high)

    print("\nSaved tables:")
    for path in saved_paths.values():
        print(f"  - {path}")

    if not config.skip_plots:
        print(f"\nSaved figures in: {FIGURES_DIR}")

    print("\nFinal optimization comparison completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()