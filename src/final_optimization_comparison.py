"""
final_optimization_comparison.py
================================

Final comparison of inverse-optimization strategies:

    1. GP + Differential Evolution, uncertainty-aware constraint, beta=0.5
    2. NN v2 ensemble + multi-start Adam, safe configuration
    3. Bayesian Optimization on the true simulator, final configuration
    4. Random Search baseline

The script reads the final result CSV files, standardizes the column names,
creates comparison tables and report-ready figures.

Main outputs
------------
results/final_optimization_comparison/final_comparison_by_target.csv
results/final_optimization_comparison/final_comparison_summary.csv
results/final_optimization_comparison/final_cost_summary.csv
results/final_optimization_comparison/final_reachable_vs_high_summary.csv

figures/final_optimization_comparison/01_final_target_tracking.png
figures/final_optimization_comparison/02_final_target_error_by_target.png
figures/final_optimization_comparison/03_final_constraint_validation.png
figures/final_optimization_comparison/04_final_residual_margin.png
figures/final_optimization_comparison/05_final_reachable_vs_high_error.png
figures/final_optimization_comparison/06_final_runtime_comparison.png
figures/final_optimization_comparison/07_final_simulator_calls_comparison.png
figures/final_optimization_comparison/08_final_error_vs_cost_pareto.png
figures/final_optimization_comparison/09_final_parameter_comparison_target065.png

Author: MatteoCasazza
Date: 2026
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GP_DIR = PROJECT_ROOT / "results" / "optimization_gp_de_v2_beta05"
NN_DIR = PROJECT_ROOT / "results" / "optimization_nn_gradient_v2_safe"
BO_DIR = PROJECT_ROOT / "results" / "optimization_bo_final"

RESULTS_DIR = PROJECT_ROOT / "results" / "final_optimization_comparison"
FIGURES_DIR = PROJECT_ROOT / "figures" / "final_optimization_comparison"

TRUE_ROBOT_LIMIT_M = 0.500
OFFLINE_DATASET_CALLS_SURROGATE = 3000
TARGETS = np.array([0.55, 0.60, 0.65, 0.70, 0.75], dtype=float)
REACHABLE_TARGET_MAX = 0.65

METHOD_ORDER = [
    "GP+DE beta=0.5",
    "NN+Adam safe",
    "BO mean",
    "Random Search mean",
]

METHOD_MARKERS = {
    "GP+DE beta=0.5": "o",
    "NN+Adam safe": "s",
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
# SMALL UTILITIES
# =============================================================================


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path, required: bool = False) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    print(f"⚠ Optional file not found: {path}")
    return None


def first_existing(paths: Sequence[Path], required: bool = False) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    if required:
        joined = "\n".join(str(p) for p in paths)
        raise FileNotFoundError(f"None of these required files were found:\n{joined}")
    return None


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column name from candidates, case-insensitive."""
    if df is None or df.empty:
        return None
    exact = {c: c for c in df.columns}
    for cand in candidates:
        if cand in exact:
            return cand
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float) > 0.5
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_target_error_mm(df: pd.DataFrame, target_col: str, peak_col: str) -> pd.Series:
    return (safe_numeric(df[peak_col]) - safe_numeric(df[target_col])).abs() * 1000.0


def infer_residual_margin_mm(df: pd.DataFrame, xr_col: str) -> pd.Series:
    return (TRUE_ROBOT_LIMIT_M - safe_numeric(df[xr_col])) * 1000.0


def infer_violation_mm(df: pd.DataFrame, xr_col: str) -> pd.Series:
    return np.maximum(0.0, (safe_numeric(df[xr_col]) - TRUE_ROBOT_LIMIT_M) * 1000.0)


def count_by_target(df: Optional[pd.DataFrame], target_col_candidates: Sequence[str]) -> Dict[float, int]:
    if df is None or df.empty:
        return {}
    target_col = find_col(df, target_col_candidates)
    if target_col is None:
        return {}
    grouped = df.groupby(safe_numeric(df[target_col]).round(6)).size()
    return {float(k): int(v) for k, v in grouped.items()}


def match_target_counts(targets: pd.Series, counts: Dict[float, int], default: int) -> List[int]:
    out = []
    for t in targets:
        key = round(float(t), 6)
        out.append(int(counts.get(key, default)))
    return out


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return 999


def sort_methods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_method_order"] = df["method"].map(method_sort_key)
    sort_cols = ["_method_order"]
    if "target" in df.columns:
        sort_cols.append("target")
    df = df.sort_values(sort_cols).drop(columns="_method_order")
    return df


# =============================================================================
# LOADING AND STANDARDIZATION
# =============================================================================


def standardize_direct_result(
    df: pd.DataFrame,
    method: str,
    offline_calls: int,
    online_calls_by_target: Optional[Dict[float, int]] = None,
    default_online_calls_per_target: int = 1,
) -> pd.DataFrame:
    """Standardize GP+DE and NN+Adam style result files."""
    if df is None or df.empty:
        return pd.DataFrame()

    target_col = find_col(df, ["target", "y_target", "target_peak_y_m"])
    peak_col = find_col(df, ["peak_y_true", "true_peak_y", "peak_y", "best_peak_y", "mean_peak_y"])
    err_col = find_col(df, ["target_error_mm", "mean_target_error_mm", "mean_error_mm", "error_mm", "best_error_mm"])
    xr_col = find_col(df, ["max_abs_xr_true", "true_max_abs_xr", "max_abs_xr", "mean_max_abs_xr", "best_max_abs_xr"])
    viol_col = find_col(df, ["constraint_violation_abs_mm", "constraint_violation_mm", "violation_mm", "mean_constraint_violation_abs_mm"])
    feas_col = find_col(df, ["feasible_abs", "feasible", "is_feasible", "mean_feasible"])
    time_col = find_col(df, ["optimization_time_s", "time_s", "runtime_s", "wall_time_s", "mean_optimization_time_s"])
    surrogate_col = find_col(df, ["surrogate_function_evaluations", "surrogate_evaluations", "n_surrogate_eval"])

    missing = []
    for name, col in [("target", target_col), ("peak_y_true", peak_col), ("max_abs_xr_true", xr_col)]:
        if col is None:
            missing.append(name)
    if missing:
        raise ValueError(f"Cannot standardize {method}: missing columns {missing}. Existing columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["target"] = safe_numeric(df[target_col]).reset_index(drop=True)
    out.insert(0, "method", method)
    out["peak_y_true"] = safe_numeric(df[peak_col]).reset_index(drop=True)

    if err_col is not None:
        out["target_error_mm"] = safe_numeric(df[err_col]).reset_index(drop=True).reset_index(drop=True)
    else:
        out["target_error_mm"] = infer_target_error_mm(df, target_col, peak_col).reset_index(drop=True)

    out["max_abs_xr_true"] = safe_numeric(df[xr_col]).reset_index(drop=True).reset_index(drop=True)
    out["residual_margin_mm"] = infer_residual_margin_mm(df, xr_col)

    if viol_col is not None:
        out["constraint_violation_abs_mm"] = safe_numeric(df[viol_col]).reset_index(drop=True)
    else:
        out["constraint_violation_abs_mm"] = infer_violation_mm(df, xr_col)

    if feas_col is not None:
        out["feasible_abs"] = to_bool_series(df[feas_col]).reset_index(drop=True)
    else:
        out["feasible_abs"] = out["constraint_violation_abs_mm"] <= 1e-6

    if time_col is not None:
        out["optimization_time_s"] = safe_numeric(df[time_col]).reset_index(drop=True).reset_index(drop=True)
    else:
        out["optimization_time_s"] = np.nan

    if surrogate_col is not None:
        out["surrogate_function_evaluations"] = safe_numeric(df[surrogate_col]).reset_index(drop=True)
    else:
        out["surrogate_function_evaluations"] = np.nan

    if online_calls_by_target is not None:
        out["online_true_simulator_calls"] = match_target_counts(
            out["target"], online_calls_by_target, default_online_calls_per_target
        )
    else:
        out["online_true_simulator_calls"] = int(default_online_calls_per_target)

    out["offline_dataset_calls"] = int(offline_calls)
    out["total_simulator_calls_equivalent"] = out["online_true_simulator_calls"] + out["offline_dataset_calls"]

    # Copy predictions and parameter columns when available.
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
    online_calls_per_target: int,
) -> pd.DataFrame:
    """Aggregate BO/Random Search per-seed result files into target-level means."""
    if df is None or df.empty:
        return pd.DataFrame()

    target_col = find_col(df, ["target", "y_target", "target_peak_y_m"])
    peak_col = find_col(df, ["peak_y_true", "true_peak_y", "peak_y", "best_peak_y", "best_true_peak_y"])
    err_col = find_col(df, ["target_error_mm", "error_mm", "best_error_mm", "best_target_error_mm"])
    xr_col = find_col(df, ["max_abs_xr_true", "true_max_abs_xr", "max_abs_xr", "best_max_abs_xr"])
    viol_col = find_col(df, ["constraint_violation_abs_mm", "constraint_violation_mm", "violation_mm"])
    feas_col = find_col(df, ["feasible_abs", "feasible", "is_feasible"])
    time_col = find_col(df, ["optimization_time_s", "time_s", "runtime_s", "wall_time_s"])
    calls_col = find_col(df, ["true_simulator_calls", "online_true_simulator_calls", "n_calls", "budget", "total_calls"])

    missing = []
    for name, col in [("target", target_col), ("peak_y_true", peak_col), ("max_abs_xr_true", xr_col)]:
        if col is None:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Cannot aggregate {method}: missing columns {missing}. Existing columns: {list(df.columns)}"
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
        work["calls_per_seed"] = float(online_calls_per_target)

    # Keep parameters if available by averaging them across seeds.
    for param in OPTIMIZED_PARAMS:
        source = find_col(df, [param, f"best_{param}"])
        if source is not None:
            work[param] = safe_numeric(df[source])

    group = work.groupby(work["target"].round(6), as_index=False)
    rows = []
    for _, g in group:
        row = {
            "method": method,
            "target": float(g["target"].iloc[0]),
            "peak_y_true": float(g["peak_y_true"].mean()),
            "peak_y_true_std": float(g["peak_y_true"].std(ddof=0)) if len(g) > 1 else 0.0,
            "target_error_mm": float(g["target_error_mm"].mean()),
            "target_error_mm_std": float(g["target_error_mm"].std(ddof=0)) if len(g) > 1 else 0.0,
            "target_error_mm_min": float(g["target_error_mm"].min()),
            "target_error_mm_max": float(g["target_error_mm"].max()),
            "max_abs_xr_true": float(g["max_abs_xr_true"].mean()),
            "max_abs_xr_true_std": float(g["max_abs_xr_true"].std(ddof=0)) if len(g) > 1 else 0.0,
            "residual_margin_mm": float(g["residual_margin_mm"].mean()),
            "residual_margin_mm_min_seed": float(g["residual_margin_mm"].min()),
            "constraint_violation_abs_mm": float(g["constraint_violation_abs_mm"].mean()),
            "feasible_abs": bool(g["feasible_abs"].all()),
            "feasibility_rate_seed_percent": float(g["feasible_abs"].mean() * 100.0),
            "optimization_time_s": float(g["optimization_time_s"].mean()) if g["optimization_time_s"].notna().any() else np.nan,
            "online_true_simulator_calls": int(g["calls_per_seed"].sum()),
            "offline_dataset_calls": int(offline_calls),
            "surrogate_function_evaluations": np.nan,
        }
        row["total_simulator_calls_equivalent"] = row["online_true_simulator_calls"] + row["offline_dataset_calls"]
        for param in OPTIMIZED_PARAMS:
            if param in g.columns:
                row[param] = float(g[param].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def standardize_summary_by_target(
    df: pd.DataFrame,
    method: str,
    offline_calls: int,
    fallback_online_calls_per_target: int,
) -> pd.DataFrame:
    """Standardize BO summary_by_target files, if available."""
    if df is None or df.empty:
        return pd.DataFrame()

    target_col = find_col(df, ["target", "y_target", "target_peak_y_m"])
    peak_col = find_col(df, ["mean_peak_y", "peak_y_mean", "peak_y_true_mean", "peak_y_true", "true_y_mean"])
    err_col = find_col(df, ["mean_target_error_mm", "mean_error_mm", "target_error_mm", "error_mm_mean"])
    xr_col = find_col(df, ["mean_max_abs_xr", "max_abs_xr_mean", "mean_max_abs_xr_true", "max_abs_xr_true"])
    margin_col = find_col(df, ["mean_residual_margin_mm", "residual_margin_mm", "mean_margin_mm"])
    min_margin_col = find_col(df, ["min_residual_margin_mm", "residual_margin_min_mm", "min_margin_mm"])
    feas_col = find_col(df, ["feasibility_rate_percent", "feasible_rate_percent", "feasible_abs"])
    calls_col = find_col(df, ["total_calls", "online_true_simulator_calls", "true_simulator_calls", "n_calls"])
    time_col = find_col(df, ["mean_optimization_time_s", "optimization_time_s", "time_s", "runtime_s"])

    if target_col is None or peak_col is None or err_col is None or xr_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["target"] = safe_numeric(df[target_col]).reset_index(drop=True)
    out.insert(0, "method", method)
    out["peak_y_true"] = safe_numeric(df[peak_col]).reset_index(drop=True)
    out["target_error_mm"] = safe_numeric(df[err_col]).reset_index(drop=True).reset_index(drop=True)
    out["max_abs_xr_true"] = safe_numeric(df[xr_col]).reset_index(drop=True).reset_index(drop=True)
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
        # If percent, use 100 as all-seed feasible. If boolean-like, this also works.
        vals = safe_numeric(df[feas_col])
        out["feasible_abs"] = vals >= 99.999
        out["feasibility_rate_seed_percent"] = vals
    else:
        out["feasible_abs"] = out["constraint_violation_abs_mm"] <= 1e-6

    if calls_col is not None:
        out["online_true_simulator_calls"] = safe_numeric(df[calls_col]).fillna(fallback_online_calls_per_target).astype(int).reset_index(drop=True)
    else:
        out["online_true_simulator_calls"] = int(fallback_online_calls_per_target)
    out["offline_dataset_calls"] = int(offline_calls)
    out["total_simulator_calls_equivalent"] = out["online_true_simulator_calls"] + out["offline_dataset_calls"]

    if time_col is not None:
        out["optimization_time_s"] = safe_numeric(df[time_col]).reset_index(drop=True).reset_index(drop=True)
    else:
        out["optimization_time_s"] = np.nan
    out["surrogate_function_evaluations"] = np.nan

    return out


def load_gp_de() -> pd.DataFrame:
    print("Loading GP+DE beta=0.5 results...")
    results = read_csv_if_exists(GP_DIR / "gp_de_results.csv", required=True)
    validated = read_csv_if_exists(GP_DIR / "gp_de_all_validated_candidates.csv", required=False)
    counts = count_by_target(validated, ["target"])
    return standardize_direct_result(
        results,
        method="GP+DE beta=0.5",
        offline_calls=OFFLINE_DATASET_CALLS_SURROGATE,
        online_calls_by_target=counts,
        default_online_calls_per_target=25,
    )


def load_nn_adam() -> pd.DataFrame:
    print("Loading NN+Adam safe results...")
    results_path = first_existing(
        [
            NN_DIR / "nn_gradient_results.csv",
            NN_DIR / "nn_adam_results.csv",
        ],
        required=True,
    )
    results = read_csv_if_exists(results_path, required=True)
    validated_path = first_existing(
        [
            NN_DIR / "nn_gradient_validated_candidates.csv",
            NN_DIR / "nn_adam_validated_candidates.csv",
        ],
        required=False,
    )
    validated = read_csv_if_exists(validated_path, required=False) if validated_path else None
    counts = count_by_target(validated, ["target"])
    return standardize_direct_result(
        results,
        method="NN+Adam safe",
        offline_calls=OFFLINE_DATASET_CALLS_SURROGATE,
        online_calls_by_target=counts,
        default_online_calls_per_target=50,
    )


def load_bo_final() -> pd.DataFrame:
    print("Loading BO final results...")
    by_seed_path = BO_DIR / "bo_final_results_by_seed.csv"
    summary_path = BO_DIR / "bo_final_summary_by_target.csv"

    by_seed = read_csv_if_exists(by_seed_path, required=False)
    if by_seed is not None and not by_seed.empty:
        return aggregate_seed_results(
            by_seed,
            method="BO mean",
            offline_calls=0,
            online_calls_per_target=200,
        )

    summary = read_csv_if_exists(summary_path, required=True)
    out = standardize_summary_by_target(
        summary,
        method="BO mean",
        offline_calls=0,
        fallback_online_calls_per_target=400,
    )
    if out.empty:
        raise ValueError(
            f"Could not standardize BO summary file. Columns: {list(summary.columns)}"
        )
    return out


def load_random_search() -> pd.DataFrame:
    print("Loading Random Search final results...")
    by_seed_path = BO_DIR / "random_search_final_results_by_seed.csv"
    by_seed = read_csv_if_exists(by_seed_path, required=True)
    return aggregate_seed_results(
        by_seed,
        method="Random Search mean",
        offline_calls=0,
        online_calls_per_target=200,
    )


# =============================================================================
# SUMMARY TABLES
# =============================================================================


def create_by_target_table(all_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "method",
        "target",
        "peak_y_true",
        "target_error_mm",
        "target_error_mm_std",
        "target_error_mm_min",
        "target_error_mm_max",
        "max_abs_xr_true",
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
    for col in keep_cols:
        if col not in all_df.columns:
            all_df[col] = np.nan
    out = all_df[keep_cols].copy()
    out = out[out["method"].notna()].copy()
    return sort_methods(out)


def create_reachable_vs_high_summary(by_target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, df_m in by_target.groupby("method", sort=False):
        reachable = df_m[df_m["target"] <= REACHABLE_TARGET_MAX]
        high = df_m[df_m["target"] > REACHABLE_TARGET_MAX]
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
    reachable_high = create_reachable_vs_high_summary(by_target)
    rows = []
    for method, df_m in by_target.groupby("method", sort=False):
        row = {
            "method": method,
            "n_targets": int(len(df_m)),
            "feasible_targets": int(df_m["feasible_abs"].sum()),
            "feasibility_rate_percent": float(df_m["feasible_abs"].mean() * 100.0),
            "mean_target_error_mm": float(df_m["target_error_mm"].mean()),
            "median_target_error_mm": float(df_m["target_error_mm"].median()),
            "max_target_error_mm": float(df_m["target_error_mm"].max()),
            "mean_true_max_abs_xr_m": float(df_m["max_abs_xr_true"].mean()),
            "max_true_max_abs_xr_m": float(df_m["max_abs_xr_true"].max()),
            "mean_residual_margin_mm": float(df_m["residual_margin_mm"].mean()),
            "min_residual_margin_mm": float(df_m["residual_margin_mm"].min()),
            "mean_constraint_violation_abs_mm": float(df_m["constraint_violation_abs_mm"].mean()),
            "max_constraint_violation_abs_mm": float(df_m["constraint_violation_abs_mm"].max()),
            "offline_simulator_calls": int(df_m["offline_dataset_calls"].max()),
            "online_simulator_calls_total": int(df_m["online_true_simulator_calls"].sum()),
            "online_simulator_calls_per_target_mean": float(df_m["online_true_simulator_calls"].mean()),
            "total_simulator_calls_equivalent": int(
                df_m["offline_dataset_calls"].max() + df_m["online_true_simulator_calls"].sum()
            ),
            "total_optimization_time_s": float(df_m["optimization_time_s"].sum(skipna=True)),
            "mean_optimization_time_s_per_target": float(df_m["optimization_time_s"].mean(skipna=True)),
            "mean_surrogate_function_evaluations": float(df_m["surrogate_function_evaluations"].mean(skipna=True)),
            "total_surrogate_function_evaluations": float(df_m["surrogate_function_evaluations"].sum(skipna=True)),
        }
        rh = reachable_high[reachable_high["method"] == method]
        if not rh.empty:
            row.update(rh.iloc[0].drop(labels="method").to_dict())
        rows.append(row)

    summary = pd.DataFrame(rows)
    return sort_methods(summary)


def create_cost_summary(summary: pd.DataFrame) -> pd.DataFrame:
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


# =============================================================================
# PLOTS
# =============================================================================


def plot_target_tracking(by_target: pd.DataFrame) -> None:
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
    fig.tight_layout()
    path = FIGURES_DIR / "01_final_target_tracking.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_grouped_bars(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
) -> None:
    pivot = df.pivot_table(index="target", columns="method", values=value_col, aggfunc="first")
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", width=0.82)
    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=2, label=hline_label or f"{hline:g}")
        ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=8)
    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticklabels([f"{float(t):.2f}" for t in pivot.index], rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_target_error(by_target: pd.DataFrame) -> None:
    plot_grouped_bars(
        by_target,
        value_col="target_error_mm",
        ylabel="True target error [mm]",
        title="Final target error by method and target",
        path=FIGURES_DIR / "02_final_target_error_by_target.png",
        hline=10.0,
        hline_label="10 mm tolerance",
    )


def plot_constraint_validation(by_target: pd.DataFrame) -> None:
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

    ax.axhspan(0.495, 0.500, alpha=0.15, label="Surrogate safety margin band")
    ax.axhline(TRUE_ROBOT_LIMIT_M, linestyle="--", linewidth=2, label="True robot limit 0.500 m")
    ax.axhline(0.495, linestyle=":", linewidth=2, label="Nominal surrogate limit 0.495 m")
    ax.set_xlabel("Target outreach [m]")
    ax.set_ylabel("True max_abs_xr [m]")
    ax.set_title("Constraint validation on true simulator")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "03_final_constraint_validation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_residual_margin(by_target: pd.DataFrame) -> None:
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
    plot_df = reachable_high.set_index("method")[[
        "mean_error_reachable_targets_mm",
        "mean_error_high_targets_mm",
    ]]
    plot_df = plot_df.reindex([m for m in METHOD_ORDER if m in plot_df.index])
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
    fig.tight_layout()
    path = FIGURES_DIR / "05_final_reachable_vs_high_error.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_runtime(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    df = summary.set_index("method").reindex([m for m in METHOD_ORDER if m in summary["method"].values])
    ax.bar(df.index, df["total_optimization_time_s"], edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("Method")
    ax.set_ylabel("Total optimization wall time [s], log scale")
    ax.set_title("Optimization runtime comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = FIGURES_DIR / "06_final_runtime_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_simulator_calls(summary: pd.DataFrame) -> None:
    df = summary.set_index("method").reindex([m for m in METHOD_ORDER if m in summary["method"].values])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(df.index, df["offline_simulator_calls"], edgecolor="black", label="Offline dataset calls")
    ax.bar(
        df.index,
        df["online_simulator_calls_total"],
        bottom=df["offline_simulator_calls"],
        edgecolor="black",
        label="Online optimization calls",
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Simulator calls")
    ax.set_title("Offline and online simulator-call cost")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "07_final_simulator_calls_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_error_vs_cost(summary: pd.DataFrame) -> None:
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
            (row["total_simulator_calls_equivalent"], row["mean_target_error_mm"]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )
    ax.set_xlabel("Total simulator calls equivalent\n(offline dataset + online validation/optimization)")
    ax.set_ylabel("Mean target error [mm]")
    ax.set_title("Accuracy-cost trade-off")
    ax.grid(True, alpha=0.3)
    # Keep log if all costs positive and spread is meaningful.
    if summary["total_simulator_calls_equivalent"].min() > 0:
        ax.set_xscale("log")
    fig.tight_layout()
    path = FIGURES_DIR / "08_final_error_vs_cost_pareto.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_parameter_comparison_target065(by_target: pd.DataFrame) -> None:
    df = by_target[np.isclose(by_target["target"], 0.65)].copy()
    available_params = [p for p in OPTIMIZED_PARAMS if p in df.columns and df[p].notna().any()]
    if df.empty or not available_params:
        print("⚠ Skipping parameter comparison at target 0.65: no parameter columns available.")
        return

    rows = []
    for _, row in df.iterrows():
        for p in available_params:
            if pd.isna(row[p]):
                continue
            low, high = PARAM_BOUNDS[p]
            rows.append(
                {
                    "method": row["method"],
                    "parameter": p,
                    "normalized_value": (float(row[p]) - low) / (high - low),
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("⚠ Skipping parameter comparison at target 0.65: empty after normalization.")
        return

    pivot = plot_df.pivot_table(index="parameter", columns="method", values="normalized_value", aggfunc="first")
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    pivot = pivot.reindex([p for p in OPTIMIZED_PARAMS if p in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_xlabel("Optimized parameter")
    ax.set_ylabel("Normalized value within bounds")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Optimized parameter comparison at target 0.65 m")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "09_final_parameter_comparison_target065.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


def generate_plots(by_target: pd.DataFrame, summary: pd.DataFrame, reachable_high: pd.DataFrame) -> None:
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
# MAIN
# =============================================================================


def print_loaded_table(by_target: pd.DataFrame) -> None:
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
    print("\nFinal comparison by target:")
    print(by_target[display_cols].to_string(index=False))


def print_summary(summary: pd.DataFrame) -> None:
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
    print("\nFinal comparison summary:")
    print(summary[display_cols].to_string(index=False))


def main() -> None:
    ensure_dirs()
    print("=" * 80)
    print("FINAL OPTIMIZATION METHOD COMPARISON")
    print("=" * 80)

    tables = []
    loaders = [load_gp_de, load_nn_adam, load_bo_final, load_random_search]
    for loader in loaders:
        try:
            table = loader()
            if table is not None and not table.empty:
                tables.append(table)
        except Exception as exc:
            print(f"⚠ Could not load one method with {loader.__name__}: {exc}")

    if not tables:
        raise RuntimeError("No comparison tables could be loaded.")

    all_df = pd.concat(tables, ignore_index=True, sort=False)
    by_target = create_by_target_table(all_df)
    reachable_high = create_reachable_vs_high_summary(by_target)
    summary = create_summary_table(by_target)
    cost_summary = create_cost_summary(summary)

    # Save tables.
    by_target_path = RESULTS_DIR / "final_comparison_by_target.csv"
    summary_path = RESULTS_DIR / "final_comparison_summary.csv"
    cost_path = RESULTS_DIR / "final_cost_summary.csv"
    rh_path = RESULTS_DIR / "final_reachable_vs_high_summary.csv"

    by_target.to_csv(by_target_path, index=False)
    summary.to_csv(summary_path, index=False)
    cost_summary.to_csv(cost_path, index=False)
    reachable_high.to_csv(rh_path, index=False)

    print_loaded_table(by_target)
    print_summary(summary)

    generate_plots(by_target, summary, reachable_high)

    print("\nSaved tables:")
    print(f"  {by_target_path}")
    print(f"  {summary_path}")
    print(f"  {cost_path}")
    print(f"  {rh_path}")
    print("\nDone.")
    print("=" * 80)


if __name__ == "__main__":
    main()
