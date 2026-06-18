"""
bo_kernel_benchmark.py
======================

Step B of the Bayesian Optimization refinement plan.

Purpose
-------
Benchmark the internal Gaussian Process kernel used by the true-simulator
Bayesian Optimization baseline, after Step A selected PI as the primary
candidate acquisition function.

This script compares:
    - Matern 3/2
    - Matern 5/2
    - RBF

using:
    - acquisition = PI
    - all official targets: 0.55, 0.60, 0.65, 0.70, 0.75 m
    - two seeds: 2026, 2027
    - 100 true simulator calls per BO run
    - 20 Latin Hypercube initial points

Outputs are written to separate folders:
    results/optimization_bo_kernel_benchmark/
    figures/optimization_bo_kernel_benchmark/

This script intentionally does not overwrite the final BO folders or the Step A
acquisition benchmark folders.

Run from project root:
    python src/bo_kernel_benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

# Import existing true-simulator BO implementation. This file must be placed in src/.
import optimization_bo as obo


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_kernel_benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_kernel_benchmark"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class KernelBenchmarkConfig:
    """Configuration for Step B kernel benchmark."""

    targets: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)
    seeds: Tuple[int, ...] = (2026, 2027)

    total_evaluations: int = 100
    initial_points: int = 20

    # Selected from Step A as the primary acquisition candidate.
    acquisition_name: str = "pi"
    acquisition_label: str = "PI"
    pi_xi: float = 0.01

    # Kernel settings to compare.
    kernels: Tuple[Tuple[str, str], ...] = (
        ("matern32", "Matern 3/2"),
        ("matern", "Matern 5/2"),
        ("rbf", "RBF"),
    )

    # BO internal settings.
    gp_restarts: int = 1
    gp_alpha: float = 1e-8
    candidate_pool_global: int = 4000
    candidate_pool_local: int = 1000
    local_sigma_unit: float = 0.07

    # Resume existing runs if the benchmark was interrupted.
    resume_existing: bool = True


# =============================================================================
# MONKEY PATCHES: PI ACQUISITION AND MATERN 3/2 KERNEL
# =============================================================================

def probability_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement for minimization.

    PI(x) = P(J(x) < J_best - xi)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (best_y - mu - xi) / sigma_safe
    pi = norm.cdf(z)
    pi[sigma <= 1e-12] = 0.0
    return np.maximum(pi, 0.0)


def patched_acquisition_scores(
    gp,
    X_cand: np.ndarray,
    best_y: float,
    acquisition: str,
    config: obo.BOConfig,
) -> np.ndarray:
    """Acquisition scores with EI, PI, and LCB support.

    Higher score is better for all acquisition functions.
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    acquisition = acquisition.lower()

    if acquisition == "ei":
        return obo.expected_improvement(mu, sigma, best_y, xi=config.ei_xi)

    if acquisition == "pi":
        xi = getattr(config, "pi_xi", config.ei_xi)
        return probability_improvement(mu, sigma, best_y, xi=xi)

    if acquisition == "lcb":
        lcb = mu - config.lcb_kappa * sigma
        return -lcb

    raise ValueError(f"Unsupported acquisition: {acquisition!r}")


def patched_build_kernel(kernel_kind: str, config: obo.BOConfig):
    """Build BO internal GP kernel with Matern 3/2, Matern 5/2, and RBF support."""
    dim = len(config.optimized_columns())
    ls_bounds = (config.gp_length_scale_bounds_low, config.gp_length_scale_bounds_high)
    noise_bounds = (config.white_noise_bounds_low, config.white_noise_bounds_high)
    kind = kernel_kind.lower()

    if kind in {"matern32", "matern_32", "matern3/2", "matern_3_2"}:
        base = Matern(length_scale=np.ones(dim), length_scale_bounds=ls_bounds, nu=1.5)
    elif kind in {"matern", "matern52", "matern_52", "matern5/2", "matern_5_2"}:
        base = Matern(length_scale=np.ones(dim), length_scale_bounds=ls_bounds, nu=2.5)
    elif kind == "rbf":
        base = RBF(length_scale=np.ones(dim), length_scale_bounds=ls_bounds)
    else:
        raise ValueError(f"Unsupported kernel_kind: {kernel_kind!r}")

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * base
        + WhiteKernel(
            noise_level=config.white_noise_level,
            noise_level_bounds=noise_bounds,
        )
    )
    return kernel


# Apply patches so optimization_bo.run_bo_single_target can use PI and Matern 3/2.
obo.acquisition_scores = patched_acquisition_scores
obo.build_kernel = patched_build_kernel


# =============================================================================
# RUNNERS
# =============================================================================

def make_bo_config(bench_cfg: KernelBenchmarkConfig) -> obo.BOConfig:
    """Create BOConfig for a kernel benchmark run."""
    cfg = obo.BOConfig()

    # Disable workflow flags; this benchmark script controls the runs.
    cfg.run_pilot = False
    cfg.stop_after_pilot = False
    cfg.run_main_bo = False
    cfg.run_random_search = False
    cfg.run_kernel_sensitivity = False
    cfg.run_acquisition_sensitivity = False
    cfg.run_budget_sweep = False
    cfg.make_time_response_plots = False

    cfg.main_acquisition = bench_cfg.acquisition_name
    cfg.gp_restarts = bench_cfg.gp_restarts
    cfg.gp_alpha = bench_cfg.gp_alpha
    cfg.candidate_pool_global = bench_cfg.candidate_pool_global
    cfg.candidate_pool_local = bench_cfg.candidate_pool_local
    cfg.local_sigma_unit = bench_cfg.local_sigma_unit

    # Dynamically attach PI xi so the patched acquisition can read it.
    cfg.pi_xi = bench_cfg.pi_xi  # type: ignore[attr-defined]

    return cfg


def safe_label(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("/", "")
        .replace("=", "")
        .replace(".", "p")
        .replace("_", "")
    )


def run_single_benchmark(
    target: float,
    kernel_kind: str,
    kernel_label: str,
    seed: int,
    bench_cfg: KernelBenchmarkConfig,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, object]]:
    """Run one BO kernel benchmark case."""
    cfg = make_bo_config(bench_cfg)
    run_label = f"kernel_{safe_label(kernel_label)}_target{int(round(target * 1000)):04d}_seed{seed}"

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=cfg,
        run_label=run_label,
        total_evaluations=bench_cfg.total_evaluations,
        n_initial_points=bench_cfg.initial_points,
        kernel_kind=kernel_kind,
        acquisition=bench_cfg.acquisition_name,
        seed=seed,
    )

    result = obo.build_result_row(best, cfg)
    result.update(
        {
            "benchmark": "kernel",
            "target": float(target),
            "kernel": kernel_kind,
            "kernel_label": kernel_label,
            "acquisition": bench_cfg.acquisition_name,
            "acquisition_label": bench_cfg.acquisition_label,
            "seed": int(seed),
            "budget": int(bench_cfg.total_evaluations),
            "initial_points": int(bench_cfg.initial_points),
            "gp_alpha": float(bench_cfg.gp_alpha),
            "optimization_time_s": float(timing["total_wall_time_s"]),
            "mean_iteration_time_s": float(timing["mean_iteration_time_s"]),
            "mean_simulation_time_s": float(timing["mean_simulation_time_s"]),
            "mean_gp_fit_time_s": float(timing["mean_gp_fit_time_s"]),
            "mean_acquisition_time_s": float(timing["mean_acquisition_time_s"]),
        }
    )

    history = history.copy()
    history["benchmark"] = "kernel"
    history["kernel_label"] = kernel_label
    history["acquisition_label"] = bench_cfg.acquisition_label
    history["budget"] = int(bench_cfg.total_evaluations)
    history["initial_points"] = int(bench_cfg.initial_points)
    history["gp_alpha"] = float(bench_cfg.gp_alpha)

    timing = dict(timing)
    timing.update(
        {
            "benchmark": "kernel",
            "kernel_label": kernel_label,
            "acquisition_label": bench_cfg.acquisition_label,
            "budget": int(bench_cfg.total_evaluations),
            "gp_alpha": float(bench_cfg.gp_alpha),
        }
    )

    return result, history, timing


# =============================================================================
# SUMMARIES AND RESUME
# =============================================================================

def load_existing_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_path = RESULTS_DIR / "bo_kernel_benchmark.csv"
    history_path = RESULTS_DIR / "bo_kernel_benchmark_history.csv"
    timing_path = RESULTS_DIR / "bo_kernel_benchmark_timing.csv"

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    timing = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()
    return results, history, timing


def already_done(
    existing_results: pd.DataFrame,
    target: float,
    kernel_label: str,
    seed: int,
    budget: int,
    acquisition_label: str,
) -> bool:
    if existing_results.empty:
        return False
    mask = (
        np.isclose(existing_results["target"].astype(float), float(target))
        & (existing_results["kernel_label"].astype(str) == kernel_label)
        & (existing_results["seed"].astype(int) == int(seed))
        & (existing_results["budget"].astype(int) == int(budget))
        & (existing_results["acquisition_label"].astype(str) == acquisition_label)
    )
    return bool(mask.any())


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate kernel benchmark results by target and kernel."""
    if results_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (target, kernel_label), g in results_df.groupby(["target", "kernel_label"], sort=False):
        g = g.copy()
        feasible_rate = 100.0 * float(g["feasible_abs"].astype(bool).mean())
        rows.append(
            {
                "target": float(target),
                "kernel_label": kernel_label,
                "n_runs": int(len(g)),
                "feasibility_rate_percent": feasible_rate,
                "mean_target_error_mm": float(g["target_error_mm"].mean()),
                "std_target_error_mm": float(g["target_error_mm"].std(ddof=0)),
                "min_target_error_mm": float(g["target_error_mm"].min()),
                "max_target_error_mm": float(g["target_error_mm"].max()),
                "mean_peak_y_true": float(g["peak_y_true"].mean()),
                "mean_max_abs_xr_true": float(g["max_abs_xr_true"].mean()),
                "mean_residual_margin_mm": float(g["residual_margin_mm"].mean()),
                "min_residual_margin_mm": float(g["residual_margin_mm"].min()),
                "mean_best_iteration": float(g["best_iteration"].mean()),
                "mean_optimization_time_s": float(g["optimization_time_s"].mean()),
                "mean_gp_fit_time_s": float(g["mean_gp_fit_time_s"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["target", "mean_target_error_mm", "mean_residual_margin_mm"])
    return summary


def build_overall_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate kernel performance across all targets."""
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["rank_by_error_within_target_seed"] = df.groupby(["target", "seed"])["target_error_mm"].rank(
        method="min", ascending=True
    )

    rows: List[Dict[str, object]] = []
    for kernel_label, g in df.groupby("kernel_label", sort=False):
        feasible_rate = 100.0 * float(g["feasible_abs"].astype(bool).mean())
        rows.append(
            {
                "kernel_label": kernel_label,
                "n_runs": int(len(g)),
                "n_targets": int(g["target"].nunique()),
                "feasibility_rate_percent": feasible_rate,
                "mean_target_error_mm": float(g["target_error_mm"].mean()),
                "median_target_error_mm": float(g["target_error_mm"].median()),
                "std_target_error_mm": float(g["target_error_mm"].std(ddof=0)),
                "max_target_error_mm": float(g["target_error_mm"].max()),
                "mean_residual_margin_mm": float(g["residual_margin_mm"].mean()),
                "min_residual_margin_mm": float(g["residual_margin_mm"].min()),
                "mean_rank_by_error": float(g["rank_by_error_within_target_seed"].mean()),
                "mean_best_iteration": float(g["best_iteration"].mean()),
                "mean_optimization_time_s": float(g["optimization_time_s"].mean()),
                "mean_gp_fit_time_s": float(g["mean_gp_fit_time_s"].mean()),
            }
        )

    overall = pd.DataFrame(rows)
    overall = overall.sort_values(
        ["feasibility_rate_percent", "mean_rank_by_error", "mean_target_error_mm"],
        ascending=[False, True, True],
    )
    return overall


# =============================================================================
# PLOTS
# =============================================================================

def kernel_order() -> List[str]:
    return ["Matern 3/2", "Matern 5/2", "RBF"]


def plot_final_error_boxplots(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_t = results_df[np.isclose(results_df["target"], target)].copy()
        labels = [x for x in kernel_order() if x in set(df_t["kernel_label"])]
        data = [df_t[df_t["kernel_label"] == label]["target_error_mm"].values for label in labels]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.ylabel("Final target error [mm]")
        plt.xlabel("Internal GP kernel")
        plt.title(f"BO kernel benchmark: final error, target = {target:.2f} m")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_kernel_final_error_boxplot_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_margin_boxplots(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_t = results_df[np.isclose(results_df["target"], target)].copy()
        labels = [x for x in kernel_order() if x in set(df_t["kernel_label"])]
        data = [df_t[df_t["kernel_label"] == label]["residual_margin_mm"].values for label in labels]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.axhline(0.0, linestyle="--", linewidth=1.8)
        plt.ylabel("Residual margin to true limit [mm]")
        plt.xlabel("Internal GP kernel")
        plt.title(f"BO kernel benchmark: residual margin, target = {target:.2f} m")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_kernel_margin_boxplot_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_mean_error_bars(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_t = summary_df[np.isclose(summary_df["target"], target)].copy()
        labels = [x for x in kernel_order() if x in set(df_t["kernel_label"])]
        df_t = df_t.set_index("kernel_label").loc[labels].reset_index()

        x = np.arange(len(df_t))
        plt.figure(figsize=(8, 6))
        plt.bar(
            x,
            df_t["mean_target_error_mm"],
            yerr=df_t["std_target_error_mm"],
            capsize=4,
            edgecolor="black",
        )
        plt.xticks(x, df_t["kernel_label"].values)
        plt.ylabel("Mean final target error [mm]")
        plt.xlabel("Internal GP kernel")
        plt.title(f"BO kernel benchmark: mean final error, target = {target:.2f} m")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_kernel_mean_error_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_convergence(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return

    for target in sorted(history_df["target"].unique()):
        df_t = history_df[np.isclose(history_df["target"], target)].copy()

        plt.figure(figsize=(10, 6))
        for label in kernel_order():
            df_k = df_t[df_t["kernel_label"] == label].copy()
            if df_k.empty:
                continue
            grouped = (
                df_k.groupby("iteration")["best_feasible_error_so_far_mm"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .sort_values("iteration")
            )
            plt.plot(grouped["iteration"], grouped["mean"], linewidth=2.2, label=label)
            plt.fill_between(grouped["iteration"], grouped["min"], grouped["max"], alpha=0.12)

        plt.xlabel("True simulator calls")
        plt.ylabel("Best feasible target error so far [mm]")
        plt.title(f"BO kernel convergence, target = {target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_kernel_convergence_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_error_margin_tradeoff(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_t = results_df[np.isclose(results_df["target"], target)].copy()
        plt.figure(figsize=(8, 6))
        for label in kernel_order():
            df_k = df_t[df_t["kernel_label"] == label]
            if df_k.empty:
                continue
            plt.scatter(
                df_k["target_error_mm"],
                df_k["residual_margin_mm"],
                s=70,
                edgecolor="black",
                label=label,
            )
        plt.axhline(0.0, linestyle="--", linewidth=1.8)
        plt.xlabel("Final target error [mm]")
        plt.ylabel("Residual margin to true limit [mm]")
        plt.title(f"BO kernel trade-off, target = {target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_kernel_error_margin_tradeoff_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_mean_error_heatmap(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    targets = sorted(summary_df["target"].unique())
    labels = [x for x in kernel_order() if x in set(summary_df["kernel_label"])]
    matrix = np.full((len(targets), len(labels)), np.nan)
    for i, target in enumerate(targets):
        for j, label in enumerate(labels):
            row = summary_df[(np.isclose(summary_df["target"], target)) & (summary_df["kernel_label"] == label)]
            if not row.empty:
                matrix[i, j] = float(row["mean_target_error_mm"].iloc[0])

    plt.figure(figsize=(8, 5.5))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label="Mean final target error [mm]")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(targets)), [f"{t:.2f}" for t in targets])
    plt.xlabel("Internal GP kernel")
    plt.ylabel("Target [m]")
    plt.title("BO kernel benchmark: mean error heatmap")
    for i in range(len(targets)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "bo_kernel_mean_error_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_overall_mean_error(overall_df: pd.DataFrame) -> None:
    if overall_df.empty:
        return

    labels = [x for x in kernel_order() if x in set(overall_df["kernel_label"])]
    df = overall_df.set_index("kernel_label").loc[labels].reset_index()
    x = np.arange(len(df))
    plt.figure(figsize=(8, 6))
    plt.bar(x, df["mean_target_error_mm"], yerr=df["std_target_error_mm"], capsize=4, edgecolor="black")
    plt.xticks(x, df["kernel_label"])
    plt.ylabel("Mean target error across all targets [mm]")
    plt.xlabel("Internal GP kernel")
    plt.title("BO kernel benchmark: overall mean error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "bo_kernel_overall_mean_error.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def make_plots(
    results_df: pd.DataFrame,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: Optional[pd.DataFrame] = None,
) -> None:
    plot_final_error_boxplots(results_df)
    plot_margin_boxplots(results_df)
    plot_mean_error_bars(summary_df)
    plot_convergence(history_df)
    plot_error_margin_tradeoff(results_df)
    plot_mean_error_heatmap(summary_df)
    if overall_df is not None:
        plot_overall_mean_error(overall_df)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()
    bench_cfg = KernelBenchmarkConfig()

    print("=" * 80)
    print("BO KERNEL BENCHMARK - STEP B")
    print("=" * 80)
    print(f"Targets:                 {bench_cfg.targets}")
    print(f"Seeds:                   {bench_cfg.seeds}")
    print(f"Acquisition:             {bench_cfg.acquisition_label}")
    print(f"Budget:                  {bench_cfg.total_evaluations} calls")
    print(f"Initial LHS points:      {bench_cfg.initial_points}")
    print(f"Kernels:                 {[k[1] for k in bench_cfg.kernels]}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)

    existing_results, existing_history, existing_timing = load_existing_results()
    result_rows: List[Dict[str, object]] = []
    history_rows: List[pd.DataFrame] = []
    timing_rows: List[Dict[str, object]] = []

    if bench_cfg.resume_existing and not existing_results.empty:
        print(f"Resuming from existing file with {len(existing_results)} completed runs.")
        result_rows.extend(existing_results.to_dict(orient="records"))
        if not existing_history.empty:
            history_rows.append(existing_history)
        if not existing_timing.empty:
            timing_rows.extend(existing_timing.to_dict(orient="records"))

    total_runs = len(bench_cfg.targets) * len(bench_cfg.kernels) * len(bench_cfg.seeds)
    run_idx = 0
    t_start = time.perf_counter()

    for target in bench_cfg.targets:
        for kernel_kind, kernel_label in bench_cfg.kernels:
            for seed in bench_cfg.seeds:
                run_idx += 1

                if bench_cfg.resume_existing and already_done(
                    existing_results,
                    target=target,
                    kernel_label=kernel_label,
                    seed=seed,
                    budget=bench_cfg.total_evaluations,
                    acquisition_label=bench_cfg.acquisition_label,
                ):
                    print(
                        f"[{run_idx}/{total_runs}] SKIP existing | "
                        f"target={target:.2f}, kernel={kernel_label}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{run_idx}/{total_runs}] target={target:.2f} | "
                    f"kernel={kernel_label} | acquisition={bench_cfg.acquisition_label} | seed={seed}"
                )

                try:
                    result, history, timing = run_single_benchmark(
                        target=target,
                        kernel_kind=kernel_kind,
                        kernel_label=kernel_label,
                        seed=seed,
                        bench_cfg=bench_cfg,
                    )
                except Exception as exc:  # keep long benchmark resumable
                    print(f"    ERROR: {exc}")
                    error_row = {
                        "benchmark": "kernel",
                        "target": float(target),
                        "kernel": kernel_kind,
                        "kernel_label": kernel_label,
                        "acquisition": bench_cfg.acquisition_name,
                        "acquisition_label": bench_cfg.acquisition_label,
                        "seed": int(seed),
                        "budget": int(bench_cfg.total_evaluations),
                        "error": repr(exc),
                    }
                    result_rows.append(error_row)
                    pd.DataFrame(result_rows).to_csv(RESULTS_DIR / "bo_kernel_benchmark.csv", index=False)
                    continue

                result_rows.append(result)
                history_rows.append(history)
                timing_rows.append(timing)

                print(
                    f"    best_y={result['peak_y_true']:.4f} m | "
                    f"err={result['target_error_mm']:.2f} mm | "
                    f"xr={result['max_abs_xr_true']:.4f} m | "
                    f"margin={result['residual_margin_mm']:.2f} mm | "
                    f"feasible={bool(result['feasible_abs'])} | "
                    f"time={result['optimization_time_s']:.1f} s"
                )

                # Incremental save after every completed run.
                pd.DataFrame(result_rows).to_csv(RESULTS_DIR / "bo_kernel_benchmark.csv", index=False)
                if history_rows:
                    pd.concat(history_rows, ignore_index=True).to_csv(
                        RESULTS_DIR / "bo_kernel_benchmark_history.csv", index=False
                    )
                if timing_rows:
                    pd.DataFrame(timing_rows).to_csv(RESULTS_DIR / "bo_kernel_benchmark_timing.csv", index=False)

    total_time = time.perf_counter() - t_start

    results_df = pd.DataFrame(result_rows)
    # Remove failed rows from summaries.
    if "target_error_mm" in results_df.columns:
        valid_results_df = results_df.dropna(subset=["target_error_mm"]).copy()
    else:
        valid_results_df = pd.DataFrame()

    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    timing_df = pd.DataFrame(timing_rows)

    summary_df = build_summary(valid_results_df)
    overall_df = build_overall_summary(valid_results_df)

    results_df.to_csv(RESULTS_DIR / "bo_kernel_benchmark.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "bo_kernel_benchmark_history.csv", index=False)
    timing_df.to_csv(RESULTS_DIR / "bo_kernel_benchmark_timing.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "bo_kernel_benchmark_summary.csv", index=False)
    overall_df.to_csv(RESULTS_DIR / "bo_kernel_benchmark_overall_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("KERNEL BENCHMARK SUMMARY")
    print("=" * 80)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No valid results to summarize.")

    print("\n" + "=" * 80)
    print("OVERALL KERNEL SUMMARY ACROSS ALL TARGETS")
    print("=" * 80)
    if not overall_df.empty:
        print(overall_df.to_string(index=False))
    else:
        print("No valid results to summarize.")

    print("\nSaved files:")
    print(f"  {RESULTS_DIR / 'bo_kernel_benchmark.csv'}")
    print(f"  {RESULTS_DIR / 'bo_kernel_benchmark_history.csv'}")
    print(f"  {RESULTS_DIR / 'bo_kernel_benchmark_timing.csv'}")
    print(f"  {RESULTS_DIR / 'bo_kernel_benchmark_summary.csv'}")
    print(f"  {RESULTS_DIR / 'bo_kernel_benchmark_overall_summary.csv'}")
    print(f"Total benchmark wall time: {total_time / 60.0:.1f} min")

    print("\nGenerating benchmark plots...")
    make_plots(valid_results_df, history_df, summary_df, overall_df)
    print("✓ Step B kernel benchmark completed.")


if __name__ == "__main__":
    main()
