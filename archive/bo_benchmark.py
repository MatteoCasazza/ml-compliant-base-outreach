"""
bo_benchmark.py
===============

Focused Bayesian Optimization benchmark for acquisition-function selection.

Step A of the BO refinement plan:
    - target 0.65 m: representative high but reachable target
    - target 0.75 m: high/saturated target
    - kernel fixed to Matern 5/2
    - compare EI, PI, and LCB with multiple kappa values
    - use multiple seeds to estimate robustness

This script intentionally writes to separate folders:
    results/optimization_bo_benchmark/
    figures/optimization_bo_benchmark/

so it does not overwrite the final BO runs produced by optimization_bo.py.

Run from project root:
    python src/bo_benchmark.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Import the existing BO implementation. This file must be placed in src/.
import optimization_bo as obo


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_benchmark"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AcquisitionBenchmarkConfig:
    """Configuration for Step A acquisition benchmark."""

    # Representative targets:
    # 0.65 = high but reachable; 0.75 = saturated/unreachable target.
    targets: Tuple[float, ...] = (0.65, 0.75)

    # Start with two seeds to keep runtime reasonable.
    # Add 2028 later if results are ambiguous.
    seeds: Tuple[int, ...] = (2026, 2027)

    # Fixed BO budget for fair acquisition comparison.
    total_evaluations: int = 100
    initial_points: int = 20

    # Fixed kernel for Step A.
    kernel_kind: str = "matern"

    # Acquisition settings.
    # acquisition_name is what run_bo_single_target receives.
    # label is what appears in tables/plots.
    acquisitions: Tuple[Tuple[str, str, Optional[float]], ...] = (
        ("ei", "EI", None),
        ("pi", "PI", None),
        ("lcb", "LCB k=1", 1.0),
        ("lcb", "LCB k=2", 2.0),
        ("lcb", "LCB k=3", 3.0),
    )

    # Internal GP and acquisition-pool settings, inherited by BOConfig.
    gp_restarts: int = 1
    candidate_pool_global: int = 4000
    candidate_pool_local: int = 1000
    local_sigma_unit: float = 0.07
    ei_xi: float = 0.01
    pi_xi: float = 0.01

    # If True, skip runs that already exist in the results CSV.
    # Useful if the benchmark is interrupted.
    resume_existing: bool = True


# =============================================================================
# MONKEY PATCH: ADD PI ACQUISITION TO optimization_bo
# =============================================================================

def probability_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement for minimization.

    PI(x) = P(J(x) < J_best - xi)

    PI is usually more greedy than EI: it rewards the probability of any
    improvement, not the magnitude of the improvement.
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
        # For minimization, LCB = mu - kappa*sigma.
        # Choose smallest LCB, therefore return negative LCB as score.
        lcb = mu - config.lcb_kappa * sigma
        return -lcb

    raise ValueError(f"Unsupported acquisition: {acquisition!r}")


# Apply the patch so run_bo_single_target can use acquisition="pi".
obo.acquisition_scores = patched_acquisition_scores


# =============================================================================
# RUNNERS
# =============================================================================

def make_bo_config(bench_cfg: AcquisitionBenchmarkConfig, kappa: Optional[float]) -> obo.BOConfig:
    """Create a BOConfig configured for benchmark runs."""
    cfg = obo.BOConfig()

    # Disable main script workflow flags; this script controls the runs.
    cfg.run_pilot = False
    cfg.stop_after_pilot = False
    cfg.run_main_bo = False
    cfg.run_random_search = False
    cfg.run_kernel_sensitivity = False
    cfg.run_acquisition_sensitivity = False
    cfg.run_budget_sweep = False
    cfg.make_time_response_plots = False

    # Keep the same physical/objective settings as optimization_bo.py.
    cfg.main_kernel = bench_cfg.kernel_kind
    cfg.gp_restarts = bench_cfg.gp_restarts
    cfg.candidate_pool_global = bench_cfg.candidate_pool_global
    cfg.candidate_pool_local = bench_cfg.candidate_pool_local
    cfg.local_sigma_unit = bench_cfg.local_sigma_unit
    cfg.ei_xi = bench_cfg.ei_xi

    # Dynamically attach PI xi so the patched function can read it.
    cfg.pi_xi = bench_cfg.pi_xi  # type: ignore[attr-defined]

    if kappa is not None:
        cfg.lcb_kappa = float(kappa)

    return cfg


def run_single_benchmark(
    target: float,
    acquisition_name: str,
    acquisition_label: str,
    kappa: Optional[float],
    seed: int,
    bench_cfg: AcquisitionBenchmarkConfig,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, object]]:
    """Run one BO acquisition benchmark case."""
    cfg = make_bo_config(bench_cfg, kappa)
    run_label = (
        f"acq_{acquisition_label.replace(' ', '').replace('=', '').replace('.', 'p')}_"
        f"target{int(round(target * 1000)):04d}_seed{seed}"
    )

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=cfg,
        run_label=run_label,
        total_evaluations=bench_cfg.total_evaluations,
        n_initial_points=bench_cfg.initial_points,
        kernel_kind=bench_cfg.kernel_kind,
        acquisition=acquisition_name,
        seed=seed,
    )

    result = obo.build_result_row(best, cfg)
    result.update(
        {
            "benchmark": "acquisition",
            "target": float(target),
            "kernel": bench_cfg.kernel_kind,
            "acquisition": acquisition_name,
            "acquisition_label": acquisition_label,
            "kappa": np.nan if kappa is None else float(kappa),
            "seed": int(seed),
            "budget": int(bench_cfg.total_evaluations),
            "initial_points": int(bench_cfg.initial_points),
            "optimization_time_s": float(timing["total_wall_time_s"]),
            "mean_iteration_time_s": float(timing["mean_iteration_time_s"]),
            "mean_simulation_time_s": float(timing["mean_simulation_time_s"]),
            "mean_gp_fit_time_s": float(timing["mean_gp_fit_time_s"]),
            "mean_acquisition_time_s": float(timing["mean_acquisition_time_s"]),
        }
    )

    history = history.copy()
    history["benchmark"] = "acquisition"
    history["acquisition_label"] = acquisition_label
    history["kappa"] = np.nan if kappa is None else float(kappa)
    history["budget"] = int(bench_cfg.total_evaluations)
    history["initial_points"] = int(bench_cfg.initial_points)

    timing = dict(timing)
    timing.update(
        {
            "benchmark": "acquisition",
            "acquisition_label": acquisition_label,
            "kappa": np.nan if kappa is None else float(kappa),
            "budget": int(bench_cfg.total_evaluations),
        }
    )

    return result, history, timing


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark results by target and acquisition."""
    if results_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    group_cols = ["target", "acquisition_label"]
    for (target, acq_label), g in results_df.groupby(group_cols, sort=False):
        g = g.copy()
        feasible_rate = 100.0 * float(g["feasible_abs"].astype(bool).mean())
        rows.append(
            {
                "target": float(target),
                "acquisition_label": acq_label,
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
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["target", "mean_target_error_mm", "mean_residual_margin_mm"])
    return summary


def load_existing_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load partial benchmark files if they exist."""
    results_path = RESULTS_DIR / "bo_acquisition_benchmark.csv"
    history_path = RESULTS_DIR / "bo_acquisition_benchmark_history.csv"
    timing_path = RESULTS_DIR / "bo_acquisition_benchmark_timing.csv"

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    timing = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()
    return results, history, timing


def already_done(
    existing_results: pd.DataFrame,
    target: float,
    acquisition_label: str,
    seed: int,
    budget: int,
) -> bool:
    if existing_results.empty:
        return False
    mask = (
        np.isclose(existing_results["target"].astype(float), float(target))
        & (existing_results["acquisition_label"].astype(str) == acquisition_label)
        & (existing_results["seed"].astype(int) == int(seed))
        & (existing_results["budget"].astype(int) == int(budget))
    )
    return bool(mask.any())


# =============================================================================
# PLOTS
# =============================================================================

def acquisition_order() -> List[str]:
    return ["EI", "PI", "LCB k=1", "LCB k=2", "LCB k=3"]


def plot_final_error_boxplots(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_t = results_df[np.isclose(results_df["target"], target)].copy()
        labels = [x for x in acquisition_order() if x in set(df_t["acquisition_label"])]
        data = [df_t[df_t["acquisition_label"] == label]["target_error_mm"].values for label in labels]

        plt.figure(figsize=(9, 6))
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.ylabel("Final target error [mm]")
        plt.xlabel("Acquisition function")
        plt.title(f"BO acquisition benchmark: final error, target = {target:.2f} m")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_acquisition_final_error_boxplot_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_margin_boxplots(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_t = results_df[np.isclose(results_df["target"], target)].copy()
        labels = [x for x in acquisition_order() if x in set(df_t["acquisition_label"])]
        data = [df_t[df_t["acquisition_label"] == label]["residual_margin_mm"].values for label in labels]

        plt.figure(figsize=(9, 6))
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.axhline(0.0, linestyle="--", linewidth=1.8)
        plt.ylabel("Residual margin to true limit [mm]")
        plt.xlabel("Acquisition function")
        plt.title(f"BO acquisition benchmark: residual margin, target = {target:.2f} m")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_acquisition_margin_boxplot_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_mean_error_bars(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_t = summary_df[np.isclose(summary_df["target"], target)].copy()
        labels = [x for x in acquisition_order() if x in set(df_t["acquisition_label"])]
        df_t = df_t.set_index("acquisition_label").loc[labels].reset_index()

        plt.figure(figsize=(9, 6))
        x = np.arange(len(df_t))
        plt.bar(
            x,
            df_t["mean_target_error_mm"],
            yerr=df_t["std_target_error_mm"],
            capsize=4,
            edgecolor="black",
        )
        plt.xticks(x, df_t["acquisition_label"].values)
        plt.ylabel("Mean final target error [mm]")
        plt.xlabel("Acquisition function")
        plt.title(f"BO acquisition benchmark: mean final error, target = {target:.2f} m")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_acquisition_mean_error_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_convergence(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return

    for target in sorted(history_df["target"].unique()):
        df_t = history_df[np.isclose(history_df["target"], target)].copy()

        plt.figure(figsize=(10, 6))
        for label in acquisition_order():
            df_a = df_t[df_t["acquisition_label"] == label].copy()
            if df_a.empty:
                continue
            grouped = (
                df_a.groupby("iteration")["best_feasible_error_so_far_mm"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .sort_values("iteration")
            )
            plt.plot(grouped["iteration"], grouped["mean"], linewidth=2.2, label=label)
            plt.fill_between(grouped["iteration"], grouped["min"], grouped["max"], alpha=0.12)

        plt.xlabel("True simulator calls")
        plt.ylabel("Best feasible target error so far [mm]")
        plt.title(f"BO acquisition convergence, target = {target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_acquisition_convergence_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_error_margin_tradeoff(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    for target in sorted(results_df["target"].unique()):
        df_t = results_df[np.isclose(results_df["target"], target)].copy()
        plt.figure(figsize=(8, 6))
        for label in acquisition_order():
            df_a = df_t[df_t["acquisition_label"] == label]
            if df_a.empty:
                continue
            plt.scatter(
                df_a["target_error_mm"],
                df_a["residual_margin_mm"],
                s=70,
                edgecolor="black",
                label=label,
            )
        plt.axhline(0.0, linestyle="--", linewidth=1.8)
        plt.xlabel("Final target error [mm]")
        plt.ylabel("Residual margin to true limit [mm]")
        plt.title(f"BO acquisition trade-off, target = {target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_acquisition_error_margin_tradeoff_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def make_plots(results_df: pd.DataFrame, history_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plot_final_error_boxplots(results_df)
    plot_margin_boxplots(results_df)
    plot_mean_error_bars(summary_df)
    plot_convergence(history_df)
    plot_error_margin_tradeoff(results_df)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()
    bench_cfg = AcquisitionBenchmarkConfig()

    print("=" * 80)
    print("BO ACQUISITION BENCHMARK - STEP A")
    print("=" * 80)
    print(f"Targets:                 {bench_cfg.targets}")
    print(f"Seeds:                   {bench_cfg.seeds}")
    print(f"Kernel:                  {bench_cfg.kernel_kind}")
    print(f"Budget:                  {bench_cfg.total_evaluations} calls")
    print(f"Initial LHS points:      {bench_cfg.initial_points}")
    print(f"Acquisitions:            {[a[1] for a in bench_cfg.acquisitions]}")
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

    total_runs = len(bench_cfg.targets) * len(bench_cfg.acquisitions) * len(bench_cfg.seeds)
    run_counter = 0
    t0_all = time.perf_counter()

    for target in bench_cfg.targets:
        for acquisition_name, acquisition_label, kappa in bench_cfg.acquisitions:
            for seed in bench_cfg.seeds:
                run_counter += 1
                if bench_cfg.resume_existing and already_done(
                    existing_results,
                    target=target,
                    acquisition_label=acquisition_label,
                    seed=seed,
                    budget=bench_cfg.total_evaluations,
                ):
                    print(
                        f"[{run_counter}/{total_runs}] SKIP existing | "
                        f"target={target:.2f}, acquisition={acquisition_label}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{run_counter}/{total_runs}] target={target:.2f} | "
                    f"acquisition={acquisition_label} | seed={seed}"
                )

                result, history, timing = run_single_benchmark(
                    target=target,
                    acquisition_name=acquisition_name,
                    acquisition_label=acquisition_label,
                    kappa=kappa,
                    seed=seed,
                    bench_cfg=bench_cfg,
                )
                result_rows.append(result)
                history_rows.append(history)
                timing_rows.append(timing)

                # Save incrementally after every run, so an interruption does not lose work.
                results_df = pd.DataFrame(result_rows)
                history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
                timing_df = pd.DataFrame(timing_rows)
                summary_df = build_summary(results_df)

                results_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark.csv", index=False)
                history_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark_history.csv", index=False)
                timing_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark_timing.csv", index=False)
                summary_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark_summary.csv", index=False)

                print(
                    f"    best_y={float(result['peak_y_true']):.4f} m | "
                    f"err={float(result['target_error_mm']):.2f} mm | "
                    f"xr={float(result['max_abs_xr_true']):.4f} m | "
                    f"margin={float(result['residual_margin_mm']):.2f} mm | "
                    f"feasible={bool(result['feasible_abs'])} | "
                    f"time={float(result['optimization_time_s']):.1f} s"
                )

    total_wall = time.perf_counter() - t0_all

    results_df = pd.DataFrame(result_rows)
    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    timing_df = pd.DataFrame(timing_rows)
    summary_df = build_summary(results_df)

    results_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark_history.csv", index=False)
    timing_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark_timing.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "bo_acquisition_benchmark_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("ACQUISITION BENCHMARK SUMMARY")
    print("=" * 80)
    if not summary_df.empty:
        cols = [
            "target",
            "acquisition_label",
            "n_runs",
            "feasibility_rate_percent",
            "mean_target_error_mm",
            "std_target_error_mm",
            "min_target_error_mm",
            "max_target_error_mm",
            "mean_residual_margin_mm",
            "min_residual_margin_mm",
            "mean_best_iteration",
        ]
        print(summary_df[cols].to_string(index=False))

    print("\nSaved files:")
    print(f"  {RESULTS_DIR / 'bo_acquisition_benchmark.csv'}")
    print(f"  {RESULTS_DIR / 'bo_acquisition_benchmark_history.csv'}")
    print(f"  {RESULTS_DIR / 'bo_acquisition_benchmark_timing.csv'}")
    print(f"  {RESULTS_DIR / 'bo_acquisition_benchmark_summary.csv'}")
    print(f"Total benchmark wall time: {total_wall / 60:.1f} min")

    print("\nGenerating benchmark plots...")
    make_plots(results_df, history_df, summary_df)

    print("\nDone. Send me the ACQUISITION BENCHMARK SUMMARY output or the summary CSV.")


if __name__ == "__main__":
    main()
