"""
bo_alpha_benchmark.py
=====================

Step C of the Bayesian Optimization refinement plan.

Purpose
-------
Benchmark the numerical regularization parameter `alpha` used by the internal
Gaussian Process of the true-simulator Bayesian Optimization baseline.

The previous benchmarks selected the following candidate configuration:
    - acquisition = PI
    - kernel      = RBF

This script compares several values of `gp_alpha` on two representative targets:
    - 0.65 m: high but reachable target
    - 0.75 m: saturated / difficult target

Outputs are written to separate folders:
    results/optimization_bo_alpha_benchmark/
    figures/optimization_bo_alpha_benchmark/

This script intentionally does not overwrite the final BO folders or previous
benchmark folders.

Run from project root:
    python src/bo_alpha_benchmark.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# This script must be placed in src/ next to optimization_bo.py
import optimization_bo as obo


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_alpha_benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_alpha_benchmark"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlphaBenchmarkConfig:
    """Configuration for Step C alpha benchmark."""

    # Representative targets.
    targets: Tuple[float, ...] = (0.65, 0.75)
    seeds: Tuple[int, ...] = (2026, 2027)

    total_evaluations: int = 100
    initial_points: int = 20

    # Selected from previous BO benchmarks.
    kernel_kind: str = "rbf"
    kernel_label: str = "RBF"
    acquisition_name: str = "pi"
    acquisition_label: str = "PI"
    pi_xi: float = 0.01

    # Values to test. These are meaningful because optimization_bo uses
    # normalize_y=True inside GaussianProcessRegressor.
    alpha_values: Tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-3)

    # BO internal settings.
    gp_restarts: int = 1
    candidate_pool_global: int = 4000
    candidate_pool_local: int = 1000
    local_sigma_unit: float = 0.07

    # Resume existing runs if interrupted.
    resume_existing: bool = True


# =============================================================================
# PI ACQUISITION PATCH
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


# Apply patch so optimization_bo.run_bo_single_target can use PI.
obo.acquisition_scores = patched_acquisition_scores


# =============================================================================
# RUNNERS
# =============================================================================


def make_bo_config(bench_cfg: AlphaBenchmarkConfig, alpha: float) -> obo.BOConfig:
    """Create BOConfig for one alpha benchmark run."""
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

    cfg.main_kernel = bench_cfg.kernel_kind
    cfg.main_acquisition = bench_cfg.acquisition_name
    cfg.gp_restarts = bench_cfg.gp_restarts
    cfg.gp_alpha = float(alpha)
    cfg.candidate_pool_global = bench_cfg.candidate_pool_global
    cfg.candidate_pool_local = bench_cfg.candidate_pool_local
    cfg.local_sigma_unit = bench_cfg.local_sigma_unit

    # Dynamically attach PI xi so the patched acquisition can read it.
    cfg.pi_xi = bench_cfg.pi_xi  # type: ignore[attr-defined]

    return cfg


def safe_alpha_label(alpha: float) -> str:
    return f"alpha{alpha:.0e}".replace("-", "m").replace("+", "p")


def run_single_benchmark(
    target: float,
    alpha: float,
    seed: int,
    bench_cfg: AlphaBenchmarkConfig,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, object]]:
    """Run one BO alpha benchmark case."""
    cfg = make_bo_config(bench_cfg, alpha)
    run_label = f"alpha_{safe_alpha_label(alpha)}_target{int(round(target * 1000)):04d}_seed{seed}"

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=cfg,
        run_label=run_label,
        total_evaluations=bench_cfg.total_evaluations,
        n_initial_points=bench_cfg.initial_points,
        kernel_kind=bench_cfg.kernel_kind,
        acquisition=bench_cfg.acquisition_name,
        seed=seed,
    )

    result = obo.build_result_row(best, cfg)
    result.update(
        {
            "benchmark": "alpha",
            "target": float(target),
            "kernel": bench_cfg.kernel_kind,
            "kernel_label": bench_cfg.kernel_label,
            "acquisition": bench_cfg.acquisition_name,
            "acquisition_label": bench_cfg.acquisition_label,
            "seed": int(seed),
            "budget": int(bench_cfg.total_evaluations),
            "initial_points": int(bench_cfg.initial_points),
            "gp_alpha": float(alpha),
            "alpha_label": f"{alpha:.0e}",
            "optimization_time_s": float(timing["total_wall_time_s"]),
            "mean_iteration_time_s": float(timing["mean_iteration_time_s"]),
            "mean_simulation_time_s": float(timing["mean_simulation_time_s"]),
            "mean_gp_fit_time_s": float(timing["mean_gp_fit_time_s"]),
            "mean_acquisition_time_s": float(timing["mean_acquisition_time_s"]),
        }
    )

    history = history.copy()
    history["benchmark"] = "alpha"
    history["kernel_label"] = bench_cfg.kernel_label
    history["acquisition_label"] = bench_cfg.acquisition_label
    history["budget"] = int(bench_cfg.total_evaluations)
    history["initial_points"] = int(bench_cfg.initial_points)
    history["gp_alpha"] = float(alpha)
    history["alpha_label"] = f"{alpha:.0e}"

    timing = dict(timing)
    timing.update(
        {
            "benchmark": "alpha",
            "kernel_label": bench_cfg.kernel_label,
            "acquisition_label": bench_cfg.acquisition_label,
            "budget": int(bench_cfg.total_evaluations),
            "gp_alpha": float(alpha),
            "alpha_label": f"{alpha:.0e}",
        }
    )

    return result, history, timing


# =============================================================================
# SUMMARIES AND RESUME
# =============================================================================


def load_existing_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_path = RESULTS_DIR / "bo_alpha_benchmark.csv"
    history_path = RESULTS_DIR / "bo_alpha_benchmark_history.csv"
    timing_path = RESULTS_DIR / "bo_alpha_benchmark_timing.csv"

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    timing = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()
    return results, history, timing


def already_done(
    existing_results: pd.DataFrame,
    target: float,
    alpha: float,
    seed: int,
    budget: int,
    kernel_label: str,
    acquisition_label: str,
) -> bool:
    if existing_results.empty:
        return False
    mask = (
        np.isclose(existing_results["target"].astype(float), float(target))
        & np.isclose(existing_results["gp_alpha"].astype(float), float(alpha))
        & (existing_results["seed"].astype(int) == int(seed))
        & (existing_results["budget"].astype(int) == int(budget))
        & (existing_results["kernel_label"].astype(str) == kernel_label)
        & (existing_results["acquisition_label"].astype(str) == acquisition_label)
    )
    return bool(mask.any())


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate alpha benchmark results by target and alpha."""
    if results_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (target, alpha), g in results_df.groupby(["target", "gp_alpha"], sort=False):
        g = g.copy()
        feasible_rate = 100.0 * float(g["feasible_abs"].astype(bool).mean())
        rows.append(
            {
                "target": float(target),
                "gp_alpha": float(alpha),
                "alpha_label": f"{float(alpha):.0e}",
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
    """Aggregate alpha performance across representative targets."""
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["rank_by_error_within_target_seed"] = df.groupby(["target", "seed"])["target_error_mm"].rank(
        method="min", ascending=True
    )

    rows: List[Dict[str, object]] = []
    for alpha, g in df.groupby("gp_alpha", sort=False):
        feasible_rate = 100.0 * float(g["feasible_abs"].astype(bool).mean())
        rows.append(
            {
                "gp_alpha": float(alpha),
                "alpha_label": f"{float(alpha):.0e}",
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


def alpha_order(values: Tuple[float, ...]) -> List[float]:
    return list(values)


def plot_alpha_error(summary_df: pd.DataFrame, cfg: AlphaBenchmarkConfig) -> None:
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_t = summary_df[np.isclose(summary_df["target"], target)].copy()
        df_t = df_t.set_index("gp_alpha").reindex(alpha_order(cfg.alpha_values)).reset_index()

        x = df_t["gp_alpha"].astype(float).values
        y = df_t["mean_target_error_mm"].astype(float).values
        yerr = df_t["std_target_error_mm"].astype(float).values

        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.2, capsize=4)
        plt.xscale("log")
        plt.xlabel("GP alpha")
        plt.ylabel("Mean final target error [mm]")
        plt.title(f"BO alpha benchmark: target error, target = {target:.2f} m")
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_alpha_error_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_alpha_margin(summary_df: pd.DataFrame, cfg: AlphaBenchmarkConfig) -> None:
    if summary_df.empty:
        return

    for target in sorted(summary_df["target"].unique()):
        df_t = summary_df[np.isclose(summary_df["target"], target)].copy()
        df_t = df_t.set_index("gp_alpha").reindex(alpha_order(cfg.alpha_values)).reset_index()

        x = df_t["gp_alpha"].astype(float).values
        y = df_t["mean_residual_margin_mm"].astype(float).values
        y_min = df_t["min_residual_margin_mm"].astype(float).values

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker="o", linewidth=2.2, label="Mean residual margin")
        plt.plot(x, y_min, marker="s", linewidth=2.0, linestyle="--", label="Minimum residual margin")
        plt.axhline(0.0, linestyle="--", linewidth=1.8)
        plt.xscale("log")
        plt.xlabel("GP alpha")
        plt.ylabel("Residual margin to true limit [mm]")
        plt.title(f"BO alpha benchmark: residual margin, target = {target:.2f} m")
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_alpha_margin_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_alpha_convergence(history_df: pd.DataFrame, cfg: AlphaBenchmarkConfig) -> None:
    if history_df.empty:
        return

    for target in sorted(history_df["target"].unique()):
        df_t = history_df[np.isclose(history_df["target"], target)].copy()

        plt.figure(figsize=(10, 6))
        for alpha in alpha_order(cfg.alpha_values):
            df_a = df_t[np.isclose(df_t["gp_alpha"].astype(float), float(alpha))].copy()
            if df_a.empty:
                continue
            grouped = (
                df_a.groupby("iteration")["best_feasible_error_so_far_mm"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .sort_values("iteration")
            )
            label = f"alpha={alpha:.0e}"
            plt.plot(grouped["iteration"], grouped["mean"], linewidth=2.2, label=label)
            plt.fill_between(grouped["iteration"], grouped["min"], grouped["max"], alpha=0.12)

        plt.xlabel("True simulator calls")
        plt.ylabel("Best feasible target error so far [mm]")
        plt.title(f"BO alpha convergence, target = {target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / f"bo_alpha_convergence_target{int(round(target * 1000)):04d}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")


def plot_overall_alpha_bars(overall_df: pd.DataFrame, cfg: AlphaBenchmarkConfig) -> None:
    if overall_df.empty:
        return

    df = overall_df.set_index("gp_alpha").reindex(alpha_order(cfg.alpha_values)).reset_index()
    labels = [f"{a:.0e}" for a in df["gp_alpha"].astype(float).values]
    x = np.arange(len(df))

    plt.figure(figsize=(8, 6))
    plt.bar(x, df["mean_target_error_mm"], edgecolor="black")
    plt.xticks(x, labels)
    plt.xlabel("GP alpha")
    plt.ylabel("Mean target error across benchmark targets [mm]")
    plt.title("BO alpha benchmark: overall mean target error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "bo_alpha_overall_mean_error.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_all(results_df: pd.DataFrame, history_df: pd.DataFrame, summary_df: pd.DataFrame, overall_df: pd.DataFrame, cfg: AlphaBenchmarkConfig) -> None:
    print("\nGenerating alpha benchmark plots...")
    plot_alpha_error(summary_df, cfg)
    plot_alpha_margin(summary_df, cfg)
    plot_alpha_convergence(history_df, cfg)
    plot_overall_alpha_bars(overall_df, cfg)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_dirs()
    cfg = AlphaBenchmarkConfig()

    print("=" * 80)
    print("BO ALPHA BENCHMARK - STEP C")
    print("=" * 80)
    print(f"Targets:                 {cfg.targets}")
    print(f"Seeds:                   {cfg.seeds}")
    print(f"Kernel:                  {cfg.kernel_label}")
    print(f"Acquisition:             {cfg.acquisition_label}")
    print(f"Budget:                  {cfg.total_evaluations} calls")
    print(f"Initial LHS points:      {cfg.initial_points}")
    print(f"Alpha values:            {[f'{a:.0e}' for a in cfg.alpha_values]}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)

    existing_results, existing_history, existing_timing = load_existing_results()
    all_results: List[Dict[str, object]] = []
    all_history: List[pd.DataFrame] = []
    all_timing: List[Dict[str, object]] = []

    if cfg.resume_existing and not existing_results.empty:
        print(f"Resuming from existing file with {len(existing_results)} completed runs.")
        all_results.extend(existing_results.to_dict(orient="records"))
        if not existing_history.empty:
            all_history.append(existing_history)
        if not existing_timing.empty:
            all_timing.extend(existing_timing.to_dict(orient="records"))

    total_runs = len(cfg.targets) * len(cfg.alpha_values) * len(cfg.seeds)
    run_counter = 0

    import time
    t_global = time.perf_counter()

    for target in cfg.targets:
        for alpha in cfg.alpha_values:
            for seed in cfg.seeds:
                run_counter += 1
                if cfg.resume_existing and already_done(
                    existing_results,
                    target=target,
                    alpha=alpha,
                    seed=seed,
                    budget=cfg.total_evaluations,
                    kernel_label=cfg.kernel_label,
                    acquisition_label=cfg.acquisition_label,
                ):
                    print(
                        f"[{run_counter}/{total_runs}] SKIP existing | "
                        f"target={target:.2f}, alpha={alpha:.0e}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{run_counter}/{total_runs}] target={target:.2f} | "
                    f"alpha={alpha:.0e} | kernel={cfg.kernel_label} | "
                    f"acquisition={cfg.acquisition_label} | seed={seed}"
                )

                result, history, timing = run_single_benchmark(target, alpha, seed, cfg)
                all_results.append(result)
                all_history.append(history)
                all_timing.append(timing)

                print(
                    f"    best_y={result['peak_y_true']:.4f} m | "
                    f"err={result['target_error_mm']:.2f} mm | "
                    f"xr={result['max_abs_xr_true']:.4f} m | "
                    f"margin={result['residual_margin_mm']:.2f} mm | "
                    f"feasible={result['feasible_abs']} | "
                    f"time={result['optimization_time_s']:.1f} s"
                )

                # Incremental saves for resume safety.
                pd.DataFrame(all_results).to_csv(RESULTS_DIR / "bo_alpha_benchmark.csv", index=False)
                pd.concat(all_history, ignore_index=True).to_csv(
                    RESULTS_DIR / "bo_alpha_benchmark_history.csv", index=False
                )
                pd.DataFrame(all_timing).to_csv(RESULTS_DIR / "bo_alpha_benchmark_timing.csv", index=False)

    total_time = time.perf_counter() - t_global

    results_df = pd.DataFrame(all_results)
    history_df = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    timing_df = pd.DataFrame(all_timing)

    summary_df = build_summary(results_df)
    overall_df = build_overall_summary(results_df)

    summary_df.to_csv(RESULTS_DIR / "bo_alpha_benchmark_summary.csv", index=False)
    overall_df.to_csv(RESULTS_DIR / "bo_alpha_benchmark_overall_summary.csv", index=False)
    results_df.to_csv(RESULTS_DIR / "bo_alpha_benchmark.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "bo_alpha_benchmark_history.csv", index=False)
    timing_df.to_csv(RESULTS_DIR / "bo_alpha_benchmark_timing.csv", index=False)

    print("\n" + "=" * 80)
    print("ALPHA BENCHMARK SUMMARY")
    print("=" * 80)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("OVERALL ALPHA SUMMARY ACROSS REPRESENTATIVE TARGETS")
    print("=" * 80)
    if not overall_df.empty:
        print(overall_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  {RESULTS_DIR / 'bo_alpha_benchmark.csv'}")
    print(f"  {RESULTS_DIR / 'bo_alpha_benchmark_history.csv'}")
    print(f"  {RESULTS_DIR / 'bo_alpha_benchmark_timing.csv'}")
    print(f"  {RESULTS_DIR / 'bo_alpha_benchmark_summary.csv'}")
    print(f"  {RESULTS_DIR / 'bo_alpha_benchmark_overall_summary.csv'}")
    print(f"Total benchmark wall time: {total_time / 60.0:.1f} min")

    plot_all(results_df, history_df, summary_df, overall_df, cfg)


if __name__ == "__main__":
    main()
