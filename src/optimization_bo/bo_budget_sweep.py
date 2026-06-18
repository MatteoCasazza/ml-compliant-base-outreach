"""
bo_budget_sweep.py
==================

Step D of the Bayesian Optimization refinement plan.

Purpose
-------
Evaluate how the selected BO configuration improves as the number of online
true-simulator calls increases.

Selected configuration from previous benchmarks:
    - acquisition = PI
    - kernel      = RBF
    - gp_alpha    = 1e-8

The script runs one long BO trajectory of 200 simulator calls for each
representative target and seed, then extracts best-so-far performance at
budgets 50, 100, 150, and 200. This avoids running four separate BO experiments
for each budget.

Outputs are written to separate folders:
    results/optimization_bo_budget_sweep/
    figures/optimization_bo_budget_sweep/

Run from project root:
    python src/bo_budget_sweep.py
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
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_budget_sweep"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_budget_sweep"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BudgetSweepConfig:
    """Configuration for Step D budget sweep."""

    # Representative targets.
    targets: Tuple[float, ...] = (0.65, 0.75)
    seeds: Tuple[int, ...] = (2026, 2027)

    # One long BO run is used to extract all these budgets.
    budgets: Tuple[int, ...] = (50, 100, 150, 200)
    long_total_evaluations: int = 200
    initial_points: int = 20

    # Selected BO configuration.
    kernel_kind: str = "rbf"
    kernel_label: str = "RBF"
    acquisition_name: str = "pi"
    acquisition_label: str = "PI"
    gp_alpha: float = 1e-8
    alpha_label: str = "1e-08"
    pi_xi: float = 0.01

    # BO internal settings.
    gp_restarts: int = 1
    candidate_pool_global: int = 4000
    candidate_pool_local: int = 1000
    local_sigma_unit: float = 0.07

    # Resume existing long BO histories if interrupted.
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


def make_bo_config(sweep_cfg: BudgetSweepConfig) -> obo.BOConfig:
    """Create BOConfig for the selected BO budget-sweep configuration."""
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

    cfg.main_kernel = sweep_cfg.kernel_kind
    cfg.main_acquisition = sweep_cfg.acquisition_name
    cfg.gp_alpha = float(sweep_cfg.gp_alpha)
    cfg.gp_restarts = sweep_cfg.gp_restarts
    cfg.candidate_pool_global = sweep_cfg.candidate_pool_global
    cfg.candidate_pool_local = sweep_cfg.candidate_pool_local
    cfg.local_sigma_unit = sweep_cfg.local_sigma_unit

    # Dynamically attach PI xi so the patched acquisition can read it.
    cfg.pi_xi = sweep_cfg.pi_xi  # type: ignore[attr-defined]

    return cfg


def run_label_for(target: float, seed: int, sweep_cfg: BudgetSweepConfig) -> str:
    target_mm = int(round(target * 1000))
    return (
        f"budget_long_{sweep_cfg.acquisition_label}_{sweep_cfg.kernel_label}_"
        f"alpha{sweep_cfg.gp_alpha:.0e}_target{target_mm:04d}_seed{seed}"
        .replace("+", "p")
        .replace("-", "m")
        .replace("/", "")
        .replace(" ", "")
    )


def run_or_load_long_history(
    target: float,
    seed: int,
    sweep_cfg: BudgetSweepConfig,
    existing_history: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run or load one long BO trajectory for a target/seed."""
    cfg = make_bo_config(sweep_cfg)
    run_label = run_label_for(target, seed, sweep_cfg)

    if sweep_cfg.resume_existing and not existing_history.empty:
        mask = existing_history["run_label"].astype(str) == run_label
        if mask.any():
            hist = existing_history.loc[mask].copy()
            if int(hist["iteration"].max()) >= sweep_cfg.long_total_evaluations:
                timing = {
                    "run_label": run_label,
                    "target": float(target),
                    "seed": int(seed),
                    "loaded_from_existing": True,
                    "total_wall_time_s": float(hist["iteration_time_s"].sum()),
                    "mean_iteration_time_s": float(hist["iteration_time_s"].mean()),
                    "mean_simulation_time_s": float(hist["simulation_time_s"].mean()),
                    "mean_gp_fit_time_s": float(hist["gp_fit_time_s"].mean()),
                    "mean_acquisition_time_s": float(hist["acquisition_time_s"].mean()),
                }
                print(f"    SKIP existing long run | target={target:.2f}, seed={seed}")
                return hist, timing

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=cfg,
        run_label=run_label,
        total_evaluations=sweep_cfg.long_total_evaluations,
        n_initial_points=sweep_cfg.initial_points,
        kernel_kind=sweep_cfg.kernel_kind,
        acquisition=sweep_cfg.acquisition_name,
        seed=seed,
    )

    history = history.copy()
    history["benchmark"] = "budget_sweep"
    history["kernel_label"] = sweep_cfg.kernel_label
    history["acquisition_label"] = sweep_cfg.acquisition_label
    history["gp_alpha"] = float(sweep_cfg.gp_alpha)
    history["alpha_label"] = sweep_cfg.alpha_label
    history["long_total_evaluations"] = int(sweep_cfg.long_total_evaluations)

    timing = dict(timing)
    timing.update(
        {
            "target": float(target),
            "seed": int(seed),
            "loaded_from_existing": False,
            "kernel": sweep_cfg.kernel_kind,
            "kernel_label": sweep_cfg.kernel_label,
            "acquisition": sweep_cfg.acquisition_name,
            "acquisition_label": sweep_cfg.acquisition_label,
            "gp_alpha": float(sweep_cfg.gp_alpha),
            "alpha_label": sweep_cfg.alpha_label,
            "long_total_evaluations": int(sweep_cfg.long_total_evaluations),
        }
    )
    return history, timing


def extract_budget_rows(
    long_history: pd.DataFrame,
    sweep_cfg: BudgetSweepConfig,
) -> pd.DataFrame:
    """Extract best feasible solution at each budget from long BO histories."""
    cfg = make_bo_config(sweep_cfg)
    rows: List[Dict[str, object]] = []

    group_cols = ["target", "seed", "run_label"]
    for (target, seed, run_label), group in long_history.groupby(group_cols, dropna=False):
        group = group.sort_values("iteration").copy()
        for budget in sweep_cfg.budgets:
            if budget > int(group["iteration"].max()):
                continue
            subset = group[group["iteration"] <= budget].copy()
            best_row = obo.select_best_from_history(subset)
            result = obo.build_result_row(best_row, cfg)
            result.update(
                {
                    "benchmark": "budget_sweep",
                    "target": float(target),
                    "seed": int(seed),
                    "budget": int(budget),
                    "long_total_evaluations": int(sweep_cfg.long_total_evaluations),
                    "initial_points": int(sweep_cfg.initial_points),
                    "kernel": sweep_cfg.kernel_kind,
                    "kernel_label": sweep_cfg.kernel_label,
                    "acquisition": sweep_cfg.acquisition_name,
                    "acquisition_label": sweep_cfg.acquisition_label,
                    "gp_alpha": float(sweep_cfg.gp_alpha),
                    "alpha_label": sweep_cfg.alpha_label,
                    "n_true_simulator_evaluations": int(budget),
                }
            )
            rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY AND PLOTS
# =============================================================================


def summarize_budget_results(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize budget-sweep results by target/budget and overall by budget."""
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        results_df.groupby(["target", "budget"], dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            feasibility_rate_percent=("feasible_abs", lambda s: 100.0 * s.astype(bool).mean()),
            mean_target_error_mm=("target_error_mm", "mean"),
            std_target_error_mm=("target_error_mm", "std"),
            min_target_error_mm=("target_error_mm", "min"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_peak_y_true=("peak_y_true", "mean"),
            mean_max_abs_xr_true=("max_abs_xr_true", "mean"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .reset_index()
        .sort_values(["target", "budget"])
    )

    overall = (
        results_df.groupby("budget", dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            n_targets=("target", "nunique"),
            feasibility_rate_percent=("feasible_abs", lambda s: 100.0 * s.astype(bool).mean()),
            mean_target_error_mm=("target_error_mm", "mean"),
            median_target_error_mm=("target_error_mm", "median"),
            std_target_error_mm=("target_error_mm", "std"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .reset_index()
        .sort_values("budget")
    )

    return summary, overall


def save_plots(results_df: pd.DataFrame, long_history: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Create budget-sweep plots."""
    if results_df.empty:
        return

    # Mean final error vs budget for each representative target.
    for target in sorted(results_df["target"].unique()):
        sub = summary_df[summary_df["target"] == target].sort_values("budget")
        if sub.empty:
            continue

        plt.figure(figsize=(7.0, 4.5))
        plt.plot(sub["budget"], sub["mean_target_error_mm"], marker="o", label="Mean across seeds")
        plt.fill_between(
            sub["budget"].astype(float),
            sub["min_target_error_mm"].astype(float),
            sub["max_target_error_mm"].astype(float),
            alpha=0.18,
            label="Seed min-max range",
        )
        plt.xlabel("True simulator calls")
        plt.ylabel("Best feasible target error [mm]")
        plt.title(f"BO budget sweep, target={target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"bo_budget_sweep_error_target{int(round(target * 1000)):04d}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7.0, 4.5))
        plt.plot(sub["budget"], sub["mean_residual_margin_mm"], marker="o", label="Mean residual margin")
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("True simulator calls")
        plt.ylabel("Residual margin [mm]")
        plt.title(f"BO residual margin vs budget, target={target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"bo_budget_sweep_margin_target{int(round(target * 1000)):04d}.png", dpi=300)
        plt.close()

        # Convergence curves by seed.
        hist_sub = long_history[long_history["target"] == target].copy()
        if not hist_sub.empty and "best_feasible_error_so_far_mm" in hist_sub.columns:
            plt.figure(figsize=(7.0, 4.5))
            for seed, g in hist_sub.groupby("seed"):
                g = g.sort_values("iteration")
                plt.plot(
                    g["iteration"],
                    g["best_feasible_error_so_far_mm"],
                    linewidth=1.8,
                    label=f"seed {int(seed)}",
                )
            for b in sorted(results_df["budget"].unique()):
                plt.axvline(int(b), linestyle=":", linewidth=0.8)
            plt.xlabel("True simulator calls")
            plt.ylabel("Best feasible target error so far [mm]")
            plt.title(f"BO convergence, target={target:.2f} m")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"bo_budget_sweep_convergence_target{int(round(target * 1000)):04d}.png", dpi=300)
            plt.close()

    # Overall mean error vs budget.
    overall = (
        results_df.groupby("budget")
        .agg(
            mean_target_error_mm=("target_error_mm", "mean"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
        )
        .reset_index()
        .sort_values("budget")
    )

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(overall["budget"], overall["mean_target_error_mm"], marker="o", label="Mean target error")
    plt.plot(overall["budget"], overall["max_target_error_mm"], marker="s", label="Worst-case target error")
    plt.xlabel("True simulator calls")
    plt.ylabel("Target error [mm]")
    plt.title("BO budget sweep summary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bo_budget_sweep_overall_error.png", dpi=300)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_dirs()
    sweep_cfg = BudgetSweepConfig()

    long_history_path = RESULTS_DIR / "bo_budget_sweep_long_history.csv"
    timing_path = RESULTS_DIR / "bo_budget_sweep_timing.csv"
    results_path = RESULTS_DIR / "bo_budget_sweep_results.csv"
    summary_path = RESULTS_DIR / "bo_budget_sweep_summary.csv"
    overall_path = RESULTS_DIR / "bo_budget_sweep_overall_summary.csv"

    print("=" * 80)
    print("BO BUDGET SWEEP - STEP D")
    print("=" * 80)
    print(f"Targets:                 {sweep_cfg.targets}")
    print(f"Seeds:                   {sweep_cfg.seeds}")
    print(f"Acquisition:             {sweep_cfg.acquisition_label}")
    print(f"Kernel:                  {sweep_cfg.kernel_label}")
    print(f"GP alpha:                {sweep_cfg.alpha_label}")
    print(f"Budgets extracted:       {sweep_cfg.budgets}")
    print(f"Long BO calls/run:       {sweep_cfg.long_total_evaluations}")
    print(f"Initial LHS points:      {sweep_cfg.initial_points}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)

    existing_history = pd.DataFrame()
    if sweep_cfg.resume_existing and long_history_path.exists():
        existing_history = pd.read_csv(long_history_path)
        print(f"Resuming from existing long history with {len(existing_history)} rows.")

    all_histories: List[pd.DataFrame] = []
    timing_rows: List[Dict[str, object]] = []

    total_runs = len(sweep_cfg.targets) * len(sweep_cfg.seeds)
    run_counter = 0
    for target in sweep_cfg.targets:
        for seed in sweep_cfg.seeds:
            run_counter += 1
            print(f"\n[{run_counter}/{total_runs}] target={target:.2f} | seed={seed}")
            history, timing = run_or_load_long_history(target, seed, sweep_cfg, existing_history)
            all_histories.append(history)
            timing_rows.append(timing)

            best_100 = obo.select_best_from_history(history[history["iteration"] <= 100])
            best_200 = obo.select_best_from_history(history[history["iteration"] <= 200])
            print(
                f"    budget100: y={best_100['peak_y_true']:.4f} m | "
                f"err={best_100['target_error_mm']:.2f} mm | "
                f"margin={best_100['residual_margin_mm']:.2f} mm"
            )
            print(
                f"    budget200: y={best_200['peak_y_true']:.4f} m | "
                f"err={best_200['target_error_mm']:.2f} mm | "
                f"margin={best_200['residual_margin_mm']:.2f} mm"
            )

    long_history = pd.concat(all_histories, ignore_index=True)
    long_history = obo.add_cumulative_best_columns(long_history)
    long_history.to_csv(long_history_path, index=False)

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(timing_path, index=False)

    results_df = extract_budget_rows(long_history, sweep_cfg)
    results_df.to_csv(results_path, index=False)

    summary_df, overall_df = summarize_budget_results(results_df)
    summary_df.to_csv(summary_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print("\n" + "=" * 80)
    print("BUDGET SWEEP SUMMARY")
    print("=" * 80)
    if summary_df.empty:
        print("No summary available.")
    else:
        print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("OVERALL BUDGET SUMMARY ACROSS REPRESENTATIVE TARGETS")
    print("=" * 80)
    if overall_df.empty:
        print("No overall summary available.")
    else:
        print(overall_df.to_string(index=False))

    print("\nSaved files:")
    for path in [long_history_path, timing_path, results_path, summary_path, overall_path]:
        print(f"  {path}")

    print("\nGenerating budget-sweep plots...")
    save_plots(results_df, long_history, summary_df)
    print("Done.")


if __name__ == "__main__":
    main()
