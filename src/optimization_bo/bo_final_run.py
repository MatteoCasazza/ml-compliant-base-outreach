"""
bo_final_run.py
===============

Final Bayesian Optimization run after the BO benchmark stages.

Selected configuration from the benchmarks:
    acquisition = PI
    kernel      = RBF
    gp_alpha    = 1e-8
    budget      = 200 true simulator calls per target
    initial LHS = 20 points

The script runs BO on all official targets. By default it uses two seeds so that
BO stochastic variability can be reported. Random Search at the same online
budget can also be run as a baseline.

Outputs are written to:
    results/optimization_bo_final/
    figures/optimization_bo_final/

Run from project root:
    python src/bo_final_run.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Place this file in src/ next to optimization_bo.py
import optimization_bo as obo


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization_bo_final"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_final"

# Make plotting helpers inside optimization_bo save in the final folders if used.
obo.RESULTS_DIR = RESULTS_DIR
obo.FIGURES_DIR = FIGURES_DIR


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FinalBOConfig:
    """Final BO configuration selected after acquisition/kernel/alpha/budget tests."""

    targets: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)

    # Use two seeds to report stochastic BO variability. Set to (2026,) for a
    # single representative run if you need a faster run.
    seeds: Tuple[int, ...] = (2026, 2027)

    # Final selected BO setup.
    acquisition_name: str = "pi"
    acquisition_label: str = "PI"
    kernel_kind: str = "rbf"
    kernel_label: str = "RBF"
    gp_alpha: float = 1e-8
    alpha_label: str = "1e-08"

    # Budget selected from Step D. 200 improves the high saturated target.
    total_evaluations: int = 200
    initial_points: int = 20

    # Internal BO settings.
    pi_xi: float = 0.01
    gp_restarts: int = 1
    candidate_pool_global: int = 4000
    candidate_pool_local: int = 1000
    local_sigma_unit: float = 0.07

    # Same-budget Random Search baseline. Set False if you only want final BO.
    run_random_search_baseline: bool = True

    # Skip already completed runs when CSV history exists.
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

    Higher score is better for all acquisitions.
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
# CONFIG HELPERS
# =============================================================================


def make_bo_config(final_cfg: FinalBOConfig) -> obo.BOConfig:
    """Create the optimization_bo.BOConfig used by the final run."""
    cfg = obo.BOConfig()

    # This script controls the workflow directly.
    cfg.run_pilot = False
    cfg.stop_after_pilot = False
    cfg.run_main_bo = False
    cfg.run_random_search = False
    cfg.run_kernel_sensitivity = False
    cfg.run_acquisition_sensitivity = False
    cfg.run_budget_sweep = False
    cfg.make_time_response_plots = False

    cfg.targets = tuple(final_cfg.targets)
    cfg.main_kernel = final_cfg.kernel_kind
    cfg.main_acquisition = final_cfg.acquisition_name
    cfg.main_total_evaluations = int(final_cfg.total_evaluations)
    cfg.main_initial_points = int(final_cfg.initial_points)

    cfg.gp_alpha = float(final_cfg.gp_alpha)
    cfg.gp_restarts = int(final_cfg.gp_restarts)
    cfg.candidate_pool_global = int(final_cfg.candidate_pool_global)
    cfg.candidate_pool_local = int(final_cfg.candidate_pool_local)
    cfg.local_sigma_unit = float(final_cfg.local_sigma_unit)

    # Dynamically attach PI xi so the patched acquisition can read it.
    cfg.pi_xi = float(final_cfg.pi_xi)  # type: ignore[attr-defined]

    return cfg


def bo_run_label(target: float, seed: int, final_cfg: FinalBOConfig) -> str:
    target_mm = int(round(target * 1000))
    return (
        f"final_BO_{final_cfg.acquisition_label}_{final_cfg.kernel_label}_"
        f"alpha{final_cfg.gp_alpha:.0e}_budget{final_cfg.total_evaluations}_"
        f"target{target_mm:04d}_seed{seed}"
        .replace("+", "p")
        .replace("-", "m")
        .replace("/", "")
        .replace(" ", "")
    )


def rs_run_label(target: float, seed: int, final_cfg: FinalBOConfig) -> str:
    target_mm = int(round(target * 1000))
    return f"final_RS_budget{final_cfg.total_evaluations}_target{target_mm:04d}_seed{seed}"


# =============================================================================
# RUNNERS
# =============================================================================


def run_or_load_bo(
    target: float,
    seed: int,
    final_cfg: FinalBOConfig,
    existing_history: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    """Run or load one final BO trajectory."""
    cfg = make_bo_config(final_cfg)
    run_label = bo_run_label(target, seed, final_cfg)

    if final_cfg.resume_existing and not existing_history.empty:
        mask = existing_history["run_label"].astype(str) == run_label
        if mask.any():
            hist = existing_history.loc[mask].copy()
            if int(hist["iteration"].max()) >= final_cfg.total_evaluations:
                hist = obo.add_cumulative_best_columns(hist)
                best = obo.select_best_from_history(hist)
                timing = {
                    "method": "BO",
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
                print(f"    SKIP existing BO | target={target:.2f}, seed={seed}")
                return best, hist, timing

    best, history, timing = obo.run_bo_single_target(
        y_target=target,
        config=cfg,
        run_label=run_label,
        total_evaluations=final_cfg.total_evaluations,
        n_initial_points=final_cfg.initial_points,
        kernel_kind=final_cfg.kernel_kind,
        acquisition=final_cfg.acquisition_name,
        seed=seed,
    )

    history = history.copy()
    history["kernel_label"] = final_cfg.kernel_label
    history["acquisition_label"] = final_cfg.acquisition_label
    history["gp_alpha"] = float(final_cfg.gp_alpha)
    history["alpha_label"] = final_cfg.alpha_label

    timing = dict(timing)
    timing.update(
        {
            "method": "BO",
            "loaded_from_existing": False,
            "kernel_label": final_cfg.kernel_label,
            "acquisition_label": final_cfg.acquisition_label,
            "gp_alpha": float(final_cfg.gp_alpha),
            "alpha_label": final_cfg.alpha_label,
        }
    )
    return best, history, timing


def run_or_load_random_search(
    target: float,
    seed: int,
    final_cfg: FinalBOConfig,
    existing_history: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    """Run or load one same-budget Random Search trajectory."""
    cfg = make_bo_config(final_cfg)
    run_label = rs_run_label(target, seed, final_cfg)

    if final_cfg.resume_existing and not existing_history.empty:
        mask = existing_history["run_label"].astype(str) == run_label
        if mask.any():
            hist = existing_history.loc[mask].copy()
            if int(hist["iteration"].max()) >= final_cfg.total_evaluations:
                hist = obo.add_cumulative_best_columns(hist)
                best = obo.select_best_from_history(hist)
                timing = {
                    "method": "RandomSearch",
                    "run_label": run_label,
                    "target": float(target),
                    "seed": int(seed),
                    "loaded_from_existing": True,
                    "total_wall_time_s": float(hist["iteration_time_s"].sum()),
                    "mean_iteration_time_s": float(hist["iteration_time_s"].mean()),
                    "mean_simulation_time_s": float(hist["simulation_time_s"].mean()),
                    "mean_gp_fit_time_s": 0.0,
                    "mean_acquisition_time_s": 0.0,
                }
                print(f"    SKIP existing Random Search | target={target:.2f}, seed={seed}")
                return best, hist, timing

    best, history, timing = obo.run_random_search_single_target(
        y_target=target,
        config=cfg,
        run_label=run_label,
        total_evaluations=final_cfg.total_evaluations,
        seed=seed,
    )

    history = history.copy()
    history["kernel_label"] = "none"
    history["acquisition_label"] = "none"
    history["gp_alpha"] = np.nan
    history["alpha_label"] = "none"

    timing = dict(timing)
    timing.update({"method": "RandomSearch", "loaded_from_existing": False})
    return best, history, timing


# =============================================================================
# SUMMARIES
# =============================================================================


def add_result_metadata(row: Dict[str, object], final_cfg: FinalBOConfig) -> Dict[str, object]:
    row = dict(row)
    if row["method"] == "BO":
        row.update(
            {
                "kernel_label": final_cfg.kernel_label,
                "acquisition_label": final_cfg.acquisition_label,
                "gp_alpha": float(final_cfg.gp_alpha),
                "alpha_label": final_cfg.alpha_label,
                "initial_points": int(final_cfg.initial_points),
                "candidate_pool_global": int(final_cfg.candidate_pool_global),
                "candidate_pool_local": int(final_cfg.candidate_pool_local),
                "pi_xi": float(final_cfg.pi_xi),
            }
        )
    return row


def summarize_by_target(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    return (
        results_df.groupby(["method", "target"], dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            feasibility_rate_percent=("feasible_abs", lambda s: 100.0 * s.astype(bool).mean()),
            mean_peak_y_true=("peak_y_true", "mean"),
            std_peak_y_true=("peak_y_true", "std"),
            mean_target_error_mm=("target_error_mm", "mean"),
            std_target_error_mm=("target_error_mm", "std"),
            min_target_error_mm=("target_error_mm", "min"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_max_abs_xr_true=("max_abs_xr_true", "mean"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_optimization_time_s=("optimization_time_s", "mean"),
            total_true_simulator_calls=("n_true_simulator_evaluations", "sum"),
        )
        .reset_index()
        .sort_values(["method", "target"])
    )


def summarize_overall(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    return (
        results_df.groupby("method", dropna=False)
        .agg(
            n_runs=("target_error_mm", "size"),
            n_targets=("target", "nunique"),
            feasibility_rate_percent=("feasible_abs", lambda s: 100.0 * s.astype(bool).mean()),
            mean_target_error_mm=("target_error_mm", "mean"),
            median_target_error_mm=("target_error_mm", "median"),
            std_target_error_mm=("target_error_mm", "std"),
            max_target_error_mm=("target_error_mm", "max"),
            mean_peak_y_true=("peak_y_true", "mean"),
            mean_max_abs_xr_true=("max_abs_xr_true", "mean"),
            max_max_abs_xr_true=("max_abs_xr_true", "max"),
            mean_residual_margin_mm=("residual_margin_mm", "mean"),
            min_residual_margin_mm=("residual_margin_mm", "min"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_optimization_time_s=("optimization_time_s", "mean"),
            total_optimization_time_s=("optimization_time_s", "sum"),
            total_true_simulator_calls=("n_true_simulator_evaluations", "sum"),
        )
        .reset_index()
        .sort_values("method")
    )


# =============================================================================
# PLOTS
# =============================================================================


def save_final_plots(
    results_df: pd.DataFrame,
    history_df: pd.DataFrame,
    by_target_df: pd.DataFrame,
    final_cfg: FinalBOConfig,
) -> None:
    """Create final BO and BO-vs-Random figures."""
    if results_df.empty:
        return

    # Use mean-over-seeds by target for final tracking/error plots.
    mean_df = by_target_df[by_target_df["method"] == "BO"].sort_values("target").copy()
    if not mean_df.empty:
        plt.figure(figsize=(8.0, 5.2))
        plt.plot(mean_df["target"], mean_df["target"], linestyle="--", linewidth=1.8, label="Ideal")
        plt.errorbar(
            mean_df["target"],
            mean_df["mean_peak_y_true"],
            yerr=mean_df["std_peak_y_true"].fillna(0.0),
            marker="o",
            capsize=4,
            label="BO mean ± std",
        )
        plt.xlabel("Target outreach [m]")
        plt.ylabel("True achieved peak_y [m]")
        plt.title("Final BO target tracking")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "bo_final_target_tracking_mean.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8.0, 5.2))
        plt.bar([f"{t:.2f}" for t in mean_df["target"]], mean_df["mean_target_error_mm"], yerr=mean_df["std_target_error_mm"].fillna(0.0), capsize=4)
        plt.xlabel("Target outreach [m]")
        plt.ylabel("Target error [mm]")
        plt.title("Final BO target error, mean ± std across seeds")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "bo_final_target_error_mean.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8.0, 5.2))
        plt.bar([f"{t:.2f}" for t in mean_df["target"]], mean_df["mean_residual_margin_mm"])
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("Target outreach [m]")
        plt.ylabel("Residual margin [mm]")
        plt.title("Final BO residual margin")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "bo_final_residual_margin_mean.png", dpi=300)
        plt.close()

    # Convergence on representative targets.
    for target in (0.65, 0.75):
        sub = history_df[(history_df["method"] == "BO") & (np.isclose(history_df["target"], target))].copy()
        if sub.empty or "best_feasible_error_so_far_mm" not in sub.columns:
            continue
        plt.figure(figsize=(8.0, 5.2))
        for seed, g in sub.groupby("seed"):
            g = g.sort_values("iteration")
            plt.plot(g["iteration"], g["best_feasible_error_so_far_mm"], linewidth=1.8, label=f"seed {int(seed)}")
        plt.axvline(100, linestyle=":", linewidth=1.0, label="100 calls")
        plt.axvline(final_cfg.total_evaluations, linestyle="--", linewidth=1.0, label=f"{final_cfg.total_evaluations} calls")
        plt.xlabel("True simulator calls")
        plt.ylabel("Best feasible target error so far [mm]")
        plt.title(f"Final BO convergence, target={target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"bo_final_convergence_target{int(round(target * 1000)):04d}.png", dpi=300)
        plt.close()

    # BO vs Random Search by target if Random Search exists.
    if "RandomSearch" in set(results_df["method"].astype(str)):
        comp = by_target_df.pivot(index="target", columns="method", values="mean_target_error_mm").reset_index()
        if {"BO", "RandomSearch"}.issubset(comp.columns):
            x = np.arange(len(comp))
            width = 0.38
            plt.figure(figsize=(8.5, 5.2))
            plt.bar(x - width / 2, comp["BO"], width, label="BO")
            plt.bar(x + width / 2, comp["RandomSearch"], width, label="Random Search")
            plt.xticks(x, [f"{t:.2f}" for t in comp["target"]])
            plt.xlabel("Target outreach [m]")
            plt.ylabel("Mean target error [mm]")
            plt.title(f"BO vs Random Search at {final_cfg.total_evaluations} simulator calls")
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "bo_final_vs_random_error.png", dpi=300)
            plt.close()

        # Convergence BO vs Random on representative target 0.75.
        for target in (0.65, 0.75):
            sub = history_df[np.isclose(history_df["target"], target)].copy()
            if sub.empty or "best_feasible_error_so_far_mm" not in sub.columns:
                continue
            plt.figure(figsize=(8.0, 5.2))
            for (method, seed), g in sub.groupby(["method", "seed"]):
                g = g.sort_values("iteration")
                plt.plot(g["iteration"], g["best_feasible_error_so_far_mm"], linewidth=1.3, label=f"{method}, seed {int(seed)}")
            plt.xlabel("True simulator calls")
            plt.ylabel("Best feasible target error so far [mm]")
            plt.title(f"BO vs Random Search convergence, target={target:.2f} m")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"bo_final_vs_random_convergence_target{int(round(target * 1000)):04d}.png", dpi=300)
            plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_dirs()
    final_cfg = FinalBOConfig()
    cfg = make_bo_config(final_cfg)

    bo_history_path = RESULTS_DIR / "bo_final_history.csv"
    bo_results_path = RESULTS_DIR / "bo_final_results_by_seed.csv"
    timing_path = RESULTS_DIR / "bo_final_timing.csv"
    by_target_path = RESULTS_DIR / "bo_final_summary_by_target.csv"
    overall_path = RESULTS_DIR / "bo_final_summary_overall.csv"

    rs_history_path = RESULTS_DIR / "random_search_final_history.csv"
    all_history_path = RESULTS_DIR / "bo_final_all_history.csv"
    all_results_path = RESULTS_DIR / "bo_final_all_results_by_seed.csv"

    print("=" * 80)
    print("FINAL BO RUN")
    print("=" * 80)
    print(f"Targets:                 {final_cfg.targets}")
    print(f"Seeds:                   {final_cfg.seeds}")
    print(f"Acquisition:             {final_cfg.acquisition_label}")
    print(f"Kernel:                  {final_cfg.kernel_label}")
    print(f"GP alpha:                {final_cfg.alpha_label}")
    print(f"Budget:                  {final_cfg.total_evaluations} true simulator calls/target/seed")
    print(f"Initial LHS points:      {final_cfg.initial_points}")
    print(f"Random Search baseline:  {final_cfg.run_random_search_baseline}")
    print(f"Output results folder:   {RESULTS_DIR}")
    print(f"Output figures folder:   {FIGURES_DIR}")
    print("=" * 80)

    existing_bo_history = pd.DataFrame()
    if final_cfg.resume_existing and bo_history_path.exists():
        existing_bo_history = pd.read_csv(bo_history_path)
        print(f"Resuming BO from existing history with {len(existing_bo_history)} rows.")

    bo_results: List[Dict[str, object]] = []
    bo_histories: List[pd.DataFrame] = []
    timing_rows: List[Dict[str, object]] = []

    total_bo_runs = len(final_cfg.targets) * len(final_cfg.seeds)
    run_counter = 0
    for target in final_cfg.targets:
        for seed in final_cfg.seeds:
            run_counter += 1
            print(f"\n[BO {run_counter}/{total_bo_runs}] target={target:.2f} | seed={seed}")
            best, history, timing = run_or_load_bo(target, seed, final_cfg, existing_bo_history)
            result = obo.build_result_row(best, cfg)
            result["optimization_time_s"] = float(timing["total_wall_time_s"])
            result = add_result_metadata(result, final_cfg)
            bo_results.append(result)
            bo_histories.append(history)
            timing_rows.append(timing)
            print(
                f"    best_y={float(best['peak_y_true']):.4f} m | "
                f"err={float(best['target_error_mm']):.2f} mm | "
                f"xr={float(best['max_abs_xr_true']):.4f} m | "
                f"margin={float(best['residual_margin_mm']):.2f} mm | "
                f"feasible={bool(best['feasible_abs'])} | "
                f"time={float(timing['total_wall_time_s']):.1f} s"
            )

    bo_history = pd.concat(bo_histories, ignore_index=True)
    bo_history = obo.add_cumulative_best_columns(bo_history)
    bo_history.to_csv(bo_history_path, index=False)

    bo_results_df = pd.DataFrame(bo_results)
    bo_results_df.to_csv(bo_results_path, index=False)

    all_histories = [bo_history]
    all_results = [bo_results_df]

    # Optional same-budget Random Search baseline.
    if final_cfg.run_random_search_baseline:
        existing_rs_history = pd.DataFrame()
        if final_cfg.resume_existing and rs_history_path.exists():
            existing_rs_history = pd.read_csv(rs_history_path)
            print(f"\nResuming Random Search from existing history with {len(existing_rs_history)} rows.")

        rs_results: List[Dict[str, object]] = []
        rs_histories: List[pd.DataFrame] = []
        total_rs_runs = len(final_cfg.targets) * len(final_cfg.seeds)
        run_counter = 0
        for target in final_cfg.targets:
            for seed in final_cfg.seeds:
                run_counter += 1
                print(f"\n[Random Search {run_counter}/{total_rs_runs}] target={target:.2f} | seed={seed}")
                best, history, timing = run_or_load_random_search(target, seed, final_cfg, existing_rs_history)
                result = obo.build_result_row(best, cfg)
                result["optimization_time_s"] = float(timing["total_wall_time_s"])
                rs_results.append(result)
                rs_histories.append(history)
                timing_rows.append(timing)
                print(
                    f"    best_y={float(best['peak_y_true']):.4f} m | "
                    f"err={float(best['target_error_mm']):.2f} mm | "
                    f"xr={float(best['max_abs_xr_true']):.4f} m | "
                    f"margin={float(best['residual_margin_mm']):.2f} mm | "
                    f"feasible={bool(best['feasible_abs'])} | "
                    f"time={float(timing['total_wall_time_s']):.1f} s"
                )

        rs_history = pd.concat(rs_histories, ignore_index=True)
        rs_history = obo.add_cumulative_best_columns(rs_history)
        rs_history.to_csv(rs_history_path, index=False)

        rs_results_df = pd.DataFrame(rs_results)
        rs_results_df.to_csv(RESULTS_DIR / "random_search_final_results_by_seed.csv", index=False)

        all_histories.append(rs_history)
        all_results.append(rs_results_df)

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(timing_path, index=False)

    all_history = pd.concat(all_histories, ignore_index=True)
    all_history = obo.add_cumulative_best_columns(all_history)
    all_history.to_csv(all_history_path, index=False)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(all_results_path, index=False)

    by_target_df = summarize_by_target(all_results_df)
    overall_df = summarize_overall(all_results_df)
    by_target_df.to_csv(by_target_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY BY TARGET")
    print("=" * 80)
    print(by_target_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FINAL OVERALL SUMMARY")
    print("=" * 80)
    print(overall_df.to_string(index=False))

    print("\nSaved files:")
    for path in [bo_history_path, bo_results_path, timing_path, by_target_path, overall_path, all_history_path, all_results_path]:
        print(f"  {path}")

    print("\nGenerating final plots...")
    save_final_plots(all_results_df, all_history, by_target_df, final_cfg)
    print("Done.")


if __name__ == "__main__":
    main()
