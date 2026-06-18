"""
bo_report_figures.py
====================

Create a complete, report-ready and presentation-ready figure set for the
Bayesian Optimization part of the project.

This script does NOT run any new optimization. It only reads the CSV files
created by:
    - bo_benchmark.py                  (Step A: acquisition benchmark)
    - bo_kernel_benchmark.py           (Step B: kernel benchmark)
    - bo_alpha_benchmark.py            (Step C: alpha benchmark)
    - bo_budget_sweep.py               (Step D: budget sweep)
    - bo_final_run.py                  (Step E: final BO + Random Search)

It saves consolidated figures to:
    figures/optimization_bo_report/

Run from the project root:
    python src/bo_report_figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional: used only for time-response plots.
try:
    import optimization_bo as obo
except Exception:  # pragma: no cover
    obo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_report"
RESULTS_DIR = PROJECT_ROOT / "results"

# Source result folders.
ACQ_DIR = RESULTS_DIR / "optimization_bo_benchmark"
KERNEL_DIR = RESULTS_DIR / "optimization_bo_kernel_benchmark"
ALPHA_DIR = RESULTS_DIR / "optimization_bo_alpha_benchmark"
BUDGET_DIR = RESULTS_DIR / "optimization_bo_budget_sweep"
FINAL_DIR = RESULTS_DIR / "optimization_bo_final"


TARGETS = [0.55, 0.60, 0.65, 0.70, 0.75]
ROBOT_LIMIT = 0.500


# =============================================================================
# SMALL UTILITIES
# =============================================================================


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    msg = f"Missing file: {path}"
    if required:
        raise FileNotFoundError(msg)
    print(f"[skip] {msg}")
    return pd.DataFrame()


def savefig(name: str) -> None:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def target_label(t: float) -> str:
    return f"{float(t):.2f}"


def add_value_labels(ax, fmt: str = "{:.1f}", dy: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * dy
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.text(
                p.get_x() + p.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )


def complete_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Make method labels consistent for BO and Random Search."""
    if df.empty:
        return df
    df = df.copy()
    if "method" not in df.columns:
        df["method"] = "BO"
    df["method"] = df["method"].astype(str)
    # Some result rows may keep default method if build_result_row had incomplete history metadata.
    if "run_label" in df.columns:
        rs_mask = df["run_label"].astype(str).str.contains("RS|Random", case=False, regex=True)
        df.loc[rs_mask, "method"] = "RandomSearch"
        bo_mask = df["run_label"].astype(str).str.contains("BO", case=False, regex=True)
        df.loc[bo_mask & ~rs_mask, "method"] = "BO"
    return df


# =============================================================================
# GENERIC HEATMAPS AND BAR PLOTS FOR BENCHMARKS
# =============================================================================


def plot_summary_heatmap(
    summary_df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    fmt: str = ".1f",
) -> None:
    if summary_df.empty:
        return
    pivot = summary_df.pivot(index=row_col, columns=col_col, values=value_col)
    # Sort rows numerically if possible.
    try:
        pivot = pivot.sort_index(key=lambda s: s.astype(float))
    except Exception:
        pivot = pivot.sort_index()

    fig_w = max(7.5, 1.25 * len(pivot.columns) + 2.5)
    fig_h = max(4.8, 0.75 * len(pivot.index) + 2.2)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label=value_col.replace("_", " "))
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(np.arange(len(pivot.index)), [target_label(v) if isinstance(v, (float, int, np.floating)) else str(v) for v in pivot.index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)
    savefig(filename)


def plot_overall_bar(
    overall_df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    filename: str,
    sort_ascending: bool = True,
) -> None:
    if overall_df.empty:
        return
    df = overall_df.sort_values(value_col, ascending=sort_ascending).copy()
    plt.figure(figsize=(8.5, 5.0))
    plt.bar(df[label_col].astype(str), df[value_col])
    plt.xlabel(label_col.replace("_", " "))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    add_value_labels(plt.gca(), "{:.1f}")
    savefig(filename)


# =============================================================================
# STEP A/B/C/D FIGURES
# =============================================================================


def make_acquisition_figures() -> None:
    summary = read_csv(ACQ_DIR / "bo_acquisition_benchmark_summary.csv")
    overall = read_csv(ACQ_DIR / "bo_acquisition_benchmark_overall_summary.csv")
    if summary.empty:
        return

    plot_summary_heatmap(
        summary,
        row_col="target",
        col_col="acquisition_label",
        value_col="mean_target_error_mm",
        title="BO acquisition benchmark: mean target error",
        xlabel="Acquisition function",
        ylabel="Target outreach [m]",
        filename="01_acquisition_mean_error_heatmap.png",
    )
    plot_summary_heatmap(
        summary,
        row_col="target",
        col_col="acquisition_label",
        value_col="mean_residual_margin_mm",
        title="BO acquisition benchmark: mean residual margin",
        xlabel="Acquisition function",
        ylabel="Target outreach [m]",
        filename="02_acquisition_mean_margin_heatmap.png",
    )
    if not overall.empty:
        plot_overall_bar(
            overall,
            label_col="acquisition_label",
            value_col="mean_target_error_mm",
            title="Overall acquisition benchmark",
            ylabel="Mean target error [mm]",
            filename="03_acquisition_overall_mean_error.png",
        )
        plot_overall_bar(
            overall,
            label_col="acquisition_label",
            value_col="mean_rank_by_error",
            title="Overall acquisition benchmark: average rank",
            ylabel="Mean rank by error",
            filename="04_acquisition_overall_rank.png",
        )


def make_kernel_figures() -> None:
    summary = read_csv(KERNEL_DIR / "bo_kernel_benchmark_summary.csv")
    overall = read_csv(KERNEL_DIR / "bo_kernel_benchmark_overall_summary.csv")
    if summary.empty:
        return

    plot_summary_heatmap(
        summary,
        row_col="target",
        col_col="kernel_label",
        value_col="mean_target_error_mm",
        title="BO kernel benchmark: mean target error",
        xlabel="Internal GP kernel",
        ylabel="Target outreach [m]",
        filename="05_kernel_mean_error_heatmap.png",
    )
    plot_summary_heatmap(
        summary,
        row_col="target",
        col_col="kernel_label",
        value_col="mean_residual_margin_mm",
        title="BO kernel benchmark: mean residual margin",
        xlabel="Internal GP kernel",
        ylabel="Target outreach [m]",
        filename="06_kernel_mean_margin_heatmap.png",
    )
    if not overall.empty:
        plot_overall_bar(
            overall,
            label_col="kernel_label",
            value_col="mean_target_error_mm",
            title="Overall kernel benchmark with PI acquisition",
            ylabel="Mean target error [mm]",
            filename="07_kernel_overall_mean_error.png",
        )
        plot_overall_bar(
            overall,
            label_col="kernel_label",
            value_col="max_target_error_mm",
            title="Overall kernel benchmark: worst-case target error",
            ylabel="Max target error [mm]",
            filename="08_kernel_overall_worst_error.png",
        )


def make_alpha_figures() -> None:
    summary = read_csv(ALPHA_DIR / "bo_alpha_benchmark_summary.csv")
    overall = read_csv(ALPHA_DIR / "bo_alpha_benchmark_overall_summary.csv")
    if summary.empty and overall.empty:
        return

    if not summary.empty:
        # Ensure numeric alpha for log x-axis.
        summary = summary.copy()
        summary["gp_alpha"] = summary["gp_alpha"].astype(float)
        for target in sorted(summary["target"].unique()):
            sub = summary[np.isclose(summary["target"], target)].sort_values("gp_alpha")
            plt.figure(figsize=(8.0, 5.0))
            plt.plot(sub["gp_alpha"], sub["mean_target_error_mm"], marker="o")
            plt.xscale("log")
            plt.xlabel("GP alpha")
            plt.ylabel("Mean target error [mm]")
            plt.title(f"BO alpha benchmark, target={target:.2f} m")
            plt.grid(True, alpha=0.3)
            savefig(f"09_alpha_error_target{int(round(float(target)*1000)):04d}.png")

            plt.figure(figsize=(8.0, 5.0))
            plt.plot(sub["gp_alpha"], sub["min_residual_margin_mm"], marker="o")
            plt.xscale("log")
            plt.axhline(0.0, linestyle="--", linewidth=1.0)
            plt.xlabel("GP alpha")
            plt.ylabel("Minimum residual margin [mm]")
            plt.title(f"BO alpha benchmark: safety margin, target={target:.2f} m")
            plt.grid(True, alpha=0.3)
            savefig(f"10_alpha_margin_target{int(round(float(target)*1000)):04d}.png")

    if not overall.empty:
        overall = overall.copy()
        overall["gp_alpha"] = overall["gp_alpha"].astype(float)
        overall = overall.sort_values("gp_alpha")
        plt.figure(figsize=(8.0, 5.0))
        plt.plot(overall["gp_alpha"], overall["mean_target_error_mm"], marker="o", label="Mean error")
        plt.plot(overall["gp_alpha"], overall["max_target_error_mm"], marker="s", label="Max error")
        plt.xscale("log")
        plt.xlabel("GP alpha")
        plt.ylabel("Target error [mm]")
        plt.title("Overall alpha benchmark")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig("11_alpha_overall_error.png")


def make_budget_figures() -> None:
    summary = read_csv(BUDGET_DIR / "bo_budget_sweep_summary.csv")
    overall = read_csv(BUDGET_DIR / "bo_budget_sweep_overall_summary.csv")
    if summary.empty:
        return

    plt.figure(figsize=(8.0, 5.0))
    for target, sub in summary.groupby("target"):
        sub = sub.sort_values("budget")
        plt.plot(sub["budget"], sub["mean_target_error_mm"], marker="o", label=f"target {float(target):.2f} m")
    plt.xlabel("True simulator calls")
    plt.ylabel("Mean target error [mm]")
    plt.title("BO budget sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("12_budget_sweep_mean_error.png")

    plt.figure(figsize=(8.0, 5.0))
    for target, sub in summary.groupby("target"):
        sub = sub.sort_values("budget")
        plt.plot(sub["budget"], sub["min_residual_margin_mm"], marker="o", label=f"target {float(target):.2f} m")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("True simulator calls")
    plt.ylabel("Minimum residual margin [mm]")
    plt.title("BO budget sweep: safety margin")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("13_budget_sweep_min_margin.png")

    if not overall.empty:
        overall = overall.sort_values("budget")
        plt.figure(figsize=(8.0, 5.0))
        plt.plot(overall["budget"], overall["mean_target_error_mm"], marker="o", label="Mean error")
        plt.plot(overall["budget"], overall["max_target_error_mm"], marker="s", label="Worst-case error")
        plt.xlabel("True simulator calls")
        plt.ylabel("Target error [mm]")
        plt.title("Overall budget sweep")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig("14_budget_sweep_overall_error.png")


# =============================================================================
# FINAL BO / RANDOM SEARCH FIGURES
# =============================================================================


def make_final_figures() -> None:
    results = read_csv(FINAL_DIR / "bo_final_all_results_by_seed.csv")
    history = read_csv(FINAL_DIR / "bo_final_all_history.csv")
    by_target = read_csv(FINAL_DIR / "bo_final_summary_by_target.csv")
    overall = read_csv(FINAL_DIR / "bo_final_summary_overall.csv")

    if results.empty:
        print("[skip] Final BO results are missing. Run bo_final_run.py first.")
        return

    results = complete_method_labels(results)
    if not history.empty:
        history = complete_method_labels(history)
    if not by_target.empty:
        by_target = complete_method_labels(by_target)

    # Final BO target tracking.
    bo_by_target = by_target[by_target["method"] == "BO"].sort_values("target").copy()
    if not bo_by_target.empty:
        plt.figure(figsize=(8.2, 5.2))
        plt.plot(bo_by_target["target"], bo_by_target["target"], linestyle="--", linewidth=1.6, label="Ideal")
        plt.errorbar(
            bo_by_target["target"],
            bo_by_target["mean_peak_y_true"],
            yerr=bo_by_target.get("std_peak_y_true", pd.Series(0, index=bo_by_target.index)).fillna(0.0),
            marker="o",
            capsize=4,
            linewidth=1.8,
            label="Final BO mean ± std",
        )
        plt.xlabel("Target outreach [m]")
        plt.ylabel("True achieved peak_y [m]")
        plt.title("Final BO target tracking")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig("15_final_bo_target_tracking.png")

        # Target error.
        plt.figure(figsize=(8.2, 5.2))
        xlabels = [target_label(t) for t in bo_by_target["target"]]
        plt.bar(xlabels, bo_by_target["mean_target_error_mm"], yerr=bo_by_target.get("std_target_error_mm", pd.Series(0, index=bo_by_target.index)).fillna(0.0), capsize=4)
        plt.xlabel("Target outreach [m]")
        plt.ylabel("Target error [mm]")
        plt.title("Final BO target error")
        plt.grid(True, axis="y", alpha=0.3)
        savefig("16_final_bo_target_error.png")

        # Constraint validation: show true max_abs_xr and the hard limit.
        plt.figure(figsize=(8.2, 5.2))
        plt.bar(xlabels, bo_by_target["mean_max_abs_xr_true"])
        plt.axhline(ROBOT_LIMIT, linestyle="--", linewidth=1.5, label="Hard limit 0.500 m")
        plt.xlabel("Target outreach [m]")
        plt.ylabel("True max |x_r| [m]")
        plt.title("Final BO constraint validation")
        plt.ylim(0.0, max(0.52, float(bo_by_target["mean_max_abs_xr_true"].max()) * 1.05))
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        savefig("17_final_bo_constraint_validation.png")

        plt.figure(figsize=(8.2, 5.2))
        plt.bar(xlabels, bo_by_target["mean_residual_margin_mm"])
        if "min_residual_margin_mm" in bo_by_target.columns:
            plt.scatter(xlabels, bo_by_target["min_residual_margin_mm"], marker="x", label="Minimum across seeds")
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("Target outreach [m]")
        plt.ylabel("Residual margin [mm]")
        plt.title("Final BO residual safety margin")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        savefig("18_final_bo_residual_margin.png")

    # BO vs Random Search error.
    if not by_target.empty and set(by_target["method"]).issuperset({"BO", "RandomSearch"}):
        comp = by_target.pivot(index="target", columns="method", values="mean_target_error_mm").sort_index()
        x = np.arange(len(comp.index))
        width = 0.38
        plt.figure(figsize=(8.6, 5.2))
        plt.bar(x - width / 2, comp["BO"], width, label="BO")
        plt.bar(x + width / 2, comp["RandomSearch"], width, label="Random Search")
        plt.xticks(x, [target_label(t) for t in comp.index])
        plt.xlabel("Target outreach [m]")
        plt.ylabel("Mean target error [mm]")
        plt.title("Final BO vs Random Search")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        savefig("19_final_bo_vs_random_error.png")

    # Error-margin trade-off across all seeds and methods.
    if {"target_error_mm", "residual_margin_mm"}.issubset(results.columns):
        plt.figure(figsize=(8.0, 5.4))
        for method, sub in results.groupby("method"):
            plt.scatter(sub["target_error_mm"], sub["residual_margin_mm"], label=method, s=55, alpha=0.85)
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("Target error [mm]")
        plt.ylabel("Residual margin [mm]")
        plt.title("Final BO safety/accuracy trade-off")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig("20_final_error_margin_tradeoff.png")

    # Optimized parameters for BO only, mean ± std across seeds.
    param_cols = [c for c in ["Kr", "hr", "f0", "f1", "A", "x_r_start"] if c in results.columns]
    bo_results = results[results["method"] == "BO"].copy()
    if param_cols and not bo_results.empty:
        for col in param_cols:
            p = bo_results.groupby("target")[col].agg(["mean", "std"]).reset_index().sort_values("target")
            plt.figure(figsize=(8.0, 5.0))
            plt.errorbar([target_label(t) for t in p["target"]], p["mean"], yerr=p["std"].fillna(0.0), marker="o", capsize=4)
            plt.xlabel("Target outreach [m]")
            plt.ylabel(col)
            plt.title(f"Final BO optimized parameter: {col}")
            plt.grid(True, alpha=0.3)
            savefig(f"21_final_parameter_{col}.png")

    # Convergence curves on key targets.
    if not history.empty and "best_feasible_error_so_far_mm" in history.columns:
        for target in [0.65, 0.75]:
            sub = history[np.isclose(history["target"], target)].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(8.4, 5.2))
            for (method, seed), g in sub.groupby(["method", "seed"]):
                g = g.sort_values("iteration")
                plt.plot(g["iteration"], g["best_feasible_error_so_far_mm"], linewidth=1.4, label=f"{method}, seed {int(seed)}")
            plt.xlabel("True simulator calls")
            plt.ylabel("Best feasible target error so far [mm]")
            plt.title(f"Convergence comparison, target={target:.2f} m")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            savefig(f"22_final_convergence_target{int(round(target*1000)):04d}.png")

    # Runtime / online calls.
    if not overall.empty:
        overall = complete_method_labels(overall)
        if {"method", "total_true_simulator_calls"}.issubset(overall.columns):
            plt.figure(figsize=(7.2, 5.0))
            plt.bar(overall["method"], overall["total_true_simulator_calls"])
            plt.ylabel("Total true simulator calls")
            plt.title("Final online evaluation cost")
            plt.grid(True, axis="y", alpha=0.3)
            savefig("23_final_online_simulator_calls.png")
        if {"method", "total_optimization_time_s"}.issubset(overall.columns):
            plt.figure(figsize=(7.2, 5.0))
            plt.bar(overall["method"], overall["total_optimization_time_s"])
            plt.ylabel("Total optimization time [s]")
            plt.title("Final optimization runtime")
            plt.grid(True, axis="y", alpha=0.3)
            savefig("24_final_runtime.png")

    make_time_response_figures(bo_results)


# =============================================================================
# TIME RESPONSES FOR REPORT/PRESENTATION
# =============================================================================


def make_time_response_figures(bo_results: pd.DataFrame) -> None:
    """Create true dynamic time-response plots for best final BO solution per target.

    For each target, use the seed with the lowest final target_error_mm.
    Also create one compact presentation plot for target 0.65 and 0.75.
    """
    if obo is None:
        print("[skip] Cannot import optimization_bo; time-response plots not created.")
        return
    if bo_results.empty:
        return

    cfg = obo.BOConfig()
    cfg.robot_limit_true = ROBOT_LIMIT

    # Best seed per target.
    best_rows = (
        bo_results.sort_values(["target", "target_error_mm"])
        .groupby("target", as_index=False)
        .first()
        .sort_values("target")
    )

    response_cache = {}
    for _, row in best_rows.iterrows():
        target = float(row["target"])
        try:
            params = obo.row_to_params(row, cfg)
            t, y, x_b, x_r, metrics = obo.simulate_candidate_with_solution(params, target, cfg)
        except Exception as exc:
            print(f"[skip] time response target={target:.2f}: {exc}")
            continue
        response_cache[target] = (t, y, x_b, x_r, metrics, row)

        fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True)
        axes[0].plot(t, y, linewidth=2.0, label="Total outreach y(t)")
        axes[0].axhline(target, linestyle="--", linewidth=1.5, label="Target")
        axes[0].set_ylabel("y [m]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t, x_r, linewidth=1.8, label="Robot displacement x_r(t)")
        axes[1].axhline(ROBOT_LIMIT, linestyle="--", linewidth=1.2, label="±0.500 m limit")
        axes[1].axhline(-ROBOT_LIMIT, linestyle="--", linewidth=1.2)
        axes[1].set_ylabel("x_r [m]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(t, x_b, linewidth=1.8, label="Base displacement x_b(t)")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("x_b [m]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        fig.suptitle(
            f"Final BO true response | target={target:.2f} m | "
            f"error={float(row['target_error_mm']):.2f} mm | "
            f"max |x_r|={float(row['max_abs_xr_true']):.4f} m | "
            f"margin={float(row['residual_margin_mm']):.2f} mm",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        savefig(f"25_time_response_target{int(round(target*1000)):04d}.png")

    # Compact presentation figure for representative targets.
    rep_targets = [0.65, 0.75]
    available = [t for t in rep_targets if any(np.isclose(list(response_cache.keys()), t))]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(11.5, 4.2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, target in zip(axes, available):
        # Retrieve key robustly for float precision.
        key = next(k for k in response_cache if np.isclose(k, target))
        t, y, x_b, x_r, metrics, row = response_cache[key]
        ax.plot(t, y, linewidth=2.0, label="y(t)")
        ax.plot(t, x_r, linewidth=1.5, label="x_r(t)")
        ax.plot(t, x_b, linewidth=1.5, label="x_b(t)")
        ax.axhline(target, linestyle="--", linewidth=1.4, label="Target")
        ax.axhline(ROBOT_LIMIT, linestyle=":", linewidth=1.2, label="0.500 m limit")
        ax.set_ylabel("Displacement [m]")
        ax.set_title(
            f"target={target:.2f} m, achieved peak={float(row['peak_y_true']):.3f} m, "
            f"error={float(row['target_error_mm']):.1f} mm"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Representative final BO time responses", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig("26_time_response_representative_targets.png")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_dirs()
    print("=" * 80)
    print("COMPLETE BO REPORT FIGURE GENERATION")
    print("=" * 80)
    print(f"Output folder: {FIGURES_DIR}")

    make_acquisition_figures()
    make_kernel_figures()
    make_alpha_figures()
    make_budget_figures()
    make_final_figures()

    print("=" * 80)
    print("Done. Use figures/optimization_bo_report for report/presentation selection.")
    print("=" * 80)


if __name__ == "__main__":
    main()
