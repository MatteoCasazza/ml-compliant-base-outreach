"""
bo_report_figures.py
====================

Create report-ready and presentation-ready figures for the Bayesian
Optimization part of the project.

This script does not run any new optimization. It only reads CSV files generated
by the BO benchmark and final-run scripts:

    - bo_benchmark.py          Step A: acquisition benchmark
    - bo_kernel_benchmark.py   Step B: internal GP kernel benchmark
    - bo_alpha_benchmark.py    Step C: internal GP alpha benchmark
    - bo_budget_sweep.py       Step D: online budget sweep
    - bo_final_run.py          Step E: final BO + Random Search baseline

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import optimization_bo as obo

    OBO_AVAILABLE = True
except Exception:
    obo = None
    OBO_AVAILABLE = False


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_report"

ACQUISITION_DIR = RESULTS_DIR / "optimization_bo_benchmark"
KERNEL_DIR = RESULTS_DIR / "optimization_bo_kernel_benchmark"
ALPHA_DIR = RESULTS_DIR / "optimization_bo_alpha_benchmark"
BUDGET_DIR = RESULTS_DIR / "optimization_bo_budget_sweep"
FINAL_DIR = RESULTS_DIR / "optimization_bo_final"


# =============================================================================
# SETTINGS
# =============================================================================

ROBOT_LIMIT = 0.500

OFFICIAL_TARGETS = [0.55, 0.60, 0.65, 0.70, 0.75]
REPRESENTATIVE_TARGETS = [0.65, 0.75]

DEFAULT_ONLY = "all"

VALID_SECTIONS = {
    "all",
    "acquisition",
    "kernel",
    "alpha",
    "budget",
    "final",
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create output directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    """Read a CSV file, optionally requiring it to exist."""
    if path.exists():
        return pd.read_csv(path)

    message = f"Missing file: {path}"

    if required:
        raise FileNotFoundError(message)

    print(f"[skip] {message}")

    return pd.DataFrame()


def savefig(filename: str) -> None:
    """Save current matplotlib figure to the report figure directory."""
    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def target_label(target: float) -> str:
    """Format target value for axis labels."""
    return f"{float(target):.2f}"


def safe_target_tag(target: float) -> str:
    """Return target tag in millimeters."""
    return f"{int(round(float(target) * 1000.0)):04d}"


def add_value_labels(
    ax: plt.Axes,
    fmt: str = "{:.1f}",
    dy: float = 0.015,
) -> None:
    """Add numeric labels above bar patches."""
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * dy

    for patch in ax.patches:
        height = patch.get_height()

        if not np.isfinite(height):
            continue

        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def complete_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Make method labels consistent for BO and Random Search tables."""
    if df.empty:
        return df

    df = df.copy()

    if "method" not in df.columns:
        df["method"] = "BO"

    df["method"] = df["method"].astype(str)

    if "run_label" in df.columns:
        run_label = df["run_label"].astype(str)

        random_mask = run_label.str.contains(
            "RS|RandomSearch|Random",
            case=False,
            regex=True,
        )
        bo_mask = run_label.str.contains("BO", case=False, regex=True)

        df.loc[random_mask, "method"] = "RandomSearch"
        df.loc[bo_mask & ~random_mask, "method"] = "BO"

    return df


def sort_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by target when available."""
    if df.empty or "target" not in df.columns:
        return df

    return df.sort_values("target").reset_index(drop=True)


def parse_sections(raw: str) -> set[str]:
    """Parse comma-separated section list."""
    sections = {item.strip().lower() for item in raw.split(",") if item.strip()}

    if not sections:
        raise ValueError("At least one section must be provided.")

    unknown = sections - VALID_SECTIONS

    if unknown:
        raise ValueError(
            f"Unknown section(s): {sorted(unknown)}. "
            f"Valid choices: {sorted(VALID_SECTIONS)}"
        )

    if "all" in sections:
        return {"acquisition", "kernel", "alpha", "budget", "final"}

    return sections


# =============================================================================
# GENERIC PLOTS
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
    """Create a heatmap from a benchmark summary table."""
    if summary_df.empty:
        return

    required_cols = {row_col, col_col, value_col}

    if not required_cols.issubset(summary_df.columns):
        print(f"[skip] {filename}: missing columns {sorted(required_cols - set(summary_df.columns))}")
        return

    pivot = summary_df.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="mean",
    )

    try:
        pivot = pivot.sort_index(key=lambda index: index.astype(float))
    except Exception:
        pivot = pivot.sort_index()

    fig_width = max(7.5, 1.30 * len(pivot.columns) + 2.5)
    fig_height = max(4.8, 0.75 * len(pivot.index) + 2.2)

    plt.figure(figsize=(fig_width, fig_height))

    image = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(image, label=value_col.replace("_", " "))

    plt.xticks(
        np.arange(len(pivot.columns)),
        [str(col) for col in pivot.columns],
        rotation=25,
        ha="right",
    )

    y_labels = []

    for value in pivot.index:
        if isinstance(value, (float, int, np.floating, np.integer)):
            y_labels.append(target_label(float(value)))
        else:
            y_labels.append(str(value))

    plt.yticks(np.arange(len(pivot.index)), y_labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]

            if np.isfinite(value):
                plt.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

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
    """Create a simple bar plot from an overall benchmark table."""
    if overall_df.empty:
        return

    required_cols = {label_col, value_col}

    if not required_cols.issubset(overall_df.columns):
        print(f"[skip] {filename}: missing columns {sorted(required_cols - set(overall_df.columns))}")
        return

    df = overall_df.sort_values(value_col, ascending=sort_ascending).copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.bar(df[label_col].astype(str), df[value_col], edgecolor="black")

    ax.set_xlabel(label_col.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    add_value_labels(ax, "{:.1f}")

    savefig(filename)


# =============================================================================
# STEP A: ACQUISITION FIGURES
# =============================================================================

def make_acquisition_figures() -> None:
    """Create consolidated acquisition benchmark figures."""
    summary = read_csv(ACQUISITION_DIR / "bo_acquisition_benchmark_summary.csv")
    overall = read_csv(ACQUISITION_DIR / "bo_acquisition_benchmark_overall_summary.csv")

    if summary.empty and overall.empty:
        print("[skip] Acquisition benchmark figures: no input tables found.")
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

    plot_overall_bar(
        overall,
        label_col="acquisition_label",
        value_col="mean_target_error_mm",
        title="Overall BO acquisition benchmark",
        ylabel="Mean target error [mm]",
        filename="03_acquisition_overall_mean_error.png",
    )

    plot_overall_bar(
        overall,
        label_col="acquisition_label",
        value_col="mean_rank_by_error",
        title="Overall BO acquisition benchmark: mean rank",
        ylabel="Mean rank by error",
        filename="04_acquisition_overall_rank.png",
    )


# =============================================================================
# STEP B: KERNEL FIGURES
# =============================================================================

def make_kernel_figures() -> None:
    """Create consolidated internal-kernel benchmark figures."""
    summary = read_csv(KERNEL_DIR / "bo_kernel_benchmark_summary.csv")
    overall = read_csv(KERNEL_DIR / "bo_kernel_benchmark_overall_summary.csv")

    if summary.empty and overall.empty:
        print("[skip] Kernel benchmark figures: no input tables found.")
        return

    plot_summary_heatmap(
        summary,
        row_col="target",
        col_col="kernel_label",
        value_col="mean_target_error_mm",
        title="BO internal-kernel benchmark: mean target error",
        xlabel="Internal GP kernel",
        ylabel="Target outreach [m]",
        filename="05_kernel_mean_error_heatmap.png",
    )

    plot_summary_heatmap(
        summary,
        row_col="target",
        col_col="kernel_label",
        value_col="mean_residual_margin_mm",
        title="BO internal-kernel benchmark: mean residual margin",
        xlabel="Internal GP kernel",
        ylabel="Target outreach [m]",
        filename="06_kernel_mean_margin_heatmap.png",
    )

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


# =============================================================================
# STEP C: ALPHA FIGURES
# =============================================================================

def make_alpha_figures() -> None:
    """Create consolidated alpha benchmark figures."""
    summary = read_csv(ALPHA_DIR / "bo_alpha_benchmark_summary.csv")
    overall = read_csv(ALPHA_DIR / "bo_alpha_benchmark_overall_summary.csv")

    if summary.empty and overall.empty:
        print("[skip] Alpha benchmark figures: no input tables found.")
        return

    if not summary.empty:
        required_cols = {
            "target",
            "gp_alpha",
            "mean_target_error_mm",
            "min_residual_margin_mm",
        }

        if required_cols.issubset(summary.columns):
            summary = summary.copy()
            summary["gp_alpha"] = summary["gp_alpha"].astype(float)

            for target in sorted(summary["target"].unique()):
                sub = summary[np.isclose(summary["target"], target)].sort_values("gp_alpha")

                plt.figure(figsize=(8.0, 5.2))
                plt.plot(
                    sub["gp_alpha"],
                    sub["mean_target_error_mm"],
                    marker="o",
                    linewidth=2.0,
                )
                plt.xscale("log")
                plt.xlabel("GP alpha")
                plt.ylabel("Mean target error [mm]")
                plt.title(f"BO alpha benchmark: target = {float(target):.2f} m")
                plt.grid(True, alpha=0.3, which="both")
                savefig(f"09_alpha_error_target{safe_target_tag(float(target))}.png")

                plt.figure(figsize=(8.0, 5.2))
                plt.plot(
                    sub["gp_alpha"],
                    sub["min_residual_margin_mm"],
                    marker="o",
                    linewidth=2.0,
                )
                plt.axhline(0.0, linestyle="--", linewidth=1.4)
                plt.xscale("log")
                plt.xlabel("GP alpha")
                plt.ylabel("Minimum residual margin [mm]")
                plt.title(f"BO alpha benchmark: safety margin, target = {float(target):.2f} m")
                plt.grid(True, alpha=0.3, which="both")
                savefig(f"10_alpha_margin_target{safe_target_tag(float(target))}.png")
        else:
            print(f"[skip] Alpha per-target plots: missing {sorted(required_cols - set(summary.columns))}")

    if not overall.empty:
        required_cols = {"gp_alpha", "mean_target_error_mm", "max_target_error_mm"}

        if required_cols.issubset(overall.columns):
            overall = overall.copy()
            overall["gp_alpha"] = overall["gp_alpha"].astype(float)
            overall = overall.sort_values("gp_alpha")

            plt.figure(figsize=(8.0, 5.2))
            plt.plot(
                overall["gp_alpha"],
                overall["mean_target_error_mm"],
                marker="o",
                linewidth=2.0,
                label="Mean error",
            )
            plt.plot(
                overall["gp_alpha"],
                overall["max_target_error_mm"],
                marker="s",
                linewidth=2.0,
                label="Worst-case error",
            )
            plt.xscale("log")
            plt.xlabel("GP alpha")
            plt.ylabel("Target error [mm]")
            plt.title("Overall BO alpha benchmark")
            plt.grid(True, alpha=0.3, which="both")
            plt.legend()
            savefig("11_alpha_overall_error.png")


# =============================================================================
# STEP D: BUDGET FIGURES
# =============================================================================

def make_budget_figures() -> None:
    """Create consolidated online-budget sweep figures."""
    summary = read_csv(BUDGET_DIR / "bo_budget_sweep_summary.csv")
    overall = read_csv(BUDGET_DIR / "bo_budget_sweep_overall_summary.csv")

    if summary.empty and overall.empty:
        print("[skip] Budget sweep figures: no input tables found.")
        return

    if not summary.empty:
        required_cols = {"target", "budget", "mean_target_error_mm", "min_residual_margin_mm"}

        if required_cols.issubset(summary.columns):
            plt.figure(figsize=(8.0, 5.2))

            for target, sub in summary.groupby("target"):
                sub = sub.sort_values("budget")

                plt.plot(
                    sub["budget"],
                    sub["mean_target_error_mm"],
                    marker="o",
                    linewidth=2.0,
                    label=f"target {float(target):.2f} m",
                )

            plt.xlabel("True simulator calls")
            plt.ylabel("Mean target error [mm]")
            plt.title("BO online-budget sweep")
            plt.grid(True, alpha=0.3)
            plt.legend()
            savefig("12_budget_sweep_mean_error.png")

            plt.figure(figsize=(8.0, 5.2))

            for target, sub in summary.groupby("target"):
                sub = sub.sort_values("budget")

                plt.plot(
                    sub["budget"],
                    sub["min_residual_margin_mm"],
                    marker="o",
                    linewidth=2.0,
                    label=f"target {float(target):.2f} m",
                )

            plt.axhline(0.0, linestyle="--", linewidth=1.4)
            plt.xlabel("True simulator calls")
            plt.ylabel("Minimum residual margin [mm]")
            plt.title("BO online-budget sweep: safety margin")
            plt.grid(True, alpha=0.3)
            plt.legend()
            savefig("13_budget_sweep_min_margin.png")
        else:
            print(f"[skip] Budget per-target plots: missing {sorted(required_cols - set(summary.columns))}")

    if not overall.empty:
        required_cols = {"budget", "mean_target_error_mm", "max_target_error_mm"}

        if required_cols.issubset(overall.columns):
            overall = overall.sort_values("budget")

            plt.figure(figsize=(8.0, 5.2))
            plt.plot(
                overall["budget"],
                overall["mean_target_error_mm"],
                marker="o",
                linewidth=2.0,
                label="Mean error",
            )
            plt.plot(
                overall["budget"],
                overall["max_target_error_mm"],
                marker="s",
                linewidth=2.0,
                label="Worst-case error",
            )
            plt.xlabel("True simulator calls")
            plt.ylabel("Target error [mm]")
            plt.title("Overall BO online-budget sweep")
            plt.grid(True, alpha=0.3)
            plt.legend()
            savefig("14_budget_sweep_overall_error.png")


# =============================================================================
# STEP E: FINAL BO FIGURES
# =============================================================================

def load_final_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final BO tables with backward-compatible filenames."""
    results = read_csv(FINAL_DIR / "bo_final_all_results_by_seed.csv")

    if results.empty:
        results = read_csv(FINAL_DIR / "bo_final_all_results.csv")

    history = read_csv(FINAL_DIR / "bo_final_all_history.csv")
    by_target = read_csv(FINAL_DIR / "bo_final_summary_by_target.csv")
    overall = read_csv(FINAL_DIR / "bo_final_summary_overall.csv")

    results = complete_method_labels(results)
    history = complete_method_labels(history)
    by_target = complete_method_labels(by_target)
    overall = complete_method_labels(overall)

    return results, history, by_target, overall


def make_final_figures(skip_time_responses: bool = False) -> None:
    """Create consolidated final BO and Random Search figures."""
    results, history, by_target, overall = load_final_tables()

    if results.empty:
        print("[skip] Final BO results are missing. Run bo_final_run.py first.")
        return

    bo_by_target = by_target[by_target["method"] == "BO"].copy()
    bo_by_target = sort_targets(bo_by_target)

    if not bo_by_target.empty:
        make_final_bo_tracking_plot(bo_by_target)
        make_final_bo_error_plot(bo_by_target)
        make_final_bo_constraint_plot(bo_by_target)
        make_final_bo_margin_plot(bo_by_target)

    make_final_bo_vs_random_error_plot(by_target)
    make_final_tradeoff_plot(results)
    make_final_parameter_plots(results)
    make_final_convergence_plots(history)
    make_final_cost_plots(overall)

    if not skip_time_responses:
        bo_results = results[results["method"] == "BO"].copy()
        make_time_response_figures(bo_results)


def make_final_bo_tracking_plot(bo_by_target: pd.DataFrame) -> None:
    """Plot final BO target tracking."""
    required_cols = {"target", "mean_peak_y_true"}

    if not required_cols.issubset(bo_by_target.columns):
        return

    yerr = (
        bo_by_target["std_peak_y_true"].fillna(0.0)
        if "std_peak_y_true" in bo_by_target.columns
        else None
    )

    plt.figure(figsize=(8.2, 5.4))
    plt.plot(
        bo_by_target["target"],
        bo_by_target["target"],
        linestyle="--",
        linewidth=1.8,
        label="Ideal tracking",
    )
    plt.errorbar(
        bo_by_target["target"],
        bo_by_target["mean_peak_y_true"],
        yerr=yerr,
        marker="o",
        capsize=4,
        linewidth=2.0,
        label="Final BO mean ± std",
    )
    plt.axhline(
        ROBOT_LIMIT,
        linestyle=":",
        linewidth=1.5,
        label="Nominal robot reach",
    )
    plt.xlabel("Target outreach [m]")
    plt.ylabel("True achieved peak_y [m]")
    plt.title("Final BO target tracking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("15_final_bo_target_tracking.png")


def make_final_bo_error_plot(bo_by_target: pd.DataFrame) -> None:
    """Plot final BO target error."""
    required_cols = {"target", "mean_target_error_mm"}

    if not required_cols.issubset(bo_by_target.columns):
        return

    yerr = (
        bo_by_target["std_target_error_mm"].fillna(0.0)
        if "std_target_error_mm" in bo_by_target.columns
        else None
    )

    xlabels = [target_label(target) for target in bo_by_target["target"]]

    plt.figure(figsize=(8.2, 5.4))
    plt.bar(
        xlabels,
        bo_by_target["mean_target_error_mm"],
        yerr=yerr,
        capsize=4,
        edgecolor="black",
    )
    plt.axhline(10.0, linestyle="--", linewidth=1.5, label="10 mm reference")
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Target error [mm]")
    plt.title("Final BO target error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    savefig("16_final_bo_target_error.png")


def make_final_bo_constraint_plot(bo_by_target: pd.DataFrame) -> None:
    """Plot final BO constraint validation."""
    required_cols = {"target", "mean_max_abs_xr_true"}

    if not required_cols.issubset(bo_by_target.columns):
        return

    xlabels = [target_label(target) for target in bo_by_target["target"]]

    plt.figure(figsize=(8.2, 5.4))
    plt.bar(
        xlabels,
        bo_by_target["mean_max_abs_xr_true"],
        edgecolor="black",
    )
    plt.axhline(
        ROBOT_LIMIT,
        linestyle="--",
        linewidth=1.8,
        label="Hard limit 0.500 m",
    )
    plt.xlabel("Target outreach [m]")
    plt.ylabel("True max |x_r| [m]")
    plt.title("Final BO constraint validation")
    plt.ylim(
        0.0,
        max(0.52, float(bo_by_target["mean_max_abs_xr_true"].max()) * 1.05),
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    savefig("17_final_bo_constraint_validation.png")


def make_final_bo_margin_plot(bo_by_target: pd.DataFrame) -> None:
    """Plot final BO residual margin."""
    required_cols = {"target", "mean_residual_margin_mm"}

    if not required_cols.issubset(bo_by_target.columns):
        return

    xlabels = [target_label(target) for target in bo_by_target["target"]]

    plt.figure(figsize=(8.2, 5.4))
    plt.bar(
        xlabels,
        bo_by_target["mean_residual_margin_mm"],
        edgecolor="black",
        label="Mean residual margin",
    )

    if "min_residual_margin_mm" in bo_by_target.columns:
        plt.scatter(
            xlabels,
            bo_by_target["min_residual_margin_mm"],
            marker="x",
            s=70,
            label="Minimum across seeds",
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Residual margin [mm]")
    plt.title("Final BO residual safety margin")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    savefig("18_final_bo_residual_margin.png")


def make_final_bo_vs_random_error_plot(by_target: pd.DataFrame) -> None:
    """Plot BO versus Random Search target error."""
    if by_target.empty:
        return

    methods = set(by_target["method"].astype(str))

    if not {"BO", "RandomSearch"}.issubset(methods):
        return

    if not {"target", "method", "mean_target_error_mm"}.issubset(by_target.columns):
        return

    comp = (
        by_target.pivot_table(
            index="target",
            columns="method",
            values="mean_target_error_mm",
            aggfunc="mean",
        )
        .sort_index()
    )

    if not {"BO", "RandomSearch"}.issubset(comp.columns):
        return

    x = np.arange(len(comp.index))
    width = 0.38

    plt.figure(figsize=(8.6, 5.4))
    plt.bar(x - width / 2, comp["BO"], width, label="BO", edgecolor="black")
    plt.bar(
        x + width / 2,
        comp["RandomSearch"],
        width,
        label="Random Search",
        edgecolor="black",
    )
    plt.xticks(x, [target_label(target) for target in comp.index])
    plt.xlabel("Target outreach [m]")
    plt.ylabel("Mean target error [mm]")
    plt.title("Final BO vs Random Search")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    savefig("19_final_bo_vs_random_error.png")


def make_final_tradeoff_plot(results: pd.DataFrame) -> None:
    """Plot final accuracy/safety trade-off."""
    required_cols = {"target_error_mm", "residual_margin_mm", "method"}

    if results.empty or not required_cols.issubset(results.columns):
        return

    plt.figure(figsize=(8.0, 5.4))

    for method, sub in results.groupby("method"):
        plt.scatter(
            sub["target_error_mm"],
            sub["residual_margin_mm"],
            label=str(method),
            s=60,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.4,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.3)
    plt.xlabel("Target error [mm]")
    plt.ylabel("Residual margin [mm]")
    plt.title("Final BO safety/accuracy trade-off")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("20_final_error_margin_tradeoff.png")


def make_final_parameter_plots(results: pd.DataFrame) -> None:
    """Plot final BO optimized parameters, mean ± std across seeds."""
    param_cols = [
        col for col in ["Kr", "hr", "f0", "f1", "A", "x_r_start"]
        if col in results.columns
    ]

    bo_results = results[results["method"] == "BO"].copy()

    if bo_results.empty or not param_cols:
        return

    for col in param_cols:
        grouped = (
            bo_results.groupby("target")[col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("target")
        )

        plt.figure(figsize=(8.0, 5.2))
        plt.errorbar(
            [target_label(target) for target in grouped["target"]],
            grouped["mean"],
            yerr=grouped["std"].fillna(0.0),
            marker="o",
            linewidth=2.0,
            capsize=4,
        )
        plt.xlabel("Target outreach [m]")
        plt.ylabel(col)
        plt.title(f"Final BO optimized parameter: {col}")
        plt.grid(True, alpha=0.3)
        savefig(f"21_final_parameter_{col}.png")


def make_final_convergence_plots(history: pd.DataFrame) -> None:
    """Plot final convergence curves for representative targets."""
    required_cols = {"target", "method", "seed", "iteration", "best_feasible_error_so_far_mm"}

    if history.empty or not required_cols.issubset(history.columns):
        return

    for target in REPRESENTATIVE_TARGETS:
        sub = history[np.isclose(history["target"].astype(float), target)].copy()

        if sub.empty:
            continue

        plt.figure(figsize=(8.4, 5.4))

        for (method, seed), group in sub.groupby(["method", "seed"]):
            group = group.sort_values("iteration")

            plt.plot(
                group["iteration"],
                group["best_feasible_error_so_far_mm"],
                linewidth=1.4,
                label=f"{method}, seed {int(seed)}",
            )

        plt.xlabel("True simulator calls")
        plt.ylabel("Best feasible target error so far [mm]")
        plt.title(f"Convergence comparison, target = {target:.2f} m")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        savefig(f"22_final_convergence_target{safe_target_tag(target)}.png")


def make_final_cost_plots(overall: pd.DataFrame) -> None:
    """Plot final online cost and runtime."""
    if overall.empty:
        return

    if {"method", "total_true_simulator_calls"}.issubset(overall.columns):
        plt.figure(figsize=(7.2, 5.2))
        plt.bar(
            overall["method"],
            overall["total_true_simulator_calls"],
            edgecolor="black",
        )
        plt.ylabel("Total true simulator calls")
        plt.title("Final online evaluation cost")
        plt.grid(True, axis="y", alpha=0.3)
        savefig("23_final_online_simulator_calls.png")

    if {"method", "total_optimization_time_s"}.issubset(overall.columns):
        plt.figure(figsize=(7.2, 5.2))
        plt.bar(
            overall["method"],
            overall["total_optimization_time_s"],
            edgecolor="black",
        )
        plt.ylabel("Total optimization time [s]")
        plt.title("Final optimization runtime")
        plt.grid(True, axis="y", alpha=0.3)
        savefig("24_final_runtime.png")


# =============================================================================
# TIME RESPONSE FIGURES
# =============================================================================

def make_time_response_figures(bo_results: pd.DataFrame) -> None:
    """
    Create true dynamic time-response plots for the best final BO solution per target.

    For each target, the seed with the lowest final target_error_mm is used.
    """
    if not OBO_AVAILABLE:
        print("[skip] Cannot import optimization_bo; time-response plots not created.")
        return

    if bo_results.empty:
        return

    required_cols = {"target", "target_error_mm"}

    if not required_cols.issubset(bo_results.columns):
        print("[skip] Time-response plots: final BO result table lacks target/error columns.")
        return

    cfg = obo.BOConfig()
    cfg.robot_limit_true = ROBOT_LIMIT

    best_rows = (
        bo_results.sort_values(["target", "target_error_mm"])
        .groupby("target", as_index=False)
        .first()
        .sort_values("target")
    )

    response_cache: dict[float, tuple[Any, Any, Any, Any, dict[str, Any], pd.Series]] = {}

    for _, row in best_rows.iterrows():
        target = float(row["target"])

        try:
            params = obo.row_to_params(row, cfg)
            t, y, x_b, x_r, metrics = obo.simulate_candidate_with_solution(
                params,
                target,
                cfg,
            )
        except Exception as exc:
            print(f"[skip] time response target={target:.2f}: {exc}")
            continue

        response_cache[target] = (t, y, x_b, x_r, metrics, row)

        plot_single_time_response(
            target=target,
            t=t,
            y=y,
            x_b=x_b,
            x_r=x_r,
            row=row,
        )

    plot_representative_time_responses(response_cache)


def plot_single_time_response(
    target: float,
    t: np.ndarray,
    y: np.ndarray,
    x_b: np.ndarray,
    x_r: np.ndarray,
    row: pd.Series,
) -> None:
    """Create detailed time-response plot for one BO solution."""
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True)

    axes[0].plot(t, y, linewidth=2.0, label="Total outreach y(t)")
    axes[0].axhline(target, linestyle="--", linewidth=1.5, label="Target")
    axes[0].set_ylabel("y [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, x_r, linewidth=1.8, label="Robot displacement x_r(t)")
    axes[1].axhline(
        ROBOT_LIMIT,
        linestyle="--",
        linewidth=1.2,
        label="+0.500 m limit",
    )
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
        f"Final BO true response | target = {target:.2f} m | "
        f"error = {float(row['target_error_mm']):.2f} mm | "
        f"max |x_r| = {float(row['max_abs_xr_true']):.4f} m | "
        f"margin = {float(row['residual_margin_mm']):.2f} mm",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(f"25_time_response_target{safe_target_tag(target)}.png")


def plot_representative_time_responses(
    response_cache: dict[float, tuple[Any, Any, Any, Any, dict[str, Any], pd.Series]],
) -> None:
    """Create compact presentation figure for representative BO time responses."""
    available_targets = []

    for target in REPRESENTATIVE_TARGETS:
        for key in response_cache:
            if np.isclose(key, target):
                available_targets.append(key)

    if not available_targets:
        return

    fig, axes = plt.subplots(
        len(available_targets),
        1,
        figsize=(11.5, 4.3 * len(available_targets)),
        sharex=True,
    )

    if len(available_targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, available_targets):
        t, y, x_b, x_r, _, row = response_cache[target]

        ax.plot(t, y, linewidth=2.0, label="y(t)")
        ax.plot(t, x_r, linewidth=1.5, label="x_r(t)")
        ax.plot(t, x_b, linewidth=1.5, label="x_b(t)")
        ax.axhline(target, linestyle="--", linewidth=1.4, label="Target")
        ax.axhline(ROBOT_LIMIT, linestyle=":", linewidth=1.2, label="0.500 m limit")

        ax.set_ylabel("Displacement [m]")
        ax.set_title(
            f"target = {target:.2f} m, achieved peak = {float(row['peak_y_true']):.3f} m, "
            f"error = {float(row['target_error_mm']):.1f} mm"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        "Representative final BO time responses",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig("26_time_response_representative_targets.png")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated report figures for the BO section."
    )

    parser.add_argument(
        "--only",
        type=str,
        default=DEFAULT_ONLY,
        help=(
            "Comma-separated sections to generate. "
            "Choices: all, acquisition, kernel, alpha, budget, final."
        ),
    )

    parser.add_argument(
        "--skip_time_responses",
        action="store_true",
        help="Skip time-response figures from the final BO solutions.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate the selected BO report figure set."""
    args = parse_args()
    sections = parse_sections(args.only)

    ensure_dirs()

    print("=" * 80)
    print("COMPLETE BO REPORT FIGURE GENERATION")
    print("=" * 80)
    print(f"Output folder: {FIGURES_DIR}")
    print(f"Sections:      {sorted(sections)}")
    print("=" * 80)

    if "acquisition" in sections:
        make_acquisition_figures()

    if "kernel" in sections:
        make_kernel_figures()

    if "alpha" in sections:
        make_alpha_figures()

    if "budget" in sections:
        make_budget_figures()

    if "final" in sections:
        make_final_figures(skip_time_responses=args.skip_time_responses)

    print("=" * 80)
    print("Done. Use figures/optimization_bo_report for report/presentation selection.")
    print("=" * 80)


if __name__ == "__main__":
    main()