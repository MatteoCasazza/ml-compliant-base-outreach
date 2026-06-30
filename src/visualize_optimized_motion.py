"""
visualize_optimized_motion.py
=============================

Create a GIF or MP4 animation of an optimized extra-reach trajectory.

The script:
    - loads a selected optimized solution;
    - lets the user choose method and target;
    - re-simulates the candidate with the true dynamic simulator;
    - saves an animation of the compliant-base robot motion.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from dynamics import simulate_system


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIS_DIR = PROJECT_ROOT / "figures" / "visualization"

PHYSICAL_VALIDATION_SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "physical_validation"
    / "physical_validation_selected_candidates.csv"
)

FALLBACK_PATHS = {
    "gpde": [
        PROJECT_ROOT / "results" / "optimization_gp_de" / "gp_de_results.csv",
    ],
    "nn": [
        PROJECT_ROOT / "results" / "optimization_nn_gradient" / "nn_gradient_results.csv",
        PROJECT_ROOT / "results" / "optimization_nn_gradient" / "nn_adam_results.csv",
    ],
    "bo": [
        PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_results_by_seed.csv",
        PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_results.csv",
        PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_all_results_by_seed.csv",
        PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_all_results.csv",
    ],
    "random": [
        PROJECT_ROOT / "results" / "optimization_bo_final" / "random_search_results.csv",
        PROJECT_ROOT / "results" / "optimization_bo_final" / "random_search_final_results_by_seed.csv",
        PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_all_results_by_seed.csv",
        PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_all_results.csv",
    ],
}


# =============================================================================
# SETTINGS
# =============================================================================

ROBOT_LIMIT_TRUE = 0.500
FEASIBILITY_TOLERANCE_M = 1e-9

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

METHOD_DISPLAY = {
    "nn": "NN+Adam",
    "gpde": "GP+DE",
    "bo": "BO",
    "random": "Random Search",
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    """Create visualization output directory."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_method_name(method: str) -> str:
    """Normalize user method name."""
    method = method.lower().strip()

    aliases = {
        "nn": "nn",
        "nn_adam": "nn",
        "nn adam": "nn",
        "nn+adam": "nn",
        "adam": "nn",
        "de": "gpde",
        "gpde": "gpde",
        "gp_de": "gpde",
        "gp de": "gpde",
        "gp+de": "gpde",
        "bo": "bo",
        "bayesian": "bo",
        "bayesian optimization": "bo",
        "random": "random",
        "random_search": "random",
        "random search": "random",
    }

    if method not in aliases:
        raise ValueError(
            f"Unknown method {method!r}. Use one of: nn, gpde, bo, random."
        )

    return aliases[method]


def method_matches(row_method: Any, requested: str) -> bool:
    """Check whether a row method label matches the requested method."""
    method = str(row_method).lower().replace("_", " ").strip()

    if requested == "nn":
        return "nn" in method or "adam" in method

    if requested == "gpde":
        return ("gp" in method and "de" in method) or "differential" in method

    if requested == "bo":
        return (
            method == "bo"
            or "bo best" in method
            or "bayesian" in method
            or method.startswith("final bo")
        )

    if requested == "random":
        return "random" in method

    return False


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


def standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create common metric columns independent of source CSV naming."""
    out = df.copy()

    rename_candidates = {
        "peak_y_true": "peak_y",
        "true_peak_y": "peak_y",
        "y_sim": "peak_y",
        "y_true": "peak_y",
        "max_abs_xr_true": "max_abs_xr",
        "true_max_abs_xr": "max_abs_xr",
        "max_xr_sim": "max_abs_xr",
        "target_error_abs_mm": "target_error_mm",
        "error_sim_mm": "target_error_mm",
        "error_mm": "target_error_mm",
    }

    for old, new in rename_candidates.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "target_error_m" not in out.columns and "target_error_mm" in out.columns:
        out["target_error_m"] = out["target_error_mm"].astype(float) / 1000.0

    if "target_error_mm" not in out.columns and "target_error_m" in out.columns:
        out["target_error_mm"] = out["target_error_m"].astype(float) * 1000.0

    if "max_abs_xr" in out.columns:
        if "residual_margin_mm" not in out.columns:
            out["residual_margin_mm"] = (
                ROBOT_LIMIT_TRUE - out["max_abs_xr"].astype(float)
            ) * 1000.0

        if "constraint_violation_abs_m" not in out.columns:
            out["constraint_violation_abs_m"] = np.maximum(
                0.0,
                out["max_abs_xr"].astype(float) - ROBOT_LIMIT_TRUE,
            )

        if "constraint_violation_abs_mm" not in out.columns:
            out["constraint_violation_abs_mm"] = (
                out["constraint_violation_abs_m"].astype(float) * 1000.0
            )

        if "feasible_abs" not in out.columns:
            out["feasible_abs"] = (
                out["max_abs_xr"].astype(float)
                <= ROBOT_LIMIT_TRUE + FEASIBILITY_TOLERANCE_M
            )

    return out


def normalize_bo_random_rows(df: pd.DataFrame, requested: str) -> pd.DataFrame:
    """Normalize method labels in BO final output files."""
    out = df.copy()

    if "method" not in out.columns:
        if requested == "random":
            out["method"] = "RandomSearch"
        elif requested == "bo":
            out["method"] = "BO"
        else:
            out["method"] = METHOD_DISPLAY.get(requested, requested)

    if "run_label" in out.columns:
        labels = out["run_label"].astype(str)

        random_mask = labels.str.contains(
            "Random|RandomSearch|RS",
            case=False,
            regex=True,
        )
        bo_mask = labels.str.contains("BO", case=False, regex=True)

        out.loc[random_mask, "method"] = "RandomSearch"
        out.loc[bo_mask & ~random_mask, "method"] = "BO"

    return out


def selection_sort_key(row: pd.Series) -> tuple[int, float, float]:
    """
    Select best candidate.

    Priority:
        feasible first;
        if feasible, minimum target error;
        if infeasible, minimum violation then target error.
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


def row_to_params(row: pd.Series) -> dict[str, float]:
    """Convert one selected row to simulator parameter dictionary."""
    return {col: float(row[col]) for col in PARAM_COLUMNS}


# =============================================================================
# CANDIDATE LOADING
# =============================================================================

def load_first_available_table(method: str) -> pd.DataFrame:
    """Load preferred physical-validation table or method fallback table."""
    if PHYSICAL_VALIDATION_SELECTED_PATH.exists():
        df = pd.read_csv(PHYSICAL_VALIDATION_SELECTED_PATH)
        df["source_file"] = str(PHYSICAL_VALIDATION_SELECTED_PATH)
        return df

    for path in FALLBACK_PATHS.get(method, []):
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            return df

    raise FileNotFoundError(
        "No candidate result file found.\n"
        f"Preferred file: {PHYSICAL_VALIDATION_SELECTED_PATH}\n"
        f"Fallback paths for method {method}:"
        + "".join(f"\n  - {path}" for path in FALLBACK_PATHS.get(method, []))
    )


def load_candidate(method: str, target: float) -> pd.Series:
    """Load one optimized candidate for the selected method and target."""
    method = normalize_method_name(method)

    df = load_first_available_table(method)
    df = normalize_bo_random_rows(df, requested=method)
    df = complete_parameter_columns(df)
    df = standardize_metric_columns(df)

    missing_params = [col for col in PARAM_COLUMNS if col not in df.columns]

    if missing_params:
        raise KeyError(
            "The candidate table does not contain all simulator parameters.\n"
            f"Missing: {missing_params}\n"
            f"Required: {PARAM_COLUMNS}"
        )

    if "method" in df.columns:
        df = df[df["method"].apply(lambda value: method_matches(value, method))].copy()

    if df.empty:
        raise ValueError(f"No rows found for method={method!r}.")

    if "target" not in df.columns:
        raise KeyError("The candidate table does not contain a 'target' column.")

    group = df[np.isclose(df["target"].astype(float), float(target), atol=1e-6)].copy()

    if group.empty:
        available_targets = sorted(df["target"].astype(float).unique())

        raise ValueError(
            f"No candidate found for method={method!r}, target={target:.3f}.\n"
            f"Available targets for this method: {available_targets}"
        )

    sorted_idx = sorted(group.index, key=lambda idx: selection_sort_key(group.loc[idx]))
    selected = group.loc[sorted_idx[0]].copy()

    print("\nSelected candidate")
    print("=" * 70)
    print(f"Method requested: {method}")
    print(f"Target:           {float(target):.3f} m")
    print(f"Source file:      {selected.get('source_file', 'unknown')}")

    if "method" in selected:
        print(f"Row method:       {selected['method']}")

    if "target_error_mm" in selected and pd.notna(selected["target_error_mm"]):
        print(f"Stored error:     {float(selected['target_error_mm']):.3f} mm")

    if "residual_margin_mm" in selected and pd.notna(selected["residual_margin_mm"]):
        print(f"Stored margin:    {float(selected['residual_margin_mm']):.3f} mm")

    print("=" * 70)

    return selected


# =============================================================================
# SIMULATION HELPERS
# =============================================================================

def extract_solution_and_metrics(output: Any) -> tuple[Any, dict[str, Any]]:
    """Extract ODE solution and metrics dictionary from simulator output."""
    sol = None
    metrics = None

    if isinstance(output, tuple):
        for item in output:
            if hasattr(item, "t") and hasattr(item, "y"):
                sol = item
            elif isinstance(item, dict) and "peak_y" in item:
                metrics = dict(item)

    elif isinstance(output, dict):
        metrics = dict(output)

    if sol is None:
        raise RuntimeError(
            "Could not extract ODE solution. The simulator must support "
            "return_full=True or return_solution=True."
        )

    if metrics is None:
        metrics = {}

    return sol, metrics


def complete_simulation_metrics(
    metrics: dict[str, Any],
    target: float,
    y: np.ndarray,
    x_b: np.ndarray,
    x_r: np.ndarray,
) -> dict[str, Any]:
    """Complete scalar metrics from the full trajectory."""
    completed = dict(metrics)

    peak_y = float(np.max(y))
    max_abs_xr = float(np.max(np.abs(x_r)))
    max_abs_xb = float(np.max(np.abs(x_b)))
    target_error_mm = abs(peak_y - float(target)) * 1000.0
    residual_margin_mm = (ROBOT_LIMIT_TRUE - max_abs_xr) * 1000.0
    violation_mm = max(0.0, max_abs_xr - ROBOT_LIMIT_TRUE) * 1000.0

    completed["peak_y"] = peak_y
    completed["max_abs_xr"] = max_abs_xr
    completed["max_abs_xb"] = max_abs_xb
    completed["target_error_mm"] = target_error_mm
    completed["residual_margin_mm"] = residual_margin_mm
    completed["constraint_violation_abs_mm"] = violation_mm
    completed["feasible_abs"] = violation_mm <= FEASIBILITY_TOLERANCE_M * 1000.0
    completed["max_xr"] = float(np.max(x_r))
    completed["min_xr"] = float(np.min(x_r))
    completed["max_xb"] = float(np.max(x_b))
    completed["min_xb"] = float(np.min(x_b))
    completed["extra_reach"] = peak_y - ROBOT_LIMIT_TRUE

    return completed


def simulate_candidate_with_solution(
    params: dict[str, float],
    target: float,
    t_sim: float = 60.0,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the true simulator and return t, y, x_b, x_r and metrics."""
    attempts = [
        {
            "T_sim": t_sim,
            "dt": dt,
            "y_target": float(target),
            "x_r_max": ROBOT_LIMIT_TRUE,
            "return_metrics": True,
            "return_full": True,
        },
        {
            "T_sim": t_sim,
            "dt": dt,
            "y_target": float(target),
            "x_r_max": ROBOT_LIMIT_TRUE,
            "return_metrics": True,
            "return_solution": True,
        },
        {
            "T_sim": t_sim,
            "dt": dt,
            "return_metrics": True,
            "return_full": True,
        },
        {
            "T_sim": t_sim,
            "dt": dt,
            "return_metrics": True,
            "return_solution": True,
        },
    ]

    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            output = simulate_system(params, **kwargs)
            sol, metrics = extract_solution_and_metrics(output)

            t = np.asarray(sol.t, dtype=float)
            x_b = np.asarray(sol.y[2], dtype=float)
            x_r = np.asarray(sol.y[3], dtype=float)
            y = x_b + x_r

            metrics = complete_simulation_metrics(
                metrics=metrics,
                target=target,
                y=y,
                x_b=x_b,
                x_r=x_r,
            )

            return t, y, x_b, x_r, metrics

        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "simulate_system could not be called with any supported full-solution interface."
    ) from last_error


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_spring(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    amplitude: float = 0.025,
    coils: int = 6,
    color: str = "black",
) -> None:
    """Draw a compact zig-zag spring between x0 and x1."""
    if x1 <= x0 + 0.005:
        ax.plot([x0, x1], [y, y], color=color, linewidth=1.5)
        return

    lead = 0.015
    xs = np.linspace(x0 + lead, x1 - lead, 2 * coils + 1)
    ys = np.full_like(xs, y)

    for index in range(1, len(xs) - 1):
        ys[index] = y + amplitude * (1 if index % 2 else -1)

    ax.plot([x0, x0 + lead], [y, y], color=color, linewidth=1.5)
    ax.plot(xs, ys, color=color, linewidth=2.0)
    ax.plot([x1 - lead, x1], [y, y], color=color, linewidth=1.5)


# =============================================================================
# ANIMATION
# =============================================================================

def create_motion_animation(
    method: str,
    target: float,
    t: np.ndarray,
    y: np.ndarray,
    x_b: np.ndarray,
    x_r: np.ndarray,
    metrics: dict[str, Any],
    save_path: Path,
    fps: int = 25,
    duration: float = 14.0,
    format_name: str = "gif",
) -> None:
    """Create and save GIF or MP4 motion animation."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = max(2, int(fps * duration))
    frame_idx = np.linspace(0, len(t) - 1, n_frames, dtype=int)

    peak_y = float(metrics.get("peak_y", np.max(y)))
    max_abs_xr = float(metrics.get("max_abs_xr", np.max(np.abs(x_r))))
    residual_margin_mm = (ROBOT_LIMIT_TRUE - max_abs_xr) * 1000.0
    target_error_mm = abs(peak_y - float(target)) * 1000.0

    x_min = min(-0.12, float(np.min(x_b)) - 0.12)
    x_max = max(float(target), peak_y, ROBOT_LIMIT_TRUE, float(np.max(y))) + 0.20

    fig, (ax_sys, ax_time) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [0.85, 1.25]},
        constrained_layout=False,
    )

    fig.suptitle(
        f"Optimized motion visualization — {METHOD_DISPLAY.get(method, method)}, "
        f"target = {target:.2f} m",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    mass_w = 0.055
    mass_h = 0.13
    y_mass = 0.50

    base_color = "#AFC7FF"
    robot_color = "#C8E7A8"

    def update(frame_number: int) -> list[Any]:
        idx = int(frame_idx[frame_number])

        current_t = float(t[idx])
        xb = float(x_b[idx])
        xr = float(x_r[idx])
        current_y = float(y[idx])
        peak_so_far = float(np.max(y[:idx + 1]))

        ax_sys.clear()
        ax_time.clear()

        # ------------------------------------------------------------------
        # Top panel: mechanical animation
        # ------------------------------------------------------------------
        ax_sys.set_xlim(x_min, x_max)
        ax_sys.set_ylim(0.24, 0.80)
        ax_sys.set_yticks([])
        ax_sys.set_xlabel("Absolute horizontal position [m]", labelpad=8)
        ax_sys.tick_params(axis="x", labelbottom=True)
        ax_sys.grid(True, axis="x", alpha=0.22)

        wall_x = 0.0
        base_x = xb
        robot_x = current_y

        wall = patches.Rectangle(
            (wall_x - 0.008, 0.27),
            0.008,
            0.42,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1.3,
        )
        ax_sys.add_patch(wall)

        draw_spring(
            ax_sys,
            wall_x,
            base_x - mass_w / 2,
            y_mass,
            amplitude=0.022,
            coils=5,
            color="black",
        )

        draw_spring(
            ax_sys,
            base_x + mass_w / 2,
            robot_x - mass_w / 2,
            y_mass,
            amplitude=0.022,
            coils=7,
            color="tab:green",
        )

        base_rect = patches.FancyBboxPatch(
            (base_x - mass_w / 2, y_mass - mass_h / 2),
            mass_w,
            mass_h,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor=base_color,
            edgecolor="black",
            linewidth=1.6,
        )

        robot_rect = patches.FancyBboxPatch(
            (robot_x - mass_w / 2, y_mass - mass_h / 2),
            mass_w,
            mass_h,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor=robot_color,
            edgecolor="black",
            linewidth=1.6,
        )

        ax_sys.add_patch(base_rect)
        ax_sys.add_patch(robot_rect)

        ax_sys.text(base_x, y_mass, r"$M_b$", ha="center", va="center", fontsize=18)
        ax_sys.text(robot_x, y_mass, r"$M_r$", ha="center", va="center", fontsize=18)

        ax_sys.axvline(
            ROBOT_LIMIT_TRUE,
            color="tab:red",
            linestyle="--",
            linewidth=1.8,
            label="Robot limit 0.500 m",
        )

        ax_sys.axvline(
            target,
            color="tab:green",
            linestyle=":",
            linewidth=2.0,
            label="Target",
        )

        ax_sys.axvline(
            current_y,
            color="black",
            linestyle="-",
            linewidth=1.4,
            alpha=0.8,
            label="Current y(t)",
        )

        ax_sys.axvline(
            peak_so_far,
            color="tab:orange",
            linestyle="-.",
            linewidth=1.8,
            label="Peak so far",
        )

        info = (
            f"t = {current_t:5.2f} s\n"
            f"x_b = {xb:+.4f} m\n"
            f"x_r = {xr:+.4f} m\n"
            f"y = {current_y:+.4f} m\n"
            f"peak_y = {peak_y:.4f} m\n"
            f"error = {target_error_mm:.2f} mm\n"
            f"margin = {residual_margin_mm:.2f} mm"
        )

        ax_sys.text(
            0.012,
            0.96,
            info,
            transform=ax_sys.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
        )

        ax_sys.legend(loc="upper right", fontsize=8)

        # ------------------------------------------------------------------
        # Bottom panel: time response
        # ------------------------------------------------------------------
        ax_time.plot(
            t[:idx + 1],
            y[:idx + 1],
            linewidth=2.2,
            color="tab:orange",
            label=r"$y(t)=x_b+x_r$",
        )

        ax_time.plot(
            t[:idx + 1],
            x_r[:idx + 1],
            linewidth=1.6,
            color="tab:green",
            label=r"$x_r(t)$",
        )

        ax_time.plot(
            t[:idx + 1],
            x_b[:idx + 1],
            linewidth=1.6,
            color="tab:cyan",
            label=r"$x_b(t)$",
        )

        ax_time.axhline(
            target,
            linestyle=":",
            linewidth=2.0,
            color="tab:green",
            label="Target",
        )

        ax_time.axhline(
            ROBOT_LIMIT_TRUE,
            linestyle="--",
            linewidth=1.8,
            color="tab:red",
            label="Robot limit",
        )

        ax_time.axhline(
            peak_y,
            linestyle="-.",
            linewidth=1.8,
            color="tab:orange",
            label="Final peak",
        )

        ax_time.plot(current_t, current_y, "o", color="black", markersize=5)

        ax_time.set_xlim(0.0, float(t[-1]))

        y_min = min(float(np.min(x_b)), float(np.min(x_r)), -0.05) - 0.03
        y_max = max(float(target), peak_y, ROBOT_LIMIT_TRUE, float(np.max(y))) + 0.07

        ax_time.set_ylim(y_min, y_max)
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Position [m]")
        ax_time.set_title("True-simulator time response", pad=12)
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(loc="upper right", fontsize=8)

        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.07,
        top=0.92,
        hspace=0.48,
    )

    if format_name.lower() == "gif":
        writer = PillowWriter(fps=fps)
        animation.save(save_path, writer=writer, dpi=110)

    elif format_name.lower() == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2500)
            animation.save(save_path, writer=writer, dpi=140)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "MP4 export requires ffmpeg, but ffmpeg was not found. "
                "Install ffmpeg or use --format gif."
            ) from exc

    else:
        raise ValueError("format_name must be 'gif' or 'mp4'.")

    plt.close(fig)

    print(f"\nSaved animation: {save_path}")
    print(f"File size: {save_path.stat().st_size / 1024.0:.1f} KB")


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create GIF/MP4 visualization of optimized compliant-base motion."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="nn",
        choices=["nn", "gpde", "de", "bo", "random"],
        help="Method to visualize: nn, gpde/de, bo, random.",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=0.65,
        help="Target outreach in meters, e.g. 0.65 or 0.75.",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Output format. GIF works without ffmpeg; MP4 requires ffmpeg.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=14.0,
        help="Animation duration in seconds.",
    )

    parser.add_argument(
        "--T_sim",
        type=float,
        default=60.0,
        help="Simulation duration in seconds.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Simulation time step.",
    )

    return parser.parse_args()


def main() -> None:
    """Create optimized-motion animation."""
    args = parse_args()

    ensure_dirs()

    method = normalize_method_name(args.method)
    target = float(args.target)

    row = load_candidate(method=method, target=target)
    params = row_to_params(row)

    t, y, x_b, x_r, metrics = simulate_candidate_with_solution(
        params=params,
        target=target,
        t_sim=float(args.T_sim),
        dt=float(args.dt),
    )

    tag = f"{method}_target{int(round(target * 1000.0)):04d}"
    save_path = VIS_DIR / f"motion_{tag}.{args.format.lower()}"

    print("\nRecomputed true-simulator metrics")
    print("=" * 70)
    print(f"peak_y:        {float(np.max(y)):.6f} m")
    print(f"target error:  {abs(float(np.max(y)) - target) * 1000.0:.3f} mm")
    print(f"max_abs_xr:    {float(np.max(np.abs(x_r))):.6f} m")
    print(f"margin:        {(ROBOT_LIMIT_TRUE - float(np.max(np.abs(x_r)))) * 1000.0:.3f} mm")
    print("=" * 70)

    create_motion_animation(
        method=method,
        target=target,
        t=t,
        y=y,
        x_b=x_b,
        x_r=x_r,
        metrics=metrics,
        save_path=save_path,
        fps=int(args.fps),
        duration=float(args.duration),
        format_name=args.format,
    )


if __name__ == "__main__":
    main()