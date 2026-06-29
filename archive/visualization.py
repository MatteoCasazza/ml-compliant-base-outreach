"""
visualization.py
================

Advanced visualizations for the final constraint-aware extra-reaching solution.

This script works with the final project pipeline:

    dynamics.py
    dataset.py
    model_peak_y.py
    optimization_constraint_aware.py
    visualization.py

Expected input files
--------------------
- data/dataset_augmented.csv
- results/gp/gp_model.pkl
- results/gp/scaler_X.pkl
- results/gp/scaler_y.pkl
- results/constraint_aware/final_solution_target064.csv

Generated outputs
-----------------
- figures/constraint_aware/final_trajectory_target064.png
- figures/animation/constraint_aware_target064.gif
- figures/gp/gp_surface_Kr_hr_constraint_aware_target064.png
- figures/sensitivity/sensitivity_constraint_aware_target064.png
- results/sensitivity/sensitivity_constraint_aware_target064.csv
- figures/robustness/monte_carlo_constraint_aware_target064.png
- results/robustness/monte_carlo_constraint_aware_target064.csv
- results/robustness/monte_carlo_constraint_aware_target064_stats.csv

Author: Matteo Casazza
Date: 2026
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from dynamics import simulate_system
from dataset import load_dataset, load_dataset_dataframe, ParameterRanges
from model_peak_y import load_model


# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "dataset_augmented.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "gp"

FINAL_SOLUTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "constraint_aware"
    / "final_solution_target064.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

X_R_MAX = 0.5
CONSTRAINT_TOLERANCE = 0.002

T_SIM = 60.0
DT = 0.001

PARAM_ORDER = ParameterRanges.get_param_names()

FINAL_TRAJECTORY_PATH = (
    FIGURES_DIR / "constraint_aware" / "final_trajectory_target064.png"
)

FINAL_ANIMATION_PATH = (
    FIGURES_DIR / "animation" / "constraint_aware_target064.gif"
)

GP_SURFACE_PATH = (
    FIGURES_DIR / "gp" / "gp_surface_Kr_hr_constraint_aware_target064.png"
)

SENSITIVITY_PLOT_PATH = (
    FIGURES_DIR / "sensitivity" / "sensitivity_constraint_aware_target064.png"
)

SENSITIVITY_CSV_PATH = (
    RESULTS_DIR / "sensitivity" / "sensitivity_constraint_aware_target064.csv"
)

MONTE_CARLO_PLOT_PATH = (
    FIGURES_DIR / "robustness" / "monte_carlo_constraint_aware_target064.png"
)

MONTE_CARLO_CSV_PATH = (
    RESULTS_DIR / "robustness" / "monte_carlo_constraint_aware_target064.csv"
)


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def ensure_output_dirs() -> None:
    """Create output folders if they do not exist."""
    for directory in [
        RESULTS_DIR,
        FIGURES_DIR,
        FIGURES_DIR / "animation",
        FIGURES_DIR / "constraint_aware",
        FIGURES_DIR / "dataset",
        FIGURES_DIR / "gp",
        FIGURES_DIR / "robustness",
        FIGURES_DIR / "sensitivity",
        RESULTS_DIR / "dataset",
        RESULTS_DIR / "gp",
        RESULTS_DIR / "robustness",
        RESULTS_DIR / "sensitivity",
        RESULTS_DIR / "constraint_aware",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_final_constraint_solution(
    filepath: str | Path = FINAL_SOLUTION_PATH,
) -> pd.DataFrame:
    """
    Load the final constraint-aware solution selected by
    optimization_constraint_aware.py.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Final solution file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_cols = {
        "method",
        "target",
        "y_sim",
        "error_sim",
        "extra_reach",
        "max_xr_sim",
        "max_xb_sim",
        "constraint_violation",
        "feasible",
        "Kb",
        "Kr",
        "Mb",
        "hb",
        "hr",
        "f0",
        "f1",
        "A",
        "x_r_start",
    }

    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {filepath}: {sorted(missing)}"
        )

    if len(df) != 1:
        print(
            f"Warning: expected one final solution, found {len(df)} rows. "
            "Using the first row."
        )

    return df


def extract_params_from_constraint_row(row: pd.Series) -> np.ndarray:
    """
    Extract full GP/simulator parameter vector from the final solution row.

    Output order:
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
    """
    return np.array(
        [
            row["Kb"],
            row["Kr"],
            row["Mb"],
            row["hb"],
            row["hr"],
            row["f0"],
            row["f1"],
            row["A"],
            row["x_r_start"],
        ],
        dtype=float,
    )


def simulate_with_metrics(
    params: np.ndarray,
    y_target: Optional[float] = None,
    return_solution: bool = False,
) -> Dict:
    """
    Run the dynamic simulation and return physical metrics.

    This helper supports both possible dynamics.py interfaces:
    - return_solution=True
    - return_full=True
    """
    if return_solution:
        try:
            output = simulate_system(
                params,
                T_sim=T_SIM,
                dt=DT,
                return_solution=True,
                return_metrics=True,
                x_r_max=X_R_MAX,
                y_target=y_target,
            )
        except TypeError:
            output = simulate_system(
                params,
                T_sim=T_SIM,
                dt=DT,
                return_full=True,
            )

        if not isinstance(output, tuple):
            raise RuntimeError(
                "Unexpected simulate_system output. Expected tuple with solution."
            )

        peak_y = None
        sol = None
        metrics = None

        for item in output:
            if hasattr(item, "t") and hasattr(item, "y"):
                sol = item
            elif isinstance(item, dict) and "peak_y" in item:
                metrics = item
            elif isinstance(item, (float, int, np.floating)):
                peak_y = float(item)

        if sol is None:
            raise RuntimeError("Could not extract ODE solution from simulator output.")

        x_b = sol.y[2]
        x_r = sol.y[3]
        y = x_b + x_r

        if metrics is None:
            peak_y_calc = float(np.max(y))
            max_xr = float(np.max(x_r))
            max_xb = float(np.max(x_b))

            metrics = {
                "peak_y": peak_y_calc if peak_y is None else peak_y,
                "t_peak": float(sol.t[np.argmax(y)]),
                "max_xr": max_xr,
                "max_xb": max_xb,
                "extra_reach": float(peak_y_calc - X_R_MAX),
                "constraint_violation": float(max(0.0, max_xr - X_R_MAX)),
                "target_error": (
                    float(abs(peak_y_calc - y_target))
                    if y_target is not None
                    else np.nan
                ),
            }

        return {
            "peak_y": float(metrics.get("peak_y", np.max(y))),
            "solution": sol,
            "metrics": metrics,
        }

    try:
        output = simulate_system(
            params,
            T_sim=T_SIM,
            dt=DT,
            return_metrics=True,
            x_r_max=X_R_MAX,
            y_target=y_target,
        )

        if isinstance(output, tuple):
            peak_y = None
            metrics = None

            for item in output:
                if isinstance(item, dict) and "peak_y" in item:
                    metrics = item
                elif isinstance(item, (float, int, np.floating)):
                    peak_y = float(item)

            if metrics is not None:
                return {
                    "peak_y": float(metrics.get("peak_y", peak_y)),
                    "solution": None,
                    "metrics": metrics,
                }

        if isinstance(output, dict):
            return {
                "peak_y": float(output["peak_y"]),
                "solution": None,
                "metrics": output,
            }

    except TypeError:
        pass

    peak_y = simulate_system(
        params,
        T_sim=T_SIM,
        dt=DT,
    )

    metrics = {
        "peak_y": float(peak_y),
        "t_peak": np.nan,
        "max_xr": np.nan,
        "max_xb": np.nan,
        "extra_reach": float(peak_y - X_R_MAX),
        "constraint_violation": np.nan,
        "target_error": (
            float(abs(peak_y - y_target))
            if y_target is not None
            else np.nan
        ),
    }

    return {
        "peak_y": float(peak_y),
        "solution": None,
        "metrics": metrics,
    }


def predict_gp_physical_units(
    gp,
    scaler_X,
    scaler_y,
    params: np.ndarray,
) -> Tuple[float, float]:
    """
    Predict peak outreach and uncertainty with the trained GP.
    """
    X_scaled = scaler_X.transform([params])
    y_scaled, y_std_scaled = gp.predict(X_scaled, return_std=True)

    y_pred = scaler_y.inverse_transform([[y_scaled[0]]])[0, 0]
    y_std = y_std_scaled[0] * scaler_y.scale_[0]

    return float(y_pred), float(y_std)


def get_dataset_bounds_from_data(X_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return min/max bounds from the actual dataset.
    """
    return X_data.min(axis=0), X_data.max(axis=0)


# ============================================================================
# A) FINAL TRAJECTORY PLOT
# ============================================================================

def plot_final_trajectory(
    params: np.ndarray,
    y_target: float,
    save_path: str | Path = FINAL_TRAJECTORY_PATH,
) -> Dict:
    """
    Plot the final constraint-aware trajectory using the true simulator.

    The plot shows:
    - total outreach y(t)
    - robot relative displacement x_r(t)
    - base displacement x_b(t)
    - target
    - robot limit
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sim = simulate_with_metrics(
        params,
        y_target=y_target,
        return_solution=True,
    )

    sol = sim["solution"]
    metrics = sim["metrics"]

    if sol is None:
        raise RuntimeError("Simulation did not return a solution. Cannot plot trajectory.")

    t = sol.t
    x_b = sol.y[2]
    x_r = sol.y[3]
    y = x_b + x_r

    peak_y = float(metrics.get("peak_y", np.max(y)))
    max_xr = float(metrics.get("max_xr", np.max(x_r)))
    max_xb = float(metrics.get("max_xb", np.max(x_b)))
    violation = float(metrics.get("constraint_violation", max(0.0, max_xr - X_R_MAX)))
    extra_reach = float(metrics.get("extra_reach", peak_y - X_R_MAX))
    error = abs(peak_y - y_target)

    # Custom colors for clearer presentation.
    color_y = "tab:blue"
    color_xr = "tab:orange"
    color_xb = "tab:green"
    color_target = "black"
    color_limit = "tab:red"
    color_peak = "tab:purple"

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # ---------------------------------------------------------------------
    # Total outreach
    # ---------------------------------------------------------------------
    axes[0].plot(
        t,
        y,
        color=color_y,
        linewidth=2.2,
        label="Total outreach y(t) = x_b(t) + x_r(t)",
    )
    axes[0].axhline(
        y_target,
        color=color_target,
        linestyle=":",
        linewidth=2.2,
        label=f"Target = {y_target:.3f} m",
    )
    axes[0].axhline(
        X_R_MAX,
        color=color_limit,
        linestyle="--",
        linewidth=2.0,
        label="Nominal robot reach = 0.500 m",
    )
    axes[0].axhline(
        peak_y,
        color=color_peak,
        linestyle="-.",
        linewidth=2.0,
        label=f"Peak y = {peak_y:.3f} m",
    )
    axes[0].set_ylabel("y(t) [m]")
    axes[0].set_title("Final constraint-aware optimized trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")

    # ---------------------------------------------------------------------
    # Robot relative displacement
    # ---------------------------------------------------------------------
    axes[1].plot(
        t,
        x_r,
        color=color_xr,
        linewidth=2.0,
        label="Robot relative displacement x_r(t)",
    )
    axes[1].axhline(
        X_R_MAX,
        color=color_limit,
        linestyle="--",
        linewidth=2.0,
        label="Robot limit = 0.500 m",
    )
    axes[1].axhline(
        max_xr,
        color=color_peak,
        linestyle="-.",
        linewidth=2.0,
        label=f"max_xr = {max_xr:.3f} m",
    )
    axes[1].set_ylabel("x_r(t) [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc="lower right")

    # ---------------------------------------------------------------------
    # Passive base displacement
    # ---------------------------------------------------------------------
    axes[2].plot(
        t,
        x_b,
        color=color_xb,
        linewidth=2.0,
        label="Passive base displacement x_b(t)",
    )
    axes[2].axhline(
        max_xb,
        color=color_peak,
        linestyle="-.",
        linewidth=2.0,
        label=f"max_xb = {max_xb:.3f} m",
    )
    axes[2].axhline(
        0.0,
        color="gray",
        linestyle="-",
        linewidth=1.0,
        alpha=0.6,
    )
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("x_b(t) [m]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, loc="lower right")

    info_text = (
        f"target = {y_target:.3f} m\n"
        f"peak_y = {peak_y:.3f} m\n"
        f"error = {error * 1000:.2f} mm\n"
        f"extra reach = {extra_reach:.3f} m\n"
        f"max_xr = {max_xr:.3f} m\n"
        f"violation = {violation * 1000:.2f} mm"
    )

    fig.text(
        0.98,
        0.02,
        info_text,
        ha="right",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

    return metrics


# ============================================================================
# B) 2D SYSTEM ANIMATION
# ============================================================================

def draw_zigzag_spring(
    ax,
    x_start: float,
    x_end: float,
    y: float = 0.2,
    amplitude: float = 0.04,
    n_coils: int = 8,
    color: str = "forestgreen",
    linewidth: float = 2.5,
    zorder: int = 2,
) -> None:
    """Draw a zig-zag spring between two horizontal positions."""
    if x_end <= x_start + 0.02:
        ax.plot(
            [x_start, x_end],
            [y, y],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
        )
        return

    margin = 0.015
    xs = np.linspace(x_start + margin, x_end - margin, 2 * n_coils + 1)
    ys = np.full_like(xs, y)

    for i in range(1, len(xs) - 1):
        ys[i] = y + amplitude * (1 if i % 2 else -1)

    ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)


class SystemAnimator:
    """
    2D animator for the compliant-base robot system.

    The animation emphasizes:
        y(t) = x_b(t) + x_r(t)

    where x_r is the robot motion relative to the passive base.
    """

    def __init__(
        self,
        params: np.ndarray,
        y_target: float,
        T_sim: float = T_SIM,
        dt: float = DT,
    ):
        self.params = params
        self.y_target = float(y_target)
        self.T_sim = T_sim
        self.dt = dt

        print(f"Running simulation for animation: T={T_sim}s, dt={dt}s")

        sim = simulate_with_metrics(
            params,
            y_target=y_target,
            return_solution=True,
        )

        self.peak_y = float(sim["metrics"].get("peak_y", sim["peak_y"]))
        self.metrics = sim["metrics"]
        self.sol = sim["solution"]

        if self.sol is None:
            raise RuntimeError("Simulation did not return an ODE solution. Cannot animate.")

        self.t = self.sol.t
        self.x_b = self.sol.y[2]
        self.x_r = self.sol.y[3]
        self.y = self.x_b + self.x_r

        self.constraint_violation = float(
            self.metrics.get("constraint_violation", np.nan)
        )
        self.feasible = bool(
            np.isfinite(self.constraint_violation)
            and self.constraint_violation <= CONSTRAINT_TOLERANCE
        )

        print(f"Simulation completed: {len(self.t)} time steps")
        print(f"  Peak outreach:          {self.peak_y:.6f} m")
        print(f"  Max robot displacement: {np.max(self.x_r):.6f} m")
        print(f"  Constraint violation:   {self.constraint_violation * 1000:.3f} mm")
        print(f"  Feasible:               {self.feasible}")

    def create_animation(
        self,
        save_path: str | Path = FINAL_ANIMATION_PATH,
        fps: int = 30,
        duration: float = 12.0,
        figsize: Tuple[float, float] = (16, 9),
    ) -> None:
        """Create and save a GIF animation of the optimized response."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print("\nCreating animation")
        print(f"  FPS:      {fps}")
        print(f"  Duration: {duration:.1f} s")

        total_frames = int(fps * duration)
        frame_indices = np.linspace(0, len(self.t) - 1, total_frames, dtype=int)

        print(f"  Frames:   {total_frames} from {len(self.t)} simulation steps")

        x_min = -0.06
        x_max = max(self.y_target, self.peak_y, X_R_MAX) + 0.16
        y_axis_max = max(self.peak_y, self.y_target, X_R_MAX) + 0.06
        y_axis_min = min(-0.10, np.min(self.x_b) - 0.05)

        base_color = "#9467bd"
        robot_color = "#d62728"
        target_color = "#2ca02c"
        nominal_color = "#1f77b4"

        fig, (ax_system, ax_plot) = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [0.9, 1.35], "hspace": 0.34},
        )

        def update(frame_number: int):
            idx = frame_indices[frame_number]

            t_current = self.t[idx]
            x_b_current = self.x_b[idx]
            x_r_current = self.x_r[idx]
            y_current = self.y[idx]
            peak_so_far = np.max(self.y[:idx + 1])

            ax_system.clear()
            ax_system.set_xlim(x_min, x_max)
            ax_system.set_ylim(0.0, 0.42)
            ax_system.set_yticks([])
            ax_system.set_title(
                "Compliant-base robot: final constraint-aware response",
                fontsize=13,
                fontweight="bold",
            )
            ax_system.grid(True, alpha=0.25, axis="x")

            wall_x = 0.0
            base_pos = x_b_current
            robot_pos = x_b_current + x_r_current

            base_size = 0.045
            robot_size = 0.045
            y_level = 0.22

            wall = patches.Rectangle(
                (-0.0025, 0.03),
                0.005,
                0.34,
                facecolor="black",
                edgecolor="black",
                zorder=1,
            )
            ax_system.add_patch(wall)

            ax_system.plot(
                [x_min, x_max],
                [0.105, 0.105],
                color="black",
                linewidth=1.0,
                alpha=0.35,
            )

            draw_zigzag_spring(
                ax_system,
                wall_x,
                base_pos - base_size / 2,
                y=y_level,
                amplitude=0.030,
                n_coils=5,
                color="orange",
                linewidth=2.3,
            )

            base_rect = patches.Rectangle(
                (base_pos - base_size / 2, y_level - base_size / 2),
                base_size,
                base_size,
                linewidth=1.8,
                edgecolor="black",
                facecolor=base_color,
                alpha=0.85,
                zorder=5,
            )
            ax_system.add_patch(base_rect)
            ax_system.text(
                base_pos,
                0.075,
                "Base",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

            draw_zigzag_spring(
                ax_system,
                base_pos + base_size / 2,
                robot_pos - robot_size / 2,
                y=y_level,
                amplitude=0.035,
                n_coils=8,
                color="forestgreen",
                linewidth=2.3,
            )

            robot_rect = patches.Rectangle(
                (robot_pos - robot_size / 2, y_level - robot_size / 2),
                robot_size,
                robot_size,
                linewidth=1.8,
                edgecolor="black",
                facecolor=robot_color,
                alpha=0.85,
                zorder=5,
            )
            ax_system.add_patch(robot_rect)
            ax_system.text(
                robot_pos,
                0.075,
                "Robot",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

            ax_system.axvline(
                X_R_MAX,
                color=nominal_color,
                linestyle="--",
                linewidth=2.0,
                label="Nominal robot reach",
            )
            ax_system.axvline(
                self.y_target,
                color=target_color,
                linestyle=":",
                linewidth=2.2,
                label=f"Target = {self.y_target:.3f} m",
            )
            ax_system.axvline(
                y_current,
                color="black",
                linestyle="-",
                linewidth=1.6,
                alpha=0.75,
                label="Current y(t)",
            )
            ax_system.axvline(
                peak_so_far,
                color="orange",
                linestyle="-.",
                linewidth=1.8,
                label=f"Peak so far = {peak_so_far:.3f} m",
            )

            status = "FEASIBLE" if self.feasible else "CONSTRAINT VIOLATION"
            info_text = (
                f"t = {t_current:5.2f} s\n"
                f"x_b = {x_b_current: .4f} m\n"
                f"x_r = {x_r_current: .4f} m\n"
                f"y = x_b + x_r = {y_current: .4f} m\n"
                f"peak_y = {self.peak_y: .4f} m\n"
                f"violation = {self.constraint_violation * 1000: .2f} mm\n"
                f"{status}"
            )

            ax_system.text(
                0.015,
                0.96,
                info_text,
                transform=ax_system.transAxes,
                fontsize=9.5,
                va="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
            )

            ax_system.legend(loc="upper right", fontsize=8.5)

            ax_plot.clear()

            t_plot = self.t[:idx + 1]
            y_plot = self.y[:idx + 1]
            xb_plot = self.x_b[:idx + 1]
            xr_plot = self.x_r[:idx + 1]

            ax_plot.plot(t_plot, y_plot, linewidth=2.0, label="Total outreach y(t)")
            ax_plot.plot(t_plot, xr_plot, linewidth=1.5, label="Robot relative displacement x_r(t)")
            ax_plot.plot(t_plot, xb_plot, linewidth=1.5, label="Base displacement x_b(t)")

            ax_plot.axhline(
                X_R_MAX,
                color=nominal_color,
                linestyle="--",
                linewidth=2.0,
                label="Nominal reach",
            )
            ax_plot.axhline(
                self.y_target,
                color=target_color,
                linestyle=":",
                linewidth=2.0,
                label="Target",
            )
            ax_plot.axhline(
                self.peak_y,
                color="orange",
                linestyle="-.",
                linewidth=1.8,
                label="Final peak",
            )
            ax_plot.plot(t_current, y_current, "o", markersize=6, color="black")

            ax_plot.set_xlim(0.0, self.T_sim)
            ax_plot.set_ylim(y_axis_min, y_axis_max)
            ax_plot.set_xlabel("Time [s]")
            ax_plot.set_ylabel("Position [m]")
            ax_plot.set_title("Time response", pad=10)
            ax_plot.grid(True, alpha=0.3)
            ax_plot.legend(loc="upper right", fontsize=8.5)

            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=1000 / fps,
            blit=False,
            repeat=True,
        )

        fig.subplots_adjust(hspace=0.34)

        print("  Saving GIF. This may take a few minutes...")
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)

        plt.close(fig)

        file_size_kb = save_path.stat().st_size / 1024
        print(f"Saved: {save_path}")
        print(f"  File size: {file_size_kb:.1f} KB")


def animate_optimized_response(
    params: np.ndarray,
    y_target: float,
    save_path: str | Path,
    T_sim: float = T_SIM,
    dt: float = DT,
    fps: int = 30,
    duration: float = 12.0,
) -> None:
    """Create a GIF animation for one optimized response."""
    animator = SystemAnimator(
        params=params,
        y_target=y_target,
        T_sim=T_sim,
        dt=dt,
    )
    animator.create_animation(
        save_path=save_path,
        fps=fps,
        duration=duration,
    )


# ============================================================================
# C) GP RESPONSE SURFACE
# ============================================================================

def plot_gp_response_surface(
    gp,
    scaler_X,
    scaler_y,
    X_data: np.ndarray,
    y_data: np.ndarray,
    fixed_values: Dict[str, float],
    x_param: str = "Kr",
    y_param: str = "hr",
    optimal_point: Optional[Dict[str, float]] = None,
    resolution: int = 70,
    save_path: str | Path = GP_SURFACE_PATH,
) -> None:
    """
    Plot a 2D GP response surface.

    Two parameters are varied on a grid while all other parameters are fixed
    to the selected optimized solution.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating GP response surface: {x_param} vs {y_param}")

    x_idx = PARAM_ORDER.index(x_param)
    y_idx = PARAM_ORDER.index(y_param)

    lb, ub = get_dataset_bounds_from_data(X_data)

    x_range = np.linspace(lb[x_idx], ub[x_idx], resolution)
    y_range = np.linspace(lb[y_idx], ub[y_idx], resolution)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    grid_points = []

    for i in range(resolution):
        for j in range(resolution):
            point = np.zeros(len(PARAM_ORDER))

            for k, name in enumerate(PARAM_ORDER):
                if name == x_param:
                    point[k] = X_grid[i, j]
                elif name == y_param:
                    point[k] = Y_grid[i, j]
                else:
                    point[k] = fixed_values[name]

            grid_points.append(point)

    grid_points = np.asarray(grid_points)

    print(f"  Predicting {len(grid_points)} grid points...")
    X_scaled = scaler_X.transform(grid_points)
    y_scaled, y_std_scaled = gp.predict(X_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * scaler_y.scale_[0]

    Z_pred = y_pred.reshape(resolution, resolution)
    Z_std = y_std.reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    contour = ax.contourf(
        X_grid,
        Y_grid,
        Z_pred,
        levels=24,
        cmap="viridis",
        alpha=0.92,
    )
    lines = ax.contour(
        X_grid,
        Y_grid,
        Z_pred,
        levels=10,
        colors="white",
        linewidths=0.5,
        alpha=0.55,
    )
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.3f")

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Predicted peak outreach [m]")

    ax.scatter(
        X_data[:, x_idx],
        X_data[:, y_idx],
        c=y_data,
        cmap="viridis",
        s=14,
        edgecolors="none",
        alpha=0.28,
        label="Dataset samples",
    )

    if optimal_point is not None:
        ax.scatter(
            optimal_point[x_param],
            optimal_point[y_param],
            s=220,
            marker="*",
            color="red",
            edgecolors="white",
            linewidth=1.8,
            label="Selected optimum",
            zorder=10,
        )

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"GP response surface: {x_param} vs {y_param}")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    ax = axes[1]
    contour_std = ax.contourf(
        X_grid,
        Y_grid,
        Z_std * 1000.0,
        levels=24,
        cmap="magma",
        alpha=0.92,
    )
    cbar_std = fig.colorbar(contour_std, ax=ax)
    cbar_std.set_label("GP predictive standard deviation [mm]")

    ax.scatter(
        X_data[:, x_idx],
        X_data[:, y_idx],
        s=12,
        color="white",
        alpha=0.18,
        edgecolors="none",
        label="Dataset samples",
    )

    if optimal_point is not None:
        ax.scatter(
            optimal_point[x_param],
            optimal_point[y_param],
            s=220,
            marker="*",
            color="cyan",
            edgecolors="black",
            linewidth=1.4,
            label="Selected optimum",
            zorder=10,
        )

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title("GP uncertainty over the same plane")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    fixed_text = "Fixed values from selected solution:\n" + "\n".join(
        f"{name} = {value:.4g}"
        for name, value in fixed_values.items()
        if name not in [x_param, y_param]
    )

    fig.text(
        0.5,
        0.01,
        fixed_text,
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    plt.suptitle(
        "Gaussian Process response and uncertainty",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================================
# D) SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis(
    gp,
    scaler_X,
    scaler_y,
    nominal_params: np.ndarray,
    X_data: np.ndarray,
    params_to_vary: List[str] = ["Mb", "hr", "f1", "A", "Kb", "Kr"],
    n_points: int = 35,
    validate_points: int = 5,
    save_plot: str | Path = SENSITIVITY_PLOT_PATH,
    save_csv: str | Path = SENSITIVITY_CSV_PATH,
) -> pd.DataFrame:
    """
    Perform one-at-a-time sensitivity analysis around the final solution.
    """
    save_plot = Path(save_plot)
    save_csv = Path(save_csv)
    save_plot.parent.mkdir(parents=True, exist_ok=True)
    save_csv.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ONE-AT-A-TIME SENSITIVITY ANALYSIS")
    print("=" * 80)

    print("Nominal selected solution:")
    for name, value in zip(PARAM_ORDER, nominal_params):
        print(f"  {name:12s} = {value:.6f}")

    y_nom, y_nom_std = predict_gp_physical_units(
        gp,
        scaler_X,
        scaler_y,
        nominal_params,
    )
    print(f"\nNominal GP prediction: {y_nom:.6f} m ± {y_nom_std:.6f} m")

    lb, ub = get_dataset_bounds_from_data(X_data)

    results = []

    n_params = len(params_to_vary)
    n_rows = int(np.ceil(n_params / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.2 * n_rows))
    axes = np.ravel(axes)

    for plot_idx, param_name in enumerate(params_to_vary):
        print(f"\nVarying parameter: {param_name}")

        param_idx = PARAM_ORDER.index(param_name)
        param_values = np.linspace(lb[param_idx], ub[param_idx], n_points)

        y_pred_values = []
        y_std_values = []

        for value in param_values:
            varied_params = nominal_params.copy()
            varied_params[param_idx] = value

            y_pred, y_std = predict_gp_physical_units(
                gp,
                scaler_X,
                scaler_y,
                varied_params,
            )

            y_pred_values.append(y_pred)
            y_std_values.append(y_std)

            results.append(
                {
                    "parameter": param_name,
                    "value": value,
                    "y_pred_gp": y_pred,
                    "y_std_gp": y_std,
                    "is_validation_point": False,
                    "y_sim": np.nan,
                    "constraint_violation": np.nan,
                    "feasible": np.nan,
                }
            )

        y_pred_values = np.asarray(y_pred_values)
        y_std_values = np.asarray(y_std_values)

        validation_indices = np.linspace(
            0,
            n_points - 1,
            validate_points,
            dtype=int,
        )

        validation_values = []
        validation_sim = []
        validation_feasible = []

        for vi in validation_indices:
            varied_params = nominal_params.copy()
            varied_params[param_idx] = param_values[vi]

            sim = simulate_with_metrics(
                varied_params,
                y_target=None,
                return_solution=False,
            )

            metrics = sim["metrics"]
            y_sim = float(metrics.get("peak_y", sim["peak_y"]))
            violation = float(metrics.get("constraint_violation", np.nan))
            feasible = bool(
                np.isfinite(violation)
                and violation <= CONSTRAINT_TOLERANCE
            )

            validation_values.append(param_values[vi])
            validation_sim.append(y_sim)
            validation_feasible.append(feasible)

            results.append(
                {
                    "parameter": param_name,
                    "value": param_values[vi],
                    "y_pred_gp": np.nan,
                    "y_std_gp": np.nan,
                    "is_validation_point": True,
                    "y_sim": y_sim,
                    "constraint_violation": violation,
                    "feasible": feasible,
                }
            )

        ax = axes[plot_idx]

        ax.plot(
            param_values,
            y_pred_values,
            linewidth=2.0,
            label="GP prediction",
        )
        ax.fill_between(
            param_values,
            y_pred_values - 1.96 * y_std_values,
            y_pred_values + 1.96 * y_std_values,
            alpha=0.22,
            label="95% predictive interval",
        )

        validation_values = np.asarray(validation_values)
        validation_sim = np.asarray(validation_sim)
        validation_feasible = np.asarray(validation_feasible, dtype=bool)

        if np.any(validation_feasible):
            ax.scatter(
                validation_values[validation_feasible],
                validation_sim[validation_feasible],
                s=75,
                c="green",
                edgecolors="black",
                linewidth=1.0,
                label="True simulation, feasible",
                zorder=5,
            )

        if np.any(~validation_feasible):
            ax.scatter(
                validation_values[~validation_feasible],
                validation_sim[~validation_feasible],
                s=75,
                c="red",
                edgecolors="black",
                linewidth=1.0,
                label="True simulation, infeasible",
                zorder=5,
            )

        ax.axvline(
            nominal_params[param_idx],
            linestyle="--",
            linewidth=1.8,
            label="Selected value",
        )
        ax.axhline(
            y_nom,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label="Nominal GP output",
        )
        ax.axhline(
            X_R_MAX,
            linestyle="-.",
            linewidth=1.5,
            alpha=0.8,
            label="Nominal reach",
        )

        ax.set_xlabel(param_name)
        ax.set_ylabel("Peak outreach [m]")
        ax.set_title(f"Sensitivity to {param_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        "One-at-a-time sensitivity analysis around final constraint-aware solution",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_plot, dpi=300, bbox_inches="tight")
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv, index=False)

    print(f"\nSaved: {save_plot}")
    print(f"Saved: {save_csv}")

    return results_df


# ============================================================================
# E) MONTE CARLO ROBUSTNESS
# ============================================================================

def monte_carlo_robustness(
    optimal_params: np.ndarray,
    y_target: float,
    n_samples: int = 100,
    noise_level: float = 0.02,   #0.05
    params_to_perturb: List[str] = ["Kb", "Mb", "hb"],
    #save_plot: str | Path = MONTE_CARLO_PLOT_PATH,
    #save_csv: str | Path = MONTE_CARLO_CSV_PATH,
    save_plot=FIGURES_DIR / "robustness" / "monte_carlo_constraint_aware_target064_noise2.png",
    save_csv=RESULTS_DIR / "robustness" / "monte_carlo_constraint_aware_target064_noise2.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run a Monte Carlo robustness test around the final optimized solution.

    Selected physical parameters are randomly perturbed using Gaussian
    multiplicative noise:
        p_perturbed = p * (1 + epsilon), epsilon ~ N(0, noise_level)
    """
    save_plot = Path(save_plot)
    save_csv = Path(save_csv)
    save_plot.parent.mkdir(parents=True, exist_ok=True)
    save_csv.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("MONTE CARLO ROBUSTNESS TEST")
    print("=" * 80)

    print("Selected optimized parameters:")
    for name, value in zip(PARAM_ORDER, optimal_params):
        marker = " (*)" if name in params_to_perturb else ""
        print(f"  {name:12s} = {value:.6f}{marker}")

    print(f"\nTarget:               {y_target:.6f} m")
    print(f"Noise standard dev.:  {noise_level * 100:.1f}%")
    print(f"Samples:              {n_samples}")
    print(f"Perturbed parameters: {params_to_perturb}")

    rng = np.random.default_rng(seed)
    perturb_indices = [PARAM_ORDER.index(p) for p in params_to_perturb]

    rows = []

    for sample_idx in tqdm(range(n_samples), desc="Monte Carlo"):
        params_perturbed = optimal_params.copy()

        noise_values = {}

        for idx in perturb_indices:
            epsilon = rng.normal(loc=0.0, scale=noise_level)
            params_perturbed[idx] *= 1.0 + epsilon
            noise_values[PARAM_ORDER[idx]] = epsilon

        sim = simulate_with_metrics(
            params_perturbed,
            y_target=y_target,
            return_solution=False,
        )

        metrics = sim["metrics"]
        y_sim = float(metrics.get("peak_y", sim["peak_y"]))
        violation = float(metrics.get("constraint_violation", np.nan))
        feasible = bool(
            np.isfinite(violation)
            and violation <= CONSTRAINT_TOLERANCE
        )
        error = abs(y_sim - y_target)

        row = {
            "sample_index": sample_idx,
            "y_achieved": y_sim,
            "error": error,
            "extra_reach": float(metrics.get("extra_reach", y_sim - X_R_MAX)),
            "max_xr": float(metrics.get("max_xr", np.nan)),
            "max_xb": float(metrics.get("max_xb", np.nan)),
            "constraint_violation": violation,
            "feasible": feasible,
        }

        for name, epsilon in noise_values.items():
            row[f"noise_{name}"] = epsilon

        for name, value in zip(PARAM_ORDER, params_perturbed):
            row[f"param_{name}"] = value

        rows.append(row)

    results_df = pd.DataFrame(rows)

    y_achieved = results_df["y_achieved"].values
    errors = results_df["error"].values
    violations = results_df["constraint_violation"].values

    stats = {
        "target": y_target,
        "n_samples": n_samples,
        "noise_level": noise_level,
        "mean_y": float(np.mean(y_achieved)),
        "std_y": float(np.std(y_achieved)),
        "min_y": float(np.min(y_achieved)),
        "max_y": float(np.max(y_achieved)),
        "q05_y": float(np.percentile(y_achieved, 5)),
        "q95_y": float(np.percentile(y_achieved, 95)),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "success_rate_10mm": float(np.mean(errors <= 0.010)),
        "feasibility_rate": float(np.mean(results_df["feasible"])),
        "mean_constraint_violation": float(np.nanmean(violations)),
        "max_constraint_violation": float(np.nanmax(violations)),
    }

    print("\nMonte Carlo summary:")
    print(f"  Target:                   {y_target:.6f} m")
    print(f"  Mean achieved:            {stats['mean_y']:.6f} m")
    print(f"  Std achieved:             {stats['std_y']:.6f} m")
    print(f"  5th-95th percentile:      [{stats['q05_y']:.6f}, {stats['q95_y']:.6f}] m")
    print(f"  Mean absolute error:      {stats['mean_error'] * 1000:.2f} mm")
    print(f"  Max absolute error:       {stats['max_error'] * 1000:.2f} mm")
    print(f"  Success rate <= 10 mm:    {stats['success_rate_10mm'] * 100:.1f}%")
    print(f"  Feasibility rate:         {stats['feasibility_rate'] * 100:.1f}%")
    print(f"  Max violation:            {stats['max_constraint_violation'] * 1000:.2f} mm")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.hist(y_achieved, bins=28, edgecolor="black", alpha=0.75)
    ax.axvline(y_target, linestyle="--", linewidth=2.2, label="Target")
    ax.axvline(stats["mean_y"], linestyle="-", linewidth=2.0, label="Mean achieved")
    ax.axvspan(
        stats["q05_y"],
        stats["q95_y"],
        alpha=0.18,
        label="5th-95th percentile",
    )
    ax.set_xlabel("Achieved peak outreach [m]")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of achieved outreach")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(errors * 1000.0, bins=28, edgecolor="black", alpha=0.75)
    ax.axvline(10.0, linestyle="--", linewidth=2.0, label="10 mm tolerance")
    ax.axvline(
        stats["mean_error"] * 1000.0,
        linestyle="-",
        linewidth=2.0,
        label="Mean error",
    )
    ax.set_xlabel("Absolute target error [mm]")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of target errors")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(y_achieved, linewidth=1.3, alpha=0.75, label="Achieved outreach")
    ax.axhline(y_target, linestyle="--", linewidth=2.0, label="Target")
    ax.fill_between(
        np.arange(n_samples),
        y_target - 0.010,
        y_target + 0.010,
        alpha=0.18,
        label="±10 mm band",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Achieved peak outreach [m]")
    ax.set_title("Monte Carlo sample sequence")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.hist(violations * 1000.0, bins=24, edgecolor="black", alpha=0.75)
    ax.axvline(
        CONSTRAINT_TOLERANCE * 1000.0,
        linestyle="--",
        linewidth=2.0,
        label="Constraint tolerance",
    )
    ax.set_xlabel("Constraint violation [mm]")
    ax.set_ylabel("Frequency")
    ax.set_title("Constraint violation under perturbations")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax.text(
        0.97,
        0.92,
        f"Feasibility rate = {stats['feasibility_rate'] * 100:.1f}%\n"
        f"Max violation = {stats['max_constraint_violation'] * 1000:.2f} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    stats_text = (
        f"mean y = {stats['mean_y']:.4f} m\n"
        f"std y = {stats['std_y']:.4f} m\n"
        f"mean error = {stats['mean_error'] * 1000:.2f} mm\n"
        f"success <=10mm = {stats['success_rate_10mm'] * 100:.1f}%\n"
        f"feasible = {stats['feasibility_rate'] * 100:.1f}%"
    )

    fig.text(
        0.98,
        0.02,
        stats_text,
        ha="right",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    plt.suptitle(
        f"Monte Carlo robustness test around final solution "
        f"(n={n_samples}, noise std={noise_level * 100:.0f}%)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_plot, dpi=300, bbox_inches="tight")
    plt.close()

    results_df.to_csv(save_csv, index=False)

    stats_path = save_csv.with_name(save_csv.stem + "_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    print(f"Saved: {save_plot}")
    print(f"Saved: {save_csv}")
    print(f"Saved: {stats_path}")

    return results_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    ensure_output_dirs()

    print("\n" + "=" * 80)
    print("ADVANCED VISUALIZATIONS - FINAL CONSTRAINT-AWARE SOLUTION")
    print("=" * 80)

    print("\nLoading model, dataset, and final solution...")

    gp, scaler_X, scaler_y = load_model(MODEL_DIR)
    X_data, y_data = load_dataset(str(DATASET_PATH))
    df_data = load_dataset_dataframe(str(DATASET_PATH))

    final_solution_df = load_final_constraint_solution(FINAL_SOLUTION_PATH)
    selected_row = final_solution_df.iloc[0]

    y_target = float(selected_row["target"])
    optimal_params = extract_params_from_constraint_row(selected_row)

    print(f"Dataset loaded: {DATASET_PATH}")
    print(f"  Samples: {len(X_data)}")
    print(f"Final solution loaded: {FINAL_SOLUTION_PATH}")

    print("\nSelected constraint-aware solution:")
    print(f"  Method:                  {selected_row['method']}")
    print(f"  Target:                  {y_target:.6f} m")
    print(f"  y_sim achieved:           {selected_row['y_sim']:.6f} m")
    print(f"  Simulation error:         {selected_row['error_sim'] * 1000:.3f} mm")
    print(f"  Extra reach:              {selected_row['extra_reach']:.6f} m")
    print(f"  max_xr:                   {selected_row['max_xr_sim']:.6f} m")
    print(f"  max_xb:                   {selected_row['max_xb_sim']:.6f} m")
    print(f"  Constraint violation:     {selected_row['constraint_violation'] * 1000:.3f} mm")
    print(f"  Feasible:                 {selected_row['feasible']}")

    print("\nOptimized parameters:")
    for name, value in zip(PARAM_ORDER, optimal_params):
        print(f"  {name:12s} = {value:.6f}")

    # ------------------------------------------------------------------------
    # A) Final trajectory
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("A) FINAL TRAJECTORY")
    print("=" * 80)

    final_metrics = plot_final_trajectory(
        params=optimal_params,
        y_target=y_target,
        save_path=FINAL_TRAJECTORY_PATH,
    )

    # ------------------------------------------------------------------------
    # B) Animation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("B) SYSTEM ANIMATION")
    print("=" * 80)

    animate_optimized_response(
        params=optimal_params,
        y_target=y_target,
        save_path=FINAL_ANIMATION_PATH,
        T_sim=T_SIM,
        dt=DT,
        fps=30,
        duration=20.0,
    )

    # ------------------------------------------------------------------------
    # C) GP response surface
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("C) GP RESPONSE SURFACE")
    print("=" * 80)

    fixed_values = {
        name: optimal_params[i]
        for i, name in enumerate(PARAM_ORDER)
        if name not in ["Kr", "hr"]
    }

    optimal_point = {
        "Kr": optimal_params[PARAM_ORDER.index("Kr")],
        "hr": optimal_params[PARAM_ORDER.index("hr")],
        "y_target": y_target,
    }

    plot_gp_response_surface(
        gp=gp,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        X_data=X_data,
        y_data=y_data,
        fixed_values=fixed_values,
        x_param="Kr",
        y_param="hr",
        optimal_point=optimal_point,
        resolution=70,
        save_path=GP_SURFACE_PATH,
    )

    # ------------------------------------------------------------------------
    # D) Sensitivity analysis
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("D) SENSITIVITY ANALYSIS")
    print("=" * 80)

    sensitivity_df = sensitivity_analysis(
        gp=gp,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        nominal_params=optimal_params,
        X_data=X_data,
        params_to_vary=["Mb", "hr", "f1", "A", "Kb", "Kr"],
        n_points=35,
        validate_points=5,
        save_plot=SENSITIVITY_PLOT_PATH,
        save_csv=SENSITIVITY_CSV_PATH,
    )

    # ------------------------------------------------------------------------
    # E) Monte Carlo robustness
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("E) MONTE CARLO ROBUSTNESS")
    print("=" * 80)

    monte_carlo_df = monte_carlo_robustness(
        optimal_params=optimal_params,
        y_target=y_target,
        n_samples=100,
        noise_level=0.02,  #0.05
        params_to_perturb=["Kb", "Mb", "hb"],
        #save_plot=MONTE_CARLO_PLOT_PATH,
        #save_csv=MONTE_CARLO_CSV_PATH,
        save_plot=FIGURES_DIR / "robustness" / "monte_carlo_constraint_aware_target064_noise2.png",
        save_csv=RESULTS_DIR / "robustness" / "monte_carlo_constraint_aware_target064_noise2.csv",
        seed=42,
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------
    monte_carlo_stats_path = MONTE_CARLO_CSV_PATH.with_name(
        MONTE_CARLO_CSV_PATH.stem + "_stats.csv"
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION PIPELINE COMPLETED")
    print("=" * 80)

    print("\nGenerated files:")
    print(f"  Final trajectory:           {FINAL_TRAJECTORY_PATH}")
    print(f"  Animation:                  {FINAL_ANIMATION_PATH}")
    print(f"  GP response surface:         {GP_SURFACE_PATH}")
    print(f"  Sensitivity plot:            {SENSITIVITY_PLOT_PATH}")
    print(f"  Sensitivity CSV:             {SENSITIVITY_CSV_PATH}")
    print(f"  Monte Carlo plot:            {MONTE_CARLO_PLOT_PATH}")
    print(f"  Monte Carlo CSV:             {MONTE_CARLO_CSV_PATH}")
    print(f"  Monte Carlo stats CSV:       {monte_carlo_stats_path}")

    print("\nRecommended figures for report/presentation:")
    print(f"  1. {FINAL_TRAJECTORY_PATH.name}")
    print(f"  2. {FINAL_ANIMATION_PATH.name}")
    print(f"  3. {GP_SURFACE_PATH.name}")
    print(f"  4. {SENSITIVITY_PLOT_PATH.name}")
    print(f"  5. {MONTE_CARLO_PLOT_PATH.name}")

    print("=" * 80 + "\n")