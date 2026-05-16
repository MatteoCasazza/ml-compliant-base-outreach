"""
visualization.py
================
Advanced visualizations for the extra-reaching robot-base system.

Modules
-------
1. 2D animation of the optimized physical response
2. Gaussian Process response surface
3. One-at-a-time sensitivity analysis
4. Monte Carlo robustness test

This script is designed to work with the final project pipeline:
    dynamics.py
    dataset.py
    models.py
    optimization.py
    visualization.py

Expected input files
--------------------
- data/dataset_augmented.csv
- results/gp_model.pkl
- results/scaler_X.pkl
- results/scaler_y.pkl
- results/inverse_results.csv

Generated outputs
-----------------
- figures/animation_target*.gif
- figures/gp_surface_Kr_hr.png
- figures/sensitivity_analysis.png
- figures/monte_carlo_robustness.png
- results/sensitivity_analysis.csv
- results/monte_carlo_robustness.csv
- results/monte_carlo_robustness_stats.csv

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
from models import load_model


# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

DATASET_PATH = "data/dataset_augmented.csv"
INVERSE_RESULTS_PATH = "results/inverse_results.csv"
MODEL_DIR = "results"

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

X_R_MAX = 0.5
CONSTRAINT_TOLERANCE = 0.002

T_SIM = 60.0
DT = 0.001

PARAM_ORDER = ParameterRanges.get_param_names()


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def ensure_output_dirs() -> None:
    """Create output folders if they do not exist."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


def load_final_inverse_results(filepath: str = INVERSE_RESULTS_PATH) -> pd.DataFrame:
    """
    Load inverse optimization results.

    The file is produced by optimization.py and contains one optimized solution
    for each target outreach.
    """
    df = pd.read_csv(filepath)

    required_cols = {
        "y_target",
        "y_sim",
        "error_sim",
        "feasible",
        "param_Kb",
        "param_Kr",
        "param_Mb",
        "param_hb",
        "param_hr",
        "param_f0",
        "param_f1",
        "param_A",
        "param_x_r_start",
    }

    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {filepath}: {sorted(missing)}"
        )

    return df


def extract_params_from_inverse_row(row: pd.Series) -> np.ndarray:
    """
    Extract full parameter vector from one inverse optimization row.

    Output order:
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
    """
    return np.array([
        row["param_Kb"],
        row["param_Kr"],
        row["param_Mb"],
        row["param_hb"],
        row["param_hr"],
        row["param_f0"],
        row["param_f1"],
        row["param_A"],
        row["param_x_r_start"],
    ], dtype=float)


def simulate_with_metrics(
    params: np.ndarray,
    y_target: Optional[float] = None,
    return_solution: bool = False,
) -> Dict:
    """
    Run the dynamic simulation and return physical metrics.

    This helper supports both the updated dynamics.py interface and older
    versions that return only peak_y or peak_y plus solution.
    """
    try:
        if return_solution:
            peak_y, sol, metrics = simulate_system(
                params,
                T_sim=T_SIM,
                dt=DT,
                return_solution=True,
                return_metrics=True,
                x_r_max=X_R_MAX,
                y_target=y_target,
            )
            return {"peak_y": peak_y, "solution": sol, "metrics": metrics}

        peak_y, metrics = simulate_system(
            params,
            T_sim=T_SIM,
            dt=DT,
            return_metrics=True,
            x_r_max=X_R_MAX,
            y_target=y_target,
        )
        return {"peak_y": peak_y, "solution": None, "metrics": metrics}

    except TypeError:
        # Backward compatibility with an older dynamics.py interface.
        if return_solution:
            peak_y, sol = simulate_system(
                params,
                T_sim=T_SIM,
                dt=DT,
                return_full=True,
            )
            x_b = sol.y[2]
            x_r = sol.y[3]
            y = x_b + x_r

            metrics = {
                "peak_y": float(np.max(y)),
                "t_peak": float(sol.t[np.argmax(y)]),
                "max_xr": float(np.max(x_r)),
                "max_xb": float(np.max(x_b)),
                "extra_reach": float(np.max(y) - X_R_MAX),
                "constraint_violation": float(max(0.0, np.max(x_r) - X_R_MAX)),
                "target_error": float(abs(np.max(y) - y_target)) if y_target is not None else np.nan,
            }
            return {"peak_y": peak_y, "solution": sol, "metrics": metrics}

        peak_y = simulate_system(params, T_sim=T_SIM, dt=DT)
        metrics = {
            "peak_y": float(peak_y),
            "t_peak": np.nan,
            "max_xr": np.nan,
            "max_xb": np.nan,
            "extra_reach": float(peak_y - X_R_MAX),
            "constraint_violation": np.nan,
            "target_error": float(abs(peak_y - y_target)) if y_target is not None else np.nan,
        }
        return {"peak_y": peak_y, "solution": None, "metrics": metrics}


def predict_gp_physical_units(gp, scaler_X, scaler_y, params: np.ndarray) -> Tuple[float, float]:
    """
    Predict peak outreach and uncertainty with the trained GP.

    Returns
    -------
    y_pred : float
        Predicted peak outreach [m].
    y_std : float
        Predictive standard deviation [m].
    """
    X_scaled = scaler_X.transform([params])
    y_scaled, y_std_scaled = gp.predict(X_scaled, return_std=True)

    y_pred = scaler_y.inverse_transform([[y_scaled[0]]])[0, 0]
    y_std = y_std_scaled[0] * scaler_y.scale_[0]

    return float(y_pred), float(y_std)


def get_dataset_bounds_from_data(X_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return min/max bounds from the actual loaded dataset.

    This is safer than relying on a hard-coded ParameterRanges object when
    the final dataset is augmented and partially targeted.
    """
    return X_data.min(axis=0), X_data.max(axis=0)


# ============================================================================
# A) 2D SYSTEM ANIMATION
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
        ax.plot([x_start, x_end], [y, y], color=color, linewidth=linewidth, zorder=zorder)
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

    Schematic representation:
        wall -- base spring -- passive base -- robot impedance -- end-effector

    The animation emphasizes the key project idea:
        y(t) = x_b(t) + x_r(t)

    where the total outreach can exceed the nominal robot reach while the robot
    relative position remains below x_r_max.
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

        self.constraint_violation = float(self.metrics.get("constraint_violation", np.nan))
        self.feasible = bool(
            np.isfinite(self.constraint_violation)
            and self.constraint_violation <= CONSTRAINT_TOLERANCE
        )

        print(f"Simulation completed: {len(self.t)} time steps")
        print(f"  Peak outreach:          {self.peak_y:.6f} m")
        print(f"  Max robot position x_r: {np.max(self.x_r):.6f} m")
        print(f"  Constraint violation:   {self.constraint_violation * 1000:.3f} mm")
        print(f"  Feasible:               {self.feasible}")

    def create_animation(
        self,
        save_path: str = "figures/animation_optimized_response.gif",
        fps: int = 30,
        duration: float = 12.0,
        figsize: Tuple[float, float] = (16, 9),
    ) -> None:
        """Create and save a GIF animation of the optimized response."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        print("\nCreating animation")
        print(f"  FPS:      {fps}")
        print(f"  Duration: {duration:.1f} s")

        total_frames = int(fps * duration)
        frame_indices = np.linspace(0, len(self.t) - 1, total_frames, dtype=int)

        print(f"  Frames:   {total_frames} from {len(self.t)} simulation steps")

        # Use a fixed horizontal scale for readability.
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

            # ----------------------------------------------------------------
            # Physical schematic
            # ----------------------------------------------------------------
            ax_system.clear()
            ax_system.set_xlim(x_min, x_max)
            ax_system.set_ylim(0.0, 0.42)
            ax_system.set_yticks([])
            ax_system.set_xlabel("")
            ax_system.set_title(
                "Compliant-base robot: optimized extra-reaching response",
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

            # Wall.
            wall = patches.Rectangle(
                (-0.0025, 0.03),
                0.005,
                0.34,
                facecolor="black",
                edgecolor="black",
                zorder=1,
            )
            ax_system.add_patch(wall)

            # Ground/reference line.
            ax_system.plot([x_min, x_max], [0.105, 0.105], color="black", linewidth=1.0, alpha=0.35)

            # Base spring.
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

            # Passive base mass.
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
            ax_system.text(base_pos, 0.075, "Base", ha="center", fontsize=10, fontweight="bold")

            # Robot impedance spring.
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

            # Robot/end-effector mass.
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
            ax_system.text(robot_pos, 0.075, "Robot", ha="center", fontsize=10, fontweight="bold")

            # Reference lines.
            ax_system.axvline(X_R_MAX, color=nominal_color, linestyle="--", linewidth=2.0, label="Nominal robot reach")
            ax_system.axvline(self.y_target, color=target_color, linestyle=":", linewidth=2.2, label=f"Target = {self.y_target:.3f} m")
            ax_system.axvline(y_current, color="black", linestyle="-", linewidth=1.6, alpha=0.75, label="Current y(t)")
            ax_system.axvline(peak_so_far, color="orange", linestyle="-.", linewidth=1.8, label=f"Peak so far = {peak_so_far:.3f} m")

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

            # ----------------------------------------------------------------
            # Time response plot
            # ----------------------------------------------------------------
            ax_plot.clear()

            t_plot = self.t[:idx + 1]
            y_plot = self.y[:idx + 1]
            xb_plot = self.x_b[:idx + 1]
            xr_plot = self.x_r[:idx + 1]

            ax_plot.plot(t_plot, y_plot, linewidth=2.0, label="Total outreach y(t)")
            ax_plot.plot(t_plot, xr_plot, linewidth=1.5, label="Robot relative position x_r(t)")
            ax_plot.plot(t_plot, xb_plot, linewidth=1.5, label="Base displacement x_b(t)")

            ax_plot.axhline(X_R_MAX, color=nominal_color, linestyle="--", linewidth=2.0, label="Nominal reach")
            ax_plot.axhline(self.y_target, color=target_color, linestyle=":", linewidth=2.0, label="Target")
            ax_plot.axhline(self.peak_y, color="orange", linestyle="-.", linewidth=1.8, label="Final peak")
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

        file_size_kb = Path(save_path).stat().st_size / 1024
        print(f"Saved: {save_path}")
        print(f"  File size: {file_size_kb:.1f} KB")


def animate_optimized_response(
    params: np.ndarray,
    y_target: float,
    save_path: str,
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
# B) GP RESPONSE SURFACE
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
    save_path: str = "figures/gp_surface_Kr_hr.png",
) -> None:
    """
    Plot a 2D GP response surface.

    Two parameters are varied on a grid while all other parameters are fixed
    to the selected optimized solution.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

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

    # ------------------------------------------------------------------------
    # Predicted outreach surface
    # ------------------------------------------------------------------------
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

    scatter = ax.scatter(
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

    # ------------------------------------------------------------------------
    # Predictive uncertainty surface
    # ------------------------------------------------------------------------
    ax = axes[1]
    contour_std = ax.contourf(
        X_grid,
        Y_grid,
        Z_std * 1000,
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
        f"{name} = {value:.4g}" for name, value in fixed_values.items()
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

    plt.suptitle("Gaussian Process response and uncertainty", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================================
# C) SENSITIVITY ANALYSIS
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
    save_plot: str = "figures/sensitivity_analysis.png",
    save_csv: str = "results/sensitivity_analysis.csv",
) -> pd.DataFrame:
    """
    Perform one-at-a-time sensitivity analysis around a selected solution.

    For each selected parameter, the GP prediction is evaluated along the
    observed dataset range while all other parameters are held fixed. A few
    points are also validated with the true simulation.
    """
    Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ONE-AT-A-TIME SENSITIVITY ANALYSIS")
    print("=" * 80)

    print("Nominal selected solution:")
    for name, value in zip(PARAM_ORDER, nominal_params):
        print(f"  {name:12s} = {value:.6f}")

    y_nom, y_nom_std = predict_gp_physical_units(gp, scaler_X, scaler_y, nominal_params)
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

            results.append({
                "parameter": param_name,
                "value": value,
                "y_pred_gp": y_pred,
                "y_std_gp": y_std,
                "is_validation_point": False,
                "y_sim": np.nan,
                "constraint_violation": np.nan,
                "feasible": np.nan,
            })

        y_pred_values = np.asarray(y_pred_values)
        y_std_values = np.asarray(y_std_values)

        validation_indices = np.linspace(0, n_points - 1, validate_points, dtype=int)
        validation_values = []
        validation_sim = []
        validation_feasible = []

        for vi in validation_indices:
            varied_params = nominal_params.copy()
            varied_params[param_idx] = param_values[vi]

            sim = simulate_with_metrics(varied_params, y_target=None, return_solution=False)
            metrics = sim["metrics"]

            y_sim = float(metrics.get("peak_y", sim["peak_y"]))
            violation = float(metrics.get("constraint_violation", np.nan))
            feasible = bool(np.isfinite(violation) and violation <= CONSTRAINT_TOLERANCE)

            validation_values.append(param_values[vi])
            validation_sim.append(y_sim)
            validation_feasible.append(feasible)

            results.append({
                "parameter": param_name,
                "value": param_values[vi],
                "y_pred_gp": np.nan,
                "y_std_gp": np.nan,
                "is_validation_point": True,
                "y_sim": y_sim,
                "constraint_violation": violation,
                "feasible": feasible,
            })

        ax = axes[plot_idx]
        ax.plot(param_values, y_pred_values, linewidth=2.0, label="GP prediction")
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
        ax.axhline(y_nom, linestyle=":", linewidth=1.5, alpha=0.8, label="Nominal GP output")
        ax.axhline(X_R_MAX, linestyle="-.", linewidth=1.5, alpha=0.8, label="Nominal reach")

        ax.set_xlabel(param_name)
        ax.set_ylabel("Peak outreach [m]")
        ax.set_title(f"Sensitivity to {param_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("One-at-a-time sensitivity analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_plot, dpi=300, bbox_inches="tight")
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv, index=False)

    print(f"\nSaved: {save_plot}")
    print(f"Saved: {save_csv}")

    return results_df


# ============================================================================
# D) MONTE CARLO ROBUSTNESS
# ============================================================================

def monte_carlo_robustness(
    optimal_params: np.ndarray,
    y_target: float,
    n_samples: int = 100,
    noise_level: float = 0.05,
    params_to_perturb: List[str] = ["Kb", "Mb", "hb"],
    save_plot: str = "figures/monte_carlo_robustness.png",
    save_csv: str = "results/monte_carlo_robustness.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run a Monte Carlo robustness test around one optimized solution.

    The selected physical parameters are randomly perturbed using Gaussian
    multiplicative noise:
        p_perturbed = p * (1 + epsilon), epsilon ~ N(0, noise_level)

    The true simulation is used for every sample.
    """
    Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)

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
            params_perturbed[idx] *= (1.0 + epsilon)
            noise_values[PARAM_ORDER[idx]] = epsilon

        sim = simulate_with_metrics(
            params_perturbed,
            y_target=y_target,
            return_solution=False,
        )

        metrics = sim["metrics"]
        y_sim = float(metrics.get("peak_y", sim["peak_y"]))
        violation = float(metrics.get("constraint_violation", np.nan))
        feasible = bool(np.isfinite(violation) and violation <= CONSTRAINT_TOLERANCE)
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

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.hist(y_achieved, bins=28, edgecolor="black", alpha=0.75)
    ax.axvline(y_target, linestyle="--", linewidth=2.2, label="Target")
    ax.axvline(stats["mean_y"], linestyle="-", linewidth=2.0, label="Mean achieved")
    ax.axvspan(stats["q05_y"], stats["q95_y"], alpha=0.18, label="5th-95th percentile")
    ax.set_xlabel("Achieved peak outreach [m]")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of achieved outreach")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(errors * 1000, bins=28, edgecolor="black", alpha=0.75)
    ax.axvline(10.0, linestyle="--", linewidth=2.0, label="10 mm tolerance")
    ax.axvline(stats["mean_error"] * 1000, linestyle="-", linewidth=2.0, label="Mean error")
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
    ax.hist(violations * 1000, bins=24, edgecolor="black", alpha=0.75)
    ax.axvline(CONSTRAINT_TOLERANCE * 1000, linestyle="--", linewidth=2.0, label="Constraint tolerance")
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
        f"Monte Carlo robustness test (n={n_samples}, noise std={noise_level * 100:.0f}%)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_plot, dpi=300, bbox_inches="tight")
    plt.close()

    results_df.to_csv(save_csv, index=False)
    stats_path = save_csv.replace(".csv", "_stats.csv")
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
    print("ADVANCED VISUALIZATIONS - COMPLETE PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load model, dataset, and inverse optimization results
    # ------------------------------------------------------------------------
    print("\nLoading model and data...")

    gp, scaler_X, scaler_y = load_model(MODEL_DIR)
    X_data, y_data = load_dataset(DATASET_PATH)
    df_data = load_dataset_dataframe(DATASET_PATH)
    inverse_results = load_final_inverse_results(INVERSE_RESULTS_PATH)

    print(f"Dataset loaded: {DATASET_PATH}")
    print(f"  Samples: {len(X_data)}")
    print(f"Inverse results loaded: {INVERSE_RESULTS_PATH}")
    print(f"  Targets: {len(inverse_results)}")

    feasible_inverse = inverse_results[inverse_results["feasible"] == True].copy()
    if len(feasible_inverse) == 0:
        raise RuntimeError("No feasible inverse optimization result found.")

    # Select the most representative target for detailed visualizations.
    # Here we use the feasible solution with the highest simulated outreach.
    selected_row = feasible_inverse.sort_values("y_sim", ascending=False).iloc[0]
    selected_idx = int(selected_row["target_index"]) if "target_index" in selected_row else int(selected_row.name)

    y_target = float(selected_row["y_target"])
    optimal_params = extract_params_from_inverse_row(selected_row)

    print("\nSelected solution for detailed visualizations:")
    print(f"  Target index:           {selected_idx}")
    print(f"  y_target:               {y_target:.6f} m")
    print(f"  y_sim achieved:         {selected_row['y_sim']:.6f} m")
    print(f"  Simulation error:       {selected_row['error_sim'] * 1000:.3f} mm")
    print(f"  Extra reach:            {selected_row['extra_reach']:.6f} m")
    print(f"  Constraint violation:   {selected_row['constraint_violation'] * 1000:.3f} mm")

    # ------------------------------------------------------------------------
    # A) Animation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("A) SYSTEM ANIMATION")
    print("=" * 80)

    animate_optimized_response(
        params=optimal_params,
        y_target=y_target,
        save_path=f"{FIGURES_DIR}/animation_target{selected_idx + 1}.gif",
        T_sim=T_SIM,
        dt=DT,
        fps=30,
        duration=12.0,
    )

    # ------------------------------------------------------------------------
    # B) GP response surface
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("B) GP RESPONSE SURFACE")
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
        save_path=f"{FIGURES_DIR}/gp_surface_Kr_hr.png",
    )

    # ------------------------------------------------------------------------
    # C) Sensitivity analysis
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("C) SENSITIVITY ANALYSIS")
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
        save_plot=f"{FIGURES_DIR}/sensitivity_analysis.png",
        save_csv=f"{RESULTS_DIR}/sensitivity_analysis.csv",
    )

    # ------------------------------------------------------------------------
    # D) Monte Carlo robustness
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("D) MONTE CARLO ROBUSTNESS")
    print("=" * 80)

    monte_carlo_df = monte_carlo_robustness(
        optimal_params=optimal_params,
        y_target=y_target,
        n_samples=100,
        noise_level=0.05,
        params_to_perturb=["Kb", "Mb", "hb"],
        save_plot=f"{FIGURES_DIR}/monte_carlo_robustness.png",
        save_csv=f"{RESULTS_DIR}/monte_carlo_robustness.csv",
        seed=42,
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VISUALIZATION PIPELINE COMPLETED")
    print("=" * 80)

    print("\nGenerated files:")
    print(f"  Animation:                  {FIGURES_DIR}/animation_target{selected_idx + 1}.gif")
    print(f"  GP response surface:         {FIGURES_DIR}/gp_surface_Kr_hr.png")
    print(f"  Sensitivity analysis:        {FIGURES_DIR}/sensitivity_analysis.png")
    print(f"  Monte Carlo robustness:      {FIGURES_DIR}/monte_carlo_robustness.png")
    print(f"  Sensitivity CSV:             {RESULTS_DIR}/sensitivity_analysis.csv")
    print(f"  Monte Carlo CSV:             {RESULTS_DIR}/monte_carlo_robustness.csv")
    print(f"  Monte Carlo statistics CSV:  {RESULTS_DIR}/monte_carlo_robustness_stats.csv")

    print("\nRecommended figures for the report/presentation:")
    print("  1. inverse_targets.png")
    print("  2. inverse_feasibility_summary.png")
    print(f"  3. animation_target{selected_idx + 1}.gif")
    print("  4. sensitivity_analysis.png")
    print("  5. monte_carlo_robustness.png")

    print("=" * 80 + "\n")
