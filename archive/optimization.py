"""
optimization.py
===============
Surrogate-based inverse optimization for extra-reaching.

Goal
----
Given a desired target outreach y_target, find a set of controllable
parameters that allows the robot-base system to reach the target while
respecting the robot nominal reach constraint.

Pipeline
--------
1. Load the trained Gaussian Process surrogate and scalers
2. Define fixed passive-system parameters
3. Optimize controllable parameters using Differential Evolution
4. Validate each candidate with the true dynamic simulation
5. Select the best feasible solution
6. Save results and generate report/presentation figures

Main optimization problem
-------------------------
Input:
    y_target

Optimized variables:
    [Kr, hr, f0, f1, A, x_r_start]

Fixed variables:
    [Kb, Mb, hb, Mr]

Physical feasibility:
    constraint_violation <= CONSTRAINT_TOLERANCE

Author: MatteoCasazza
Date: 2026
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from model_peak_y import load_model
from dynamics import simulate_system


# ============================================================================
# GLOBAL SETTINGS AND PATHS
# ============================================================================

# This makes paths robust when running from the project root:
#     python src/optimization.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

GP_RESULTS_DIR = RESULTS_DIR / "gp"
INV_RESULTS_DIR = RESULTS_DIR / "inverse_optimization"
INV_FIG_DIR = FIGURES_DIR / "inverse_optimization"

# Model/scaler folder produced by src/models.py
GP_MODEL_DIR = GP_RESULTS_DIR

X_R_MAX = 0.5
CONSTRAINT_TOLERANCE = 0.002

T_SIM = 60.0
DT = 0.001

# Penalizes uncertain GP predictions during optimization.
UNCERTAINTY_WEIGHT = 0.15

# Multiple independent optimization attempts per target.
N_OPTIMIZATION_ATTEMPTS = 5


def ensure_output_dirs() -> None:
    """Create output folders used by this script."""
    for d in [RESULTS_DIR, FIGURES_DIR, GP_RESULTS_DIR, INV_RESULTS_DIR, INV_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# INVERSE PROBLEM CONFIGURATION
# ============================================================================

@dataclass
class InverseProblemConfig:
    """
    Inverse problem configuration.

    Fixed parameters describe the passive base/environment.
    Controllable parameters are optimized by the inverse optimizer.

    The bounds below are conservative and consistent with the feasible
    high-outreach region of the augmented dataset.
    """

    # Fixed passive-system parameters
    Kb: float = 1000.0
    Mb: float = 20.0
    hb: float = 0.10
    Mr: float = 10.0

    # Controllable bounds
    # Order: [Kr, hr, f0, f1, A, x_r_start]
    Kr_min: float = 1500.0
    Kr_max: float = 5000.0

    hr_min: float = 0.10
    hr_max: float = 0.45

    f0_min: float = 0.10
    f0_max: float = 0.45

    f1_min: float = 1.0
    f1_max: float = 4.0

    A_min: float = 0.09
    A_max: float = 0.105

    x_r_start_min: float = 0.35
    x_r_start_max: float = 0.37

    def get_fixed_params(self) -> Dict[str, float]:
        """Return fixed passive-system parameters."""
        return {
            "Kb": self.Kb,
            "Mb": self.Mb,
            "hb": self.hb,
            "Mr": self.Mr,
        }

    def get_controllable_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for controllable parameters."""
        return [
            (self.Kr_min, self.Kr_max),
            (self.hr_min, self.hr_max),
            (self.f0_min, self.f0_max),
            (self.f1_min, self.f1_max),
            (self.A_min, self.A_max),
            (self.x_r_start_min, self.x_r_start_max),
        ]

    @staticmethod
    def get_controllable_names() -> List[str]:
        """Return controllable parameter names in optimization order."""
        return ["Kr", "hr", "f0", "f1", "A", "x_r_start"]

    @staticmethod
    def get_full_param_order() -> List[str]:
        """Return full parameter order used by the GP training dataset."""
        return ["Kb", "Kr", "Mb", "hb", "hr", "f0", "f1", "A", "x_r_start"]


def reconstruct_full_params(
    controllable_params: np.ndarray,
    fixed_params: Dict[str, float],
) -> np.ndarray:
    """
    Reconstruct the full GP input vector from controllable and fixed parameters.

    Input order required by the trained GP:
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
    """
    Kr, hr, f0, f1, A, x_r_start = controllable_params

    full_params = np.array([
        fixed_params["Kb"],
        Kr,
        fixed_params["Mb"],
        fixed_params["hb"],
        hr,
        f0,
        f1,
        A,
        x_r_start,
    ], dtype=float)

    return full_params


# ============================================================================
# SIMULATION HELPERS
# ============================================================================

def simulate_with_metrics(
    full_params: np.ndarray,
    y_target: Optional[float] = None,
    return_solution: bool = False,
) -> Dict:
    """
    Run the true dynamic simulation and return physical metrics.

    This helper is compatible with the updated dynamics.py where simulate_system
    can return metrics and optionally the ODE solution.
    """
    try:
        if return_solution:
            peak_y, sol, metrics = simulate_system(
                full_params,
                T_sim=T_SIM,
                dt=DT,
                return_solution=True,
                return_metrics=True,
                x_r_max=X_R_MAX,
                y_target=y_target,
            )
            return {
                "peak_y": peak_y,
                "solution": sol,
                "metrics": metrics,
            }

        peak_y, metrics = simulate_system(
            full_params,
            T_sim=T_SIM,
            dt=DT,
            return_metrics=True,
            x_r_max=X_R_MAX,
            y_target=y_target,
        )

        return {
            "peak_y": peak_y,
            "solution": None,
            "metrics": metrics,
        }

    except TypeError:
        # Backward compatibility with older dynamics.py versions.
        if return_solution:
            peak_y, sol = simulate_system(
                full_params,
                T_sim=T_SIM,
                dt=DT,
                return_full=True,
            )

            x_b = sol.y[2]
            x_r = sol.y[3]
            y = x_b + x_r

            metrics = {
                "peak_y": float(np.max(y)),
                "max_xr": float(np.max(x_r)),
                "max_xb": float(np.max(x_b)),
                "extra_reach": float(np.max(y) - X_R_MAX),
                "constraint_violation": float(max(0.0, np.max(x_r) - X_R_MAX)),
                "target_error": float(abs(np.max(y) - y_target)) if y_target is not None else np.nan,
            }

            return {
                "peak_y": peak_y,
                "solution": sol,
                "metrics": metrics,
            }

        peak_y = simulate_system(
            full_params,
            T_sim=T_SIM,
            dt=DT,
        )

        metrics = {
            "peak_y": float(peak_y),
            "max_xr": np.nan,
            "max_xb": np.nan,
            "extra_reach": float(peak_y - X_R_MAX),
            "constraint_violation": np.nan,
            "target_error": float(abs(peak_y - y_target)) if y_target is not None else np.nan,
        }

        return {
            "peak_y": peak_y,
            "solution": None,
            "metrics": metrics,
        }


# ============================================================================
# INVERSE OPTIMIZER
# ============================================================================

class InverseOptimizer:
    """
    Inverse optimizer using a trained Gaussian Process surrogate.

    The GP is used only inside the optimizer.
    Final performance and feasibility are always evaluated with the true
    dynamic simulation.
    """

    def __init__(
        self,
        gp,
        scaler_X,
        scaler_y,
        config: InverseProblemConfig,
    ):
        self.gp = gp
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.config = config

        self.fixed_params = config.get_fixed_params()
        self.bounds = config.get_controllable_bounds()

    def predict_gp(self, full_params: np.ndarray) -> Tuple[float, float]:
        """
        Predict peak outreach using the GP surrogate.

        Returns
        -------
        y_pred : float
            Predicted peak outreach [m].
        y_std : float
            Predictive standard deviation [m].
        """
        X_scaled = self.scaler_X.transform([full_params])

        y_pred_scaled, y_std_scaled = self.gp.predict(
            X_scaled,
            return_std=True,
        )

        y_pred = self.scaler_y.inverse_transform(
            [[y_pred_scaled[0]]]
        )[0, 0]

        y_std = y_std_scaled[0] * self.scaler_y.scale_[0]

        return float(y_pred), float(y_std)

    def objective_function(
        self,
        controllable_params: np.ndarray,
        y_target: float,
    ) -> float:
        """
        Objective function minimized by Differential Evolution.

        The main term is target tracking error.
        A small uncertainty penalty discourages solutions in poorly known
        GP regions.
        """
        full_params = reconstruct_full_params(
            controllable_params,
            self.fixed_params,
        )

        y_pred, y_std = self.predict_gp(full_params)

        tracking_cost = (y_pred - y_target) ** 2
        uncertainty_cost = UNCERTAINTY_WEIGHT * (y_std ** 2)

        return tracking_cost + uncertainty_cost

    def optimize_once(
        self,
        y_target: float,
        seed: int,
        maxiter: int = 120,
        popsize: int = 12,
        verbose: bool = False,
    ) -> Dict:
        """
        Run one Differential Evolution optimization attempt.
        """
        start_time = time.time()

        result_opt = differential_evolution(
            func=lambda x: self.objective_function(x, y_target),
            bounds=self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            workers=1,
            updating="immediate",
            polish=True,
            atol=1e-7,
            tol=1e-7,
        )

        elapsed = time.time() - start_time

        controllable_opt = result_opt.x
        full_params_opt = reconstruct_full_params(
            controllable_opt,
            self.fixed_params,
        )

        y_gp_pred, y_gp_std = self.predict_gp(full_params_opt)

        if verbose:
            print(
                f"    Seed {seed}: GP={y_gp_pred:.6f} m, "
                f"std={y_gp_std:.6f} m, "
                f"cost={result_opt.fun:.6e}, "
                f"time={elapsed:.2f} s"
            )

        return {
            "seed": seed,
            "success": bool(result_opt.success),
            "optimizer_message": result_opt.message,
            "objective_value": float(result_opt.fun),
            "controllable_opt": controllable_opt,
            "full_params_opt": full_params_opt,
            "y_gp_pred": y_gp_pred,
            "y_gp_std": y_gp_std,
            "optimization_time": elapsed,
        }

    def optimize_target(
        self,
        y_target: float,
        target_index: int,
        n_attempts: int = N_OPTIMIZATION_ATTEMPTS,
        verbose: bool = True,
    ) -> Dict:
        """
        Optimize and validate multiple candidates for one target.

        The final selected candidate is:
        1. the feasible candidate with the smallest simulation error, if any;
        2. otherwise, the candidate with the best penalty score.
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"TARGET {target_index + 1}: y_target = {y_target:.4f} m")
            print("=" * 80)

        candidates = []

        for attempt in range(n_attempts):
            seed = 1000 + 100 * target_index + attempt

            opt_result = self.optimize_once(
                y_target=y_target,
                seed=seed,
                verbose=verbose,
            )

            validation = validate_with_simulation(
                full_params=opt_result["full_params_opt"],
                y_target=y_target,
                y_gp_pred=opt_result["y_gp_pred"],
                verbose=False,
            )

            candidate = {
                **opt_result,
                **validation,
                "y_target": y_target,
            }

            candidates.append(candidate)

            if verbose:
                print(
                    f"    Validation: sim={candidate['y_sim']:.6f} m | "
                    f"sim error={candidate['error_sim'] * 1000:.2f} mm | "
                    f"violation={candidate['constraint_violation'] * 1000:.2f} mm | "
                    f"feasible={candidate['feasible']}"
                )

        feasible_candidates = [c for c in candidates if c["feasible"]]

        if feasible_candidates:
            best = min(feasible_candidates, key=lambda c: c["error_sim"])
            selection_reason = "best feasible simulation error"
        else:
            best = min(
                candidates,
                key=lambda c: c["error_sim"] + 10.0 * c["constraint_violation"],
            )
            selection_reason = "no feasible candidate found; selected minimum penalized error"

        best["selection_reason"] = selection_reason
        best["n_feasible_candidates"] = len(feasible_candidates)
        best["n_attempts"] = n_attempts

        if verbose:
            print("\n  Selected candidate:")
            print(f"    Reason:          {selection_reason}")
            print(f"    GP prediction:   {best['y_gp_pred']:.6f} m")
            print(f"    Simulation:      {best['y_sim']:.6f} m")
            print(f"    Target error:    {best['error_sim'] * 1000:.2f} mm")
            print(f"    GP vs sim:       {best['error_gp_vs_sim'] * 1000:.2f} mm")
            print(f"    Max x_r:         {best['max_xr']:.6f} m")
            print(f"    Max x_b:         {best['max_xb']:.6f} m")
            print(f"    Extra reach:     {best['extra_reach']:.6f} m")
            print(f"    Violation:       {best['constraint_violation'] * 1000:.2f} mm")
            print(f"    Feasible:        {best['feasible']}")

            print("\n  Optimized controllable parameters:")
            for name, val in zip(
                self.config.get_controllable_names(),
                best["controllable_opt"],
            ):
                print(f"    {name:12s} = {val:.6f}")

        return best


# ============================================================================
# VALIDATION
# ============================================================================

def validate_with_simulation(
    full_params: np.ndarray,
    y_target: float,
    y_gp_pred: float,
    verbose: bool = True,
) -> Dict:
    """
    Validate optimized parameters with the true dynamic simulation.
    """
    sim = simulate_with_metrics(
        full_params,
        y_target=y_target,
        return_solution=False,
    )

    metrics = sim["metrics"]
    y_sim = float(metrics.get("peak_y", sim["peak_y"]))

    error_gp = abs(y_gp_pred - y_target)
    error_sim = abs(y_sim - y_target)
    error_gp_vs_sim = abs(y_gp_pred - y_sim)

    constraint_violation = float(metrics.get("constraint_violation", np.nan))
    extra_reach = float(metrics.get("extra_reach", y_sim - X_R_MAX))
    max_xr = float(metrics.get("max_xr", np.nan))
    max_xb = float(metrics.get("max_xb", np.nan))

    feasible = bool(
        np.isfinite(constraint_violation)
        and constraint_violation <= CONSTRAINT_TOLERANCE
    )

    if verbose:
        print("\n--- TRUE SIMULATION VALIDATION ---")
        print(f"  Target:                 {y_target:.6f} m")
        print(f"  GP prediction:          {y_gp_pred:.6f} m")
        print(f"  Simulated peak_y:       {y_sim:.6f} m")
        print(f"  Simulation error:       {error_sim:.6f} m")
        print(f"  GP vs simulation:       {error_gp_vs_sim:.6f} m")
        print(f"  Extra reach:            {extra_reach:.6f} m")
        print(f"  Max robot position x_r: {max_xr:.6f} m")
        print(f"  Max base motion x_b:    {max_xb:.6f} m")
        print(f"  Constraint violation:   {constraint_violation:.6f} m")
        print(f"  Feasible:               {feasible}")

    return {
        "y_sim": y_sim,
        "error_gp": error_gp,
        "error_sim": error_sim,
        "error_gp_vs_sim": error_gp_vs_sim,
        "extra_reach": extra_reach,
        "max_xr": max_xr,
        "max_xb": max_xb,
        "constraint_violation": constraint_violation,
        "feasible": feasible,
    }


# ============================================================================
# PIPELINE
# ============================================================================

def run_inverse_optimization_pipeline(
    targets: np.ndarray,
    config: Optional[InverseProblemConfig] = None,
    gp_model_dir: Path = GP_MODEL_DIR,
    save_results: bool = True,
    plot_trajectories: bool = True,
    trajectory_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run inverse optimization for multiple target values.
    """
    ensure_output_dirs()

    print("\n" + "=" * 80)
    print("SURROGATE-BASED INVERSE OPTIMIZATION")
    print("=" * 80)
    print(f"Number of targets:        {len(targets)}")
    print(f"Target range:             [{targets.min():.3f}, {targets.max():.3f}] m")
    print(f"Robot nominal reach:      {X_R_MAX:.3f} m")
    print(f"Constraint tolerance:     {CONSTRAINT_TOLERANCE * 1000:.1f} mm")
    print(f"Optimization attempts:    {N_OPTIMIZATION_ATTEMPTS} per target")

    if config is None:
        config = InverseProblemConfig()

    print("\nFixed parameters:")
    for key, value in config.get_fixed_params().items():
        print(f"  {key:4s} = {value}")

    print("\nControllable bounds:")
    for name, bound in zip(config.get_controllable_names(), config.get_controllable_bounds()):
        print(f"  {name:12s}: [{bound[0]}, {bound[1]}]")

    print(f"\nLoading GP model from: {gp_model_dir}/")
    gp, scaler_X, scaler_y = load_model(gp_model_dir)

    optimizer = InverseOptimizer(
        gp=gp,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        config=config,
    )

    results = []
    selected_solutions = []

    for i, y_target in enumerate(targets):
        best = optimizer.optimize_target(
            y_target=float(y_target),
            target_index=i,
            n_attempts=N_OPTIMIZATION_ATTEMPTS,
            verbose=True,
        )

        row = {
            "target_index": i,
            "y_target": float(y_target),
            "y_gp_pred": best["y_gp_pred"],
            "y_gp_std": best["y_gp_std"],
            "y_sim": best["y_sim"],
            "error_gp": best["error_gp"],
            "error_sim": best["error_sim"],
            "error_gp_vs_sim": best["error_gp_vs_sim"],
            "extra_reach": best["extra_reach"],
            "max_xr": best["max_xr"],
            "max_xb": best["max_xb"],
            "constraint_violation": best["constraint_violation"],
            "feasible": best["feasible"],
            "n_feasible_candidates": best["n_feasible_candidates"],
            "n_attempts": best["n_attempts"],
            "selection_reason": best["selection_reason"],
            "optimization_time": best["optimization_time"],
            "success": best["success"],
            "seed": best["seed"],
        }

        for name, value in zip(config.get_full_param_order(), best["full_params_opt"]):
            row[f"param_{name}"] = value

        results.append(row)
        selected_solutions.append(best)

    results_df = pd.DataFrame(results)

    print_final_summary(results_df)

    if save_results:
        csv_path = INV_RESULTS_DIR / "inverse_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        plot_inverse_results(
            results_df,
            save_path=INV_FIG_DIR / "inverse_targets.png",
        )

        plot_optimized_parameters(
            results_df,
            save_path=INV_FIG_DIR / "inverse_optimized_parameters.png",
        )

        plot_feasibility_summary(
            results_df,
            save_path=INV_FIG_DIR / "inverse_feasibility_summary.png",
        )

        if plot_trajectories:
            if trajectory_indices is None:
                trajectory_indices = list(range(len(targets)))

            plot_trajectories_for_targets(
                selected_solutions,
                trajectory_indices,
                save_dir=INV_FIG_DIR,
            )

    return results_df


def print_final_summary(results_df: pd.DataFrame) -> None:
    """
    Print final optimization summary.
    """
    print("\n" + "=" * 80)
    print("FINAL INVERSE OPTIMIZATION RESULTS")
    print("=" * 80)

    display_cols = [
        "y_target",
        "y_gp_pred",
        "y_sim",
        "error_sim",
        "extra_reach",
        "max_xr",
        "max_xb",
        "constraint_violation",
        "feasible",
    ]

    print("\n" + results_df[display_cols].to_string(index=False))

    feasible_rate = 100.0 * results_df["feasible"].mean()

    print("\nError statistics:")
    print(f"  Mean simulation error:     {results_df['error_sim'].mean() * 1000:.2f} mm")
    print(f"  Max simulation error:      {results_df['error_sim'].max() * 1000:.2f} mm")
    print(f"  Mean GP vs simulation:     {results_df['error_gp_vs_sim'].mean() * 1000:.2f} mm")
    print(f"  Feasible targets:          {results_df['feasible'].sum()}/{len(results_df)}")
    print(f"  Feasibility rate:          {feasible_rate:.1f}%")
    print(f"  Mean extra reach:          {results_df['extra_reach'].mean():.4f} m")
    print(f"  Max extra reach:           {results_df['extra_reach'].max():.4f} m")


# ============================================================================
# PLOTS
# ============================================================================

def plot_inverse_results(
    results_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Plot target, GP prediction, simulation result and errors.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(results_df))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.bar(x - width, results_df["y_target"], width, label="Target", alpha=0.75, edgecolor="black")
    ax.bar(x, results_df["y_gp_pred"], width, label="GP prediction", alpha=0.75, edgecolor="black")
    ax.bar(x + width, results_df["y_sim"], width, label="True simulation", alpha=0.75, edgecolor="black")
    ax.errorbar(
        x,
        results_df["y_gp_pred"],
        yerr=results_df["y_gp_std"],
        fmt="none",
        capsize=5,
        linewidth=1.8,
        label="GP ±1σ",
    )
    ax.axhline(X_R_MAX, linestyle="--", linewidth=2, label="Nominal reach")
    ax.set_xlabel("Target index")
    ax.set_ylabel("Peak outreach [m]")
    ax.set_title("Target vs surrogate vs simulation")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(x, results_df["error_gp"] * 1000, "o-", linewidth=2, markersize=7, label="GP target error")
    ax.plot(x, results_df["error_sim"] * 1000, "s-", linewidth=2, markersize=7, label="Simulation target error")
    ax.plot(x, results_df["error_gp_vs_sim"] * 1000, "^-", linewidth=2, markersize=7, label="GP vs simulation")
    ax.set_xlabel("Target index")
    ax.set_ylabel("Error [mm]")
    ax.set_title("Optimization and validation errors")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.bar(x, results_df["constraint_violation"] * 1000, alpha=0.75, edgecolor="black")
    ax.axhline(CONSTRAINT_TOLERANCE * 1000, linestyle="--", linewidth=2, label="Tolerance")
    ax.set_xlabel("Target index")
    ax.set_ylabel("Constraint violation [mm]")
    ax.set_title("Robot reach constraint")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_optimized_parameters(
    results_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Plot optimized controllable parameters across target values.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    controllable_cols = [
        "param_Kr",
        "param_hr",
        "param_f0",
        "param_f1",
        "param_A",
        "param_x_r_start",
    ]

    labels = ["Kr [N/m]", "hr [-]", "f0 [Hz]", "f1 [Hz]", "A [m]", "x_r_start [m]"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    x = results_df["y_target"].values

    for ax, col, label in zip(axes, controllable_cols, labels):
        ax.plot(x, results_df[col].values, "o-", linewidth=2, markersize=7)
        ax.set_xlabel("Target outreach [m]")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Optimized controllable parameters", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_feasibility_summary(
    results_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Plot maximum robot/base motion indicators and feasibility.

    Note
    ----
    max_xr and max_xb may occur at different time instants.
    Therefore this plot should be interpreted as a summary of maximum observed
    contributions, not as an exact instantaneous decomposition of peak_y.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(results_df))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.bar(
        x,
        results_df["max_xr"],
        label="Maximum robot position x_r",
        alpha=0.75,
        edgecolor="black",
    )
    ax.bar(
        x,
        results_df["max_xb"],
        bottom=results_df["max_xr"],
        label="Maximum base displacement x_b",
        alpha=0.75,
        edgecolor="black",
    )
    ax.axhline(X_R_MAX, linestyle="--", linewidth=2, label="Nominal robot reach")
    ax.set_xlabel("Target index")
    ax.set_ylabel("Position indicator [m]")
    ax.set_title("Maximum robot/base motion indicators")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[1]
    colors = ["green" if f else "red" for f in results_df["feasible"]]
    ax.bar(
        x,
        results_df["error_sim"] * 1000,
        color=colors,
        alpha=0.75,
        edgecolor="black",
    )
    ax.set_xlabel("Target index")
    ax.set_ylabel("Simulation target error [mm]")
    ax.set_title("Validation error and feasibility")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    for i, feasible in enumerate(results_df["feasible"]):
        label = "OK" if feasible else "VIOL."
        ax.text(
            i,
            results_df["error_sim"].iloc[i] * 1000,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_trajectories_for_targets(
    solutions: List[Dict],
    indices: List[int],
    save_dir: Path,
) -> None:
    """
    Plot simulated trajectories for selected optimized solutions.

    Each figure shows:
    - total outreach y(t)
    - robot relative position x_r(t)
    - passive base displacement x_b(t)
    - target outreach
    - nominal robot reach
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        if idx >= len(solutions):
            continue

        sol_data = solutions[idx]
        full_params = sol_data["full_params_opt"]
        target_value = sol_data.get("y_target", np.nan)

        sim = simulate_with_metrics(
            full_params,
            y_target=target_value,
            return_solution=True,
        )

        sol = sim["solution"]
        metrics = sim["metrics"]

        if sol is None:
            print(f"Could not plot trajectory for target {idx + 1}: no solution returned.")
            continue

        t = sol.t
        x_b = sol.y[2]
        x_r = sol.y[3]
        y = x_b + x_r

        peak_y = metrics.get("peak_y", np.max(y))

        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

        ax = axes[0]
        ax.plot(t, y, linewidth=2.2, label="Total outreach y(t) = x_b + x_r")
        ax.plot(t, x_r, linewidth=1.8, label="Robot relative position x_r(t)")
        ax.plot(t, x_b, linewidth=1.8, label="Base displacement x_b(t)")
        ax.axhline(X_R_MAX, linestyle="--", linewidth=2, label="Nominal robot reach")

        if np.isfinite(target_value):
            ax.axhline(target_value, linestyle=":", linewidth=2, label=f"Target = {target_value:.3f} m")

        ax.axhline(peak_y, linestyle="-.", linewidth=2, label=f"Peak outreach = {peak_y:.3f} m")

        ax.set_ylabel("Position [m]")
        ax.set_title(f"Optimized trajectory for target {idx + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.plot(t, x_r, linewidth=2, label="x_r(t)")
        ax.axhline(X_R_MAX, linestyle="--", linewidth=2, label="x_r_max")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Robot relative position [m]")
        ax.set_title("Robot reach constraint over time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        filepath = save_dir / f"inverse_trajectory_target{idx + 1}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INVERSE OPTIMIZATION - COMPLETE PIPELINE")
    print("=" * 80)

    # Conservative final targets for the extra-reaching task.
    # They are above the nominal robot reach of 0.5 m and should be feasible
    # with the stricter bounds used in this file.
    targets = np.array([0.52, 0.58, 0.60, 0.62])

    config = InverseProblemConfig(
        Kb=1000.0,
        Mb=20.0,
        hb=0.10,
        Mr=10.0,
    )

    results_df = run_inverse_optimization_pipeline(
        targets=targets,
        config=config,
        gp_model_dir=GP_MODEL_DIR,
        save_results=True,
        plot_trajectories=True,
        trajectory_indices=[0, 1, 2, 3],
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)

    print("\nPerformance:")
    print(f"  Targets tested:              {len(targets)}")
    print(f"  Mean simulation error:       {results_df['error_sim'].mean() * 1000:.2f} mm")
    print(f"  Max simulation error:        {results_df['error_sim'].max() * 1000:.2f} mm")
    print(f"  Feasible targets:            {results_df['feasible'].sum()}/{len(results_df)}")
    print(f"  Mean constraint violation:   {results_df['constraint_violation'].mean() * 1000:.2f} mm")

    print("\nOutputs:")
    print(f"  Results table:               {INV_RESULTS_DIR / 'inverse_results.csv'}")
    print(f"  Target comparison plot:      {INV_FIG_DIR / 'inverse_targets.png'}")
    print(f"  Parameter plot:              {INV_FIG_DIR / 'inverse_optimized_parameters.png'}")
    print(f"  Feasibility plot:            {INV_FIG_DIR / 'inverse_feasibility_summary.png'}")
    print(f"  Trajectories:                {INV_FIG_DIR / 'inverse_trajectory_target*.png'}")

    print("\nNext possible step:")
    print("  If all targets are feasible, use these results in the report.")
    print("  If some targets still violate the constraint, reduce A_max or x_r_start_max further.")
    print("=" * 80 + "\n")
