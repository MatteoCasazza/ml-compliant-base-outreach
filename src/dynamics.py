"""
dynamics.py
===========

Physics-based simulator for the two-degree-of-freedom compliant-base system.

Coordinates
-----------
x_b : passive base position
x_r : robot position relative to the base
y   : total end-effector outreach, defined as y = x_b + x_r

Input
-----
x_rd(t) : commanded robot position, generated as a linear chirp signal.

Main output
-----------
peak_y : maximum total outreach reached during the simulation.

The model is a simplified version of the compliant-base robot system inspired by:
Roveda et al. (2016), Mechatronics 39.

Author: Matteo Casazza
Date: 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

PARAM_ORDER = (
    "Kb",
    "Kr",
    "Mb",
    "hb",
    "hr",
    "f0",
    "f1",
    "A",
    "x_r_start",
)

DEFAULT_ROBOT_MASS = 10.0
DEFAULT_T_SIM = 60.0
DEFAULT_DT = 0.001
DEFAULT_ROBOT_LIMIT = 0.500
DEFAULT_FEASIBILITY_TOL = 1e-9

SOLVER_METHOD = "RK45"
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 1e-9


# =============================================================================
# PARAMETER AND SIGNAL UTILITIES
# =============================================================================

def chirp_signal(
    t: float | np.ndarray,
    f0: float,
    f1: float,
    T: float,
    A: float,
    x_r_start: float,
) -> np.ndarray:
    """
    Generate the commanded robot position using a linear-frequency chirp.

    The signal is defined so that:

        x_rd(0) = x_r_start

    This avoids an artificial jump between the initial robot position and the
    initial commanded position.

    Parameters
    ----------
    t : float or ndarray
        Time instant(s) [s].
    f0 : float
        Initial chirp frequency [Hz].
    f1 : float
        Final chirp frequency [Hz].
    T : float
        Chirp duration [s].
    A : float
        Chirp amplitude [m].
    x_r_start : float
        Initial robot relative position [m].

    Returns
    -------
    ndarray
        Commanded robot position x_rd(t) [m].
    """
    t_array = np.asarray(t, dtype=float)

    if T <= 0:
        return x_r_start * np.ones_like(t_array, dtype=float)

    chirp_rate = (f1 - f0) / T
    phase = 2.0 * np.pi * (f0 * t_array + 0.5 * chirp_rate * t_array**2)

    return x_r_start + A * np.sin(phase)


def prepare_params(
    params: Mapping[str, float] | Sequence[float],
    T_sim: float,
) -> dict[str, float]:
    """
    Convert input parameters to a complete parameter dictionary.

    Parameters
    ----------
    params : dict or sequence
        If a dictionary is provided, it must contain the simulator parameters.
        If a sequence is provided, the expected order is PARAM_ORDER.
    T_sim : float
        Simulation duration [s]. It is also used as the chirp duration.

    Returns
    -------
    dict
        Complete parameter dictionary, including:
        - T  : chirp duration [s]
        - Mr : robot mass [kg]
    """
    if isinstance(params, Mapping):
        prepared = dict(params)
    else:
        if len(params) != len(PARAM_ORDER):
            raise ValueError(
                f"Expected {len(PARAM_ORDER)} parameters in order {PARAM_ORDER}, "
                f"received {len(params)}."
            )
        prepared = dict(zip(PARAM_ORDER, params))

    missing = [name for name in PARAM_ORDER if name not in prepared]
    if missing:
        raise KeyError(f"Missing required simulator parameters: {missing}")

    prepared["T"] = float(T_sim)
    prepared.setdefault("Mr", DEFAULT_ROBOT_MASS)

    return {key: float(value) for key, value in prepared.items()}


def compute_damping(K: float, M: float, h: float) -> float:
    """
    Compute the viscous damping coefficient from the damping ratio.

    Formula:

        D = 2 h sqrt(K M)

    Parameters
    ----------
    K : float
        Stiffness [N/m].
    M : float
        Mass [kg].
    h : float
        Damping ratio [-].

    Returns
    -------
    float
        Viscous damping coefficient [Ns/m].
    """
    if K <= 0:
        raise ValueError("Stiffness K must be positive.")
    if M <= 0:
        raise ValueError("Mass M must be positive.")

    return float(2.0 * h * np.sqrt(K * M))


def compute_natural_frequency(K: float, M: float) -> tuple[float, float]:
    """
    Compute the undamped natural frequency of a mass-spring subsystem.

    Parameters
    ----------
    K : float
        Stiffness [N/m].
    M : float
        Mass [kg].

    Returns
    -------
    omega_n : float
        Natural angular frequency [rad/s].
    f_n : float
        Natural frequency [Hz].
    """
    if K <= 0:
        raise ValueError("Stiffness K must be positive.")
    if M <= 0:
        raise ValueError("Mass M must be positive.")

    omega_n = np.sqrt(K / M)
    f_n = omega_n / (2.0 * np.pi)

    return float(omega_n), float(f_n)


# =============================================================================
# DYNAMIC MODEL
# =============================================================================

def system_dynamics(
    t: float,
    state: Sequence[float],
    params: Mapping[str, float],
) -> list[float]:
    """
    Compute the time derivative of the 2-DoF system state.

    State definition:

        state = [dx_b, dx_r, x_b, x_r]

    where:
    - dx_b is the base velocity [m/s];
    - dx_r is the robot relative velocity [m/s];
    - x_b is the base position [m];
    - x_r is the robot position relative to the base [m].

    Parameters
    ----------
    t : float
        Current time [s].
    state : sequence of float
        Current system state [dx_b, dx_r, x_b, x_r].
    params : dict
        Complete parameter dictionary.

    Returns
    -------
    list of float
        State derivative [ddx_b, ddx_r, dx_b, dx_r].
    """
    dx_b, dx_r, x_b, x_r = state

    Kb = params["Kb"]
    Kr = params["Kr"]
    Mb = params["Mb"]
    Mr = params.get("Mr", DEFAULT_ROBOT_MASS)

    hb = params["hb"]
    hr = params["hr"]

    f0 = params["f0"]
    f1 = params["f1"]
    A = params["A"]
    x_r_start = params["x_r_start"]
    T = params.get("T", DEFAULT_T_SIM)

    Db = compute_damping(Kb, Mb, hb)
    Dr = compute_damping(Kr, Mr, hr)

    x_rd = float(chirp_signal(t, f0, f1, T, A, x_r_start))

    ddx_b = (Dr * dx_r + Kr * (x_r - x_rd) - Db * dx_b - Kb * x_b) / Mb
    ddx_r = (-Dr * dx_r - Kr * (x_r - x_rd)) / Mr - ddx_b

    return [ddx_b, ddx_r, dx_b, dx_r]


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    sol: Any,
    x_r_max: float = DEFAULT_ROBOT_LIMIT,
    y_target: float | None = None,
    feasibility_tol: float = DEFAULT_FEASIBILITY_TOL,
) -> dict[str, Any]:
    """
    Compute physical performance metrics from the simulation result.

    Parameters
    ----------
    sol : OdeResult
        Output object returned by scipy.integrate.solve_ivp.
    x_r_max : float, optional
        Maximum admissible robot relative displacement [m].
    y_target : float, optional
        Desired total outreach target [m].
    feasibility_tol : float, optional
        Numerical tolerance used to classify feasibility.

    Returns
    -------
    dict
        Physical performance metrics.
    """
    _, _, x_b, x_r = sol.y
    y = x_b + x_r

    peak_idx = int(np.argmax(y))
    peak_y = float(y[peak_idx])

    max_xr = float(np.max(x_r))
    min_xr = float(np.min(x_r))
    max_abs_xr = float(np.max(np.abs(x_r)))

    max_xb = float(np.max(x_b))
    min_xb = float(np.min(x_b))
    max_abs_xb = float(np.max(np.abs(x_b)))

    constraint_violation = max(0.0, max_xr - x_r_max)
    constraint_violation_abs = max(0.0, max_abs_xr - x_r_max)

    metrics: dict[str, Any] = {
        "peak_y": peak_y,
        "t_peak": float(sol.t[peak_idx]),
        "final_y": float(y[-1]),
        "max_xr": max_xr,
        "min_xr": min_xr,
        "max_abs_xr": max_abs_xr,
        "max_xb": max_xb,
        "min_xb": min_xb,
        "max_abs_xb": max_abs_xb,
        "x_r_max": float(x_r_max),
        "extra_reach": float(peak_y - x_r_max),
        "constraint_violation": float(constraint_violation),
        "constraint_violation_abs": float(constraint_violation_abs),
        "feasible": bool(constraint_violation <= feasibility_tol),
        "feasible_abs": bool(constraint_violation_abs <= feasibility_tol),
    }

    if y_target is not None:
        target_error = abs(peak_y - float(y_target))
        metrics.update(
            {
                "y_target": float(y_target),
                "target_error": float(target_error),
                "target_error_mm": float(target_error * 1000.0),
                "target_reached": bool(peak_y >= float(y_target)),
            }
        )

    return metrics


# =============================================================================
# SIMULATION INTERFACE
# =============================================================================

def simulate_system(
    params: Mapping[str, float] | Sequence[float],
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    return_full: bool = False,
    return_metrics: bool = False,
    x_r_max: float = DEFAULT_ROBOT_LIMIT,
    y_target: float | None = None,
    feasibility_tol: float = DEFAULT_FEASIBILITY_TOL,
):
    """
    Simulate the 2-DoF compliant-base robot system.

    This function keeps the same return interface used by the rest of the
    project.

    Parameters
    ----------
    params : dict or sequence
        System and excitation parameters.
    T_sim : float, optional
        Simulation duration [s].
    dt : float, optional
        Time step used to store the solution [s].
    return_full : bool, optional
        If True, return the full ODE solution.
    return_metrics : bool, optional
        If True, return the physical metrics dictionary.
    x_r_max : float, optional
        Maximum admissible robot relative displacement [m].
    y_target : float, optional
        Desired total outreach target [m].
    feasibility_tol : float, optional
        Numerical tolerance used for feasibility classification.

    Returns
    -------
    float or tuple
        Depending on return_full and return_metrics:
        - peak_y
        - peak_y, sol
        - peak_y, metrics
        - peak_y, sol, metrics
    """
    if T_sim <= 0:
        raise ValueError("T_sim must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if x_r_max <= 0:
        raise ValueError("x_r_max must be positive.")

    prepared_params = prepare_params(params, T_sim)

    initial_state = [
        0.0,
        0.0,
        0.0,
        prepared_params["x_r_start"],
    ]

    t_eval = np.arange(0.0, T_sim + 0.5 * dt, dt)

    sol = solve_ivp(
        system_dynamics,
        t_span=(0.0, T_sim),
        y0=initial_state,
        t_eval=t_eval,
        args=(prepared_params,),
        method=SOLVER_METHOD,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
    )

    if not sol.success:
        if return_full and return_metrics:
            return np.nan, sol, {}
        if return_full:
            return np.nan, sol
        if return_metrics:
            return np.nan, {}
        return np.nan

    metrics = compute_metrics(
        sol=sol,
        x_r_max=x_r_max,
        y_target=y_target,
        feasibility_tol=feasibility_tol,
    )

    peak_y = metrics["peak_y"]

    if return_full and return_metrics:
        return peak_y, sol, metrics

    if return_full:
        return peak_y, sol

    if return_metrics:
        return peak_y, metrics

    return peak_y


# =============================================================================
# PLOTTING UTILITY
# =============================================================================

def plot_simulation_example(
    params: Mapping[str, float] | Sequence[float],
    T_sim: float = DEFAULT_T_SIM,
    dt: float = DEFAULT_DT,
    y_target: float | None = None,
    x_r_max: float = DEFAULT_ROBOT_LIMIT,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Plot a complete simulation example.

    The figure shows:
    1. commanded chirp signal x_rd(t);
    2. passive base position x_b(t);
    3. robot relative position x_r(t);
    4. total outreach y(t) = x_b(t) + x_r(t).

    Parameters
    ----------
    params : dict or sequence
        System and excitation parameters.
    T_sim : float, optional
        Simulation duration [s].
    dt : float, optional
        Time step used to store the solution [s].
    y_target : float, optional
        Desired total outreach target [m].
    x_r_max : float, optional
        Maximum admissible robot relative displacement [m].
    save_path : str or Path, optional
        Path where the figure is saved.
    show : bool, optional
        If True, display the figure.

    Returns
    -------
    peak_y : float
        Maximum total outreach [m].
    metrics : dict
        Physical performance metrics.
    """
    peak_y, sol, metrics = simulate_system(
        params=params,
        T_sim=T_sim,
        dt=dt,
        return_full=True,
        return_metrics=True,
        x_r_max=x_r_max,
        y_target=y_target,
    )

    if sol is None or not np.isfinite(peak_y):
        raise RuntimeError("Simulation failed. Cannot generate plot.")

    t = sol.t
    _, _, x_b, x_r = sol.y
    y = x_b + x_r

    prepared_params = prepare_params(params, T_sim)

    x_rd = chirp_signal(
        t=t,
        f0=prepared_params["f0"],
        f1=prepared_params["f1"],
        T=T_sim,
        A=prepared_params["A"],
        x_r_start=prepared_params["x_r_start"],
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, x_rd, linewidth=1.5, label=r"$x_{rd}(t)$ command")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("Excitation signal: chirp command", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, x_b, linewidth=1.5, label=r"$x_b(t)$ passive base")
    axes[1].set_ylabel("Position [m]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, x_r, linewidth=1.5, label=r"$x_r(t)$ robot relative position")
    axes[2].axhline(
        x_r_max,
        linestyle="--",
        linewidth=1.5,
        label=f"Robot limit = {x_r_max:.2f} m",
    )
    axes[2].set_ylabel("Position [m]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, y, linewidth=2.0, label=r"$y(t)=x_b(t)+x_r(t)$")
    axes[3].axhline(
        peak_y,
        linestyle="--",
        linewidth=1.5,
        label=f"Peak = {peak_y:.3f} m",
    )
    axes[3].axhline(
        x_r_max,
        linestyle=":",
        linewidth=1.5,
        label=f"Nominal reach = {x_r_max:.2f} m",
    )

    if y_target is not None:
        axes[3].axhline(
            y_target,
            linestyle="-.",
            linewidth=1.5,
            label=f"Target = {y_target:.3f} m",
        )

    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Outreach [m]")
    axes[3].set_title(
        f"Total outreach: peak = {peak_y:.3f} m, "
        f"extra reach = {metrics['extra_reach']:.3f} m",
        fontsize=11,
    )
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return peak_y, metrics


# =============================================================================
# SELF-TEST
# =============================================================================

def main() -> None:
    """Run a simple simulator test."""
    print("=" * 70)
    print("TEST: src/dynamics.py")
    print("=" * 70)

    test_params = {
        "Kb": 2000.0,
        "Kr": 1500.0,
        "Mb": 50.0,
        "hb": 0.2,
        "hr": 0.5,
        "f0": 0.001,
        "f1": 5.0,
        "A": 0.10,
        "x_r_start": 0.4,
        "Mr": 10.0,
    }

    x_r_max = DEFAULT_ROBOT_LIMIT
    y_target = 0.55

    print("\nTest parameters:")
    for key, value in test_params.items():
        print(f"  {key:12s} = {value}")

    print("\nRunning simulation...")

    peak_y, metrics = simulate_system(
        params=test_params,
        T_sim=DEFAULT_T_SIM,
        dt=DEFAULT_DT,
        return_metrics=True,
        x_r_max=x_r_max,
        y_target=y_target,
    )

    print("\nSimulation completed.")
    print(f"  Peak outreach:          {peak_y:.4f} m")
    print(f"  Nominal robot reach:    {metrics['x_r_max']:.4f} m")
    print(f"  Extra reach:            {metrics['extra_reach']:.4f} m")
    print(f"  Max robot position xr:  {metrics['max_xr']:.4f} m")
    print(f"  Max abs robot xr:       {metrics['max_abs_xr']:.4f} m")
    print(f"  Constraint violation:   {metrics['constraint_violation']:.4f} m")
    print(f"  Abs. constraint viol.:  {metrics['constraint_violation_abs']:.4f} m")
    print(f"  Feasible:               {metrics['feasible']}")
    print(f"  Feasible abs.:          {metrics['feasible_abs']}")
    print(f"  Target error:           {metrics.get('target_error', np.nan):.4f} m")

    print("\nGenerating validation plot...")

    plot_simulation_example(
        params=test_params,
        T_sim=DEFAULT_T_SIM,
        dt=DEFAULT_DT,
        y_target=y_target,
        x_r_max=x_r_max,
        save_path="figures/test_simulation.png",
        show=True,
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()