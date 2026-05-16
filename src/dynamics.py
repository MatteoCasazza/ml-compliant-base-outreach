"""
dynamics.py
===========
Two-degree-of-freedom system: controllable robot mounted on a compliant base.

Coordinates:
- x_b: passive base position
- x_r: robot position relative to the base
- y = x_b + x_r: total end-effector outreach

Input:
- x_rd(t): commanded robot position, generated as a chirp excitation signal

Main output:
- peak_y: maximum total outreach reached during the simulation

This module is used as the physics-based simulator of the project.
It generates the dynamic response of the system for a given set of physical
and control parameters.

Reference:
Roveda et al. (2016) - Mechatronics 39
Simplified model without external environment interaction.

Author: MatteoCasazza
Date: 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Parameter order used when params are passed as an array.
# This order must be consistent with dataset.py and models.py.
PARAM_ORDER = ['Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start']


def chirp_signal(t, f0, f1, T, A, x_r_start):
    """
    Generate a linear-frequency chirp command signal.

    The signal is generated using a sine function so that the initial command
    is consistent with the initial robot position:

        x_rd(0) = x_r_start

    This avoids an artificial initial jump in the commanded trajectory.

    Inputs
    ------
    t : float or array-like
        Time instant(s) [s].
    f0 : float
        Initial chirp frequency [Hz].
    f1 : float
        Final chirp frequency [Hz].
    T : float
        Total chirp duration [s].
    A : float
        Chirp amplitude [m].
    x_r_start : float
        Initial/offset robot position relative to the base [m].

    Output
    ------
    x_rd : float or ndarray
        Commanded robot position [m].
    """
    t = np.asarray(t)

    if T <= 0:
        return x_r_start * np.ones_like(t, dtype=float)

    chirp_rate = (f1 - f0) / T
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)

    return x_r_start + A * np.sin(phase)


def prepare_params(params, T_sim):
    """
    Convert input parameters to a dictionary and add default values.

    This function avoids modifying the original input dictionary.

    Inputs
    ------
    params : dict or array-like
        If dict, it must contain:
            Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start

        If array-like, the expected order is:
            [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]

    T_sim : float
        Simulation time [s]. It is also used as chirp duration.

    Output
    ------
    params : dict
        Complete parameter dictionary, including:
            T  : chirp duration [s]
            Mr : robot mass [kg], default 10.0
    """
    if isinstance(params, (list, tuple, np.ndarray)):
        params = dict(zip(PARAM_ORDER, params))
    else:
        params = params.copy()

    params['T'] = T_sim
    params.setdefault('Mr', 10.0)

    return params


def compute_damping(K, M, h):
    """
    Compute the viscous damping coefficient from the damping ratio.

    Formula:
        D = 2 h sqrt(K M)

    Inputs
    ------
    K : float
        Stiffness [N/m].
    M : float
        Mass [kg].
    h : float
        Damping ratio [-].

    Output
    ------
    D : float
        Viscous damping coefficient [Ns/m].
    """
    return 2.0 * h * np.sqrt(K * M)


def system_dynamics(t, state, params):
    """
    Compute the time derivative of the 2-DoF system state.

    State definition:
        state = [dx_b, dx_r, x_b, x_r]

    where:
    - dx_b: base velocity [m/s]
    - dx_r: robot relative velocity [m/s]
    - x_b : base position [m]
    - x_r : robot position relative to the base [m]

    The absolute end-effector position is:
        y = x_b + x_r

    Inputs
    ------
    t : float
        Current time [s].
    state : array-like, shape (4,)
        Current state [dx_b, dx_r, x_b, x_r].
    params : dict
        System and excitation parameters:
            Kb, Kr : base and robot stiffness [N/m]
            Mb, Mr : base and robot mass [kg]
            hb, hr : base and robot damping ratio [-]
            f0, f1 : chirp initial/final frequency [Hz]
            A      : chirp amplitude [m]
            x_r_start : initial/offset robot position [m]
            T      : chirp duration [s]

    Output
    ------
    dstate : list
        State derivative [ddx_b, ddx_r, dx_b, dx_r].
    """
    dx_b, dx_r, x_b, x_r = state

    Kb = params['Kb']
    Kr = params['Kr']
    Mb = params['Mb']
    Mr = params.get('Mr', 10.0)

    hb = params['hb']
    hr = params['hr']

    f0 = params['f0']
    f1 = params['f1']
    A = params['A']
    x_r_start = params['x_r_start']
    T = params.get('T', 60.0)

    Db = compute_damping(Kb, Mb, hb)
    Dr = compute_damping(Kr, Mr, hr)

    x_rd = chirp_signal(t, f0, f1, T, A, x_r_start)

    # Dynamic equations.
    # Since x_r is relative to the base, the relative acceleration of the robot
    # also depends on the acceleration of the passive base.
    ddx_b = (Dr * dx_r + Kr * (x_r - x_rd) - Db * dx_b - Kb * x_b) / Mb
    ddx_r = (-Dr * dx_r - Kr * (x_r - x_rd)) / Mr - ddx_b

    return [ddx_b, ddx_r, dx_b, dx_r]


def compute_metrics(sol, x_r_max=0.5, y_target=None):
    """
    Compute physical performance metrics from the simulation result.

    Inputs
    ------
    sol : OdeResult
        Output object returned by scipy.integrate.solve_ivp.
    x_r_max : float, optional
        Maximum admissible robot relative position [m].
        This value also represents the nominal robot reach.
        Default is 0.5 m.
    y_target : float, optional
        Desired total outreach target [m].

    Outputs
    -------
    metrics : dict
        Dictionary containing:
            peak_y               : maximum total outreach [m]
            t_peak               : time at which peak_y occurs [s]
            final_y              : final total outreach [m]
            max_xr               : maximum robot relative position [m]
            min_xr               : minimum robot relative position [m]
            max_abs_xr           : maximum absolute robot relative position [m]
            max_xb               : maximum base position [m]
            min_xb               : minimum base position [m]
            max_abs_xb           : maximum absolute base position [m]
            x_r_max              : nominal robot reach limit [m]
            extra_reach          : peak_y - x_r_max [m]
            constraint_violation : violation of robot upper limit [m]

        If y_target is provided:
            y_target       : target outreach [m]
            target_error   : absolute error between peak_y and target [m]
            target_reached : True if peak_y >= y_target
    """
    dx_b, dx_r, x_b, x_r = sol.y
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
    extra_reach = peak_y - x_r_max

    metrics = {
        'peak_y': peak_y,
        't_peak': float(sol.t[peak_idx]),
        'final_y': float(y[-1]),
        'max_xr': max_xr,
        'min_xr': min_xr,
        'max_abs_xr': max_abs_xr,
        'max_xb': max_xb,
        'min_xb': min_xb,
        'max_abs_xb': max_abs_xb,
        'x_r_max': float(x_r_max),
        'extra_reach': float(extra_reach),
        'constraint_violation': float(constraint_violation),
    }

    if y_target is not None:
        metrics['y_target'] = float(y_target)
        metrics['target_error'] = float(abs(peak_y - y_target))
        metrics['target_reached'] = bool(peak_y >= y_target)

    return metrics


def simulate_system(
    params,
    T_sim=60.0,
    dt=0.001,
    return_full=False,
    return_metrics=False,
    x_r_max=0.5,
    y_target=None
):
    """
    Simulate the 2-DoF system.

    Inputs
    ------
    params : dict or array-like
        System and excitation parameters.

        If dict, expected keys are:
            Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start

        Optional key:
            Mr, robot mass [kg]. If not provided, Mr = 10.0 kg.

        If array-like, expected order is:
            [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]

    T_sim : float, optional
        Simulation duration [s].
        Default is 60.0 s.
    dt : float, optional
        Time step used for storing the solution [s].
        Default is 0.001 s.
    return_full : bool, optional
        If True, return the complete ODE solution.
    return_metrics : bool, optional
        If True, return additional physical metrics.
    x_r_max : float, optional
        Maximum admissible robot relative position [m].
        Used only for metric computation.
    y_target : float, optional
        Desired total outreach target [m].
        Used only for metric computation.

    Outputs
    -------
    Default:
        peak_y : float
            Maximum total outreach [m].

    If return_full=True:
        peak_y : float
            Maximum total outreach [m].
        sol : OdeResult
            Full simulation result.

    If return_metrics=True:
        peak_y : float
            Maximum total outreach [m].
        metrics : dict
            Physical performance metrics.

    If return_full=True and return_metrics=True:
        peak_y : float
            Maximum total outreach [m].
        sol : OdeResult
            Full simulation result.
        metrics : dict
            Physical performance metrics.
    """
    params = prepare_params(params, T_sim)

    # Initial state: [dx_b, dx_r, x_b, x_r]
    x0 = [0.0, 0.0, 0.0, params['x_r_start']]

    t_eval = np.arange(0.0, T_sim, dt)

    sol = solve_ivp(
        system_dynamics,
        t_span=[0.0, T_sim],
        y0=x0,
        t_eval=t_eval,
        args=(params,),
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    if not sol.success:
        if return_full and return_metrics:
            return np.nan, sol, {}
        if return_full:
            return np.nan, sol
        if return_metrics:
            return np.nan, {}
        return np.nan

    metrics = compute_metrics(sol, x_r_max=x_r_max, y_target=y_target)
    peak_y = metrics['peak_y']

    if return_full and return_metrics:
        return peak_y, sol, metrics

    if return_full:
        return peak_y, sol

    if return_metrics:
        return peak_y, metrics

    return peak_y


def plot_simulation_example(
    params,
    T_sim=60.0,
    dt=0.001,
    y_target=None,
    x_r_max=0.5,
    save_path=None
):
    """
    Plot a complete simulation example for validation.

    The figure shows:
    1. Commanded chirp signal x_rd(t)
    2. Passive base position x_b(t)
    3. Robot relative position x_r(t)
    4. Total outreach y(t) = x_b(t) + x_r(t)

    Optional horizontal lines are added for:
    - target outreach y_target
    - nominal robot reach x_r_max
    - achieved peak outreach

    Inputs
    ------
    params : dict or array-like
        System and excitation parameters.
    T_sim : float, optional
        Simulation duration [s].
    dt : float, optional
        Time step used for storing the solution [s].
    y_target : float, optional
        Desired total outreach target [m].
    x_r_max : float, optional
        Maximum admissible robot relative position [m].
    save_path : str, optional
        If provided, path where the figure is saved.

    Outputs
    -------
    peak_y : float
        Maximum total outreach [m].
    metrics : dict
        Physical performance metrics.
    """
    peak_y, sol, metrics = simulate_system(
        params,
        T_sim=T_sim,
        dt=dt,
        return_full=True,
        return_metrics=True,
        x_r_max=x_r_max,
        y_target=y_target
    )

    t = sol.t
    dx_b, dx_r, x_b, x_r = sol.y
    y = x_b + x_r

    params_prepared = prepare_params(params, T_sim)

    x_rd = chirp_signal(
        t,
        params_prepared['f0'],
        params_prepared['f1'],
        T_sim,
        params_prepared['A'],
        params_prepared['x_r_start']
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Commanded chirp signal
    axes[0].plot(t, x_rd, 'b-', linewidth=1.5, label='$x_{rd}$ command')
    axes[0].set_ylabel('Position [m]')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Excitation signal: chirp command', fontsize=11)

    # Passive base position
    axes[1].plot(t, x_b, 'purple', linewidth=1.5, label='$x_b$ passive base')
    axes[1].set_ylabel('Position [m]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Robot relative position
    axes[2].plot(t, x_r, 'red', linewidth=1.5, label='$x_r$ robot relative position')
    axes[2].axhline(
        x_r_max,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'Robot max = {x_r_max:.2f} m'
    )
    axes[2].set_ylabel('Position [m]')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Total outreach
    axes[3].plot(t, y, 'k-', linewidth=2, label='$y = x_b + x_r$ outreach')
    axes[3].axhline(
        peak_y,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Peak = {peak_y:.3f} m'
    )
    axes[3].axhline(
        x_r_max,
        color='black',
        linestyle=':',
        linewidth=1.5,
        label=f'Nominal reach = {x_r_max:.2f} m'
    )

    if y_target is not None:
        axes[3].axhline(
            y_target,
            color='green',
            linestyle='-.',
            linewidth=1.5,
            label=f'Target = {y_target:.3f} m'
        )

    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Outreach [m]')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title(
        f'Total outreach: peak = {peak_y:.3f} m, '
        f'extra reach = {metrics["extra_reach"]:.3f} m',
        fontsize=11
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")

    plt.show()

    return peak_y, metrics


# ============================================================================
# TEST: run this file to verify that the simulator works correctly
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: src/dynamics.py")
    print("=" * 70)

    test_params = {
        'Kb': 2000,
        'Kr': 1500,
        'Mb': 50,
        'hb': 0.2,
        'hr': 0.5,
        'f0': 0.001,
        'f1': 5,
        'A': 0.10,
        'x_r_start': 0.4,
        'Mr': 10
    }

    x_r_max = 0.5
    y_target = 0.55

    print("\nTest parameters:")
    for key, val in test_params.items():
        print(f"  {key:12s} = {val}")

    print("\nRunning simulation...")
    peak_y, metrics = simulate_system(
        test_params,
        T_sim=60,
        dt=0.001,
        return_metrics=True,
        x_r_max=x_r_max,
        y_target=y_target
    )

    print("\n✓ Simulation completed!")
    print(f"  Peak outreach:          {peak_y:.4f} m")
    print(f"  Nominal robot reach:    {metrics['x_r_max']:.4f} m")
    print(f"  Extra reach:            {metrics['extra_reach']:.4f} m")
    print(f"  Max robot position xr:  {metrics['max_xr']:.4f} m")
    print(f"  Constraint violation:   {metrics['constraint_violation']:.4f} m")
    print(f"  Target error:           {metrics.get('target_error', np.nan):.4f} m")

    print("\nGenerating plot...")
    plot_simulation_example(
        test_params,
        T_sim=60,
        dt=0.001,
        y_target=y_target,
        x_r_max=x_r_max,
        save_path='figures/test_simulation.png'
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)