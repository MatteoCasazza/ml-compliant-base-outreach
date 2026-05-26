"""
MPC Control Experiment for ML Compliant Base Outreach Project

This module implements a Model Predictive Control (MPC) approach as an 
INDEPENDENT, ADVANCED EXTENSION to the main project pipeline.

Context:
--------
The main pipeline uses:
- Dynamic simulator
- Gaussian Process surrogate model  
- Inverse optimization with Differential Evolution
- Bayesian Optimization

This MPC extension:
------------------
- Uses the TRUE PHYSICAL MODEL (not data-driven surrogate)
- Implements closed-loop receding horizon control
- Solves a convex QP at each time step using cvxpy + OSQP
- Tracks a desired outreach reference trajectory
- Demonstrates advanced control theory application

The MPC formulation uses:
- Discretized linear state-space model (c2d from continuous dynamics)
- Finite horizon N with quadratic cost
- Hard constraints on input bounds and rate limits
- Terminal cost for stability

This is NOT replacing the inverse optimization methods.
This is an exploratory control-theoretic comparison showing what happens
when you have perfect model knowledge and can replan at each step.

Author: MatteoCasazza
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path
import warnings

# Check for cvxpy availability
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("\n" + "="*70)
    print("ERROR: cvxpy not found!")
    print("="*70)
    print("\nTo run MPC, please install cvxpy and OSQP solver:")
    print("\n  pip install cvxpy osqp\n")
    print("Then run this script again.")
    print("="*70 + "\n")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths (relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mpc"
FIGURES_DIR = PROJECT_ROOT / "figures" / "mpc"

# Physical parameters (fixed base parameters)
PARAMS = {
    'Kb': 2000.0,      # Base stiffness [N/m]
    'Mb': 50.0,        # Base mass [kg]
    'hb': 0.2,         # Base damping ratio [-]
    'Mr': 10.0,        # Robot mass [kg]
}

# Robot parameters from BEST Bayesian Optimization solution (central target)
PARAMS['Kr'] = 3482.193396   # Robot stiffness [N/m]
PARAMS['hr'] = 0.430359      # Robot damping ratio [-]

# Target reference
Y_REF = 0.592779  # [m] - Central target from Bayesian Optimization

# Initial state: z0 = [dxb0, dxr0, xb0, xr0]
Z0 = np.array([0.0, 0.0, 0.0, 0.455427])
# Initial outreach: y0 = xb0 + xr0 = 0.455427 m

# Input constraints
U_MIN = 0.30  # [m] minimum robot relative position command
U_MAX = 0.65  # [m] maximum robot relative position command
DU_MAX = 0.02 # [m] maximum input rate of change per time step

# Simulation settings
DT = 0.02          # [s] time step
T_TOTAL = 5.0      # [s] total simulation time
N_SIM = int(T_TOTAL / DT)  # number of simulation steps

# MPC settings
N_HORIZON = 30     # MPC prediction horizon (steps)

# MPC cost weights
QY = 1000.0         # Output tracking weight
QY_TERMINAL = 5000.0  # Terminal output cost weight
RU = 0.1            # Input magnitude penalty
RDU = 10.0          # Input rate penalty (smoothness)

# Output observation matrix: y = C @ z = xb + xr
C = np.array([0.0, 0.0, 1.0, 1.0])

# Reference errors from inverse methods (for comparison plot)
DE_ERROR_MM = 0.339   # Differential Evolution final error [mm]
BO_ERROR_MM = 0.039   # Bayesian Optimization final error [mm]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Create output directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


def compute_damping(K, M, h):
    """
    Compute damping coefficient from stiffness, mass, and damping ratio.
    
    D = 2 * h * sqrt(K * M)
    
    Parameters
    ----------
    K : float
        Stiffness [N/m]
    M : float
        Mass [kg]
    h : float
        Damping ratio [-]
    
    Returns
    -------
    float
        Damping coefficient [N·s/m]
    """
    return 2.0 * h * np.sqrt(K * M)


# =============================================================================
# DYNAMICS MODEL
# =============================================================================

def build_continuous_state_space(params):
    """
    Build continuous-time state-space matrices A, B from physical parameters.
    
    State: z = [dxb, dxr, xb, xr]
    Input: u = xrd (robot relative position command)
    Output: y = xb + xr (total outreach)
    
    Dynamics:
    ---------
    ddxb = (Dr*dxr + Kr*(xr - xrd) - Db*dxb - Kb*xb) / Mb
    ddxr = (-Dr*dxr - Kr*(xr - xrd)) / Mr - ddxb
    
    State-space:
    ------------
    z_dot = A z + B u
    y = C z
    
    Parameters
    ----------
    params : dict
        Physical parameters {Kb, Mb, hb, Mr, Kr, hr}
    
    Returns
    -------
    A : ndarray (4, 4)
        Continuous state matrix
    B : ndarray (4, 1)
        Continuous input matrix
    """
    Kb = params['Kb']
    Mb = params['Mb']
    hb = params['hb']
    Mr = params['Mr']
    Kr = params['Kr']
    hr = params['hr']
    
    # Compute damping coefficients
    Db = compute_damping(Kb, Mb, hb)
    Dr = compute_damping(Kr, Mr, hr)
    
    # Build A matrix (4x4)
    # Row 0: ddxb equation
    # Row 1: ddxr equation
    # Row 2: dxb = integral of dxb
    # Row 3: dxr = integral of dxr
    
    A = np.array([
        [-Db/Mb,              Dr/Mb,             -Kb/Mb,              Kr/Mb],
        [ Db/Mb, -(Dr/Mr + Dr/Mb),              Kb/Mb, -(Kr/Mr + Kr/Mb)],
        [    1.0,                0.0,                0.0,                0.0],
        [    0.0,                1.0,                0.0,                0.0],
    ])
    
    # Build B matrix (4x1)
    B = np.array([
        [-Kr/Mb],
        [ Kr/Mr + Kr/Mb],
        [0.0],
        [0.0],
    ])
    
    return A, B


def discretize_system(A, B, dt):
    """
    Discretize continuous-time system using matrix exponential.
    
    Uses scipy.signal.cont2discrete with 'zoh' (zero-order hold) method.
    
    Parameters
    ----------
    A : ndarray (n, n)
        Continuous state matrix
    B : ndarray (n, m)
        Continuous input matrix  
    dt : float
        Time step [s]
    
    Returns
    -------
    Ad : ndarray (n, n)
        Discrete state matrix
    Bd : ndarray (n, m)
        Discrete input matrix
    """
    # Use scipy's cont2discrete for robust conversion
    sys_d = cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1]))), 
                          dt, method='zoh')
    Ad = sys_d[0]
    Bd = sys_d[1]
    
    return Ad, Bd


# =============================================================================
# MPC OPTIMIZATION
# =============================================================================

def solve_mpc_step(z_current, previous_u, Ad, Bd, C, y_ref, config):
    """
    Solve one MPC optimization step using cvxpy.
    
    Formulation:
    ------------
    minimize:
        sum_{k=0}^{N-1} [ Qy*(y[k]-y_ref)^2 + Ru*(u[k]-y_ref)^2 + Rdu*(u[k]-u_prev)^2 ]
        + Qy_terminal * (y[N] - y_ref)^2
    
    subject to:
        z[k+1] = Ad @ z[k] + Bd * u[k]
        y[k] = C @ z[k]
        u_min <= u[k] <= u_max
        -du_max <= u[0] - previous_u <= du_max
        -du_max <= u[k] - u[k-1] <= du_max  for k > 0
    
    Parameters
    ----------
    z_current : ndarray (4,)
        Current state
    previous_u : float
        Previously applied input (for rate constraint)
    Ad : ndarray (4, 4)
        Discrete state matrix
    Bd : ndarray (4, 1)
        Discrete input matrix
    C : ndarray (4,)
        Output matrix
    y_ref : float
        Reference output to track
    config : dict
        MPC configuration {N, Qy, Qy_terminal, Ru, Rdu, u_min, u_max, du_max}
    
    Returns
    -------
    u_opt : float or None
        Optimal first input u[0], or None if solve failed
    """
    N = config['N']
    Qy = config['Qy']
    Qy_terminal = config['Qy_terminal']
    Ru = config['Ru']
    Rdu = config['Rdu']
    u_min = config['u_min']
    u_max = config['u_max']
    du_max = config['du_max']
    
    n_states = Ad.shape[0]
    
    # Decision variables
    z = cp.Variable((n_states, N+1))  # States z[k] for k=0..N
    u = cp.Variable(N)                 # Inputs u[k] for k=0..N-1
    
    # Cost function
    cost = 0.0
    
    # Stage costs (k = 0 to N-1)
    for k in range(N):
        y_k = C @ z[:, k]
        cost += Qy * cp.square(y_k - y_ref)
        cost += Ru * cp.square(u[k] - y_ref)
        
        # Input rate penalty
        if k == 0:
            cost += Rdu * cp.square(u[k] - previous_u)
        else:
            cost += Rdu * cp.square(u[k] - u[k-1])
    
    # Terminal cost (k = N)
    y_N = C @ z[:, N]
    cost += Qy_terminal * cp.square(y_N - y_ref)
    
    # Constraints
    constraints = []
    
    # Initial condition
    constraints.append(z[:, 0] == z_current)
    
    # Dynamics constraints
    for k in range(N):
        constraints.append(z[:, k+1] == Ad @ z[:, k] + Bd.flatten() * u[k])
    
    # Input bounds
    constraints.append(u >= u_min)
    constraints.append(u <= u_max)
    
    # Input rate constraints
    constraints.append(u[0] - previous_u >= -du_max)
    constraints.append(u[0] - previous_u <= du_max)
    
    for k in range(1, N):
        constraints.append(u[k] - u[k-1] >= -du_max)
        constraints.append(u[k] - u[k-1] <= du_max)
    
    # Solve
    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return u.value[0]
        else:
            warnings.warn(f"MPC solve status: {problem.status}")
            return None
            
    except Exception as e:
        warnings.warn(f"MPC solve exception: {e}")
        return None


# =============================================================================
# CLOSED-LOOP SIMULATION
# =============================================================================

def simulate_mpc():
    """
    Run closed-loop MPC simulation.
    
    At each time step:
    1. Solve MPC optimization problem over horizon N
    2. Apply only first input u[0]
    3. Propagate discrete dynamics one step
    4. Repeat
    
    Returns
    -------
    results_df : pd.DataFrame
        Simulation results with columns [time, dxb, dxr, xb, xr, y, u, tracking_error_m, tracking_error_mm]
    n_failures : int
        Number of MPC solve failures
    """
    print("\n" + "="*70)
    print("STARTING MPC CLOSED-LOOP SIMULATION")
    print("="*70)
    
    # Build continuous model
    A, B = build_continuous_state_space(PARAMS)
    print(f"\n✓ Continuous state-space model built")
    print(f"  A shape: {A.shape}")
    print(f"  B shape: {B.shape}")
    
    # Discretize
    Ad, Bd = discretize_system(A, B, DT)
    print(f"✓ System discretized with dt = {DT} s")
    print(f"  Ad shape: {Ad.shape}")
    print(f"  Bd shape: {Bd.shape}")
    
    # MPC configuration
    mpc_config = {
        'N': N_HORIZON,
        'Qy': QY,
        'Qy_terminal': QY_TERMINAL,
        'Ru': RU,
        'Rdu': RDU,
        'u_min': U_MIN,
        'u_max': U_MAX,
        'du_max': DU_MAX,
    }
    
    print(f"\n✓ MPC configuration:")
    print(f"  Horizon N = {N_HORIZON} steps = {N_HORIZON * DT} s")
    print(f"  Qy = {QY}, Qy_terminal = {QY_TERMINAL}")
    print(f"  Ru = {RU}, Rdu = {RDU}")
    print(f"  Input bounds: [{U_MIN}, {U_MAX}] m")
    print(f"  Rate limit: ±{DU_MAX} m/step")
    
    # Initialize simulation
    z_current = Z0.copy()
    previous_u = Z0[3]  # Start from initial robot position xr0
    
    # Storage
    time_history = []
    state_history = []
    input_history = []
    output_history = []
    error_history = []
    
    n_failures = 0
    
    print(f"\n✓ Simulation setup:")
    print(f"  Initial state z0 = {Z0}")
    print(f"  Initial output y0 = {Z0[2] + Z0[3]:.6f} m")
    print(f"  Target y_ref = {Y_REF:.6f} m")
    print(f"  Initial error = {(Z0[2] + Z0[3]) - Y_REF:.6f} m = {((Z0[2] + Z0[3]) - Y_REF)*1000:.2f} mm")
    print(f"  Total time = {T_TOTAL} s")
    print(f"  Steps = {N_SIM}")
    
    print(f"\n{'='*70}")
    print("Running MPC loop...")
    print(f"{'='*70}\n")
    
    # MPC loop
    for step in range(N_SIM):
        t = step * DT
        
        # Current output
        y_current = C @ z_current
        
        # Solve MPC
        u_opt = solve_mpc_step(z_current, previous_u, Ad, Bd, C, Y_REF, mpc_config)
        
        if u_opt is None:
            # MPC solve failed - reuse previous input
            u_opt = previous_u
            n_failures += 1
            if n_failures == 1:
                print(f"⚠ MPC solve failed at step {step} (t={t:.2f}s) - reusing previous input")
        
        # Apply input and propagate dynamics
        z_next = Ad @ z_current + Bd.flatten() * u_opt
        
        # Store results
        time_history.append(t)
        state_history.append(z_current.copy())
        input_history.append(u_opt)
        output_history.append(y_current)
        error_history.append(y_current - Y_REF)
        
        # Update for next iteration
        z_current = z_next
        previous_u = u_opt
        
        # Progress indicator
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{N_SIM} | t={t:.2f}s | y={y_current:.6f}m | u={u_opt:.6f}m | error={error_history[-1]*1000:+.2f}mm")
    
    # Final state
    t_final = T_TOTAL
    y_final = C @ z_current
    time_history.append(t_final)
    state_history.append(z_current.copy())
    input_history.append(previous_u)  # Pad for alignment
    output_history.append(y_final)
    error_history.append(y_final - Y_REF)
    
    print(f"\n{'='*70}")
    print("MPC simulation complete!")
    print(f"{'='*70}\n")
    
    if n_failures > 0:
        print(f"⚠ Warning: {n_failures} MPC solve failures occurred")
        print(f"  ({n_failures/N_SIM*100:.1f}% of steps)\n")
    else:
        print(f"✓ All {N_SIM} MPC solves succeeded\n")
    
    # Convert to DataFrame
    state_array = np.array(state_history)
    
    results_df = pd.DataFrame({
        'time': time_history,
        'dxb': state_array[:, 0],
        'dxr': state_array[:, 1],
        'xb': state_array[:, 2],
        'xr': state_array[:, 3],
        'y': output_history,
        'u': input_history,
        'tracking_error_m': error_history,
        'tracking_error_mm': np.array(error_history) * 1000.0,
    })
    
    return results_df, n_failures


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def compute_metrics(df):
    """
    Compute performance metrics from simulation results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation results
    
    Returns
    -------
    dict
        Performance metrics
    """
    metrics = {
        'y_ref': Y_REF,
        'initial_y': df['y'].iloc[0],
        'final_y': df['y'].iloc[-1],
        'peak_y': df['y'].max(),
        'final_error_m': df['tracking_error_m'].iloc[-1],
        'final_error_mm': df['tracking_error_mm'].iloc[-1],
        'mean_abs_error_m': df['tracking_error_m'].abs().mean(),
        'mean_abs_error_mm': df['tracking_error_mm'].abs().mean(),
        'max_abs_error_m': df['tracking_error_m'].abs().max(),
        'max_abs_error_mm': df['tracking_error_mm'].abs().max(),
        'u_min_used': df['u'].min(),
        'u_max_used': df['u'].max(),
    }

    abs_err_mm = df["tracking_error_mm"].abs()
    post_transient = df[df["time"] >= 0.5]
    post_abs_err_mm = post_transient["tracking_error_mm"].abs()

    within_1mm = abs_err_mm <= 1.0
    settling_time = np.nan
    for i in range(len(df)):
        if np.all(within_1mm.iloc[i:]):
            settling_time = df["time"].iloc[i]
            break
    
    metrics.update({
        "overshoot_m": max(0.0, df["y"].max() - Y_REF),
        "overshoot_mm": max(0.0, df["y"].max() - Y_REF) * 1000.0,
        "settling_time_1mm_s": settling_time,
        "mean_abs_error_after_0p5s_mm": post_abs_err_mm.mean(),
        "max_abs_error_after_0p5s_mm": post_abs_err_mm.max(),
    })

    return metrics


def save_results(df, metrics, n_failures):
    """
    Save results and summary to CSV files.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation results
    metrics : dict
        Performance metrics
    n_failures : int
        Number of MPC failures
    """
    # Save detailed results
    results_path = RESULTS_DIR / "mpc_results.csv"
    df.to_csv(results_path, index=False)
    print(f"✓ Detailed results saved to: {results_path}")
    
    # Save summary
    summary = {
        **metrics,
        'n_failures': n_failures,
        'dt': DT,
        'T_total': T_TOTAL,
        'N_horizon': N_HORIZON,
        'Qy': QY,
        'Qy_terminal': QY_TERMINAL,
        'Ru': RU,
        'Rdu': RDU,
        'du_max': DU_MAX,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = RESULTS_DIR / "mpc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")


def plot_tracking(df):
    """Plot output tracking: y(t) vs y_ref."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['time'], df['y'], 'b-', linewidth=2, label='Output y(t)')
    ax.axhline(Y_REF, color='r', linestyle='--', linewidth=2, label=f'Reference y_ref = {Y_REF:.4f} m')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Outreach [m]', fontsize=12)
    ax.set_title('MPC Output Tracking Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_tracking.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tracking plot saved to: {path}")


def plot_control_signal(df):
    """Plot control input u(t) with bounds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['time'], df['u'], 'g-', linewidth=2, label='Control input u(t)')
    ax.axhline(U_MIN, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'u_min = {U_MIN} m')
    ax.axhline(U_MAX, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'u_max = {U_MAX} m')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Control Input u (xrd) [m]', fontsize=12)
    ax.set_title('MPC Control Signal', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([U_MIN - 0.05, U_MAX + 0.05])
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_control_signal.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Control signal plot saved to: {path}")


def plot_tracking_error(df):
    """Plot tracking error over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['time'], df['tracking_error_mm'], 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Tracking Error [mm]', fontsize=12)
    ax.set_title('MPC Tracking Error (y - y_ref)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_tracking_error.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tracking error plot saved to: {path}")


def plot_phase_space(df):
    """Plot phase space: xb vs xr."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color by time
    scatter = ax.scatter(df['xb'], df['xr'], c=df['time'], cmap='viridis', 
                        s=30, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # Mark start and end
    ax.plot(df['xb'].iloc[0], df['xr'].iloc[0], 'go', markersize=12, 
            label='Start', markeredgecolor='k', markeredgewidth=1.5)
    ax.plot(df['xb'].iloc[-1], df['xr'].iloc[-1], 'rs', markersize=12, 
            label='End', markeredgecolor='k', markeredgewidth=1.5)
    
    ax.set_xlabel('Base Displacement xb [m]', fontsize=12)
    ax.set_ylabel('Robot Relative Displacement xr [m]', fontsize=12)
    ax.set_title('MPC Phase Space Trajectory', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time [s]', fontsize=11)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_phase_space.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Phase space plot saved to: {path}")


def plot_mpc_vs_inverse_methods(metrics):
    """
    Compare MPC final error with DE and BO on the central target.
    
    This plot connects the MPC extension to the main inverse optimization
    pipeline, showing how closed-loop MPC compares to open-loop inverse
    design methods when tracking the same target.
    
    Parameters
    ----------
    metrics : dict
        Performance metrics from MPC simulation
    """
    de_error_mm = DE_ERROR_MM
    bo_error_mm = BO_ERROR_MM
    mpc_error_mm = abs(metrics['final_error_mm'])
    
    methods = ['DE + GP', 'Bayesian Opt.', 'MPC']
    errors = [de_error_mm, bo_error_mm, mpc_error_mm]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Final Tracking Error [mm]', fontsize=12)
    ax.set_title('Central Target: Nominal MPC Tracking vs Open-Loop Inverse Design', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate bars with values
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{err:.3f} mm',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add note
    note_text = (
        "Note: DE and BO optimize open-loop parameters;\n"
        "MPC uses closed-loop receding horizon control."
    )
    ax.text(0.98, 0.98, note_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_vs_inverse_methods.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ MPC comparison plot saved to: {path}")


def print_summary(metrics, n_failures):
    """Print formatted summary of MPC performance."""
    print("\n" + "="*70)
    print("MPC PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nTarget Configuration:")
    print(f"  Reference y_ref        = {metrics['y_ref']:.6f} m")
    print(f"\nTracking Performance:")
    print(f"  Initial outreach       = {metrics['initial_y']:.6f} m")
    print(f"  Final outreach         = {metrics['final_y']:.6f} m")
    print(f"  Peak outreach          = {metrics['peak_y']:.6f} m")
    print(f"\nError Metrics:")
    print(f"  Final error            = {metrics['final_error_m']:.6f} m  ({metrics['final_error_mm']:+.2f} mm)")
    print(f"  Mean absolute error    = {metrics['mean_abs_error_m']:.6f} m  ({metrics['mean_abs_error_mm']:.2f} mm)")
    print(f"  Max absolute error     = {metrics['max_abs_error_m']:.6f} m  ({metrics['max_abs_error_mm']:.2f} mm)")
    print(f"\nControl Performance:")
    print(f"  Input range used       = [{metrics['u_min_used']:.6f}, {metrics['u_max_used']:.6f}] m")
    print(f"  Input bounds           = [{U_MIN}, {U_MAX}] m")
    
    if n_failures > 0:
        print(f"\nSolver Status:")
        print(f"  ⚠ MPC failures         = {n_failures} / {N_SIM} ({n_failures/N_SIM*100:.1f}%)")
    else:
        print(f"\nSolver Status:")
        print(f"  ✓ All solves succeeded = {N_SIM} / {N_SIM} (100%)")
    
    print(f"\nComparison with Inverse Methods (Central Target):")
    print(f"  DE + GP final error    = {DE_ERROR_MM:.3f} mm")
    print(f"  Bayesian Opt. error    = {BO_ERROR_MM:.3f} mm")
    print(f"  MPC final error        = {abs(metrics['final_error_mm']):.3f} mm")
    
    print(f"  Overshoot              = {metrics['overshoot_m']:.6f} m  ({metrics['overshoot_mm']:.2f} mm)")
    print(f"  Settling time ±1 mm    = {metrics['settling_time_1mm_s']:.3f} s")
    print(f"  Mean error after 0.5s  = {metrics['mean_abs_error_after_0p5s_mm']:.3f} mm")
    print(f"  Max error after 0.5s   = {metrics['max_abs_error_after_0p5s_mm']:.3f} mm")

    print("="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MPC CONTROL EXPERIMENT")
    print("ML Compliant Base Outreach Project")
    print("="*70)
    print("\nThis is an ADVANCED EXTENSION using true Model Predictive Control.")
    print("It demonstrates closed-loop receding horizon optimization with the")
    print("physical model, as a control-theoretic comparison to the main")
    print("inverse optimization pipeline.\n")
    
    # Setup
    ensure_dirs()
    
    # Run MPC simulation
    df, n_failures = simulate_mpc()
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING PERFORMANCE METRICS")
    print("="*70)
    metrics = compute_metrics(df)
    print("✓ Metrics computed")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    save_results(df, metrics, n_failures)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_tracking(df)
    plot_control_signal(df)
    plot_tracking_error(df)
    plot_phase_space(df)
    plot_mpc_vs_inverse_methods(metrics)
    
    # Print summary
    print_summary(metrics, n_failures)
    
    print("="*70)
    print("MPC EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nOutputs generated:")
    print(f"  • results/mpc_results.csv")
    print(f"  • results/mpc_summary.csv")
    print(f"  • figures/mpc_tracking.png")
    print(f"  • figures/mpc_control_signal.png")
    print(f"  • figures/mpc_tracking_error.png")
    print(f"  • figures/mpc_phase_space.png")
    print(f"  • figures/mpc_vs_inverse_methods.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()