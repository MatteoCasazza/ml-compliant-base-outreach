"""
MPC Robustness Analysis for ML Compliant Base Outreach Project

This module extends the MPC control experiment by testing robustness under
plant/model mismatch through Monte Carlo simulation.

Context:
--------
The nominal MPC (src/mpc_control.py) achieves near-perfect tracking because
it uses the same model for both control and simulation. This robustness
analysis is more realistic: the MPC controller uses a nominal model while
the true plant has perturbed parameters.

Key Concept:
------------
- Controller model = NOMINAL parameters (what MPC thinks the system is)
- Plant model = PERTURBED parameters (what the system actually is)

This tests how well the closed-loop MPC handles model uncertainty, which is
critical for real-world deployment where perfect model knowledge is never
available.

Monte Carlo Setup:
------------------
- N = 50 runs with random parameter perturbations
- Parameters perturbed: Kb, Mb, hb, Kr, hr (±5%)
- Mr kept fixed (actuator mass assumed known)
- Same initial condition, target, and MPC settings for all runs

This is NOT replacing the GP + DE/BO inverse design methods.
This is an exploratory control-theoretic robustness extension.

Author: Generated for ML Compliant Base Outreach Project
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path
import warnings
from typing import Dict, Tuple, Optional

# Check for cvxpy availability
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("\n" + "="*70)
    print("ERROR: cvxpy not found!")
    print("="*70)
    print("\nTo run MPC robustness analysis, please install cvxpy and OSQP:")
    print("\n  pip install cvxpy osqp\n")
    print("Then run this script again.")
    print("="*70 + "\n")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mpc"
FIGURES_DIR = PROJECT_ROOT / "figures" / "mpc"

# Nominal physical parameters (from best BO solution)
NOMINAL_PARAMS = {
    'Kb': 2000.0,
    'Mb': 50.0,
    'hb': 0.2,
    'Mr': 10.0,      # Fixed (not perturbed)
    'Kr': 3482.193396,
    'hr': 0.430359,
}

# Target reference
Y_REF = 0.592779  # [m] - Central target

# Initial state: z0 = [dxb0, dxr0, xb0, xr0]
Z0 = np.array([0.0, 0.0, 0.0, 0.455427])

# Input constraints
U_MIN = 0.30
U_MAX = 0.65
DU_MAX = 0.02

# Simulation settings
DT = 0.02
T_TOTAL = 5.0
N_SIM = int(T_TOTAL / DT)

# MPC settings
N_HORIZON = 30
QY = 1000.0
QY_TERMINAL = 5000.0
RU = 0.1
RDU = 10.0

# Output matrix
C = np.array([0.0, 0.0, 1.0, 1.0])

# Robustness analysis settings
N_ROBUST = 30           # Number of Monte Carlo runs
PERTURBATION_LEVEL = 0.15  # ±5% perturbation
RANDOM_STATE = 42       # For reproducibility

RUN_REFERENCE_CHANGE = False

# Time-varying reference for optional experiment
REF_SCHEDULE = [
    (0.0, 1.5, 0.520767),  # Low target
    (1.5, 3.0, 0.592779),  # Central target
    (3.0, 5.0, 0.628967),  # High target
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Create output directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


def compute_damping(K: float, M: float, h: float) -> float:
    """
    Compute damping coefficient from stiffness, mass, and damping ratio.
    
    D = 2 * h * sqrt(K * M)
    """
    return 2.0 * h * np.sqrt(K * M)


# =============================================================================
# DYNAMICS MODEL
# =============================================================================

def build_continuous_state_space(params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time state-space matrices A, B.
    
    State: z = [dxb, dxr, xb, xr]
    Input: u = xrd
    Output: y = xb + xr
    
    Returns
    -------
    A : ndarray (4, 4)
    B : ndarray (4, 1)
    """
    Kb = params['Kb']
    Mb = params['Mb']
    hb = params['hb']
    Mr = params['Mr']
    Kr = params['Kr']
    hr = params['hr']
    
    Db = compute_damping(Kb, Mb, hb)
    Dr = compute_damping(Kr, Mr, hr)
    
    A = np.array([
        [-Db/Mb,              Dr/Mb,             -Kb/Mb,              Kr/Mb],
        [ Db/Mb, -(Dr/Mr + Dr/Mb),              Kb/Mb, -(Kr/Mr + Kr/Mb)],
        [    1.0,                0.0,                0.0,                0.0],
        [    0.0,                1.0,                0.0,                0.0],
    ])
    
    B = np.array([
        [-Kr/Mb],
        [ Kr/Mr + Kr/Mb],
        [0.0],
        [0.0],
    ])
    
    return A, B


def discretize_system(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous system using zero-order hold.
    
    Returns
    -------
    Ad : ndarray (4, 4)
    Bd : ndarray (4, 1)
    """
    sys_d = cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1]))), 
                          dt, method='zoh')
    return sys_d[0], sys_d[1]


def build_discrete_model(params: Dict, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete-time model from parameters.
    
    Returns
    -------
    Ad : ndarray (4, 4)
    Bd : ndarray (4, 1)
    """
    A, B = build_continuous_state_space(params)
    Ad, Bd = discretize_system(A, B, dt)
    return Ad, Bd


# =============================================================================
# MPC OPTIMIZATION
# =============================================================================

def solve_mpc_step(z_current: np.ndarray, 
                   previous_u: float,
                   Ad_nominal: np.ndarray,
                   Bd_nominal: np.ndarray,
                   C: np.ndarray,
                   y_ref: float,
                   config: Dict) -> Optional[float]:
    """
    Solve one MPC optimization step using nominal model.
    
    This is the CONTROLLER - it only knows the nominal model.
    The solution will be applied to the (possibly different) plant.
    
    Returns
    -------
    u_opt : float or None
        Optimal first input, or None if solve failed
    """
    N = config['N']
    Qy = config['Qy']
    Qy_terminal = config['Qy_terminal']
    Ru = config['Ru']
    Rdu = config['Rdu']
    u_min = config['u_min']
    u_max = config['u_max']
    du_max = config['du_max']
    
    n_states = Ad_nominal.shape[0]
    
    # Decision variables
    z = cp.Variable((n_states, N+1))
    u = cp.Variable(N)
    
    # Cost function
    cost = 0.0
    
    for k in range(N):
        y_k = C @ z[:, k]
        cost += Qy * cp.square(y_k - y_ref)
        cost += Ru * cp.square(u[k] - y_ref)
        
        if k == 0:
            cost += Rdu * cp.square(u[k] - previous_u)
        else:
            cost += Rdu * cp.square(u[k] - u[k-1])
    
    y_N = C @ z[:, N]
    cost += Qy_terminal * cp.square(y_N - y_ref)
    
    # Constraints
    constraints = [z[:, 0] == z_current]
    
    for k in range(N):
        constraints.append(z[:, k+1] == Ad_nominal @ z[:, k] + Bd_nominal.flatten() * u[k])
    
    constraints.append(u >= u_min)
    constraints.append(u <= u_max)
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
            return None
            
    except Exception:
        return None


# =============================================================================
# PARAMETER PERTURBATION
# =============================================================================

def perturb_params(nominal_params: Dict, 
                   rng: np.random.Generator,
                   perturbation_level: float) -> Dict:
    """
    Generate perturbed plant parameters.
    
    Perturbs: Kb, Mb, hb, Kr, hr by ±perturbation_level
    Keeps: Mr fixed
    
    Parameters
    ----------
    nominal_params : dict
        Nominal parameter values
    rng : np.random.Generator
        Random number generator
    perturbation_level : float
        Perturbation magnitude (e.g., 0.05 for ±5%)
    
    Returns
    -------
    perturbed_params : dict
        Perturbed parameter values
    """
    perturbed = nominal_params.copy()
    
    # Parameters to perturb
    perturb_keys = ['Kb', 'Mb', 'hb', 'Kr', 'hr']
    
    for key in perturb_keys:
        nominal_value = nominal_params[key]
        # Uniform perturbation in [nominal * (1-δ), nominal * (1+δ)]
        perturbation_factor = rng.uniform(1 - perturbation_level, 1 + perturbation_level)
        perturbed[key] = nominal_value * perturbation_factor
    
    # Mr stays fixed
    perturbed['Mr'] = nominal_params['Mr']
    
    return perturbed


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(df: pd.DataFrame, n_failures: int) -> Dict:
    """
    Compute performance metrics from simulation results.
    
    Returns
    -------
    metrics : dict
        Performance metrics
    """
    # Basic tracking metrics
    metrics = {
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
        'n_failures': n_failures,
    }
    
    # Post-transient metrics (after 0.5s)
    df_post = df[df['time'] >= 0.5]
    if len(df_post) > 0:
        metrics['mean_abs_error_after_0p5s_mm'] = df_post['tracking_error_mm'].abs().mean()
        metrics['max_abs_error_after_0p5s_mm'] = df_post['tracking_error_mm'].abs().max()
    else:
        metrics['mean_abs_error_after_0p5s_mm'] = np.nan
        metrics['max_abs_error_after_0p5s_mm'] = np.nan
    
    # Overshoot
    overshoot_m = max(0.0, df['y'].max() - Y_REF)
    metrics['overshoot_m'] = overshoot_m
    metrics['overshoot_mm'] = overshoot_m * 1000.0
    
    # Settling time (1mm threshold)
    settling_time = np.nan
    abs_errors = df['tracking_error_mm'].abs().values
    times = df['time'].values
    
    for i in range(len(abs_errors)):
        if np.all(abs_errors[i:] <= 1.0):
            settling_time = times[i]
            break
    
    metrics['settling_time_1mm_s'] = settling_time
    
    return metrics


# =============================================================================
# CLOSED-LOOP SIMULATION WITH PLANT MISMATCH
# =============================================================================

def simulate_mpc_with_plant_mismatch(run_id: int,
                                      plant_params: Dict,
                                      nominal_model: Tuple[np.ndarray, np.ndarray],
                                      config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate closed-loop MPC with plant/model mismatch.
    
    Key: MPC uses nominal_model, but dynamics evolve with plant_params.
    
    Parameters
    ----------
    run_id : int
        Monte Carlo run identifier
    plant_params : dict
        True plant parameters (perturbed)
    nominal_model : tuple
        (Ad_nominal, Bd_nominal) for MPC controller
    config : dict
        MPC configuration
    
    Returns
    -------
    df : pd.DataFrame
        Trajectory data
    metrics : dict
        Performance metrics
    """
    Ad_nominal, Bd_nominal = nominal_model
    
    # Build PLANT model (true system)
    Ad_plant, Bd_plant = build_discrete_model(plant_params, DT)
    
    # Initialize
    z_current = Z0.copy()
    previous_u = Z0[3]
    
    time_history = []
    state_history = []
    input_history = []
    output_history = []
    error_history = []
    
    n_failures = 0
    
    # MPC loop
    for step in range(N_SIM):
        t = step * DT
        y_current = C @ z_current
        
        # Solve MPC using NOMINAL model
        u_opt = solve_mpc_step(z_current, previous_u, Ad_nominal, Bd_nominal, 
                               C, Y_REF, config)
        
        if u_opt is None:
            u_opt = previous_u
            n_failures += 1
        
        # Apply input to TRUE PLANT
        z_next = Ad_plant @ z_current + Bd_plant.flatten() * u_opt
        
        # Store
        time_history.append(t)
        state_history.append(z_current.copy())
        input_history.append(u_opt)
        output_history.append(y_current)
        error_history.append(y_current - Y_REF)
        
        z_current = z_next
        previous_u = u_opt
    
    # Final state
    y_final = C @ z_current
    time_history.append(T_TOTAL)
    state_history.append(z_current.copy())
    input_history.append(previous_u)
    output_history.append(y_final)
    error_history.append(y_final - Y_REF)
    
    # Build DataFrame
    state_array = np.array(state_history)
    df = pd.DataFrame({
        'run_id': run_id,
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
    
    # Compute metrics
    metrics = compute_metrics(df, n_failures)
    metrics['run_id'] = run_id
    
    # Add plant parameters to metrics
    for key in ['Kb', 'Mb', 'hb', 'Kr', 'hr']:
        metrics[f'{key}_plant'] = plant_params[key]
    
    return df, metrics


# =============================================================================
# MONTE CARLO ROBUSTNESS ANALYSIS
# =============================================================================

def run_robustness_monte_carlo() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Monte Carlo robustness analysis.
    
    Returns
    -------
    runs_df : pd.DataFrame
        One row per run with metrics
    trajectories_df : pd.DataFrame
        Long format trajectories
    summary_df : pd.DataFrame
        Aggregate statistics
    """
    print("\n" + "="*70)
    print("MPC ROBUSTNESS ANALYSIS - MONTE CARLO SIMULATION")
    print("="*70)
    print(f"\nSetup:")
    print(f"  Number of runs      = {N_ROBUST}")
    print(f"  Perturbation level  = ±{PERTURBATION_LEVEL*100:.0f}%")
    print(f"  Perturbed params    = Kb, Mb, hb, Kr, hr")
    print(f"  Fixed params        = Mr")
    print(f"  Random seed         = {RANDOM_STATE}")
    print(f"\nNominal parameters:")
    for key, value in NOMINAL_PARAMS.items():
        print(f"  {key:3s} = {value:.6f}")
    
    # Build nominal model for controller
    Ad_nominal, Bd_nominal = build_discrete_model(NOMINAL_PARAMS, DT)
    nominal_model = (Ad_nominal, Bd_nominal)
    
    # MPC config
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
    
    print(f"\nMPC configuration:")
    print(f"  Horizon N           = {N_HORIZON}")
    print(f"  Weights             = Qy={QY}, Qy_term={QY_TERMINAL}, Ru={RU}, Rdu={RDU}")
    print(f"  Input bounds        = [{U_MIN}, {U_MAX}] m")
    print(f"  Rate limit          = ±{DU_MAX} m/step")
    
    print(f"\n{'='*70}")
    print("Running Monte Carlo simulations...")
    print(f"{'='*70}\n")
    
    # Initialize random generator
    rng = np.random.default_rng(RANDOM_STATE)
    
    # Storage
    all_trajectories = []
    all_metrics = []
    
    # Monte Carlo loop
    for run_id in range(N_ROBUST):
        # Perturb plant parameters
        plant_params = perturb_params(NOMINAL_PARAMS, rng, PERTURBATION_LEVEL)
        
        # Simulate
        df_traj, metrics = simulate_mpc_with_plant_mismatch(
            run_id, plant_params, nominal_model, mpc_config
        )
        
        all_trajectories.append(df_traj)
        all_metrics.append(metrics)
        
        # Progress
        if (run_id + 1) % 10 == 0 or run_id == 0:
            print(f"  Run {run_id+1:2d}/{N_ROBUST} | "
                  f"final_y={metrics['final_y']:.6f}m | "
                  f"error={metrics['final_error_mm']:+.2f}mm | "
                  f"failures={metrics['n_failures']}")
    
    print(f"\n{'='*70}")
    print("Monte Carlo simulation complete!")
    print(f"{'='*70}\n")
    
    # Combine results
    trajectories_df = pd.concat(all_trajectories, ignore_index=True)
    runs_df = pd.DataFrame(all_metrics)
    
    # Compute summary statistics
    summary = {
        'n_runs': N_ROBUST,
        'perturbation_level': PERTURBATION_LEVEL,
        'target': Y_REF,
        'mean_final_error_mm': runs_df['final_error_mm'].mean(),
        'std_final_error_mm': runs_df['final_error_mm'].std(),
        'max_final_error_mm': runs_df['final_error_mm'].abs().max(),
        'mean_abs_final_error_mm': runs_df['final_error_mm'].abs().mean(),
        'p95_abs_final_error_mm': runs_df['final_error_mm'].abs().quantile(0.95),
        'mean_tracking_error_after_0p5s_mm': runs_df['mean_abs_error_after_0p5s_mm'].mean(),
        'max_tracking_error_after_0p5s_mm': runs_df['max_abs_error_after_0p5s_mm'].max(),
        'mean_overshoot_mm': runs_df['overshoot_mm'].mean(),
        'max_overshoot_mm': runs_df['overshoot_mm'].max(),
        'mean_settling_time_1mm_s': runs_df['settling_time_1mm_s'].mean(),
        'success_rate_1mm_final': (runs_df['final_error_mm'].abs() <= 1.0).mean() * 100,
        'success_rate_5mm_final': (runs_df['final_error_mm'].abs() <= 5.0).mean() * 100,
        'success_rate_no_failures': (runs_df['n_failures'] == 0).mean() * 100,
    }
    
    summary_df = pd.DataFrame([summary])
    
    return runs_df, trajectories_df, summary_df


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_robustness_results(runs_df: pd.DataFrame,
                            trajectories_df: pd.DataFrame,
                            summary_df: pd.DataFrame):
    """Save robustness analysis results to CSV files."""
    print("="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Runs
    runs_path = RESULTS_DIR / "mpc_robustness_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"✓ Run metrics saved to: {runs_path}")
    
    # Trajectories
    traj_path = RESULTS_DIR / "mpc_robustness_trajectories.csv"
    trajectories_df.to_csv(traj_path, index=False)
    print(f"✓ Trajectories saved to: {traj_path}")
    
    # Summary
    summary_path = RESULTS_DIR / "mpc_robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}\n")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_robustness_tracking(trajectories_df: pd.DataFrame):
    """Plot all tracking trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all runs with low alpha
    for run_id in trajectories_df['run_id'].unique():
        df_run = trajectories_df[trajectories_df['run_id'] == run_id]
        ax.plot(df_run['time'], df_run['y'], 'b-', alpha=0.15, linewidth=1)
    
    # Target line
    ax.axhline(Y_REF, color='r', linestyle='--', linewidth=2, label=f'Target y_ref = {Y_REF:.4f} m')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Outreach y [m]', fontsize=12)
    ax.set_title(f'MPC Robustness: All Trajectories (N={N_ROBUST}, ±{PERTURBATION_LEVEL*100:.0f}%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_robustness_tracking.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tracking plot saved to: {path}")


def plot_robustness_error_band(trajectories_df: pd.DataFrame):
    """Plot mean ± std error band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by time and compute statistics
    stats = trajectories_df.groupby('time')['y'].agg(['mean', 'std']).reset_index()
    
    # Mean trajectory
    ax.plot(stats['time'], stats['mean'], 'b-', linewidth=2.5, label='Mean trajectory')
    
    # ±1 std band
    ax.fill_between(stats['time'], 
                    stats['mean'] - stats['std'],
                    stats['mean'] + stats['std'],
                    alpha=0.3, color='blue', label='Mean ± 1 std')
    
    # Target
    ax.axhline(Y_REF, color='r', linestyle='--', linewidth=2, label=f'Target y_ref = {Y_REF:.4f} m')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Outreach y [m]', fontsize=12)
    ax.set_title(f'MPC Robustness: Mean ± Std Band (N={N_ROBUST}, ±{PERTURBATION_LEVEL*100:.0f}%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_robustness_error_band.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error band plot saved to: {path}")


def plot_final_error_histogram(runs_df: pd.DataFrame):
    """Plot histogram of final tracking errors (signed)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(runs_df['final_error_mm'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Reference lines
    ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Zero error')
    ax.axvline(-1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='±1 mm')
    ax.axvline(1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(-5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='±5 mm')
    ax.axvline(5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Final Tracking Error [mm]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'MPC Robustness: Final Error Distribution (N={N_ROBUST})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_robustness_final_error_histogram.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Final error histogram saved to: {path}")


def plot_abs_final_error_histogram(runs_df: pd.DataFrame):
    """Plot histogram of absolute final errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(runs_df['final_error_mm'].abs(), bins=20, edgecolor='black', alpha=0.7, color='teal')
    
    ax.axvline(1, color='orange', linestyle='--', linewidth=2, label='1 mm threshold')
    ax.axvline(5, color='green', linestyle='--', linewidth=2, label='5 mm threshold')
    
    ax.set_xlabel('Absolute Final Tracking Error [mm]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'MPC Robustness: Absolute Final Error Distribution (N={N_ROBUST})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_robustness_abs_final_error_histogram.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Absolute final error histogram saved to: {path}")


def plot_control_band(trajectories_df: pd.DataFrame):
    """Plot mean ± std control signal band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by time
    stats = trajectories_df.groupby('time')['u'].agg(['mean', 'std']).reset_index()
    
    ax.plot(stats['time'], stats['mean'], 'g-', linewidth=2.5, label='Mean control input')
    ax.fill_between(stats['time'],
                    stats['mean'] - stats['std'],
                    stats['mean'] + stats['std'],
                    alpha=0.3, color='green', label='Mean ± 1 std')
    
    # Bounds
    ax.axhline(U_MIN, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'u_min = {U_MIN} m')
    ax.axhline(U_MAX, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'u_max = {U_MAX} m')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Control Input u [m]', fontsize=12)
    ax.set_title(f'MPC Robustness: Control Signal Band (N={N_ROBUST})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_robustness_control_band.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Control band plot saved to: {path}")


def plot_delta_u_example(trajectories_df: pd.DataFrame):
    """Plot Δu for one example run to show rate constraint satisfaction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use run_id = 0 as example
    df_run = trajectories_df[trajectories_df['run_id'] == 0].copy()
    df_run = df_run.sort_values('time')
    
    # Compute Δu
    delta_u = np.diff(df_run['u'].values)
    time_delta = df_run['time'].values[1:]
    
    ax.plot(time_delta, delta_u * 1000, 'b-', linewidth=2, label='Δu (rate of change)')
    
    # Rate limits
    ax.axhline(DU_MAX * 1000, color='r', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Rate limit = ±{DU_MAX*1000:.0f} mm/step')
    ax.axhline(-DU_MAX * 1000, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Input Rate Δu [mm/step]', fontsize=12)
    ax.set_title('MPC Input Rate: Rate Constraint Satisfaction (Example Run 0)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_robustness_delta_u.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Delta u plot saved to: {path}")


# =============================================================================
# OPTIONAL: REFERENCE CHANGE EXPERIMENT
# =============================================================================

def run_reference_change_experiment():
    """
    Optional: Run MPC with time-varying reference.
    
    Reference schedule:
    - 0.0 - 1.5s: y_ref = 0.520767 (low)
    - 1.5 - 3.0s: y_ref = 0.592779 (central)
    - 3.0 - 5.0s: y_ref = 0.628967 (high)
    """
    print("\n" + "="*70)
    print("OPTIONAL: REFERENCE CHANGE EXPERIMENT")
    print("="*70)
    print("\nReference schedule:")
    for t_start, t_end, ref_val in REF_SCHEDULE:
        print(f"  {t_start:.1f}s - {t_end:.1f}s: y_ref = {ref_val:.6f} m")
    
    # Build nominal model
    Ad_nominal, Bd_nominal = build_discrete_model(NOMINAL_PARAMS, DT)
    
    # MPC config
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
    
    # Initialize
    z_current = Z0.copy()
    previous_u = Z0[3]
    
    time_history = []
    state_history = []
    input_history = []
    output_history = []
    ref_history = []
    error_history = []
    
    n_failures = 0
    
    print("\nRunning simulation...")
    
    # MPC loop
    for step in range(N_SIM):
        t = step * DT
        
        # Get current reference
        y_ref_current = Y_REF
        for t_start, t_end, ref_val in REF_SCHEDULE:
            if t_start <= t < t_end:
                y_ref_current = ref_val
                break
        
        y_current = C @ z_current
        
        # Solve MPC
        u_opt = solve_mpc_step(z_current, previous_u, Ad_nominal, Bd_nominal,
                               C, y_ref_current, mpc_config)
        
        if u_opt is None:
            u_opt = previous_u
            n_failures += 1
        
        # Propagate (using nominal plant)
        z_next = Ad_nominal @ z_current + Bd_nominal.flatten() * u_opt
        
        # Store
        time_history.append(t)
        state_history.append(z_current.copy())
        input_history.append(u_opt)
        output_history.append(y_current)
        ref_history.append(y_ref_current)
        error_history.append(y_current - y_ref_current)
        
        z_current = z_next
        previous_u = u_opt
    
    # Final
    y_final = C @ z_current
    y_ref_final = REF_SCHEDULE[-1][2]
    time_history.append(T_TOTAL)
    state_history.append(z_current.copy())
    input_history.append(previous_u)
    output_history.append(y_final)
    ref_history.append(y_ref_final)
    error_history.append(y_final - y_ref_final)
    
    print("✓ Simulation complete\n")
    
    # Build DataFrame
    state_array = np.array(state_history)
    df = pd.DataFrame({
        'time': time_history,
        'dxb': state_array[:, 0],
        'dxr': state_array[:, 1],
        'xb': state_array[:, 2],
        'xr': state_array[:, 3],
        'y': output_history,
        'u': input_history,
        'y_ref': ref_history,
        'tracking_error_m': error_history,
        'tracking_error_mm': np.array(error_history) * 1000.0,
    })
    
    # Save results
    results_path = RESULTS_DIR / "mpc_reference_change_results.csv"
    df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")
    
    # Summary
    summary = {
        'n_failures': n_failures,
        'mean_abs_error_mm': df['tracking_error_mm'].abs().mean(),
        'max_abs_error_mm': df['tracking_error_mm'].abs().max(),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = RESULTS_DIR / "mpc_reference_change_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Plots
    print("\nGenerating plots...")
    
    # Tracking
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time'], df['y'], 'b-', linewidth=2, label='Output y(t)')
    ax.plot(df['time'], df['y_ref'], 'r--', linewidth=2, label='Reference y_ref(t)')
    
    # Mark transition times
    for t_start, t_end, ref_val in REF_SCHEDULE[:-1]:
        ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Outreach [m]', fontsize=12)
    ax.set_title('MPC Reference Change: Tracking Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_reference_change_tracking.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tracking plot saved to: {path}")
    
    # Error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time'], df['tracking_error_mm'], 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    for t_start, t_end, ref_val in REF_SCHEDULE[:-1]:
        ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Tracking Error [mm]', fontsize=12)
    ax.set_title('MPC Reference Change: Tracking Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_reference_change_error.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error plot saved to: {path}")
    
    # Control
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time'], df['u'], 'g-', linewidth=2, label='Control input u(t)')
    ax.axhline(U_MIN, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'u_min = {U_MIN} m')
    ax.axhline(U_MAX, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'u_max = {U_MAX} m')
    for t_start, t_end, ref_val in REF_SCHEDULE[:-1]:
        ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Control Input u [m]', fontsize=12)
    ax.set_title('MPC Reference Change: Control Signal', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "mpc_reference_change_control.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Control plot saved to: {path}")
    
    print("\n✓ Reference change experiment complete\n")


# =============================================================================
# SUMMARY PRINTING
# =============================================================================

def print_summary(summary_df: pd.DataFrame):
    """Print formatted robustness summary."""
    s = summary_df.iloc[0]
    
    print("="*70)
    print("MPC ROBUSTNESS SUMMARY")
    print("="*70)
    print(f"\nTest Configuration:")
    print(f"  Number of runs         = {int(s['n_runs'])}")
    print(f"  Perturbation level     = ±{s['perturbation_level']*100:.0f}%")
    print(f"  Target reference       = {s['target']:.6f} m")
    
    print(f"\nFinal Error Statistics:")
    print(f"  Mean final error       = {s['mean_final_error_mm']:+.2f} mm")
    print(f"  Std final error        = {s['std_final_error_mm']:.2f} mm")
    print(f"  Mean |final error|     = {s['mean_abs_final_error_mm']:.2f} mm")
    print(f"  Max |final error|      = {s['max_final_error_mm']:.2f} mm")
    print(f"  95th percentile        = {s['p95_abs_final_error_mm']:.2f} mm")
    
    print(f"\nPost-Transient Performance (t >= 0.5s):")
    print(f"  Mean tracking error    = {s['mean_tracking_error_after_0p5s_mm']:.2f} mm")
    print(f"  Max tracking error     = {s['max_tracking_error_after_0p5s_mm']:.2f} mm")
    
    print(f"\nTransient Behavior:")
    print(f"  Mean overshoot         = {s['mean_overshoot_mm']:.2f} mm")
    print(f"  Max overshoot          = {s['max_overshoot_mm']:.2f} mm")
    print(f"  Mean settling time     = {s['mean_settling_time_1mm_s']:.2f} s (1mm threshold)")
    
    print(f"\nSuccess Rates:")
    print(f"  Within 1mm final       = {s['success_rate_1mm_final']:.1f}%")
    print(f"  Within 5mm final       = {s['success_rate_5mm_final']:.1f}%")
    print(f"  No solver failures     = {s['success_rate_no_failures']:.1f}%")
    
    print("="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MPC ROBUSTNESS ANALYSIS")
    print("ML Compliant Base Outreach Project")
    print("="*70)
    print("\nThis module tests the nominal MPC controller under plant/model mismatch.")
    print("The MPC uses a nominal model while the true plant has perturbed parameters.")
    print("This is more realistic than the nominal case and demonstrates closed-loop")
    print("robustness to model uncertainty.\n")
    
    # Setup
    ensure_dirs()
    
    # Run Monte Carlo robustness analysis
    runs_df, trajectories_df, summary_df = run_robustness_monte_carlo()
    
    # Save results
    print("\n" + "="*70)
    save_robustness_results(runs_df, trajectories_df, summary_df)
    
    # Generate plots
    print("="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_robustness_tracking(trajectories_df)
    plot_robustness_error_band(trajectories_df)
    plot_final_error_histogram(runs_df)
    plot_abs_final_error_histogram(runs_df)
    plot_control_band(trajectories_df)
    plot_delta_u_example(trajectories_df)
    
    # Print summary
    print("\n")
    print_summary(summary_df)
    
    # Optional: reference change experiment
    if RUN_REFERENCE_CHANGE:
        print("="*70)
        print("OPTIONAL EXPERIMENT")
        print("="*70)
        run_reference_change_experiment()
    
    # Final message
    print("="*70)
    print("MPC ROBUSTNESS ANALYSIS COMPLETE!")
    print("="*70)
    print("\nMain outputs generated:")
    print("  • results/mpc_robustness_runs.csv")
    print("  • results/mpc_robustness_trajectories.csv")
    print("  • results/mpc_robustness_summary.csv")
    print("  • figures/mpc_robustness_tracking.png")
    print("  • figures/mpc_robustness_error_band.png")
    print("  • figures/mpc_robustness_final_error_histogram.png")
    print("  • figures/mpc_robustness_abs_final_error_histogram.png")
    print("  • figures/mpc_robustness_control_band.png")
    print("  • figures/mpc_robustness_delta_u.png")
    print("\nOptional reference change outputs:")
    print("  • results/mpc_reference_change_results.csv")
    print("  • results/mpc_reference_change_summary.csv")
    print("  • figures/mpc_reference_change_tracking.png")
    print("  • figures/mpc_reference_change_error.png")
    print("  • figures/mpc_reference_change_control.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()