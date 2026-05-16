"""
mpc_robustness.py
=================

MPC disturbance rejection and noisy closed-loop robustness experiment.

This script tests the MPC controller in a more realistic scenario:

1. The controller tracks a fixed target y_ref.
2. At t = DISTURBANCE_TIME, a sudden displacement disturbance is applied
   to the robot relative coordinate xr.
3. Measurement noise is added to the state used by the MPC.
4. Small process noise is added to the true plant evolution.
5. The MPC must recover and bring y = xb + xr back to the target.

This is an additional closed-loop control experiment.
It does NOT replace the main GP + DE/BO inverse optimization pipeline.

State:
    z = [dxb, dxr, xb, xr]

Input:
    u = xrd

Output:
    y = xb + xr

Generated files:
    results/mpc/mpc_disturbance_results.csv
    results/mpc/mpc_disturbance_summary.csv

    figures/mpc/mpc_disturbance_tracking.png
    figures/mpc/mpc_disturbance_error.png
    figures/mpc/mpc_disturbance_control.png
    figures/mpc/mpc_disturbance_states.png
    figures/mpc/mpc_disturbance_zoom.png
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

try:
    import cvxpy as cp
except ImportError:
    print("\nERROR: cvxpy not found.")
    print("Install it with:")
    print("  pip install cvxpy osqp\n")
    raise


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mpc"
FIGURES_DIR = PROJECT_ROOT / "figures" / "mpc"


# =============================================================================
# NOMINAL MODEL PARAMETERS
# =============================================================================

NOMINAL_PARAMS = {
    "Kb": 2000.0,
    "Mb": 50.0,
    "hb": 0.2,
    "Mr": 10.0,
    "Kr": 3482.193396,
    "hr": 0.430359,
}

# Target reference
Y_REF = 0.592779

# Initial state: z = [dxb, dxr, xb, xr]
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

# Output matrix: y = xb + xr
C = np.array([0.0, 0.0, 1.0, 1.0])


# =============================================================================
# DISTURBANCE + NOISE SETTINGS
# =============================================================================

DISTURBANCE_TIME = 2.0
DISTURBANCE_ON_XR = 0.050   # 50 mm sudden displacement on xr

ADD_MEASUREMENT_NOISE = True
ADD_PROCESS_NOISE = True

# Measurement noise used only by the controller.
# Units:
#   dxb, dxr -> m/s
#   xb, xr   -> m
MEASUREMENT_NOISE_STD = np.array([
    0.002,   # dxb measurement noise [m/s]
    0.002,   # dxr measurement noise [m/s]
    0.001,   # xb measurement noise [m]
    0.001,   # xr measurement noise [m]
])

# Process noise added to the true plant evolution.
# Units:
#   dxb, dxr -> m/s
#   xb, xr   -> m
PROCESS_NOISE_STD = np.array([
    0.0005,    # dxb process noise [m/s]
    0.0005,    # dxr process noise [m/s]
    0.00005,   # xb process noise [m]
    0.00005,   # xr process noise [m]
])

RANDOM_STATE = 42

# Cases to run
RUN_NOMINAL_CASE = True
RUN_DISTURBANCE_ONLY_CASE = True
RUN_DISTURBANCE_NOISY_CASE = True


# =============================================================================
# UTILITY
# =============================================================================

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_damping(K: float, M: float, h: float) -> float:
    """
    Compute viscous damping from stiffness, mass and damping ratio.

    D = 2 h sqrt(KM)
    """
    return 2.0 * h * np.sqrt(K * M)


# =============================================================================
# DYNAMICS
# =============================================================================

def build_continuous_state_space(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time state-space matrices.

    State:
        z = [dxb, dxr, xb, xr]

    Input:
        u = xrd

    Output:
        y = xb + xr
    """
    Kb = params["Kb"]
    Mb = params["Mb"]
    hb = params["hb"]
    Mr = params["Mr"]
    Kr = params["Kr"]
    hr = params["hr"]

    Db = compute_damping(Kb, Mb, hb)
    Dr = compute_damping(Kr, Mr, hr)

    A = np.array([
        [-Db / Mb,                Dr / Mb,              -Kb / Mb,                Kr / Mb],
        [ Db / Mb, -(Dr / Mr + Dr / Mb),                 Kb / Mb, -(Kr / Mr + Kr / Mb)],
        [1.0,                     0.0,                    0.0,                    0.0],
        [0.0,                     1.0,                    0.0,                    0.0],
    ])

    B = np.array([
        [-Kr / Mb],
        [ Kr / Mr + Kr / Mb],
        [0.0],
        [0.0],
    ])

    return A, B


def build_discrete_model(params: Dict[str, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete-time state-space model using zero-order hold.
    """
    A, B = build_continuous_state_space(params)

    sys_d = cont2discrete(
        (A, B, np.eye(4), np.zeros((4, 1))),
        dt,
        method="zoh",
    )

    Ad = sys_d[0]
    Bd = sys_d[1]

    return Ad, Bd


# =============================================================================
# MPC SOLVER
# =============================================================================

def solve_mpc_step(
    z_measured: np.ndarray,
    previous_u: float,
    Ad: np.ndarray,
    Bd: np.ndarray,
    y_ref: float,
) -> Optional[float]:
    """
    Solve one MPC step.

    Important:
    ----------
    The MPC uses z_measured, which may contain measurement noise.
    The plant itself evolves from the true state.
    """
    n_states = Ad.shape[0]
    N = N_HORIZON

    z = cp.Variable((n_states, N + 1))
    u = cp.Variable(N)

    cost = 0.0
    constraints = [z[:, 0] == z_measured]

    for k in range(N):
        y_k = C @ z[:, k]

        cost += QY * cp.square(y_k - y_ref)
        cost += RU * cp.square(u[k] - y_ref)

        if k == 0:
            cost += RDU * cp.square(u[k] - previous_u)
            constraints += [
                u[k] - previous_u <= DU_MAX,
                u[k] - previous_u >= -DU_MAX,
            ]
        else:
            cost += RDU * cp.square(u[k] - u[k - 1])
            constraints += [
                u[k] - u[k - 1] <= DU_MAX,
                u[k] - u[k - 1] >= -DU_MAX,
            ]

        constraints += [
            z[:, k + 1] == Ad @ z[:, k] + Bd.flatten() * u[k],
            u[k] >= U_MIN,
            u[k] <= U_MAX,
        ]

    y_N = C @ z[:, N]
    cost += QY_TERMINAL * cp.square(y_N - y_ref)

    problem = cp.Problem(cp.Minimize(cost), constraints)

    try:
        problem.solve(
            solver=cp.OSQP,
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
        )

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return float(u.value[0])

        return None

    except Exception:
        return None


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_case(
    case_name: str,
    apply_disturbance: bool,
    add_measurement_noise: bool,
    add_process_noise: bool,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate one closed-loop MPC case.
    """
    rng = np.random.default_rng(random_seed)

    Ad, Bd = build_discrete_model(NOMINAL_PARAMS, DT)

    z_true = Z0.copy()
    previous_u = Z0[3]

    disturbance_applied = False
    n_failures = 0

    history = []

    for step in range(N_SIM + 1):
        t = step * DT

        y_true = float(C @ z_true)
        error_true_m = y_true - Y_REF

        # Measurement used by MPC
        if add_measurement_noise:
            measurement_noise = rng.normal(0.0, MEASUREMENT_NOISE_STD)
            z_measured = z_true + measurement_noise
        else:
            measurement_noise = np.zeros(4)
            z_measured = z_true.copy()

        y_measured = float(C @ z_measured)

        history.append({
            "case": case_name,
            "time": t,
            "dxb": z_true[0],
            "dxr": z_true[1],
            "xb": z_true[2],
            "xr": z_true[3],
            "y": y_true,
            "dxb_measured": z_measured[0],
            "dxr_measured": z_measured[1],
            "xb_measured": z_measured[2],
            "xr_measured": z_measured[3],
            "y_measured": y_measured,
            "u": previous_u,
            "tracking_error_m": error_true_m,
            "tracking_error_mm": error_true_m * 1000.0,
            "measurement_noise_dxb": measurement_noise[0],
            "measurement_noise_dxr": measurement_noise[1],
            "measurement_noise_xb": measurement_noise[2],
            "measurement_noise_xr": measurement_noise[3],
            "disturbance_applied": int(disturbance_applied),
            "n_failures_so_far": n_failures,
        })

        if step == N_SIM:
            break

        # Solve MPC using measured state
        u_opt = solve_mpc_step(
            z_measured=z_measured,
            previous_u=previous_u,
            Ad=Ad,
            Bd=Bd,
            y_ref=Y_REF,
        )

        if u_opt is None:
            u_opt = previous_u
            n_failures += 1

        # Propagate true plant
        z_next = Ad @ z_true + Bd.flatten() * u_opt

        # Apply process noise to the true plant
        if add_process_noise:
            process_noise = rng.normal(0.0, PROCESS_NOISE_STD)
            z_next = z_next + process_noise

        # Apply sudden displacement disturbance once
        if apply_disturbance and (not disturbance_applied) and (t >= DISTURBANCE_TIME):
            z_next[3] += DISTURBANCE_ON_XR
            disturbance_applied = True

        z_true = z_next
        previous_u = u_opt

    df = pd.DataFrame(history)
    metrics = compute_case_metrics(df, case_name)

    return df, metrics


# =============================================================================
# METRICS
# =============================================================================

def compute_recovery_time(df: pd.DataFrame, threshold_mm: float = 1.0) -> float:
    """
    Recovery time after disturbance.

    First time after DISTURBANCE_TIME such that all subsequent true output
    errors remain below threshold_mm.
    """
    df_post = df[df["time"] >= DISTURBANCE_TIME].copy()

    if df_post.empty:
        return np.nan

    times = df_post["time"].to_numpy()
    errors = df_post["tracking_error_mm"].abs().to_numpy()

    for i in range(len(errors)):
        if np.all(errors[i:] <= threshold_mm):
            return times[i] - DISTURBANCE_TIME

    return np.nan


def compute_case_metrics(df: pd.DataFrame, case_name: str) -> Dict[str, float]:
    """
    Compute metrics for one case.
    """
    df_post_dist = df[df["time"] >= DISTURBANCE_TIME].copy()
    df_post_0p5 = df[df["time"] >= DISTURBANCE_TIME + 0.5].copy()

    if df_post_dist.empty:
        max_error_after_dist_mm = np.nan
    else:
        max_error_after_dist_mm = df_post_dist["tracking_error_mm"].abs().max()

    if df_post_0p5.empty:
        mean_error_after_0p5_mm = np.nan
        max_error_after_0p5_mm = np.nan
    else:
        mean_error_after_0p5_mm = df_post_0p5["tracking_error_mm"].abs().mean()
        max_error_after_0p5_mm = df_post_0p5["tracking_error_mm"].abs().max()

    metrics = {
        "case": case_name,
        "target_y_ref": Y_REF,
        "disturbance_time_s": DISTURBANCE_TIME,
        "disturbance_on_xr_m": DISTURBANCE_ON_XR,
        "final_y": df["y"].iloc[-1],
        "final_error_mm": df["tracking_error_mm"].iloc[-1],
        "mean_abs_error_mm": df["tracking_error_mm"].abs().mean(),
        "max_abs_error_mm": df["tracking_error_mm"].abs().max(),
        "max_abs_error_after_disturbance_mm": max_error_after_dist_mm,
        "mean_abs_error_0p5s_after_disturbance_mm": mean_error_after_0p5_mm,
        "max_abs_error_0p5s_after_disturbance_mm": max_error_after_0p5_mm,
        "recovery_time_1mm_s": compute_recovery_time(df, threshold_mm=1.0),
        "recovery_time_5mm_s": compute_recovery_time(df, threshold_mm=5.0),
        "u_min_used": df["u"].min(),
        "u_max_used": df["u"].max(),
        "n_solver_failures": df["n_failures_so_far"].iloc[-1],
    }

    return metrics


# =============================================================================
# PLOTS
# =============================================================================

def plot_tracking(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    for case_name, df_case in df_all.groupby("case"):
        ax.plot(
            df_case["time"],
            df_case["y"],
            linewidth=2.2,
            label=case_name,
        )

    ax.axhline(Y_REF, color="red", linestyle="--", linewidth=2, label=f"Target = {Y_REF:.4f} m")
    ax.axvline(DISTURBANCE_TIME, color="black", linestyle=":", linewidth=2, label="Disturbance")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Outreach y [m]", fontsize=12)
    ax.set_title("MPC Disturbance Rejection with Noise: Output Tracking", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "mpc_disturbance_tracking.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_error(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    for case_name, df_case in df_all.groupby("case"):
        ax.plot(
            df_case["time"],
            df_case["tracking_error_mm"],
            linewidth=2.0,
            label=case_name,
        )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(1.0, color="orange", linestyle="--", linewidth=1.5, label="±1 mm")
    ax.axhline(-1.0, color="orange", linestyle="--", linewidth=1.5)
    ax.axhline(5.0, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="±5 mm")
    ax.axhline(-5.0, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(DISTURBANCE_TIME, color="black", linestyle=":", linewidth=2, label="Disturbance")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Tracking error [mm]", fontsize=12)
    ax.set_title("MPC Disturbance Rejection with Noise: Tracking Error", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "mpc_disturbance_error.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_control(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    for case_name, df_case in df_all.groupby("case"):
        ax.plot(
            df_case["time"],
            df_case["u"],
            linewidth=2.0,
            label=case_name,
        )

    ax.axhline(U_MIN, color="red", linestyle="--", linewidth=1.5, label=f"u_min = {U_MIN:.2f} m")
    ax.axhline(U_MAX, color="red", linestyle="--", linewidth=1.5, label=f"u_max = {U_MAX:.2f} m")
    ax.axvline(DISTURBANCE_TIME, color="black", linestyle=":", linewidth=2, label="Disturbance")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Control input u = xrd [m]", fontsize=12)
    ax.set_title("MPC Disturbance Rejection with Noise: Control Input", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "mpc_disturbance_control.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_states(df_all: pd.DataFrame):
    """
    Plot states for the noisy disturbed case only.
    """
    case_name = "disturbance_plus_noise"

    if case_name not in df_all["case"].unique():
        case_name = df_all["case"].unique()[-1]

    df_case = df_all[df_all["case"] == case_name].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_case["time"], df_case["xb"], linewidth=2, label="xb")
    ax.plot(df_case["time"], df_case["xr"], linewidth=2, label="xr")
    ax.plot(df_case["time"], df_case["y"], linewidth=2.4, label="y = xb + xr")

    ax.axhline(Y_REF, color="red", linestyle="--", linewidth=2, label="Target")
    ax.axvline(DISTURBANCE_TIME, color="black", linestyle=":", linewidth=2, label="Disturbance")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position [m]", fontsize=12)
    ax.set_title("MPC Disturbance Rejection with Noise: State Response", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "mpc_disturbance_states.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_zoom(df_all: pd.DataFrame):
    """
    Zoom around the disturbance time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    t_min = DISTURBANCE_TIME - 0.4
    t_max = DISTURBANCE_TIME + 1.2

    for case_name, df_case in df_all.groupby("case"):
        df_zoom = df_case[(df_case["time"] >= t_min) & (df_case["time"] <= t_max)]
        ax.plot(
            df_zoom["time"],
            df_zoom["tracking_error_mm"],
            linewidth=2.2,
            label=case_name,
        )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(1.0, color="orange", linestyle="--", linewidth=1.5, label="±1 mm")
    ax.axhline(-1.0, color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(DISTURBANCE_TIME, color="black", linestyle=":", linewidth=2, label="Disturbance")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Tracking error [mm]", fontsize=12)
    ax.set_title("Zoom Around Disturbance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "mpc_disturbance_zoom.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def make_plots(df_all: pd.DataFrame):
    print("\nGenerating plots...")
    plot_tracking(df_all)
    plot_error(df_all)
    plot_control(df_all)
    plot_states(df_all)
    plot_zoom(df_all)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("MPC DISTURBANCE REJECTION WITH NOISE")
    print("=" * 70)

    ensure_dirs()

    print("\nConfiguration:")
    print(f"  Target y_ref             = {Y_REF:.6f} m")
    print(f"  Disturbance time         = {DISTURBANCE_TIME:.2f} s")
    print(f"  Disturbance on xr        = {DISTURBANCE_ON_XR * 1000:.1f} mm")
    print(f"  Measurement noise        = {ADD_MEASUREMENT_NOISE}")
    print(f"  Process noise            = {ADD_PROCESS_NOISE}")
    print(f"  Measurement noise std    = {MEASUREMENT_NOISE_STD}")
    print(f"  Process noise std        = {PROCESS_NOISE_STD}")
    print(f"  Input bounds             = [{U_MIN:.2f}, {U_MAX:.2f}] m")
    print(f"  Rate limit               = ±{DU_MAX * 1000:.1f} mm/step")
    print(f"  MPC horizon              = {N_HORIZON} steps = {N_HORIZON * DT:.2f} s")

    all_dfs: List[pd.DataFrame] = []
    all_metrics: List[Dict[str, float]] = []

    if RUN_NOMINAL_CASE:
        print("\nRunning case: nominal")
        df, metrics = simulate_case(
            case_name="nominal",
            apply_disturbance=False,
            add_measurement_noise=False,
            add_process_noise=False,
            random_seed=RANDOM_STATE,
        )
        all_dfs.append(df)
        all_metrics.append(metrics)

    if RUN_DISTURBANCE_ONLY_CASE:
        print("Running case: disturbance_only")
        df, metrics = simulate_case(
            case_name="disturbance_only",
            apply_disturbance=True,
            add_measurement_noise=False,
            add_process_noise=False,
            random_seed=RANDOM_STATE + 1,
        )
        all_dfs.append(df)
        all_metrics.append(metrics)

    if RUN_DISTURBANCE_NOISY_CASE:
        print("Running case: disturbance_plus_noise")
        df, metrics = simulate_case(
            case_name="disturbance_plus_noise",
            apply_disturbance=True,
            add_measurement_noise=ADD_MEASUREMENT_NOISE,
            add_process_noise=ADD_PROCESS_NOISE,
            random_seed=RANDOM_STATE + 2,
        )
        all_dfs.append(df)
        all_metrics.append(metrics)

    df_all = pd.concat(all_dfs, ignore_index=True)
    summary_df = pd.DataFrame(all_metrics)

    results_path = RESULTS_DIR / "mpc_disturbance_results.csv"
    summary_path = RESULTS_DIR / "mpc_disturbance_summary.csv"

    df_all.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n✓ Saved results: {results_path}")
    print(f"✓ Saved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        print(f"\nCase: {row['case']}")
        print(f"  Final error                         = {row['final_error_mm']:+.3f} mm")
        print(f"  Mean |error|                        = {row['mean_abs_error_mm']:.3f} mm")
        print(f"  Max |error|                         = {row['max_abs_error_mm']:.3f} mm")
        print(f"  Max |error| after disturbance        = {row['max_abs_error_after_disturbance_mm']:.3f} mm")
        print(f"  Mean |error| after 0.5s from disturb = {row['mean_abs_error_0p5s_after_disturbance_mm']:.3f} mm")
        print(f"  Recovery time within 1mm            = {row['recovery_time_1mm_s']}")
        print(f"  Recovery time within 5mm            = {row['recovery_time_5mm_s']}")
        print(f"  Input range used                    = [{row['u_min_used']:.4f}, {row['u_max_used']:.4f}] m")
        print(f"  Solver failures                     = {int(row['n_solver_failures'])}")

    make_plots(df_all)

    print("\n" + "=" * 70)
    print("MPC DISTURBANCE REJECTION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  results/mpc/mpc_disturbance_results.csv")
    print("  results/mpc/mpc_disturbance_summary.csv")
    print("  figures/mpc/mpc_disturbance_tracking.png")
    print("  figures/mpc/mpc_disturbance_error.png")
    print("  figures/mpc/mpc_disturbance_control.png")
    print("  figures/mpc/mpc_disturbance_states.png")
    print("  figures/mpc/mpc_disturbance_zoom.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()