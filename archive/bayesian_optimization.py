"""
Bayesian Optimization experiment for inverse outreach design.

This script is an additional experiment and does NOT replace the main
inverse optimization pipeline in optimization.py.

It supports two modes:

1. RUN_ALL_TARGETS = False
   Run BO only on the central target.

2. RUN_ALL_TARGETS = True
   Run BO on the same 5 targets used in the official inverse optimization.

Goal:
    Given a target outreach, find controllable parameters using
    Bayesian Optimization with the TRUE dynamic simulator as black-box.

Optimized parameters:
    Kr, hr, f0, f1, A, x_r_start

Fixed parameters:
    Kb, Mb, hb, Mr

Main outputs when RUN_ALL_TARGETS = True:
    results/bo/bo_results_all_targets.csv
    results/bo/bo_summary_all_targets.csv
    figures/bo/bo_vs_de_all_targets.png
    figures/bo/bo_convergence_all_targets.png
    figures/bo/bo_error_by_target.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RESULTS_BO_DIR = PROJECT_ROOT / "results" / "bo"
FIGURES_BO_DIR = PROJECT_ROOT / "figures" / "bo"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dynamics import simulate_system
except ImportError as e:
    raise ImportError(
        "Could not import simulate_system from dynamics.py. "
        "Check that src/dynamics.py exists and defines simulate_system()."
    ) from e


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
RUN_ALL_TARGETS = True

FIXED_PARAMS = {
    "Kb": 2000.0,
    "Mb": 50.0,
    "hb": 0.2,
    "Mr": 10.0,
}

CONTROL_NAMES = ["Kr", "hr", "f0", "f1", "A", "x_r_start"]

BOUNDS_DICT = {
    "Kr": (100.0, 5000.0),
    "hr": (0.1, 1.1),
    "f0": (1e-4, 1e-2),
    "f1": (3.0, 10.0),
    "A": (0.1, 0.2),
    "x_r_start": (0.3, 0.5),
}

BOUNDS = np.array([BOUNDS_DICT[name] for name in CONTROL_NAMES], dtype=float)

# BO settings
N_INITIAL = 10
N_ITER = 40
N_CANDIDATES = 3000
KAPPA = 0.5
RANDOM_STATE = 42

# Official Differential Evolution results from optimization.py
TARGETS = [
    {
        "label": "target1_p20",
        "percentile": 20,
        "target": 0.520767,
        "de_y_sim": 0.519726,
    },
    {
        "label": "target2_p35",
        "percentile": 35,
        "target": 0.557278,
        "de_y_sim": 0.560575,
    },
    {
        "label": "target3_p50",
        "percentile": 50,
        "target": 0.592779,
        "de_y_sim": 0.593118,
    },
    {
        "label": "target4_p65",
        "percentile": 65,
        "target": 0.628967,
        "de_y_sim": 0.631119,
    },
    {
        "label": "target5_p80",
        "percentile": 80,
        "target": 0.681966,
        "de_y_sim": 0.681470,
    },
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def ensure_dirs():
    """Create output directories if needed."""
    RESULTS_BO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_BO_DIR.mkdir(parents=True, exist_ok=True)


def scale_unit_to_bounds(X_unit: np.ndarray) -> np.ndarray:
    """Map samples from [0, 1]^d to physical bounds."""
    lower = BOUNDS[:, 0]
    upper = BOUNDS[:, 1]
    return lower + X_unit * (upper - lower)


def sample_lhs_points(n_samples: int, random_state: int) -> np.ndarray:
    """Generate initial LHS points in the controllable parameter space."""
    sampler = qmc.LatinHypercube(d=len(CONTROL_NAMES), seed=random_state)
    X_unit = sampler.random(n_samples)
    return scale_unit_to_bounds(X_unit)


def sample_random_candidates(n_candidates: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random candidate points for acquisition optimization."""
    X_unit = rng.random((n_candidates, len(CONTROL_NAMES)))
    return scale_unit_to_bounds(X_unit)


def vector_to_params(x: np.ndarray) -> dict:
    """Convert controllable vector to full parameter dictionary."""
    control_params = {name: float(value) for name, value in zip(CONTROL_NAMES, x)}

    full_params = {
        "Kb": FIXED_PARAMS["Kb"],
        "Kr": control_params["Kr"],
        "Mb": FIXED_PARAMS["Mb"],
        "hb": FIXED_PARAMS["hb"],
        "hr": control_params["hr"],
        "f0": control_params["f0"],
        "f1": control_params["f1"],
        "A": control_params["A"],
        "x_r_start": control_params["x_r_start"],
        "Mr": FIXED_PARAMS["Mr"],
    }

    return full_params


def extract_peak_y(sim_result) -> float:
    """
    Extract peak_y from simulate_system() output.

    In this project, simulate_system() returns directly peak_y
    as a numpy.float64 scalar. Other cases are kept for robustness.
    """
    if isinstance(sim_result, (int, float, np.integer, np.floating)):
        return float(sim_result)

    if isinstance(sim_result, dict):
        if "peak_y" in sim_result:
            return float(sim_result["peak_y"])

        if "y" in sim_result:
            return float(np.max(sim_result["y"]))

        if "outreach" in sim_result:
            return float(np.max(sim_result["outreach"]))

        if "xb" in sim_result and "xr" in sim_result:
            return float(
                np.max(np.asarray(sim_result["xb"]) + np.asarray(sim_result["xr"]))
            )

    for attr in ["peak_y", "max_y", "y_peak"]:
        if hasattr(sim_result, attr):
            return float(getattr(sim_result, attr))

    if hasattr(sim_result, "y"):
        return float(np.max(np.asarray(getattr(sim_result, "y"))))

    if hasattr(sim_result, "xb") and hasattr(sim_result, "xr"):
        xb = np.asarray(getattr(sim_result, "xb"))
        xr = np.asarray(getattr(sim_result, "xr"))
        return float(np.max(xb + xr))

    if isinstance(sim_result, (tuple, list)):
        for item in sim_result:
            try:
                return extract_peak_y(item)
            except Exception:
                pass

    raise ValueError(
        f"Could not extract peak_y from simulate_system() output. "
        f"Type: {type(sim_result)}, value: {repr(sim_result)}"
    )


def evaluate_simulator(x: np.ndarray, target: float) -> dict:
    """Evaluate true dynamic simulator for one controllable parameter vector."""
    full_params = vector_to_params(x)

    try:
        sim_result = simulate_system(**full_params)
    except TypeError:
        try:
            sim_result = simulate_system(full_params)
        except TypeError as e:
            raise TypeError(
                "simulate_system() could not be called with either "
                "simulate_system(**full_params) or simulate_system(full_params). "
                "Check the function signature in src/dynamics.py."
            ) from e

    y_sim = extract_peak_y(sim_result)
    error_m = abs(y_sim - target)
    squared_error = error_m**2

    result = {
        "y_sim": y_sim,
        "error_m": error_m,
        "error_mm": error_m * 1000.0,
        "squared_error": squared_error,
    }

    for name, value in zip(CONTROL_NAMES, x):
        result[name] = float(value)

    return result


# ---------------------------------------------------------------------
# Bayesian Optimization core
# ---------------------------------------------------------------------
def fit_bo_gp(X: np.ndarray, y: np.ndarray):
    """
    Fit internal GP model for Bayesian Optimization.

    This GP models:
        controllable parameters -> squared simulator error
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-3))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=False,
        n_restarts_optimizer=5,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_scaled, y_scaled)

    return gp, scaler_X, scaler_y


def select_next_point_lcb(
    gp: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select next BO point using Lower Confidence Bound for minimization.

    LCB(x) = mu(x) - kappa * std(x)
    """
    candidates = sample_random_candidates(N_CANDIDATES, rng)
    candidates_scaled = scaler_X.transform(candidates)

    mu_scaled, std_scaled = gp.predict(candidates_scaled, return_std=True)

    mu = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
    std = std_scaled * scaler_y.scale_[0]

    acquisition = mu - KAPPA * std
    best_idx = int(np.argmin(acquisition))

    return candidates[best_idx]


def run_bayesian_optimization_for_target(
    target: float,
    label: str,
    de_y_sim: float,
    percentile: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run Bayesian Optimization for one target."""
    rng = np.random.default_rng(RANDOM_STATE)

    de_error_m = abs(de_y_sim - target)

    print("\n" + "=" * 80)
    print(f"BAYESIAN OPTIMIZATION - {label}")
    print("=" * 80)
    print(f"Target:   {target:.6f} m")
    print(f"DE y_sim: {de_y_sim:.6f} m")
    print(f"DE error: {de_error_m * 1000.0:.3f} mm")

    rows = []
    X_evaluated = []
    y_evaluated = []

    X_initial = sample_lhs_points(N_INITIAL, RANDOM_STATE)

    print("\nInitial simulator evaluations...")
    for i, x in enumerate(X_initial, start=1):
        eval_result = evaluate_simulator(x, target)

        X_evaluated.append(x)
        y_evaluated.append(eval_result["squared_error"])

        row = {
            "label": label,
            "percentile": percentile,
            "target": target,
            "iteration": i,
            "is_initial": True,
            **eval_result,
        }
        rows.append(row)

        print(
            f"  Init {i:02d}/{N_INITIAL}: "
            f"y_sim={eval_result['y_sim']:.6f} m, "
            f"error={eval_result['error_mm']:.3f} mm"
        )

    print("\nBayesian Optimization iterations...")
    for it in range(1, N_ITER + 1):
        X_arr = np.asarray(X_evaluated)
        y_arr = np.asarray(y_evaluated)

        gp, scaler_X, scaler_y = fit_bo_gp(X_arr, y_arr)
        x_next = select_next_point_lcb(gp, scaler_X, scaler_y, rng)

        eval_result = evaluate_simulator(x_next, target)

        X_evaluated.append(x_next)
        y_evaluated.append(eval_result["squared_error"])

        global_iteration = N_INITIAL + it

        row = {
            "label": label,
            "percentile": percentile,
            "target": target,
            "iteration": global_iteration,
            "is_initial": False,
            **eval_result,
        }
        rows.append(row)

        best_error_mm = min(r["error_mm"] for r in rows)

        print(
            f"  BO {it:02d}/{N_ITER}: "
            f"y_sim={eval_result['y_sim']:.6f} m, "
            f"error={eval_result['error_mm']:.3f} mm, "
            f"best={best_error_mm:.3f} mm"
        )

    df = pd.DataFrame(rows)

    df["best_error_m"] = df["error_m"].cummin()
    df["best_error_mm"] = df["error_mm"].cummin()
    df["de_y_sim"] = de_y_sim
    df["de_error_m"] = de_error_m
    df["de_error_mm"] = de_error_m * 1000.0

    best_idx = df["error_m"].idxmin()
    best = df.loc[best_idx]

    summary = {
        "label": label,
        "percentile": percentile,
        "target": target,
        "bo_best_y_sim": best["y_sim"],
        "bo_best_error_m": best["error_m"],
        "bo_best_error_mm": best["error_mm"],
        "bo_best_iteration": int(best["iteration"]),
        "bo_best_Kr": best["Kr"],
        "bo_best_hr": best["hr"],
        "bo_best_f0": best["f0"],
        "bo_best_f1": best["f1"],
        "bo_best_A": best["A"],
        "bo_best_x_r_start": best["x_r_start"],
        "de_y_sim": de_y_sim,
        "de_error_m": de_error_m,
        "de_error_mm": de_error_m * 1000.0,
        "bo_minus_de_error_mm": best["error_mm"] - de_error_m * 1000.0,
        "bo_better_than_de": bool(best["error_mm"] < de_error_m * 1000.0),
        "n_initial": N_INITIAL,
        "n_iter": N_ITER,
        "total_evaluations": N_INITIAL + N_ITER,
        "kappa": KAPPA,
        "n_candidates": N_CANDIDATES,
    }

    print(f"\nBest BO for {label}:")
    print(f"  y_sim:    {summary['bo_best_y_sim']:.6f} m")
    print(f"  error:    {summary['bo_best_error_mm']:.3f} mm")
    print(f"  DE error: {summary['de_error_mm']:.3f} mm")

    return df, summary


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------
def plot_single_convergence(df: pd.DataFrame, summary: dict):
    label = summary["label"]
    fig_path = FIGURES_BO_DIR / f"bo_convergence_{label}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(df["iteration"], df["best_error_mm"], marker="o", linewidth=2)
    plt.axhline(
        summary["de_error_mm"],
        linestyle="--",
        linewidth=2,
        label=f"DE reference: {summary['de_error_mm']:.3f} mm",
    )
    plt.xlabel("Simulator evaluation")
    plt.ylabel("Best error so far [mm]")
    plt.title(f"Bayesian Optimization convergence - {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_single_convergence_zoom(df: pd.DataFrame, summary: dict):
    label = summary["label"]
    fig_path = FIGURES_BO_DIR / f"bo_convergence_zoom_{label}.png"

    df_zoom = df[df["iteration"] >= N_INITIAL]

    plt.figure(figsize=(8, 5))
    plt.plot(df_zoom["iteration"], df_zoom["best_error_mm"], marker="o", linewidth=2)
    plt.axhline(
        summary["de_error_mm"],
        linestyle="--",
        linewidth=2,
        label=f"DE reference: {summary['de_error_mm']:.3f} mm",
    )
    plt.ylim(0, max(1.0, min(5.0, df_zoom["best_error_mm"].max() * 1.2)))
    plt.xlabel("Simulator evaluation")
    plt.ylabel("Best error so far [mm]")
    plt.title(f"Bayesian Optimization convergence zoom - {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_single_evaluations(df: pd.DataFrame, summary: dict):
    label = summary["label"]
    fig_path = FIGURES_BO_DIR / f"bo_evaluations_{label}.png"

    initial = df[df["is_initial"] == True]
    bo_df = df[df["is_initial"] == False]

    plt.figure(figsize=(8, 5))
    plt.scatter(initial["iteration"], initial["y_sim"], marker="o", label="Initial evaluations")
    plt.scatter(bo_df["iteration"], bo_df["y_sim"], marker="x", label="BO evaluations")
    plt.axhline(
        summary["target"],
        linestyle="--",
        linewidth=2,
        label=f"Target = {summary['target']:.6f} m",
    )
    plt.xlabel("Simulator evaluation")
    plt.ylabel("Simulated peak outreach [m]")
    plt.title(f"BO simulator evaluations - {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_bo_vs_de_single(summary: dict):
    label = summary["label"]
    fig_path = FIGURES_BO_DIR / f"bo_vs_de_{label}.png"

    labels = ["Bayesian Optimization", "Differential Evolution"]
    errors = [summary["bo_best_error_mm"], summary["de_error_mm"]]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, errors)
    plt.ylabel("Simulator error [mm]")
    plt.title(f"BO vs DE - {label}")
    plt.grid(axis="y", alpha=0.3)

    for bar, error in zip(bars, errors):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{error:.3f} mm",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_bo_vs_de_all_targets(summary_df: pd.DataFrame):
    fig_path = FIGURES_BO_DIR / "bo_vs_de_all_targets.png"

    labels = [f"P{int(p)}" for p in summary_df["percentile"]]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    bars_bo = plt.bar(
        x - width / 2,
        summary_df["bo_best_error_mm"],
        width,
        label="Bayesian Optimization",
    )
    bars_de = plt.bar(
        x + width / 2,
        summary_df["de_error_mm"],
        width,
        label="Differential Evolution",
    )

    plt.xticks(x, labels)
    plt.ylabel("Simulator error [mm]")
    plt.xlabel("Target percentile")
    plt.title("BO vs Differential Evolution across target values")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in [bars_bo, bars_de]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_bo_convergence_all_targets(all_results_df: pd.DataFrame):
    fig_path = FIGURES_BO_DIR / "bo_convergence_all_targets.png"

    plt.figure(figsize=(9, 5))

    for label, group in all_results_df.groupby("label"):
        percentile = int(group["percentile"].iloc[0])
        plt.plot(
            group["iteration"],
            group["best_error_mm"],
            marker="o",
            linewidth=1.5,
            label=f"P{percentile}",
        )

    plt.xlabel("Simulator evaluation")
    plt.ylabel("Best error so far [mm]")
    plt.title("Bayesian Optimization convergence across targets")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Target")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_bo_error_by_target(summary_df: pd.DataFrame):
    fig_path = FIGURES_BO_DIR / "bo_error_by_target.png"

    plt.figure(figsize=(8, 5))
    plt.plot(
        summary_df["target"],
        summary_df["bo_best_error_mm"],
        marker="o",
        linewidth=2,
        label="Bayesian Optimization",
    )
    plt.plot(
        summary_df["target"],
        summary_df["de_error_mm"],
        marker="s",
        linewidth=2,
        label="Differential Evolution",
    )

    plt.xlabel("Target outreach [m]")
    plt.ylabel("Simulator error [mm]")
    plt.title("Inverse optimization error vs target")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


def plot_bo_convergence_all_targets_zoom(all_results_df: pd.DataFrame):
    fig_path = FIGURES_BO_DIR / "bo_convergence_all_targets_zoom.png"

    plt.figure(figsize=(9, 5))

    for label, group in all_results_df.groupby("label"):
        percentile = int(group["percentile"].iloc[0])
        group_zoom = group[group["iteration"] >= N_INITIAL]

        plt.plot(
            group_zoom["iteration"],
            group_zoom["best_error_mm"],
            marker="o",
            linewidth=1.5,
            label=f"P{percentile}",
        )

    plt.ylim(0, 5)
    plt.xlabel("Simulator evaluation")
    plt.ylabel("Best error so far [mm]")
    plt.title("Bayesian Optimization convergence across targets - zoom")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Target")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {fig_path}")


# ---------------------------------------------------------------------
# Save and summary functions
# ---------------------------------------------------------------------
def save_single_target_outputs(df: pd.DataFrame, summary: dict):
    label = summary["label"]

    results_path = RESULTS_BO_DIR / f"bo_results_{label}.csv"
    summary_path = RESULTS_BO_DIR / f"bo_summary_{label}.csv"

    df.to_csv(results_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"✓ Saved results: {results_path}")
    print(f"✓ Saved summary: {summary_path}")

    plot_single_convergence(df, summary)
    plot_single_convergence_zoom(df, summary)
    plot_single_evaluations(df, summary)
    plot_bo_vs_de_single(summary)


def print_single_summary(summary: dict):
    print("\n" + "=" * 80)
    print(f"BO SUMMARY - {summary['label']}")
    print("=" * 80)

    print(f"Target:              {summary['target']:.6f} m")
    print(f"BO best y_sim:       {summary['bo_best_y_sim']:.6f} m")
    print(f"BO best error:       {summary['bo_best_error_mm']:.3f} mm")
    print(f"DE y_sim:            {summary['de_y_sim']:.6f} m")
    print(f"DE error:            {summary['de_error_mm']:.3f} mm")
    print(f"BO - DE error:       {summary['bo_minus_de_error_mm']:.3f} mm")
    print(f"BO better than DE:   {summary['bo_better_than_de']}")

    print("\nBest BO parameters:")
    print(f"  Kr:         {summary['bo_best_Kr']:.6f}")
    print(f"  hr:         {summary['bo_best_hr']:.6f}")
    print(f"  f0:         {summary['bo_best_f0']:.6f}")
    print(f"  f1:         {summary['bo_best_f1']:.6f}")
    print(f"  A:          {summary['bo_best_A']:.6f}")
    print(f"  x_r_start:  {summary['bo_best_x_r_start']:.6f}")


def print_final_multi_summary(summary_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("MULTI-TARGET BO SUMMARY")
    print("=" * 80)

    cols = [
        "percentile",
        "target",
        "bo_best_y_sim",
        "bo_best_error_mm",
        "de_y_sim",
        "de_error_mm",
        "bo_minus_de_error_mm",
        "bo_better_than_de",
    ]

    print(summary_df[cols].to_string(index=False))

    print("\nAggregate metrics:")
    print(f"  BO mean error: {summary_df['bo_best_error_mm'].mean():.3f} mm")
    print(f"  BO max error:  {summary_df['bo_best_error_mm'].max():.3f} mm")
    print(f"  DE mean error: {summary_df['de_error_mm'].mean():.3f} mm")
    print(f"  DE max error:  {summary_df['de_error_mm'].max():.3f} mm")

    n_better = int(summary_df["bo_better_than_de"].sum())
    print(f"\nBO better than DE on {n_better}/{len(summary_df)} targets.")
    print(f"  BO evaluations per target: {N_INITIAL + N_ITER}")
    print(f"  Total true simulator evaluations: {(N_INITIAL + N_ITER) * len(summary_df)}")


# ---------------------------------------------------------------------
# Main execution modes
# ---------------------------------------------------------------------
def run_single_target_mode():
    target_info = TARGETS[2]

    df, summary = run_bayesian_optimization_for_target(
        target=target_info["target"],
        label=target_info["label"],
        de_y_sim=target_info["de_y_sim"],
        percentile=target_info["percentile"],
    )

    save_single_target_outputs(df, summary)
    print_single_summary(summary)


def run_all_targets_mode():
    all_results = []
    summaries = []

    for target_info in TARGETS:
        df, summary = run_bayesian_optimization_for_target(
            target=target_info["target"],
            label=target_info["label"],
            de_y_sim=target_info["de_y_sim"],
            percentile=target_info["percentile"],
        )

        save_single_target_outputs(df, summary)

        all_results.append(df)
        summaries.append(summary)

    all_results_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    all_results_path = RESULTS_BO_DIR / "bo_results_all_targets.csv"
    summary_path = RESULTS_BO_DIR / "bo_summary_all_targets.csv"

    all_results_df.to_csv(all_results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n✓ Saved all-target BO results: {all_results_path}")
    print(f"✓ Saved all-target BO summary: {summary_path}")

    plot_bo_vs_de_all_targets(summary_df)
    plot_bo_convergence_all_targets(all_results_df)
    plot_bo_convergence_all_targets_zoom(all_results_df)
    plot_bo_error_by_target(summary_df)

    print_final_multi_summary(summary_df)


def print_header():
    print("=" * 80)
    print("BAYESIAN OPTIMIZATION - TRUE SIMULATOR BLACK-BOX")
    print("=" * 80)

    print("\nThis is an additional experiment.")
    print("It does NOT replace the official GP + Differential Evolution pipeline.")

    print("\nFixed parameters:")
    for key, value in FIXED_PARAMS.items():
        print(f"  {key} = {value}")

    print("\nOptimized parameters:")
    for name in CONTROL_NAMES:
        print(f"  {name}: {BOUNDS_DICT[name]}")

    print("\nBO configuration:")
    print(f"  RUN_ALL_TARGETS:      {RUN_ALL_TARGETS}")
    print(f"  Initial evaluations:  {N_INITIAL}")
    print(f"  BO iterations:        {N_ITER}")
    print(f"  Total evaluations:    {N_INITIAL + N_ITER} per target")
    print(f"  Candidates/iteration: {N_CANDIDATES}")
    print(f"  Kappa:                {KAPPA}")
    print(f"  Random state:         {RANDOM_STATE}")
    print(f"  Results directory:    {RESULTS_BO_DIR}")
    print(f"  Figures directory:    {FIGURES_BO_DIR}")


def main():
    ensure_dirs()
    print_header()

    if RUN_ALL_TARGETS:
        run_all_targets_mode()
    else:
        run_single_target_mode()

    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION EXPERIMENT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()