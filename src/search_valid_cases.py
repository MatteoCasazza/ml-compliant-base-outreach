"""
search_valid_cases.py
=====================
Search for physically feasible extra-reach cases using the true simulator.

Goal:
- maximize peak_y
- keep the robot relative position within its bound:
      max_xr <= x_r_max + tolerance

This script is not part of the ML training pipeline.
It is used to understand which targets are physically realistic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc
from joblib import Parallel, delayed

from dynamics import simulate_system


PARAM_NAMES = ['Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start']


def sample_parameters(n_samples=2000, seed=123):
    """
    Generate a targeted parameter set for feasible extra-reach search.
    """
    bounds = {
        'Kb': (100.0, 1500.0),
        'Kr': (500.0, 5000.0),
        'Mb': (10.0, 80.0),
        'hb': (0.05, 0.25),
        'hr': (0.1, 0.8),
        'f0': (0.05, 0.5),
        'f1': (1.0, 8.0),
        'A': (0.03, 0.12),
        'x_r_start': (0.35, 0.45),
    }

    lb = np.array([bounds[name][0] for name in PARAM_NAMES])
    ub = np.array([bounds[name][1] for name in PARAM_NAMES])

    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    X_unit = sampler.random(n_samples)
    X = qmc.scale(X_unit, lb, ub)

    return X


def simulate_one(params_array, idx, x_r_max=0.5, tolerance=0.002):
    """
    Simulate one sample and return feasibility information.
    """
    try:
        peak_y, metrics = simulate_system(
            params_array,
            T_sim=60.0,
            dt=0.001,
            return_metrics=True,
            x_r_max=x_r_max
        )

        feasible = metrics['constraint_violation'] <= tolerance
        extra_reach = metrics['peak_y'] - x_r_max

        row = dict(zip(PARAM_NAMES, params_array))
        row.update({
            'peak_y': metrics['peak_y'],
            'max_xr': metrics['max_xr'],
            'max_xb': metrics['max_xb'],
            'extra_reach': extra_reach,
            'constraint_violation': metrics['constraint_violation'],
            'feasible': feasible
        })

        return row

    except Exception as e:
        print(f"Simulation {idx} failed: {e}")
        row = dict(zip(PARAM_NAMES, params_array))
        row.update({
            'peak_y': np.nan,
            'max_xr': np.nan,
            'max_xb': np.nan,
            'extra_reach': np.nan,
            'constraint_violation': np.nan,
            'feasible': False
        })
        return row


if __name__ == "__main__":
    N_SAMPLES = 2000
    X_R_MAX = 0.5
    TOLERANCE = 0.002

    print("=" * 70)
    print("TARGETED SEARCH FOR FEASIBLE EXTRA-REACH CASES")
    print("=" * 70)
    print(f"Samples:      {N_SAMPLES}")
    print(f"x_r_max:      {X_R_MAX}")
    print(f"Tolerance:    {TOLERANCE}")
    print("=" * 70)

    X = sample_parameters(n_samples=N_SAMPLES, seed=123)

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulate_one)(X[i], i, X_R_MAX, TOLERANCE)
        for i in range(N_SAMPLES)
    )

    df = pd.DataFrame(results)
    df = df.dropna(subset=['peak_y']).reset_index(drop=True)

    feasible = df[df['feasible']]
    good = feasible[feasible['extra_reach'] > 0]

    print("\nResults:")
    print(f"Total valid simulations:       {len(df)}")
    print(f"Feasible samples:              {len(feasible)}")
    print(f"Good feasible extra-reach:     {len(good)}")

    if len(good) > 0:
        print(f"Best feasible peak_y:          {good['peak_y'].max():.6f} m")
        print(f"Best feasible extra_reach:     {good['extra_reach'].max():.6f} m")

        best = good.sort_values('peak_y', ascending=False).head(10)

        print("\nTop 10 feasible extra-reach cases:")
        cols = [
            'peak_y', 'extra_reach', 'max_xr', 'max_xb',
            'constraint_violation',
            'Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start'
        ]
        print(best[cols].to_string(index=False))

    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/targeted_search_results.csv', index=False)

    print("\n✓ Results saved to: data/targeted_search_results.csv")