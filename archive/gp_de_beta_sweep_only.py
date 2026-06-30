"""
gp_de_beta_sweep_only.py
=========================

Run only the beta_constraint_std sweep for optimization_gp_de_v2.py,
without rerunning the full GP+DE optimization on all targets.

Place this file in src/ and run:

    python src/gp_de_beta_sweep_only.py

It saves:
    results/optimization_gp_de_v2/gp_de_beta_sweep.csv
    figures/optimization_gp_de_v2/gp_de_beta_sensitivity.png
"""

from __future__ import annotations

import pandas as pd

from optimization_gp_de_v2 import (
    GPDEConfig,
    ensure_output_dirs,
    load_surrogates,
    run_beta_sweep,
    plot_beta_sweep,
)


def main() -> None:
    ensure_output_dirs()

    config = GPDEConfig(
        run_uncertainty_diagnostics=False,
        run_lambda_sweep=False,
        run_beta_sweep=True,
        beta_sweep_values=(0.0, 0.5, 1.0, 1.96),
        beta_sweep_targets=(0.65, 0.75),
        beta_sweep_attempts=2,
        beta_sweep_maxiter=70,
        beta_sweep_popsize=12,
    )

    print("=" * 80)
    print("GP + DE BETA SWEEP ONLY")
    print("=" * 80)
    print(f"Targets: {config.beta_sweep_targets}")
    print(f"Beta values: {config.beta_sweep_values}")
    print(f"Attempts per setting: {config.beta_sweep_attempts}")
    print(f"DE maxiter/popsize: {config.beta_sweep_maxiter}/{config.beta_sweep_popsize}")
    print("=" * 80)

    models = load_surrogates()
    sweep_df = run_beta_sweep(models, config)
    plot_beta_sweep(sweep_df, config)

    display_cols = [
        "sweep_beta_constraint_std",
        "target",
        "peak_y_true",
        "target_error_mm",
        "max_abs_xr_true",
        "residual_margin_mm",
        "constraint_violation_abs_mm",
        "feasible_abs",
        "peak_y_pred",
        "max_abs_xr_pred",
        "constraint_value_m",
    ]
    existing_cols = [c for c in display_cols if c in sweep_df.columns]

    print("\n" + "=" * 80)
    print("BETA SWEEP SUMMARY")
    print("=" * 80)
    print(sweep_df[existing_cols].sort_values(["sweep_beta_constraint_std", "target"]).to_string(index=False))

    path = config.__class__  # keeps linters quiet; output paths are defined in optimization_gp_de_v2
    print("\nSaved beta sweep files in:")
    print("  results/optimization_gp_de_v2/gp_de_beta_sweep.csv")
    print("  figures/optimization_gp_de_v2/gp_de_beta_sensitivity.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
