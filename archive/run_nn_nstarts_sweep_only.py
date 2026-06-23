"""
run_nn_nstarts_sweep_only.py
----------------------------

Run only the NN+Adam multi-start sweep and save:

results/optimization_nn_gradient_v2_safe/nn_gradient_nstarts_sweep.csv
figures/optimization_nn_gradient_v2_safe/nn_gradient_nstarts_sensitivity_target065.png
"""

from dataclasses import replace

from optimization_nn_gradient_ensemble_safe_final import (
    NNGradientConfig,
    ensure_output_dirs,
    get_device,
    load_nn_surrogate,
    run_nstarts_sweep,
    plot_nstarts_sweep,
)


def main() -> None:
    ensure_output_dirs()

    config = NNGradientConfig()

    # Run only the multi-start sweep.
    config = replace(
        config,
        run_learning_rate_sweep=False,
        run_nstarts_sweep=True,
        nstarts_sweep_target=0.65,
        nstarts_sweep_values=(50, 100, 150, 200),
        nstarts_sweep_steps=300,
        nstarts_sweep_top_k_validate=3,
    )

    device = get_device()
    models = load_nn_surrogate(config, device)

    sweep_df = run_nstarts_sweep(
        models=models,
        base_config=config,
        device=device,
    )

    plot_nstarts_sweep(sweep_df, config)

    print("\nSaved:")
    print("  results/optimization_nn_gradient_v2_safe/nn_gradient_nstarts_sweep.csv")
    print("  figures/optimization_nn_gradient_v2_safe/nn_gradient_nstarts_sensitivity_target065.png")


if __name__ == "__main__":
    main()