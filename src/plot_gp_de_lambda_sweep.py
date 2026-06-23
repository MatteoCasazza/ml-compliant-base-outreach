"""
plot_gp_de_lambda_sweep.py
--------------------------

Create a compact appendix figure for the GP+DE lambda sweep.

Expected input:
    results/optimization_gp_de_v2/gp_de_lambda_sweep.csv

Expected output:
    figures/optimization_gp_de_v2/gp_de_lambda_sensitivity.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "results" / "optimization_gp_de_v2" / "gp_de_lambda_sweep.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "optimization_gp_de_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = FIG_DIR / "gp_de_lambda_sensitivity.png"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Lambda sweep CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Sort by lambda
    df = df.sort_values("sweep_lambda_constraint").reset_index(drop=True)

    # If residual margin is not already saved, compute it from true constraint value
    if "residual_margin_mm" not in df.columns:
        if "max_abs_xr_true" not in df.columns:
            raise ValueError(
                "Need either 'residual_margin_mm' or 'max_abs_xr_true' in gp_de_lambda_sweep.csv"
            )
        df["residual_margin_mm"] = (0.500 - df["max_abs_xr_true"]) * 1000.0

    # Short console summary
    cols_to_show = [
        "sweep_lambda_constraint",
        "target",
        "peak_y_true",
        "target_error_mm",
        "max_abs_xr_true",
        "residual_margin_mm",
        "feasible_abs",
    ]
    existing_cols = [c for c in cols_to_show if c in df.columns]

    print("\nLAMBDA SWEEP SUMMARY")
    print(df[existing_cols].to_string(index=False))

    # ---------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = df["sweep_lambda_constraint"]

    # Panel 1: target error
    axes[0].plot(x, df["target_error_mm"], marker="o", linewidth=2)
    axes[0].set_xlabel(r"Constraint penalty weight $\lambda_c$")
    axes[0].set_ylabel("True target error [mm]")
    axes[0].set_title("Lambda Sweep: Target Error")
    axes[0].grid(True, alpha=0.3)

    # annotate values
    for xi, yi in zip(x, df["target_error_mm"]):
        axes[0].annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center")

    # Panel 2: residual margin
    axes[1].plot(x, df["residual_margin_mm"], marker="o", linewidth=2)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.5)
    axes[1].set_xlabel(r"Constraint penalty weight $\lambda_c$")
    axes[1].set_ylabel("Residual margin [mm]")
    axes[1].set_title("Lambda Sweep: Workspace Margin")
    axes[1].grid(True, alpha=0.3)

    # annotate feasibility
    for xi, yi, feas in zip(x, df["residual_margin_mm"], df["feasible_abs"]):
        label = "F" if bool(feas) else "NF"
        axes[1].annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center")

    target_text = ""
    if "target" in df.columns and len(df["target"].unique()) == 1:
        target_text = f" (target = {df['target'].iloc[0]:.2f} m)"

    fig.suptitle(f"GP+DE Constraint Penalty Sweep{target_text}", fontsize=14)
    plt.tight_layout()

    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure: {OUT_PATH}")


if __name__ == "__main__":
    main()