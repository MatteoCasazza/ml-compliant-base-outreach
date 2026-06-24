from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = PROJECT_ROOT / "results" / "optimization_bo_final" / "bo_final_all_history.csv"
FIGURES_DIR = PROJECT_ROOT / "figures" / "optimization_bo_final"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FIGURES_DIR / "bo_final_convergence_all_targets_bo_vs_random.png"


def prepare_best_feasible_curve(df_target):
    """
    Build one monotonic best-feasible-error curve per seed.
    Missing values before the first feasible point remain NaN.
    Missing values after the first feasible point are forward-filled.
    """
    curves = []

    max_iter = int(df_target["iteration"].max())
    full_iter = pd.DataFrame({"iteration": np.arange(1, max_iter + 1)})

    for seed, df_seed in df_target.groupby("seed"):
        df_seed = df_seed.sort_values("iteration").copy()

        curve = df_seed[["iteration", "best_feasible_error_so_far_mm"]].copy()
        curve = full_iter.merge(curve, on="iteration", how="left")

        curve["best_feasible_error_so_far_mm"] = curve[
            "best_feasible_error_so_far_mm"
        ].ffill()

        curve["seed"] = seed
        curves.append(curve)

    return pd.concat(curves, ignore_index=True)


def main():
    df = pd.read_csv(HISTORY_PATH)

    targets = [0.55, 0.60, 0.65, 0.70, 0.75]
    methods = ["BO", "RandomSearch"]

    df = df[df["target"].round(2).isin(targets)].copy()
    df = df[df["method"].isin(methods)].copy()

    if df.empty:
        raise ValueError("No BO or Random Search data found for the selected targets.")

    metric = "best_feasible_error_so_far_mm"

    if metric not in df.columns:
        raise KeyError(f"Column '{metric}' not found in {HISTORY_PATH}")

    # un colore fisso per ogni target
    color_map = {
        0.55: "C0",
        0.60: "C1",
        0.65: "C2",
        0.70: "C3",
        0.75: "C4",
    }

    plt.figure(figsize=(10.5, 6.5))

    for target in targets:
        color = color_map[target]

        for method in methods:
            df_tm = df[
                (df["target"].round(2) == target)
                & (df["method"] == method)
            ].copy()

            if df_tm.empty:
                continue

            curves = prepare_best_feasible_curve(df_tm)

            grouped = (
                curves.groupby("iteration")[metric]
                .agg(["mean"])
                .reset_index()
                .sort_values("iteration")
            )

            linestyle = "-" if method == "BO" else "--"
            label = f"{target:.2f} m - {method}"

            plt.plot(
                grouped["iteration"],
                grouped["mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )

    plt.xlabel("True simulator calls")
    plt.ylabel("Best feasible target error so far [mm]")
    plt.title("BO vs Random Search convergence for all targets")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()