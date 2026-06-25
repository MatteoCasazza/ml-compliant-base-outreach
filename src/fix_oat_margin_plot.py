from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "sensitivity_oat"
    / "sensitivity_oat_summary.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "sensitivity_oat"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = 0.75
METRIC = "min_residual_margin_mm"

OUTPUT_NAME = "sensitivity_residual_margin_tornado_target0750_fixed.png"


def plot_fixed_oat_margin_tornado() -> None:
    df = pd.read_csv(SUMMARY_PATH)

    df_t = df[np.isclose(df["target"].astype(float), TARGET)].copy()
    df_t = df_t.sort_values(METRIC, ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    ax.barh(
        df_t["parameter"],
        df_t[METRIC],
        edgecolor="black",
        linewidth=0.8,
        color="#1f77b4",
    )

    xmin = min(float(df_t[METRIC].min()), 0.0)
    xmax = max(float(df_t[METRIC].max()), 0.0)
    x_range = xmax - xmin if xmax > xmin else 1.0

    # Regione infeasible + linea boundary
    ax.axvspan(
        xmin,
        0.0,
        alpha=0.12,
        color="#9ecae1",
    )
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=2.0,
        color="#1f77b4",
    )

    ax.set_xlabel("min residual margin [mm]")
    ax.set_ylabel("Perturbed parameter")
    ax.set_title(
        "Minimum residual margin under OAT perturbations, target = 0.75 m"
    )
    ax.grid(True, axis="x", alpha=0.3)

    # spazio extra a destra per etichette C/P
    ax.set_xlim(xmin - 0.08 * x_range, xmax + 0.14 * x_range)

    # Etichette C/P sui bar
    for i, (_, row) in enumerate(df_t.iterrows()):
        value = float(row[METRIC])
        group_label = "C" if row["parameter_group"] == "controllable" else "P"

        offset = 0.025 * x_range
        if value >= 0:
            x_text = value + offset
            ha = "left"
        else:
            x_text = value - offset
            ha = "right"

        ax.text(
            x_text,
            i,
            group_label,
            va="center",
            ha=ha,
            fontsize=9,
        )

    # -------------------------------------------------------------------------
    # LEGEND 1: constraint boundary + infeasible margin
    # -------------------------------------------------------------------------
    handles_main = [
        Line2D([0], [0], color="#1f77b4", lw=2.0, linestyle="--", label="Constraint boundary"),
        Patch(facecolor="#9ecae1", edgecolor="none", alpha=0.12, label="Infeasible margin"),
    ]

    legend_main = ax.legend(
        handles=handles_main,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.12),   # questa sta sopra
        fontsize=8,
        frameon=True,
    )
    ax.add_artist(legend_main)

    # -------------------------------------------------------------------------
    # LEGEND 2: spiegazione C / P
    # -------------------------------------------------------------------------
    handles_cp = [
        Line2D([], [], linestyle="None", label="C = controllable"),
        Line2D([], [], linestyle="None", label="P = physical/model"),
    ]

    legend_cp = ax.legend(
        handles=handles_cp,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.01),   # questa sta sotto
        fontsize=8,
        frameon=True,
        handlelength=0,
        handletextpad=0.0,
        borderpad=0.4,
        labelspacing=0.3,
    )

    plt.tight_layout()

    out_path = FIGURES_DIR / OUTPUT_NAME
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved fixed figure: {out_path}")


if __name__ == "__main__":
    plot_fixed_oat_margin_tornado()