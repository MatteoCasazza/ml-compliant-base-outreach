"""
mpc_animation.py
================

Animazione MPC closed-loop nello stesso stile grafico di visualization.py.

Schema:
    [Parete] ~~~ [Massa Base] ~~~ [Massa Robot] --> Target

Nota importante:
- xb, xr, y e u sono i valori fisici veri letti da results/mpc/mpc_results.csv
- nella visualizzazione viene usato un offset grafico per rendere visibile
  la molla base, perché xb può essere molto vicino a zero

Input:
    results/mpc/mpc_results.csv

Output:
    figures/mpc/mpc_nominal_animation.gif
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "mpc"
FIGURES_DIR = PROJECT_ROOT / "figures" / "mpc"

INPUT_CSV = RESULTS_DIR / "mpc_results.csv"
OUTPUT_GIF = FIGURES_DIR / "mpc_nominal_animation.gif"

Y_REF = 0.592779

U_MIN = 0.30
U_MAX = 0.65

FPS = 20
DURATION_GIF = 8.0
DPI = 80

# Offset solo grafico: non modifica i valori fisici
VISUAL_OFFSET = 0.12

# Colori coerenti con visualization.py
COLOR_BASE = "#9467bd"    # viola
COLOR_ROBOT = "#d62728"   # rosso
COLOR_TARGET = "#27ae60"  # verde
COLOR_CURRENT = "blue"
COLOR_PEAK = "orange"


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dirs():
    """Create output directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_mpc_data() -> pd.DataFrame:
    """Load nominal MPC results."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}\n"
            "Run first:\n"
            "  python src/mpc_control.py"
        )

    df = pd.read_csv(INPUT_CSV)

    required = ["time", "xb", "xr", "y", "u"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

    if "tracking_error_mm" not in df.columns:
        df["tracking_error_mm"] = (df["y"] - Y_REF) * 1000.0

    print(f"✓ Loaded MPC data: {INPUT_CSV}")
    print(f"✓ Samples: {len(df)}")
    return df


def draw_zigzag_spring(
    ax,
    x_start,
    x_end,
    y=0.2,
    amplitude=0.04,
    n_coils=8,
    color="green",
    linewidth=2.5,
    zorder=2,
):
    """
    Disegna una molla zig-zag tra x_start e x_end.
    Se la distanza è troppo piccola, disegna una linea semplice.
    """
    if x_end <= x_start:
        ax.plot([x_start, x_end], [y, y], color=color, linewidth=linewidth, zorder=zorder)
        return

    margin = min(0.02, 0.15 * abs(x_end - x_start))

    if x_end - x_start < 0.04:
        ax.plot([x_start, x_end], [y, y], color=color, linewidth=linewidth, zorder=zorder)
        return

    xs = np.linspace(x_start + margin, x_end - margin, 2 * n_coils + 1)
    ys = np.full_like(xs, y)

    for i in range(1, len(xs) - 1):
        ys[i] = y + amplitude * (1 if i % 2 else -1)

    # piccoli tratti dritti iniziale/finale
    x_full = np.concatenate([[x_start], xs, [x_end]])
    y_full = np.concatenate([[y], ys, [y]])

    ax.plot(x_full, y_full, color=color, linewidth=linewidth, zorder=zorder)


# =============================================================================
# ANIMATION
# =============================================================================

def create_animation(df: pd.DataFrame):
    """
    Crea GIF animata del caso MPC nominale.
    """
    t = df["time"].to_numpy()
    xb = df["xb"].to_numpy()
    xr = df["xr"].to_numpy()
    y = df["y"].to_numpy()
    u = df["u"].to_numpy()
    err_mm = df["tracking_error_mm"].to_numpy()

    total_frames = int(FPS * DURATION_GIF)
    frame_indices = np.linspace(0, len(t) - 1, total_frames, dtype=int)

    print("\nCreazione animazione MPC...")
    print(f"  FPS: {FPS}")
    print(f"  Durata GIF: {DURATION_GIF:.1f} s")
    print(f"  Frame totali: {total_frames}")

    fig, (ax_system, ax_plot) = plt.subplots(
        2,
        1,
        figsize=(22, 9),
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    # Limiti asse sistema
    visual_y = VISUAL_OFFSET + y
    visual_target = VISUAL_OFFSET + Y_REF
    y_max_axis = max(visual_target * 1.12, visual_y.max() * 1.12, VISUAL_OFFSET + U_MAX + 0.1)

    def init():
        ax_system.clear()
        ax_plot.clear()
        return []

    def update(frame_idx):
        idx = frame_indices[frame_idx]

        t_current = t[idx]
        xb_current = xb[idx]
        xr_current = xr[idx]
        y_current = y[idx]
        u_current = u[idx]
        err_current = err_mm[idx]

        # ---------------------------------------------------------------------
        # Coordinate fisiche e coordinate grafiche
        # ---------------------------------------------------------------------
        # Fisico:
        #   base_pos_phys = xb
        #   robot_pos_phys = xb + xr = y
        #
        # Grafico:
        #   aggiungo VISUAL_OFFSET per vedere bene la molla base.
        # ---------------------------------------------------------------------
        base_pos = VISUAL_OFFSET + xb_current
        robot_pos = VISUAL_OFFSET + y_current
        target_pos = VISUAL_OFFSET + Y_REF

        base_size = 0.045
        robot_size = 0.045

        wall_x = 0.0
        spring_y = 0.2

        base_left = base_pos - base_size / 2
        base_right = base_pos + base_size / 2

        robot_left = robot_pos - robot_size / 2
        robot_right = robot_pos + robot_size / 2

        # ---------------------------------------------------------------------
        # SISTEMA FISICO
        # ---------------------------------------------------------------------
        ax_system.clear()
        ax_system.set_xlim(-0.08, y_max_axis + 0.08)
        ax_system.set_ylim(0.0, 0.42)
        ax_system.set_aspect("auto")
        ax_system.set_yticks([])
        ax_system.set_xlabel("Position [m]  (visual offset used only for drawing)", fontsize=11)
        ax_system.grid(True, alpha=0.2)

        # Parete verticale
        wall_rect = patches.Rectangle(
            (-0.0025, -0.04),
            0.005,
            0.50,
            facecolor="black",
            edgecolor="black",
            zorder=0,
        )
        ax_system.add_patch(wall_rect)

        # Tratteggio parete, simile allo stile MATLAB/schema professore
        for y_w in np.linspace(0.02, 0.38, 10):
            ax_system.plot(
                [-0.005, -0.045],
                [y_w, y_w - 0.025],
                color="black",
                linewidth=1,
                zorder=0,
            )

        # Molla base: parete -> lato sinistro base
        draw_zigzag_spring(
            ax_system,
            wall_x,
            base_left,
            y=spring_y,
            amplitude=0.030,
            n_coils=5,
            color="orange",
            linewidth=2.5,
            zorder=2,
        )

        # Massa base
        base_rect = patches.Rectangle(
            (base_left, spring_y - base_size / 2),
            base_size,
            base_size,
            linewidth=2,
            edgecolor="black",
            facecolor=COLOR_BASE,
            alpha=0.85,
            zorder=5,
        )
        ax_system.add_patch(base_rect)

        ax_system.text(
            base_pos,
            0.055,
            "Base",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=COLOR_BASE,
        )

        # Molla robot: lato destro base -> lato sinistro robot
        draw_zigzag_spring(
            ax_system,
            base_right,
            robot_left,
            y=spring_y,
            amplitude=0.040,
            n_coils=8,
            color="forestgreen",
            linewidth=2.5,
            zorder=2,
        )

        # Massa robot
        robot_rect = patches.Rectangle(
            (robot_left, spring_y - robot_size / 2),
            robot_size,
            robot_size,
            linewidth=2,
            edgecolor="black",
            facecolor=COLOR_ROBOT,
            alpha=0.85,
            zorder=5,
        )
        ax_system.add_patch(robot_rect)

        ax_system.text(
            robot_pos,
            0.055,
            "Robot",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=COLOR_ROBOT,
        )

        # Marker end-effector / outreach corrente
        ax_system.plot(
            robot_pos,
            spring_y,
            "ko",
            markersize=6,
            zorder=7,
            label="Current y",
        )

        # Linea outreach corrente
        ax_system.plot(
            [robot_pos, robot_pos],
            [-0.03, 0.43],
            color=COLOR_CURRENT,
            linestyle="--",
            linewidth=2,
            alpha=0.55,
            label=f"Current y = {y_current:.3f} m",
        )

        # Linea target
        ax_system.plot(
            [target_pos, target_pos],
            [-0.03, 0.43],
            color=COLOR_TARGET,
            linestyle="--",
            linewidth=2.5,
            label=f"Target = {Y_REF:.3f} m",
        )

        ax_system.text(
            target_pos,
            0.36,
            "Target",
            ha="center",
            color=COLOR_TARGET,
            fontsize=12,
            fontweight="bold",
        )

        # Peak raggiunto finora
        peak_so_far = y[: idx + 1].max()
        peak_pos = VISUAL_OFFSET + peak_so_far

        ax_system.plot(
            [peak_pos, peak_pos],
            [-0.02, 0.41],
            color=COLOR_PEAK,
            linestyle=":",
            linewidth=2,
            alpha=0.75,
            label=f"Peak so far = {peak_so_far:.3f} m",
        )

        # Info box valori fisici veri
        info_text = (
            f"t = {t_current:.2f} s\n"
            f"x_b = {xb_current:+.4f} m\n"
            f"x_r = {xr_current:+.4f} m\n"
            f"y = x_b + x_r = {y_current:.4f} m\n"
            f"u = x_rd = {u_current:.4f} m\n"
            f"error = {err_current:+.2f} mm"
        )

        ax_system.text(
            0.02,
            0.95,
            info_text,
            transform=ax_system.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.82),
            family="monospace",
        )

        ax_system.text(
            0.02,
            0.10,
            "MPC input u is x_rd(t): desired relative robot position.\n"
            "Unlike the dataset stage, here x_rd is computed online, not a chirp.",
            transform=ax_system.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        )

        ax_system.legend(loc="upper right", fontsize=11)
        ax_system.set_title(
            "MPC Closed-Loop Control over Compliant Base",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        # ---------------------------------------------------------------------
        # PLOT INFERIORE: y(t) e u(t)
        # ---------------------------------------------------------------------
        ax_plot.clear()

        t_plot = t[: idx + 1]
        y_plot = y[: idx + 1]
        u_plot = u[: idx + 1]

        ax_plot.plot(
            t_plot,
            y_plot,
            color="blue",
            linewidth=2.5,
            label="Outreach y(t)",
        )

        ax_plot.axhline(
            Y_REF,
            color=COLOR_TARGET,
            linestyle="--",
            linewidth=2,
            label="Target",
        )

        ax_plot.plot(
            t_current,
            y_current,
            "ro",
            markersize=7,
            label="Current y" if frame_idx == 0 else None,
        )

        ax_plot.set_xlim(0, t.max())
        ax_plot.set_ylim(min(y.min(), Y_REF) - 0.015, max(y.max(), Y_REF) + 0.015)
        ax_plot.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
        ax_plot.set_ylabel("Outreach y [m]", fontsize=11, fontweight="bold", color="blue")
        ax_plot.tick_params(axis="y", labelcolor="blue")
        ax_plot.grid(True, alpha=0.3)

        # Secondo asse per u(t)
        ax_u = ax_plot.twinx()

        ax_u.plot(
            t_plot,
            u_plot,
            color="green",
            linewidth=2,
            alpha=0.85,
            label="MPC input u(t) = x_rd(t)",
        )

        ax_u.axhline(
            U_MIN,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
            label="u bounds",
        )

        ax_u.axhline(
            U_MAX,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
        )

        ax_u.set_ylim(U_MIN - 0.04, U_MAX + 0.04)
        ax_u.set_ylabel("Control input u [m]", fontsize=11, fontweight="bold", color="green")
        ax_u.tick_params(axis="y", labelcolor="green")

        # Legenda combinata
        lines1, labels1 = ax_plot.get_legend_handles_labels()
        lines2, labels2 = ax_u.get_legend_handles_labels()
        ax_plot.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

        ax_plot.set_title(
            "Closed-loop tracking and MPC control input",
            fontsize=12,
            fontweight="bold",
        )

        return []

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )

    ensure_dirs()

    print("  Salvataggio GIF...")
    writer = PillowWriter(fps=FPS)
    anim.save(OUTPUT_GIF, writer=writer, dpi=DPI)

    plt.close(fig)

    print(f"✓ Animazione salvata: {OUTPUT_GIF}")
    print(f"  Dimensione: {OUTPUT_GIF.stat().st_size / 1024:.1f} KB")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("MPC ANIMATION - VISUALIZATION STYLE")
    print("=" * 70)

    df = load_mpc_data()
    create_animation(df)

    print("\n" + "=" * 70)
    print("MPC ANIMATION COMPLETE")
    print("=" * 70)
    print(f"Output:")
    print(f"  {OUTPUT_GIF}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()