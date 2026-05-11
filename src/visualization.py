"""
visualization.py
================
Visualizzazioni avanzate per analisi interpretativa del sistema.

Moduli:
1. Animazione 2D del sistema fisico
2. GP response surface (heatmap 2D)
3. Sensitivity analysis (one-at-a-time)
4. Monte Carlo robustness test

Author: Matteo Casazza
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import time
from tqdm import tqdm

# Import moduli progetto
from dynamics import simulate_system
from dataset import load_dataset, ParameterRanges
from models import load_model


# ============================================================================
# A) ANIMAZIONE 2D SISTEMA FISICO
# ============================================================================

def draw_zigzag_spring(ax, x_start, x_end, y=0.2, amplitude=0.04, n_coils=8,
                       color='green', linewidth=2.5, zorder=2):
    """
    Disegna una molla come spezzata zig-zag tra x_start e x_end.
    """
    if x_end <= x_start:
        return

    margin = 0.02
    xs = np.linspace(x_start + margin, x_end - margin, 2 * n_coils + 1)
    ys = np.full_like(xs, y)

    for i in range(1, len(xs) - 1):
        ys[i] = y + amplitude * (1 if i % 2 else -1)

    ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)

class SystemAnimator:
    """
    Animatore per visualizzazione 2D del sistema massa-molla-smorzatore.
    
    Schema:
    [Parete] ~~~ [Massa Base] ~~~ [Massa Robot] --> Target
    """
    
    def __init__(
        self,
        params: np.ndarray,
        y_target: float,
        T_sim: float = 60.0,
        dt: float = 0.001
    ):
        """
        Parameters
        ----------
        params : array (9,)
            Parametri sistema completi
        y_target : float
            Target outreach [m]
        T_sim : float
            Tempo simulazione [s]
        dt : float
            Step temporale [s]
        """
        self.params = params
        self.y_target = y_target
        self.T_sim = T_sim
        self.dt = dt
        
        # Simula sistema
        print(f"Simulazione per animazione (T={T_sim}s, dt={dt}s)...")
        self.peak_y, self.sol = simulate_system(
            params, T_sim=T_sim, dt=dt, return_full=True
        )
        
        # Estrai traiettorie
        self.t = self.sol.t
        self.x_b = self.sol.y[2]  # Posizione base
        self.x_r = self.sol.y[3]  # Posizione robot
        self.y = self.x_b + self.x_r  # Outreach totale
        
        print(f"✓ Simulazione completata: {len(self.t)} timesteps")
        print(f"  Peak achieved: {self.peak_y:.6f} m")
    
    def create_animation(
        self,
        save_path: str = 'figures/animation.gif',
        fps: int = 30,
        duration: float = 10.0,
        figsize: Tuple[float, float] = (22, 8)
    ) -> None:
        """
        Crea animazione GIF del sistema.
        
        Parameters
        ----------
        save_path : str
            Path output GIF
        fps : int
            Frame per secondo
        duration : float
            Durata animazione [s] (se < T_sim, accelera)
        figsize : tuple
            Dimensioni figura
        """
        print(f"\nCreazione animazione...")
        print(f"  FPS: {fps}")
        print(f"  Durata: {duration}s")
        
        # Calcola sottocampionamento
        total_frames = int(fps * duration)
        frame_indices = np.linspace(0, len(self.t) - 1, total_frames, dtype=int)
        
        print(f"  Frame totali: {total_frames} (da {len(self.t)} timesteps)")
        
        # Setup figura
        fig, (ax_system, ax_plot) = plt.subplots(
            2, 1, figsize=figsize, 
            gridspec_kw={'height_ratios': [1, 1]}
        )
        
        # Limiti assi sistema
        y_max = max(self.y_target * 1.2, self.y.max() * 1.2)
        
        # Colori
        color_base = '#9467bd'  # Viola
        color_robot = '#d62728'  # Rosso
        color_target = '#27ae60'  # Verde
        
        def init():
            """Inizializza plot."""
            ax_system.clear()
            ax_plot.clear()
            return []
        
        def update(frame_idx):
            """Aggiorna frame."""
            idx = frame_indices[frame_idx]
            t_current = self.t[idx]
            x_b_current = self.x_b[idx]
            x_r_current = self.x_r[idx]
            y_current = self.y[idx]
            
            # ========== SISTEMA FISICO ==========
            ax_system.clear()
            ax_system.set_xlim(-0.3, y_max + 0.2)
            ax_system.set_ylim(0.0, 0.4)
            # ax_system.set_aspect('equal', adjustable='box')
            ax_system.set_aspect('auto')
            ax_system.axis('on')
            ax_system.set_yticks([])
            ax_system.set_ylabel('')
            
            # Parete verticale (barra nera)
            wall_x = 0

            wall_rect = patches.Rectangle(
                (-0.0025, -0.05),   # x, y
                0.005,              # larghezza
                0.50,              # altezza
                facecolor='black',
                edgecolor='black',
                zorder=0
            )
            ax_system.add_patch(wall_rect)

            # Tratteggio parete
            #for y_w in np.linspace(-0.12, 0.32, 9):
            #    ax_system.plot(
            #        [-0.015, -0.055],
            #        [y_w, y_w - 0.02],
            #        color='black',
            #        linewidth=1
            #    )
            
            # Posizioni e dimensioni masse
            base_pos = x_b_current
            base_size = 0.04

            robot_pos = base_pos + x_r_current
            robot_size = 0.04

            # Molla base (da parete a massa base)
            draw_zigzag_spring(
                ax_system,
                wall_x,
                base_pos - base_size / 2,
                y=0.2,
                amplitude=0.035,
                n_coils=5,
                color='orange',
                linewidth=2.5,
            )

            # Massa base (quadrato viola)
            base_rect = patches.Rectangle(
                (base_pos - base_size/2, 0.2 - base_size/2),
                base_size, base_size,
                linewidth=2,
                edgecolor='black',
                facecolor=color_base,
                alpha=0.8,
                zorder=5
            )
            ax_system.add_patch(base_rect)
            ax_system.text(
                base_pos, 0.06,
                'Base',
                ha='center',
                fontsize=13,
                fontweight='bold'
            )

            # Molla robot (da base a robot)
            draw_zigzag_spring(
                ax_system,
                base_pos + base_size / 2,
                robot_pos - robot_size / 2,
                y=0.2,
                amplitude=0.045,
                n_coils=8,
                color='forestgreen',
                linewidth=2.5,
            )

            # Massa robot (quadrato rosso)
            robot_rect = patches.Rectangle(
                (robot_pos - robot_size/2, 0.2 - robot_size/2),
                robot_size, robot_size,
                linewidth=2,
                edgecolor='black',
                facecolor=color_robot,
                alpha=0.8,
                zorder=5
            )
            ax_system.add_patch(robot_rect)
            ax_system.text(
                robot_pos, 0.06,
                'Robot',
                ha='center',
                fontsize=13,
                fontweight='bold'
            )
            
            # Linea outreach corrente
            ax_system.plot([y_current, y_current], [-0.1, 0.5],
                          'b--', linewidth=2, alpha=0.6, label='Current y')
            
            # Linea target
            ax_system.plot([self.y_target, self.y_target], [-0.1, 0.5],
                          color_target, linestyle='--', linewidth=2.5,
                          label=f'Target = {self.y_target:.3f} m')
            
            # Linea peak raggiunto
            peak_so_far = self.y[:idx+1].max()
            ax_system.plot([peak_so_far, peak_so_far], [-0.08, 0.48],
                          'orange', linestyle=':', linewidth=2,
                          alpha=0.7, label=f'Peak = {peak_so_far:.3f} m')
            
            # Info text
            error_current = abs(y_current - self.y_target)
            info_text = (
                f"t = {t_current:.2f} s\n"
                f"x_b = {x_b_current:.4f} m\n"
                f"x_r = {x_r_current:.4f} m\n"
                f"y = {y_current:.4f} m\n"
                f"Error = {error_current*1000:.2f} mm"
            )
            ax_system.text(
                0.02, 0.95, info_text,
                transform=ax_system.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace'
            )
            
            ax_system.legend(loc='upper right', fontsize=12)
            ax_system.set_title(
                f'Robot System over Compliant Base - Target {self.y_target:.3f} m',
                fontsize=12, fontweight='bold', pad=15
            )
            
            # ========== PLOT TRAIETTORIA ==========
            ax_plot.clear()
            
            # Plot fino a frame corrente
            t_plot = self.t[:idx+1]
            y_plot = self.y[:idx+1]
            
            ax_plot.plot(t_plot, y_plot, 'b-', linewidth=2, label='Outreach y(t)')
            ax_plot.axhline(self.y_target, color=color_target, 
                           linestyle='--', linewidth=2, label='Target')
            ax_plot.axhline(self.peak_y, color='orange',
                           linestyle=':', linewidth=2, label='Peak final')
            
            # Punto corrente
            ax_plot.plot(t_current, y_current, 'ro', markersize=8)
            
            ax_plot.set_xlim(0, self.T_sim)
            ax_plot.set_ylim(0.2, y_max)
            ax_plot.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
            ax_plot.set_ylabel('Outreach [m]', fontsize=11, fontweight='bold')
            ax_plot.grid(True, alpha=0.3)
            ax_plot.legend(loc='lower right', fontsize=12)
            
            return []
        
        # Crea animazione
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=total_frames, interval=1000/fps,
            blit=False, repeat=True
        )
        
        # Salva GIF
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  Salvataggio GIF (può richiedere 1-2 minuti)...")
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        
        plt.close(fig)
        print(f"✓ Animazione salvata: {save_path}")
        print(f"  Dimensione: {Path(save_path).stat().st_size / 1024:.1f} KB")


def animate_optimized_response(
    params: np.ndarray,
    y_target: float,
    save_path: str = "figures/animation_target3.gif",
    T_sim: float = 60.0,
    dt: float = 0.001,
    fps: int = 30,
    duration: float = 10.0
) -> None:
    """
    Wrapper per creare animazione di una risposta ottimizzata.
    
    Parameters
    ----------
    params : array (9,)
        Parametri sistema completi
    y_target : float
        Target outreach [m]
    save_path : str
        Path output
    T_sim : float
        Tempo simulazione [s]
    dt : float
        Step temporale [s]
    fps : int
        Frame per secondo
    duration : float
        Durata animazione desiderata [s]
    """
    animator = SystemAnimator(params, y_target, T_sim, dt)
    animator.create_animation(save_path, fps, duration)


# ============================================================================
# B) GP RESPONSE SURFACE
# ============================================================================

def plot_gp_response_surface(
    gp,
    scaler_X,
    scaler_y,
    X_data: np.ndarray,
    y_data: np.ndarray,
    fixed_values: Dict[str, float],
    x_param: str = "Kr",
    y_param: str = "hr",
    optimal_point: Optional[Dict[str, float]] = None,
    resolution: int = 50,
    save_path: str = "figures/gp_surface_Kr_hr.png"
) -> None:
    """
    Plot response surface del GP su piano 2D.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Modello GP
    scaler_X, scaler_y : StandardScaler
        Scalers
    X_data : array (n, 9)
        Dataset input (per scatter)
    y_data : array (n,)
        Dataset output
    fixed_values : dict
        Valori fissi per altri parametri
    x_param, y_param : str
        Parametri da variare (assi)
    optimal_point : dict, optional
        Punto ottimale da evidenziare
    resolution : int
        Risoluzione griglia
    save_path : str
        Path output
    """
    print(f"\nGenerazione GP response surface ({x_param} vs {y_param})...")
    
    # Ordine parametri
    param_order = ParameterRanges.get_param_names()
    x_idx = param_order.index(x_param)
    y_idx = param_order.index(y_param)
    
    # Range parametri
    param_ranges = ParameterRanges()
    lb, ub = param_ranges.get_bounds()
    
    x_range = np.linspace(lb[x_idx], ub[x_idx], resolution)
    y_range = np.linspace(lb[y_idx], ub[y_idx], resolution)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Costruisci griglia parametri completi
    grid_points = []
    for i in range(resolution):
        for j in range(resolution):
            point = np.zeros(9)
            # Riempi con valori fissi
            for k, name in enumerate(param_order):
                if name == x_param:
                    point[k] = X_grid[i, j]
                elif name == y_param:
                    point[k] = Y_grid[i, j]
                else:
                    point[k] = fixed_values[name]
            grid_points.append(point)
    
    grid_points = np.array(grid_points)
    
    # Predici con GP
    print(f"  Predizione su {len(grid_points)} punti...")
    X_grid_scaled = scaler_X.transform(grid_points)
    y_grid_scaled = gp.predict(X_grid_scaled)
    y_grid = scaler_y.inverse_transform(y_grid_scaled.reshape(-1, 1)).ravel()
    Z_grid = y_grid.reshape(resolution, resolution)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    contour = ax.contourf(
        X_grid, Y_grid, Z_grid,
        levels=20, cmap='viridis', alpha=0.9
    )
    
    # Contour lines
    contour_lines = ax.contour(
        X_grid, Y_grid, Z_grid,
        levels=10, colors='white', alpha=0.4, linewidths=0.5
    )
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Predicted Peak Outreach [m]', 
                   fontsize=12, fontweight='bold')
    
    # Scatter dataset
    ax.scatter(
        X_data[:, x_idx], X_data[:, y_idx],
        c=y_data, cmap='viridis', s=30,
        edgecolors='white', linewidth=0.5,
        alpha=0.6, label='Training data'
    )
    
    # Punto ottimale
    if optimal_point is not None:
        ax.scatter(
            optimal_point[x_param], optimal_point[y_param],
            c='red', s=200, marker='*',
            edgecolors='white', linewidth=2,
            label=f'Optimal (y={optimal_point["y_target"]:.3f}m)',
            zorder=10
        )
    
    ax.set_xlabel(x_param, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_param, fontsize=13, fontweight='bold')
    ax.set_title(
        f'GP Response Surface: {x_param} vs {y_param}',
        fontsize=15, fontweight='bold', pad=20
    )
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.2)
    
    # Annotazione parametri fissi
    fixed_text = "Fixed parameters:\n" + "\n".join(
        f"{k}={v:.3f}" for k, v in fixed_values.items()
        if k not in [x_param, y_param]
    )
    ax.text(
        0.02, 0.98, fixed_text,
        transform=ax.transAxes,
        fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        family='monospace'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Surface plot salvato: {save_path}")
    plt.close()


# ============================================================================
# C) SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis(
    gp,
    scaler_X,
    scaler_y,
    nominal_params: np.ndarray,
    params_to_vary: List[str] = ['hr', 'Mb', 'Kr', 'A', 'Kb'],
    n_points: int = 30,
    validate_points: int = 5,
    save_plot: str = "figures/sensitivity_analysis.png",
    save_csv: str = "results/sensitivity_analysis.csv"
) -> pd.DataFrame:
    """
    Sensitivity analysis one-at-a-time.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
    scaler_X, scaler_y : StandardScaler
    nominal_params : array (9,)
        Configurazione nominale
    params_to_vary : list
        Parametri da variare
    n_points : int
        Punti per parametro
    validate_points : int
        Quanti punti validare con simulazione
    save_plot : str
    save_csv : str
    
    Returns
    -------
    results_df : DataFrame
    """
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS (One-at-a-time)")
    print(f"{'='*70}")
    print(f"Configurazione nominale:")
    
    param_order = ParameterRanges.get_param_names()
    for i, name in enumerate(param_order):
        print(f"  {name:12s} = {nominal_params[i]:.6f}")
    
    # Predizione nominale
    X_nom_scaled = scaler_X.transform([nominal_params])
    y_nom_scaled = gp.predict(X_nom_scaled)
    y_nom = scaler_y.inverse_transform([[y_nom_scaled[0]]])[0, 0]
    print(f"\nPeak nominale (GP): {y_nom:.6f} m")
    
    # Range parametri
    param_ranges = ParameterRanges()
    lb, ub = param_ranges.get_bounds()
    
    # Storage
    results = []
    
    # Setup plot
    n_params = len(params_to_vary)
    fig, axes = plt.subplots(
        (n_params + 1) // 2, 2,
        figsize=(16, 4 * ((n_params + 1) // 2))
    )
    axes = axes.ravel() if n_params > 1 else [axes]
    
    for param_idx, param_name in enumerate(params_to_vary):
        print(f"\nVariazione parametro: {param_name}")
        
        # Indice nel vettore
        idx_in_vec = param_order.index(param_name)
        
        # Range
        param_values = np.linspace(lb[idx_in_vec], ub[idx_in_vec], n_points)
        y_pred_list = []
        
        # Varia parametro
        for val in param_values:
            params_varied = nominal_params.copy()
            params_varied[idx_in_vec] = val
            
            # Predici GP
            X_scaled = scaler_X.transform([params_varied])
            y_scaled = gp.predict(X_scaled)
            y_pred = scaler_y.inverse_transform([[y_scaled[0]]])[0, 0]
            y_pred_list.append(y_pred)
            
            results.append({
                'parameter': param_name,
                'value': val,
                'y_pred_gp': y_pred
            })
        
        y_pred_array = np.array(y_pred_list)
        
        # Validazione con simulazione (pochi punti)
        val_indices = np.linspace(0, n_points - 1, validate_points, dtype=int)
        y_sim_vals = []
        val_values = []
        
        for vi in val_indices:
            params_varied = nominal_params.copy()
            params_varied[idx_in_vec] = param_values[vi]
            
            y_sim = simulate_system(params_varied, T_sim=60, dt=0.001)
            y_sim_vals.append(y_sim)
            val_values.append(param_values[vi])
        
        # Plot
        ax = axes[param_idx]
        
        # GP prediction
        ax.plot(param_values, y_pred_array, 'b-', linewidth=2,
                label='GP Prediction')
        
        # Simulated validation points
        ax.scatter(val_values, y_sim_vals, c='red', s=100,
                  marker='o', edgecolors='black', linewidth=1.5,
                  label='Simulated', zorder=5)
        
        # Nominal point
        ax.axvline(nominal_params[idx_in_vec], color='green',
                   linestyle='--', linewidth=2, alpha=0.7,
                   label='Nominal')
        ax.axhline(y_nom, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Outreach [m]', fontsize=12)
        ax.set_title(f'Sensitivity to {param_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Rimuovi subplot extra
    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Sensitivity Analysis (One-at-a-Time)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300, bbox_inches='tight')
    print(f"\n✓ Sensitivity plot salvato: {save_plot}")
    plt.close()
    
    # Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv, index=False)
    print(f"✓ Sensitivity CSV salvato: {save_csv}")
    
    return results_df


# ============================================================================
# D) MONTE CARLO ROBUSTNESS
# ============================================================================

def monte_carlo_robustness(
    optimal_params: np.ndarray,
    y_target: float,
    n_samples: int = 100,
    noise_level: float = 0.05,
    params_to_perturb: List[str] = ['Kb', 'Mb', 'hb'],
    save_plot: str = "figures/monte_carlo_robustness.png",
    save_csv: str = "results/monte_carlo_robustness.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Monte Carlo robustness test con rumore sui parametri fissi.
    
    Parameters
    ----------
    optimal_params : array (9,)
        Parametri ottimali
    y_target : float
        Target desiderato
    n_samples : int
        Numero campioni Monte Carlo
    noise_level : float
        Livello rumore (es. 0.05 = ±5%)
    params_to_perturb : list
        Parametri da perturbare
    save_plot : str
    save_csv : str
    seed : int
    
    Returns
    -------
    results_df : DataFrame
    """
    print(f"\n{'='*70}")
    print("MONTE CARLO ROBUSTNESS TEST")
    print(f"{'='*70}")
    print(f"Parametri ottimali:")
    
    param_order = ParameterRanges.get_param_names()
    for i, name in enumerate(param_order):
        marker = " (*)" if name in params_to_perturb else ""
        print(f"  {name:12s} = {optimal_params[i]:.6f}{marker}")
    
    print(f"\nTarget: {y_target:.6f} m")
    print(f"Noise level: ±{noise_level*100:.1f}%")
    print(f"Samples: {n_samples}")
    print(f"Parametri perturbati: {params_to_perturb}")
    
    np.random.seed(seed)
    
    # Indici parametri da perturbare
    perturb_indices = [param_order.index(p) for p in params_to_perturb]
    
    # Storage
    y_achieved = []
    
    print(f"\nSimulazione {n_samples} campioni Monte Carlo...")
    for i in tqdm(range(n_samples), desc="Monte Carlo"):
        # Perturba parametri
        params_perturbed = optimal_params.copy()
        for idx in perturb_indices:
            noise = np.random.normal(-noise_level, noise_level)
            params_perturbed[idx] *= (1 + noise)
        
        # Simula
        y_sim = simulate_system(params_perturbed, T_sim=60, dt=0.001)
        y_achieved.append(y_sim)
    
    y_achieved = np.array(y_achieved)
    errors = np.abs(y_achieved - y_target)
    
    # Statistiche
    stats = {
        'mean': y_achieved.mean(),
        'std': y_achieved.std(),
        'min': y_achieved.min(),
        'max': y_achieved.max(),
        'q05': np.percentile(y_achieved, 5),
        'q95': np.percentile(y_achieved, 95),
        'mean_error': errors.mean(),
        'max_error': errors.max(),
        'success_rate': np.sum(errors < 0.01) / n_samples  # <10mm = success
    }
    
    print(f"\n{'='*70}")
    print("RISULTATI MONTE CARLO")
    print(f"{'='*70}")
    print(f"Target:               {y_target:.6f} m")
    print(f"Mean achieved:        {stats['mean']:.6f} m")
    print(f"Std:                  {stats['std']:.6f} m")
    print(f"Range:                [{stats['min']:.6f}, {stats['max']:.6f}] m")
    print(f"5th-95th percentile:  [{stats['q05']:.6f}, {stats['q95']:.6f}] m")
    print(f"Mean error:           {stats['mean_error']*1000:.3f} mm")
    print(f"Max error:            {stats['max_error']*1000:.3f} mm")
    print(f"Success rate (<10mm): {stats['success_rate']*100:.1f}%")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== Histogram ==========
    ax = axes[0, 0]
    ax.hist(y_achieved, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(y_target, color='red', linestyle='--', linewidth=2.5,
               label=f'Target = {y_target:.3f} m')
    ax.axvline(stats['mean'], color='green', linestyle='-', linewidth=2,
               label=f'Mean = {stats["mean"]:.3f} m')
    ax.axvline(stats['q05'], color='orange', linestyle=':', linewidth=1.5,
               label='5th-95th pct')
    ax.axvline(stats['q95'], color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Achieved Peak Outreach [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Achieved Outreach', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== Error histogram ==========
    ax = axes[0, 1]
    ax.hist(errors * 1000, bins=30, edgecolor='black', alpha=0.7, color='salmon')
    ax.axvline(stats['mean_error'] * 1000, color='red', linestyle='--',
               linewidth=2, label=f'Mean = {stats["mean_error"]*1000:.2f} mm')
    ax.set_xlabel('Absolute Error [mm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Errors', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== Time series ==========
    ax = axes[1, 0]
    ax.plot(y_achieved, 'b-', alpha=0.5, linewidth=1)
    ax.axhline(y_target, color='red', linestyle='--', linewidth=2,
               label='Target')
    ax.fill_between(range(n_samples),
                     y_target - 0.01, y_target + 0.01,
                     alpha=0.2, color='green', label='±10mm tolerance')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Achieved Outreach [m]', fontsize=12)
    ax.set_title('Monte Carlo Samples', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ========== Box plot ==========
    ax = axes[1, 1]
    box = ax.boxplot([y_achieved], vert=True, patch_artist=True,
                     tick_labels=['Robustness Test'])
    box['boxes'][0].set_facecolor('lightblue')
    ax.axhline(y_target, color='red', linestyle='--', linewidth=2,
               label='Target')
    ax.set_ylabel('Achieved Outreach [m]', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Summary', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotazioni statistiche
    stats_text = (
        f"μ = {stats['mean']:.4f} m\n"
        f"σ = {stats['std']:.4f} m\n"
        f"Range = [{stats['min']:.4f}, {stats['max']:.4f}]\n"
        f"Error: {stats['mean_error']*1000:.2f} ± "
        f"{stats['std']*1000:.2f} mm"
    )
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.suptitle(
        f'Monte Carlo Robustness Test (n={n_samples}, noise=±{noise_level*100:.0f}%)',
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300, bbox_inches='tight')
    print(f"\n✓ Monte Carlo plot salvato: {save_plot}")
    plt.close()
    
    # Save CSV
    results_df = pd.DataFrame({
        'sample_index': range(n_samples),
        'y_achieved': y_achieved,
        'error': errors
    })
    results_df.to_csv(save_csv, index=False)
    print(f"✓ Monte Carlo CSV salvato: {save_csv}")
    
    # Summary CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(save_csv.replace('.csv', '_stats.csv'), index=False)
    print(f"✓ Statistics salvate: {save_csv.replace('.csv', '_stats.csv')}")
    
    return results_df


# ============================================================================
# MAIN: Pipeline completa visualizzazioni
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED VISUALIZATIONS - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # ========== Setup ==========
    print("Caricamento modelli e dati...")
    
    # Carica GP
    gp, scaler_X, scaler_y = load_model('results')
    
    # Carica dataset
    X_data, y_data = load_dataset('data/dataset_outreach.csv')
    
    # Carica risultati inverse optimization
    inverse_results = pd.read_csv('results/inverse_results.csv')
    print(f"\n✓ Caricati {len(inverse_results)} risultati inverse optimization")
    
    # Seleziona target centrale
    middle_idx = len(inverse_results) // 2
    target_row = inverse_results.iloc[middle_idx]
    
    y_target = target_row['y_target']
    param_cols = [c for c in inverse_results.columns if c.startswith('param_')]
    optimal_params = target_row[param_cols].values.astype(float)
    
    print(f"\nTarget selezionato per visualizzazioni:")
    print(f"  Index: {middle_idx}")
    print(f"  y_target: {y_target:.6f} m")
    print(f"  y_sim achieved: {target_row['y_sim']:.6f} m")
    print(f"  Error: {target_row['error_sim']*1000:.3f} mm")
    
    # ========== A) ANIMAZIONE ==========
    print(f"\n{'='*70}")
    print("A) GENERAZIONE ANIMAZIONE")
    print(f"{'='*70}")
    
    animate_optimized_response(
        params=optimal_params,
        y_target=y_target,
        save_path=f"figures/animation_target{middle_idx+1}.gif",
        T_sim=60.0,
        dt=0.001,
        fps= 30,     # 30 per prova finale
        duration=20.0  # 10 secondi di animazione
    )
    
    # ========== B) GP SURFACE ==========
    print(f"\n{'='*70}")
    print("B) GP RESPONSE SURFACE")
    print(f"{'='*70}")
    
    # Parametri fissi (mediane dataset)
    param_order = ParameterRanges.get_param_names()
    fixed_values = {}
    for i, name in enumerate(param_order):
        if name not in ['Kr', 'hr']:
            fixed_values[name] = optimal_params[i]
    
    # Punto ottimale
    optimal_point = {
        'Kr': optimal_params[param_order.index('Kr')],
        'hr': optimal_params[param_order.index('hr')],
        'y_target': y_target
    }
    
    plot_gp_response_surface(
        gp, scaler_X, scaler_y,
        X_data, y_data,
        fixed_values=fixed_values,
        x_param='Kr',
        y_param='hr',
        optimal_point=optimal_point,
        resolution=50,
        save_path='figures/gp_surface_Kr_hr.png'
    )
    
    # ========== C) SENSITIVITY ANALYSIS ==========
    print(f"\n{'='*70}")
    print("C) SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    
    sensitivity_df = sensitivity_analysis(
        gp, scaler_X, scaler_y,
        nominal_params=optimal_params,
        params_to_vary=['hr', 'Mb', 'Kr', 'A', 'Kb'],
        n_points=30,
        validate_points=5,
        save_plot='figures/sensitivity_analysis.png',
        save_csv='results/sensitivity_analysis.csv'
    )
    
    # ========== D) MONTE CARLO ROBUSTNESS ==========
    print(f"\n{'='*70}")
    print("D) MONTE CARLO ROBUSTNESS")
    print(f"{'='*70}")
    
    monte_carlo_df = monte_carlo_robustness(
        optimal_params=optimal_params,
        y_target=y_target,
        n_samples=100,   # 100 per finale
        noise_level=0.05,  # ±5%
        params_to_perturb=['Kb', 'Mb', 'hb'],
        save_plot='figures/monte_carlo_robustness.png',
        save_csv='results/monte_carlo_robustness.csv',
        seed=42
    )
    
    # ========== SUMMARY FINALE ==========
    print(f"\n{'='*70}")
    print("VISUALIZZAZIONI COMPLETATE!")
    print(f"{'='*70}")
    
    print(f"\n📊 FILE GENERATI:")
    print(f"\n  Animazioni:")
    print(f"    - figures/animation_target{middle_idx+1}.gif")
    
    print(f"\n  Surface Plots:")
    print(f"    - figures/gp_surface_Kr_hr.png")
    
    print(f"\n  Analysis Plots:")
    print(f"    - figures/sensitivity_analysis.png")
    print(f"    - figures/monte_carlo_robustness.png")
    
    print(f"\n  Data CSV:")
    print(f"    - results/sensitivity_analysis.csv")
    print(f"    - results/monte_carlo_robustness.csv")
    print(f"    - results/monte_carlo_robustness_stats.csv")
    
    print(f"\n✅ PROGETTO COMPLETO CON VISUALIZZAZIONI AVANZATE!")
    print(f"   Pronto per presentazione e relazione.")
    print("="*70 + "\n")