"""
dynamics.py
===========
Sistema a 2 gradi di libertà: robot controllabile su base compliante.

Coordinate:
- x_b: posizione base passiva
- x_r: posizione robot (relativa alla base)
- y = x_b + x_r: outreach totale

Input:
- x_rd(t): posizione comandata del robot (chirp)

Reference:
Roveda et al. (2016) - Mechatronics 39
Modello semplificato senza ambiente esterno.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def chirp_signal(t, f0, f1, T, A, x_r_start):
    """
    Genera segnale chirp con frequenza linearmente crescente.
    
    Parameters
    ----------
    t : float or array
        Tempo [s]
    f0 : float
        Frequenza iniziale [Hz]
    f1 : float
        Frequenza finale [Hz]
    T : float
        Durata totale [s]
    A : float
        Ampiezza [m]
    x_r_start : float
        Posizione iniziale [m]
    
    Returns
    -------
    x_rd : float or array
        Posizione comandata [m]
    """
    if T <= 0:
        return x_r_start * np.ones_like(t)
    
    k = (f1 - f0) / T
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return x_r_start + A * np.cos(phase)


def system_dynamics(t, state, params):
    """
    Equazioni del sistema a 2 DoF.
    
    State: [dx_b, dx_r, x_b, x_r]
    
    Parameters
    ----------
    t : float
        Tempo attuale
    state : array (4,)
        [dx_b, dx_r, x_b, x_r]
    params : dict
        Parametri fisici e di controllo
    
    Returns
    -------
    dstate : array (4,)
        Derivata dello stato
    """
    dx_b, dx_r, x_b, x_r = state
    
    # Estrai parametri
    Kb = params['Kb']
    Kr = params['Kr']
    Mb = params['Mb']
    hb = params['hb']
    hr = params['hr']
    f0 = params['f0']
    f1 = params['f1']
    A = params['A']
    x_r_start = params['x_r_start']
    Mr = params.get('Mr', 10.0)  # Default 10 kg
    T = params.get('T', 60.0)    # Default 60 s
    
    # Calcola damping da damping ratio
    Db = 2 * hb * np.sqrt(Kb * Mb)
    Dr = 2 * hr * np.sqrt(Kr * Mr)
    
    # Posizione comandata (chirp)
    x_rd = chirp_signal(t, f0, f1, T, A, x_r_start)
    
    # Equazioni dinamiche (da paper Roveda et al.)
    ddx_b = (Dr * dx_r + Kr * (x_r - x_rd) - Db * dx_b - Kb * x_b) / Mb
    ddx_r = (-Dr * dx_r - Kr * (x_r - x_rd)) / Mr - ddx_b
    
    return [ddx_b, ddx_r, dx_b, dx_r]


def simulate_system(params, T_sim=60.0, dt=0.001, return_full=False):
    """
    Simula il sistema e ritorna il massimo outreach.
    
    Parameters
    ----------
    params : dict or array
        Se dict: {'Kb': ..., 'Kr': ..., ecc.}
        Se array: [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
    T_sim : float
        Tempo simulazione [s]
    dt : float
        Step temporale [s]
    return_full : bool
        Se True, ritorna anche la soluzione completa
    
    Returns
    -------
    peak_y : float
        Massimo outreach [m]
    sol : OdeResult (opzionale)
        Soluzione completa se return_full=True
    """
    # Converte array in dict se necessario
    if isinstance(params, (list, np.ndarray)):
        param_names = ['Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start']
        params = dict(zip(param_names, params))
    
    # Aggiungi parametri di default
    params['T'] = T_sim
    if 'Mr' not in params:
        params['Mr'] = 10.0
    
    # Stato iniziale: [dx_b, dx_r, x_b, x_r]
    x0 = [0, 0, 0, params['x_r_start']]
    
    # Integrazione
    t_eval = np.arange(0, T_sim, dt)
    sol = solve_ivp(
        system_dynamics,
        t_span=[0, T_sim],
        y0=x0,
        t_eval=t_eval,
        args=(params,),
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Calcola outreach: y = x_b + x_r
    x_b = sol.y[2]
    x_r = sol.y[3]
    y = x_b + x_r
    
    peak_y = np.max(y)
    
    if return_full:
        return peak_y, sol
    return peak_y


def plot_simulation_example(params, T_sim=60.0, dt=0.001, save_path=None):
    """
    Plot completo di una simulazione per validazione.
    
    Parameters
    ----------
    params : dict
        Parametri sistema
    T_sim : float
        Tempo simulazione [s]
    dt : float
        Step temporale [s]
    save_path : str, optional
        Path per salvare figura
    """
    peak_y, sol = simulate_system(params, T_sim, dt, return_full=True)
    
    t = sol.t
    dx_b, dx_r, x_b, x_r = sol.y
    y = x_b + x_r
    
    # Ricostruisci chirp
    x_rd = chirp_signal(
        t, 
        params['f0'], 
        params['f1'], 
        T_sim, 
        params['A'], 
        params['x_r_start']
    )
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Chirp command
    axes[0].plot(t, x_rd, 'b-', linewidth=1.5, label='$x_{rd}$ (comando)')
    axes[0].set_ylabel('Posizione [m]')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Segnale di eccitazione (chirp)', fontsize=11)
    
    # Base position
    axes[1].plot(t, x_b, 'purple', linewidth=1.5, label='$x_b$ (base passiva)')
    axes[1].set_ylabel('Posizione [m]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Robot position (relative)
    axes[2].plot(t, x_r, 'red', linewidth=1.5, label='$x_r$ (robot)')
    axes[2].set_ylabel('Posizione [m]')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Total outreach
    axes[3].plot(t, y, 'k-', linewidth=2, label='$y = x_b + x_r$ (outreach)')
    axes[3].axhline(peak_y, color='red', linestyle='--', linewidth=1.5, 
                    label=f'Peak = {peak_y:.3f} m')
    axes[3].set_xlabel('Tempo [s]')
    axes[3].set_ylabel('Outreach [m]')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title(f'Outreach totale (max = {peak_y:.3f} m)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura salvata: {save_path}")
    
    plt.show()
    
    return peak_y


# ============================================================================
# TEST: Esegui questo file per verificare che tutto funzioni
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST: src/dynamics.py")
    print("="*70)
    
    # Parametri di test
    test_params = {
        'Kb': 2000,
        'Kr': 1500,
        'Mb': 50,
        'hb': 0.2,
        'hr': 0.5,
        'f0': 0.001,
        'f1': 5,
        'A': 0.15,
        'x_r_start': 0.4,
        'Mr': 10
    }
    
    print("\nParametri di test:")
    for key, val in test_params.items():
        print(f"  {key:12s} = {val}")
    
    print("\nEsecuzione simulazione...")
    peak_y = simulate_system(test_params, T_sim=60, dt=0.001)
    print(f"\n✓ Simulazione completata!")
    print(f"  Peak outreach: {peak_y:.4f} m")
    
    print("\nGenerazione plot...")
    plot_simulation_example(test_params, save_path='figures/test_simulation.png')
    
    print("\n" + "="*70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("="*70)