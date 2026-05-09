"""
optimization.py
===============
Problema inverso: dato y_target, trova parametri controllabili ottimali.

Workflow:
1. Carica GP e scalers
2. Definisce parametri fissi (Kb, Mb, hb, Mr)
3. Ottimizza parametri controllabili (Kr, hr, f0, f1, A, x_r_start)
4. Valida con simulazione dinamica
5. Genera plot e tabella risultati

Author: [Il tuo nome]
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
import time
from dataclasses import dataclass

# Import moduli progetto
from models import load_model
from dynamics import simulate_system, plot_simulation_example
from dataset import load_dataset, ParameterRanges


# ============================================================================
# CONFIGURAZIONE PROBLEMA INVERSO
# ============================================================================

@dataclass
class InverseProblemConfig:
    """
    Configurazione problema inverso.
    
    Separa parametri FISSI (proprietà sistema identificate offline)
    da parametri CONTROLLABILI (da ottimizzare).
    """
    # Parametri FISSI (proprietà base + robot)
    Kb: float = 2000.0      # Stiffness base [N/m]
    Mb: float = 50.0        # Massa base [kg]
    hb: float = 0.2         # Damping ratio base [-]
    Mr: float = 10.0        # Massa robot [kg]
    
    # Bounds parametri CONTROLLABILI
    # Ordine: [Kr, hr, f0, f1, A, x_r_start]
    Kr_min: float = 100.0
    Kr_max: float = 5000.0
    
    hr_min: float = 0.1
    hr_max: float = 1.1
    
    f0_min: float = 0.0001
    f0_max: float = 0.01
    
    f1_min: float = 3.0
    f1_max: float = 10.0
    
    A_min: float = 0.1
    A_max: float = 0.2
    
    x_r_start_min: float = 0.3
    x_r_start_max: float = 0.5
    
    def get_fixed_params(self) -> Dict[str, float]:
        """Ritorna dizionario parametri fissi."""
        return {
            'Kb': self.Kb,
            'Mb': self.Mb,
            'hb': self.hb,
            'Mr': self.Mr
        }
    
    def get_controllable_bounds(self) -> List[Tuple[float, float]]:
        """
        Ritorna bounds parametri controllabili.
        
        Returns
        -------
        bounds : list of tuples
            [(Kr_min, Kr_max), (hr_min, hr_max), ...]
        """
        return [
            (self.Kr_min, self.Kr_max),      # Kr
            (self.hr_min, self.hr_max),      # hr
            (self.f0_min, self.f0_max),      # f0
            (self.f1_min, self.f1_max),      # f1
            (self.A_min, self.A_max),        # A
            (self.x_r_start_min, self.x_r_start_max)  # x_r_start
        ]
    
    @staticmethod
    def get_controllable_names() -> List[str]:
        """Nomi parametri controllabili in ordine."""
        return ['Kr', 'hr', 'f0', 'f1', 'A', 'x_r_start']
    
    @staticmethod
    def get_full_param_order() -> List[str]:
        """Ordine completo parametri (come nel dataset)."""
        return ['Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start']


def reconstruct_full_params(
    controllable_params: np.ndarray,
    fixed_params: Dict[str, float]
) -> np.ndarray:
    """
    Ricostruisce vettore completo parametri [9,] da controllabili + fissi.
    
    Parameters
    ----------
    controllable_params : array (6,)
        [Kr, hr, f0, f1, A, x_r_start]
    fixed_params : dict
        {'Kb': ..., 'Mb': ..., 'hb': ..., 'Mr': ...}
    
    Returns
    -------
    full_params : array (9,)
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
        
    Notes
    -----
    IMPORTANTE: L'ordine deve corrispondere esattamente a quello usato
    nel dataset e nel training del GP!
    """
    # Estrai controllabili
    Kr, hr, f0, f1, A, x_r_start = controllable_params
    
    # Ricostruisci vettore completo
    full_params = np.array([
        fixed_params['Kb'],    # 0
        Kr,                     # 1
        fixed_params['Mb'],    # 2
        fixed_params['hb'],    # 3
        hr,                     # 4
        f0,                     # 5
        f1,                     # 6
        A,                      # 7
        x_r_start              # 8
    ])
    
    return full_params


# ============================================================================
# OTTIMIZZAZIONE INVERSA
# ============================================================================

class InverseOptimizer:
    """
    Ottimizzatore per problema inverso usando GP come surrogate.
    """
    
    def __init__(
        self,
        gp,
        scaler_X,
        scaler_y,
        config: InverseProblemConfig
    ):
        """
        Parameters
        ----------
        gp : GaussianProcessRegressor
            Modello GP addestrato
        scaler_X : StandardScaler
            Scaler per input
        scaler_y : StandardScaler
            Scaler per output
        config : InverseProblemConfig
            Configurazione problema
        """
        self.gp = gp
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.config = config
        self.fixed_params = config.get_fixed_params()
        self.bounds = config.get_controllable_bounds()
        
    def predict_gp(self, full_params: np.ndarray) -> Tuple[float, float]:
        """
        Predice y con GP (con de-normalizzazione).
        
        Parameters
        ----------
        full_params : array (9,)
            Parametri completi
        
        Returns
        -------
        y_pred : float
            Predizione [m]
        y_std : float
            Incertezza [m]
        """
        # Normalizza input
        X_scaled = self.scaler_X.transform([full_params])
        
        # Predici (scaled)
        y_pred_scaled, y_std_scaled = self.gp.predict(X_scaled, return_std=True)
        
        # De-normalizza
        y_pred = self.scaler_y.inverse_transform([[y_pred_scaled[0]]])[0, 0]
        y_std = y_std_scaled[0] * self.scaler_y.scale_[0]
        
        return y_pred, y_std
    
    def objective_function(
        self,
        controllable_params: np.ndarray,
        y_target: float
    ) -> float:
        """
        Funzione obiettivo per differential_evolution.
        
        Parameters
        ----------
        controllable_params : array (6,)
            [Kr, hr, f0, f1, A, x_r_start]
        y_target : float
            Target outreach [m]
        
        Returns
        -------
        cost : float
            (y_pred - y_target)²
        """
        # Ricostruisci parametri completi
        full_params = reconstruct_full_params(
            controllable_params,
            self.fixed_params
        )
        
        # Predici con GP
        y_pred, _ = self.predict_gp(full_params)
        
        # Errore quadratico
        cost = (y_pred - y_target)**2
        
        return cost
    
    def optimize(
        self,
        y_target: float,
        maxiter: int = 1000,
        popsize: int = 15,
        seed: int = 42,
        verbose: bool = True
    ) -> Dict:
        """
        Ottimizza parametri controllabili per raggiungere y_target.
        
        Parameters
        ----------
        y_target : float
            Target outreach [m]
        maxiter : int
            Iterazioni max differential evolution
        popsize : int
            Population size
        seed : int
            Random seed
        verbose : bool
            Stampa progress
        
        Returns
        -------
        result : dict
            {
                'y_target': target richiesto,
                'controllable_opt': parametri ottimizzati [6],
                'full_params_opt': parametri completi [9],
                'y_gp_pred': predizione GP,
                'y_gp_std': incertezza GP,
                'optimization_time': tempo ottimizzazione [s],
                'success': bool
            }
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"OTTIMIZZAZIONE INVERSA: y_target = {y_target:.4f} m")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Ottimizzazione
        result_opt = differential_evolution(
            func=lambda x: self.objective_function(x, y_target),
            bounds=self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            workers= 1,  # Parallelizza con -1 ma su windows dice che non si può
            updating='immediate',     # 'deferred',
            polish=True,  # Refine locale
            atol=1e-6,
            tol=1e-6
        )
        
        elapsed = time.time() - start_time
        
        # Estrai risultato
        controllable_opt = result_opt.x
        full_params_opt = reconstruct_full_params(
            controllable_opt,
            self.fixed_params
        )
        
        # Predizione finale GP
        y_gp_pred, y_gp_std = self.predict_gp(full_params_opt)
        
        if verbose:
            print(f"\n✓ Ottimizzazione completata in {elapsed:.2f} s")
            print(f"  Target:      {y_target:.6f} m")
            print(f"  GP pred:     {y_gp_pred:.6f} m")
            print(f"  GP std:      {y_gp_std:.6f} m")
            print(f"  Errore GP:   {abs(y_gp_pred - y_target):.6f} m")
            print(f"  Success:     {result_opt.success}")
            
            print(f"\nParametri ottimizzati (controllabili):")
            for name, val in zip(self.config.get_controllable_names(), 
                                 controllable_opt):
                print(f"  {name:12s} = {val:.6f}")
        
        return {
            'y_target': y_target,
            'controllable_opt': controllable_opt,
            'full_params_opt': full_params_opt,
            'y_gp_pred': y_gp_pred,
            'y_gp_std': y_gp_std,
            'optimization_time': elapsed,
            'success': result_opt.success
        }


# ============================================================================
# VALIDAZIONE CON SIMULAZIONE
# ============================================================================

def validate_with_simulation(
    full_params: np.ndarray,
    y_target: float,
    y_gp_pred: float,
    T_sim: float = 60.0,
    dt: float = 0.001,
    verbose: bool = True
) -> Dict:
    """
    Valida parametri ottimizzati con simulazione dinamica vera.
    
    Parameters
    ----------
    full_params : array (9,)
        Parametri completi
    y_target : float
        Target richiesto [m]
    y_gp_pred : float
        Predizione GP [m]
    T_sim : float
        Tempo simulazione [s]
    dt : float
        Step temporale [s]
    verbose : bool
        Stampa risultati
    
    Returns
    -------
    validation : dict
        {
            'y_sim': outreach simulato,
            'error_gp': |y_gp - y_target|,
            'error_sim': |y_sim - y_target|,
            'error_gp_vs_sim': |y_gp - y_sim|
        }
    """
    if verbose:
        print(f"\n--- VALIDAZIONE SIMULAZIONE ---")
    
    # Simula sistema
    y_sim = simulate_system(full_params, T_sim=T_sim, dt=dt)
    
    # Errori
    error_gp = abs(y_gp_pred - y_target)
    error_sim = abs(y_sim - y_target)
    error_gp_vs_sim = abs(y_gp_pred - y_sim)
    
    if verbose:
        print(f"  Simulato:    {y_sim:.6f} m")
        print(f"  Errore sim:  {error_sim:.6f} m ({100*error_sim/y_target:.2f}%)")
        print(f"  GP vs sim:   {error_gp_vs_sim:.6f} m")
    
    return {
        'y_sim': y_sim,
        'error_gp': error_gp,
        'error_sim': error_sim,
        'error_gp_vs_sim': error_gp_vs_sim
    }


# ============================================================================
# SELEZIONE TARGET
# ============================================================================

def select_targets_from_dataset(
    y_data: np.ndarray,
    percentiles: List[float] = [20, 35, 50, 65, 80],
    verbose: bool = True
) -> np.ndarray:
    """
    Seleziona target realistici dai percentili del dataset.
    
    Parameters
    ----------
    y_data : array (n,)
        Valori peak_y dal dataset
    percentiles : list
        Percentili da usare
    verbose : bool
        Stampa target selezionati
    
    Returns
    -------
    targets : array (len(percentiles),)
        Target selezionati [m]
    """
    targets = np.percentile(y_data, percentiles)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TARGET SELEZIONATI (dai percentili dataset)")
        print(f"{'='*70}")
        print(f"Dataset: min={y_data.min():.3f}, max={y_data.max():.3f}, "
              f"mean={y_data.mean():.3f} m")
        print(f"\nTarget:")
        for p, t in zip(percentiles, targets):
            print(f"  {p:3.0f}° percentile: {t:.6f} m")
    
    return targets


# ============================================================================
# PIPELINE COMPLETA
# ============================================================================

def run_inverse_optimization_pipeline(
    targets: np.ndarray,
    config: Optional[InverseProblemConfig] = None,
    gp_path: str = 'results/gp_model.pkl',
    save_results: bool = True,
    plot_trajectories: bool = True,
    trajectory_indices: List[int] = [0, 2, 4]  # Quali target plottare
) -> pd.DataFrame:
    """
    Pipeline completa ottimizzazione inversa su target multipli.
    
    Parameters
    ----------
    targets : array (n,)
        Target da testare [m]
    config : InverseProblemConfig, optional
        Configurazione (default: usa valori standard)
    gp_path : str
        Path al modello GP
    save_results : bool
        Salva CSV e plot
    plot_trajectories : bool
        Plot traiettorie simulazioni
    trajectory_indices : list
        Indici target da plottare
    
    Returns
    -------
    results_df : DataFrame
        Tabella risultati completi
    """
    print("\n" + "="*70)
    print("INVERSE OPTIMIZATION PIPELINE")
    print("="*70)
    print(f"Numero target: {len(targets)}")
    print(f"Range target:  [{targets.min():.3f}, {targets.max():.3f}] m")
    
    # ========== Setup ==========
    if config is None:
        config = InverseProblemConfig()
    
    # Carica modello
    print(f"\nCaricamento modello da: {Path(gp_path).parent}/")
    gp, scaler_X, scaler_y = load_model(str(Path(gp_path).parent))
    
    # Crea ottimizzatore
    optimizer = InverseOptimizer(gp, scaler_X, scaler_y, config)
    
    print(f"\nParametri fissi:")
    for k, v in config.get_fixed_params().items():
        print(f"  {k:4s} = {v}")
    
    # ========== Ottimizzazione per ogni target ==========
    results = []
    all_solutions = []
    
    for i, y_target in enumerate(targets):
        print(f"\n{'-'*70}")
        print(f"TARGET {i+1}/{len(targets)}")
        print(f"{'-'*70}")
        
        # Ottimizza
        opt_result = optimizer.optimize(
            y_target=y_target,
            maxiter=100,
            popsize=10,
            seed=42 + i,  # Seed diverso per ogni target
            verbose=True
        )
        
        # Valida con simulazione
        validation = validate_with_simulation(
            full_params=opt_result['full_params_opt'],
            y_target=y_target,
            y_gp_pred=opt_result['y_gp_pred'],
            verbose=True
        )
        
        # Combina risultati
        result_entry = {
            'target_index': i,
            'y_target': y_target,
            'y_gp_pred': opt_result['y_gp_pred'],
            'y_gp_std': opt_result['y_gp_std'],
            'y_sim': validation['y_sim'],
            'error_gp': validation['error_gp'],
            'error_sim': validation['error_sim'],
            'error_gp_vs_sim': validation['error_gp_vs_sim'],
            'optimization_time': opt_result['optimization_time'],
            'success': opt_result['success']
        }
        
        # Aggiungi parametri ottimizzati
        param_names = config.get_full_param_order()
        for name, val in zip(param_names, opt_result['full_params_opt']):
            result_entry[f'param_{name}'] = val
        
        results.append(result_entry)
        all_solutions.append(opt_result)
    
    # ========== Crea DataFrame ==========
    results_df = pd.DataFrame(results)
    
    # ========== Stampa summary ==========
    print("\n" + "="*70)
    print("RISULTATI COMPLETI")
    print("="*70)
    print(f"\n{results_df[['y_target', 'y_gp_pred', 'y_sim', 'error_sim']].to_string(index=False)}")
    
    print(f"\n--- STATISTICHE ERRORI ---")
    print(f"Errore GP medio:   {results_df['error_gp'].mean():.6f} m")
    print(f"Errore sim medio:  {results_df['error_sim'].mean():.6f} m")
    print(f"Errore sim max:    {results_df['error_sim'].max():.6f} m")
    print(f"GP vs sim medio:   {results_df['error_gp_vs_sim'].mean():.6f} m")
    
    # ========== Salvataggio ==========
    if save_results:
        # CSV
        csv_path = 'results/inverse_results.csv'
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Risultati salvati: {csv_path}")
        
        # Plot comparativo
        plot_inverse_results(results_df, save_path='figures/inverse_targets.png')
        
        # Plot traiettorie
        if plot_trajectories:
            plot_trajectories_for_targets(
                all_solutions,
                trajectory_indices,
                save_dir='figures'
            )
    
    return results_df


# ============================================================================
# VISUALIZZAZIONI
# ============================================================================

def plot_inverse_results(
    results_df: pd.DataFrame,
    save_path: str = 'figures/inverse_targets.png'
) -> None:
    """
    Plot comparativo risultati ottimizzazione inversa.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(results_df))
    
    # ========== PLOT 1: Target vs Achieved ==========
    width = 0.25
    
    ax1.bar(x - width, results_df['y_target'], width, 
            label='Target', color='gray', alpha=0.7, edgecolor='black')
    ax1.bar(x, results_df['y_gp_pred'], width,
            label='GP Prediction', color='blue', alpha=0.7, edgecolor='black')
    ax1.bar(x + width, results_df['y_sim'], width,
            label='Simulated', color='green', alpha=0.7, edgecolor='black')
    
    # Error bars su GP
    ax1.errorbar(x, results_df['y_gp_pred'], yerr=results_df['y_gp_std'],
                 fmt='none', ecolor='darkblue', capsize=5, linewidth=2,
                 label='GP Uncertainty (±1σ)')
    
    ax1.set_xlabel('Target Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peak Outreach [m]', fontsize=12, fontweight='bold')
    ax1.set_title('Target vs GP vs Simulated Outreach', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x)
    
    # ========== PLOT 2: Errori ==========
    ax2.plot(x, results_df['error_gp'] * 1000, 'o-', linewidth=2, 
             markersize=8, label='GP Error', color='blue')
    ax2.plot(x, results_df['error_sim'] * 1000, 's-', linewidth=2,
             markersize=8, label='Simulation Error', color='green')
    ax2.plot(x, results_df['error_gp_vs_sim'] * 1000, '^-', linewidth=2,
             markersize=8, label='GP vs Sim', color='orange')
    
    ax2.set_xlabel('Target Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error [mm]', fontsize=12, fontweight='bold')
    ax2.set_title('Optimization Errors', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot salvato: {save_path}")
    plt.close()


def plot_trajectories_for_targets(
    solutions: List[Dict],
    indices: List[int],
    save_dir: str = 'figures'
) -> None:
    """
    Plot traiettorie simulazioni per target selezionati.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for idx in indices:
        if idx >= len(solutions):
            continue
        
        sol = solutions[idx]
        params = sol['full_params_opt']
        y_target = sol['y_target']
        
        # Simula con return_full
        peak_y, sim_result = simulate_system(
            params, T_sim=60.0, dt=0.001, return_full=True
        )
        
        # Plot
        t = sim_result.t
        x_b = sim_result.y[2]
        x_r = sim_result.y[3]
        y = x_b + x_r
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(t, y, 'k-', linewidth=2, label='Outreach y(t)')
        ax.axhline(y_target, color='red', linestyle='--', linewidth=2,
                   label=f'Target = {y_target:.3f} m')
        ax.axhline(peak_y, color='blue', linestyle=':', linewidth=2,
                   label=f'Achieved = {peak_y:.3f} m')
        
        ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Outreach [m]', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Optimized Trajectory for Target {idx+1} '
            f'(y_target = {y_target:.3f} m)',
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        filepath = f'{save_dir}/inverse_trajectory_target{idx+1}.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Traiettoria salvata: {filepath}")
        plt.close()


# ============================================================================
# MAIN: Test completo ottimizzazione inversa
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INVERSE OPTIMIZATION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # ========== 1. Carica dataset per scegliere target ==========
    X, y = load_dataset('data/dataset_outreach.csv')
    
    # ========== 2. Seleziona target dai percentili ==========
    targets = select_targets_from_dataset(
        y,
        percentiles=[20, 35, 50, 65, 80],
        verbose=True
    )
    
    # ========== 3. Configurazione problema ==========
    config = InverseProblemConfig(
        # Parametri fissi
        Kb=2000.0,
        Mb=50.0,
        hb=0.2,
        Mr=10.0
    )
    
    # ========== 4. Esegui pipeline completa ==========
    results_df = run_inverse_optimization_pipeline(
        targets=targets,
        config=config,
        gp_path='results/gp_model.pkl',
        save_results=True,
        plot_trajectories=True,
        trajectory_indices=[0, 2, 4]  # Plot per target 1, 3, 5
    )
    
    # ========== 5. Summary finale ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETATA!")
    print("="*70)
    print(f"\n📊 PERFORMANCE:")
    print(f"  Numero target testati: {len(targets)}")
    print(f"  Errore simulazione medio: {results_df['error_sim'].mean()*1000:.3f} mm")
    print(f"  Errore simulazione max:   {results_df['error_sim'].max()*1000:.3f} mm")
    print(f"  Success rate: {100*results_df['success'].sum()/len(results_df):.0f}%")
    
    print(f"\n📁 OUTPUT:")
    print(f"  Risultati: results/inverse_results.csv")
    print(f"  Plot:      figures/inverse_targets.png")
    print(f"  Traiet.:   figures/inverse_trajectory_target*.png")
    
    print("\n✅ Progetto ML completo!")
    print("   Prossimi step opzionali:")
    print("     - Sensitivity analysis")
    print("     - Monte Carlo robustness")
    print("     - Confronto con Random Forest")
    print("="*70 + "\n")