"""
dataset.py
==========
Generazione dataset per training ML usando Latin Hypercube Sampling.

Workflow:
1. Definisce range parametri (9D)
2. Campiona con LHS
3. Simula sistema per ogni campione (parallelo)
4. Salva dataset in CSV
5. Genera statistiche e plot

Author: MatteoCasazza
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc
from joblib import Parallel, delayed
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import time

# Import del simulatore
from dynamics import simulate_system


# ============================================================================
# CONFIGURAZIONE PARAMETRI
# ============================================================================

@dataclass
class ParameterRanges:
    """
    Range dei parametri del sistema.
    
    Tutti i valori seguono la convenzione della roadmap.
    """
    # Parametri base (passiva)
    Kb_min: float = 100.0      # Stiffness base [N/m]
    Kb_max: float = 5000.0
    
    Mb_min: float = 10.0       # Massa base [kg]
    Mb_max: float = 100.0
    
    hb_min: float = 0.1        # Damping ratio base [-]
    hb_max: float = 0.3
    
    # Parametri robot (controllabile)
    Kr_min: float = 100.0      # Stiffness robot [N/m]
    Kr_max: float = 5000.0
    
    hr_min: float = 0.1        # Damping ratio robot [-]
    hr_max: float = 1.1
    
    # Parametri chirp
    f0_min: float = 0.0001     # Frequenza iniziale [Hz]
    f0_max: float = 0.01
    
    f1_min: float = 3.0        # Frequenza finale [Hz]
    f1_max: float = 10.0
    
    A_min: float = 0.1         # Ampiezza [m]
    A_max: float = 0.2
    
    x_r_start_min: float = 0.3 # Posizione iniziale [m]
    x_r_start_max: float = 0.5
    
    # Parametro fisso
    Mr: float = 10.0           # Massa robot [kg] - FISSO
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ritorna lower e upper bounds come array.
        
        Returns
        -------
        lb : array (9,)
            Lower bounds
        ub : array (9,)
            Upper bounds
        """
        lb = np.array([
            self.Kb_min,
            self.Kr_min,
            self.Mb_min,
            self.hb_min,
            self.hr_min,
            self.f0_min,
            self.f1_min,
            self.A_min,
            self.x_r_start_min
        ])
        
        ub = np.array([
            self.Kb_max,
            self.Kr_max,
            self.Mb_max,
            self.hb_max,
            self.hr_max,
            self.f0_max,
            self.f1_max,
            self.A_max,
            self.x_r_start_max
        ])
        
        return lb, ub
    
    @staticmethod
    def get_param_names() -> list:
        """Nomi dei parametri in ordine."""
        return ['Kb', 'Kr', 'Mb', 'hb', 'hr', 'f0', 'f1', 'A', 'x_r_start']


# ============================================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================================

def generate_lhs_samples(
    n_samples: int,
    param_ranges: ParameterRanges,
    seed: int = 42
) -> np.ndarray:
    """
    Genera campioni usando Latin Hypercube Sampling.
    
    Parameters
    ----------
    n_samples : int
        Numero di campioni da generare
    param_ranges : ParameterRanges
        Range dei parametri
    seed : int
        Random seed per riproducibilità
    
    Returns
    -------
    samples : array (n_samples, 9)
        Matrice dei campioni
    
    Notes
    -----
    Usa scipy.stats.qmc.LatinHypercube per garantire:
    - Copertura uniforme dello spazio
    - Evitare clustering
    - Ottimizzazione centered per simmetria
    """
    lb, ub = param_ranges.get_bounds()
    n_dims = len(lb)
    
    print(f"Generazione {n_samples} campioni LHS in spazio {n_dims}D...")
    
    # Crea sampler LHS
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    
    # Genera campioni normalizzati [0, 1]^d
    samples_normalized = sampler.random(n=n_samples)
    
    # Scala ai bounds reali
    samples = qmc.scale(samples_normalized, lb, ub)
    
    print(f"✓ Campioni generati: shape {samples.shape}")
    
    return samples


# ============================================================================
# SIMULAZIONE DATASET
# ============================================================================

def simulate_parameter_set(
    params_array: np.ndarray,
    idx: int,
    T_sim: float = 60.0,
    dt: float = 0.001
) -> Tuple[int, float]:
    """
    Simula il sistema per un set di parametri.
    
    Parameters
    ----------
    params_array : array (9,)
        [Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
    idx : int
        Indice campione (per tracking)
    T_sim : float
        Tempo simulazione [s]
    dt : float
        Step temporale [s]
    
    Returns
    -------
    idx : int
        Indice campione
    peak_y : float
        Massimo outreach raggiunto [m]
    
    Notes
    -----
    Questa funzione è chiamata in parallelo da joblib.
    Deve essere top-level (non nested) per serializzazione.
    """
    try:
        peak_y = simulate_system(params_array, T_sim=T_sim, dt=dt)
        return idx, peak_y
    except Exception as e:
        print(f"⚠️  Errore simulazione #{idx}: {e}")
        return idx, np.nan


def generate_dataset(
    n_samples: int = 220,
    param_ranges: Optional[ParameterRanges] = None,
    T_sim: float = 60.0,
    dt: float = 0.001,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera dataset completo con simulazioni parallele.
    
    Parameters
    ----------
    n_samples : int
        Numero di campioni
    param_ranges : ParameterRanges, optional
        Range parametri (default: usa valori standard)
    T_sim : float
        Tempo simulazione [s]
    dt : float
        Step temporale [s]
    n_jobs : int
        Numero core (-1 = tutti disponibili)
    seed : int
        Random seed
    verbose : bool
        Stampa progress
    
    Returns
    -------
    X : array (n_samples, 9)
        Matrice parametri input
    y : array (n_samples,)
        Vettore peak outreach
    
    Examples
    --------
    >>> X, y = generate_dataset(n_samples=100, n_jobs=4)
    >>> print(X.shape, y.shape)
    (100, 9) (100,)
    """
    if param_ranges is None:
        param_ranges = ParameterRanges()
    
    print("="*70)
    print("GENERAZIONE DATASET")
    print("="*70)
    print(f"Numero campioni:  {n_samples}")
    print(f"Tempo simulazione: {T_sim} s")
    print(f"Step temporale:    {dt} s")
    print(f"Parallelizzazione: {n_jobs} core")
    print("="*70)
    
    # 1. Genera campioni LHS
    X = generate_lhs_samples(n_samples, param_ranges, seed=seed)
    
    # 2. Simula in parallelo
    print(f"\nEsecuzione {n_samples} simulazioni parallele...")
    start_time = time.time()
    
    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(simulate_parameter_set)(X[i], i, T_sim, dt)
        for i in range(n_samples)
    )
    
    elapsed = time.time() - start_time
    
    # 3. Estrai risultati
    y = np.zeros(n_samples)
    for idx, peak_y in results:
        y[idx] = peak_y
    
    # 4. Check NaN
    n_failed = np.sum(np.isnan(y))
    if n_failed > 0:
        print(f"\n⚠️  {n_failed}/{n_samples} simulazioni fallite (NaN)")
        print("Rimuovo campioni falliti...")
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
    
    print(f"\n✓ Dataset generato in {elapsed:.1f} s")
    print(f"  Campioni validi: {len(y)}/{n_samples}")
    print(f"  Tempo medio/simulazione: {elapsed/n_samples:.3f} s")
    
    return X, y


# ============================================================================
# SALVATAGGIO E I/O
# ============================================================================

def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    filepath: str = 'data/dataset_outreach.csv',
    param_ranges: Optional[ParameterRanges] = None
) -> pd.DataFrame:
    """
    Salva dataset in CSV con metadata.
    
    Parameters
    ----------
    X : array (n, 9)
        Parametri input
    y : array (n,)
        Peak outreach
    filepath : str
        Path file output
    param_ranges : ParameterRanges, optional
        Per includere bounds in header
    
    Returns
    -------
    df : DataFrame
        Dataset salvato
    """
    if param_ranges is None:
        param_ranges = ParameterRanges()
    
    # Crea DataFrame
    param_names = ParameterRanges.get_param_names()
    df = pd.DataFrame(X, columns=param_names)
    df['peak_y'] = y
    
    # Crea directory se non esiste
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Scrivi metadata come commenti
    with open(filepath, 'w') as f:
        f.write(f"# Dataset Outreach - ML Compliant Base Project\n")
        f.write(f"# Samples: {len(y)}\n")
        f.write(f"# Sampling: Latin Hypercube\n")
        f.write(f"# Peak y statistics:\n")
        f.write(f"#   Mean:   {y.mean():.6f} m\n")
        f.write(f"#   Std:    {y.std():.6f} m\n")
        f.write(f"#   Min:    {y.min():.6f} m\n")
        f.write(f"#   Max:    {y.max():.6f} m\n")
        f.write(f"#   Median: {np.median(y):.6f} m\n")
        f.write(f"# Parameter ranges:\n")
        lb, ub = param_ranges.get_bounds()
        for i, name in enumerate(param_names):
            f.write(f"#   {name:12s}: [{lb[i]:.6f}, {ub[i]:.6f}]\n")
        f.write("#\n")
    
    # Appendi dati
    df.to_csv(filepath, index=False, mode='a')
    
    print(f"\n✓ Dataset salvato: {filepath}")
    print(f"  Dimensioni: {df.shape}")
    
    return df


def load_dataset(filepath: str = 'data/dataset_outreach.csv') -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica dataset da CSV.
    
    Parameters
    ----------
    filepath : str
        Path al CSV
    
    Returns
    -------
    X : array (n, 9)
        Parametri input
    y : array (n,)
        Peak outreach
    """
    df = pd.read_csv(filepath, comment='#')
    
    param_names = ParameterRanges.get_param_names()
    X = df[param_names].values
    y = df['peak_y'].values
    
    print(f"✓ Dataset caricato: {filepath}")
    print(f"  Campioni: {len(y)}")
    
    return X, y


# ============================================================================
# VISUALIZZAZIONE
# ============================================================================

def plot_dataset_stats(
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = 'figures'
) -> None:
    """
    Genera plot statistici del dataset.
    
    Parameters
    ----------
    X : array (n, 9)
        Parametri input
    y : array (n,)
        Peak outreach
    save_dir : str
        Directory output figure
    
    Generates
    ---------
    - dataset_distribution.png: istogramma peak_y
    - dataset_correlations.png: heatmap correlazioni
    - dataset_scatter.png: scatter parametri chiave vs peak_y
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    param_names = ParameterRanges.get_param_names()
    
    # ========== PLOT 1: Distribuzione peak_y ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(y, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(y.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {y.mean():.3f} m')
    ax.axvline(np.median(y), color='orange', linestyle='--', linewidth=2,
               label=f'Median = {np.median(y):.3f} m')
    
    ax.set_xlabel('Peak Outreach [m]', fontsize=12)
    ax.set_ylabel('Frequenza', fontsize=12)
    ax.set_title(f'Distribuzione Peak Outreach (n={len(y)})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    filepath = f'{save_dir}/dataset_distribution.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()
    
    # ========== PLOT 2: Correlation heatmap ==========
    df = pd.DataFrame(X, columns=param_names)
    df['peak_y'] = y
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlazione'},
        ax=ax
    )
    
    ax.set_title('Matrice di Correlazione (Parametri → Peak Outreach)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    filepath = f'{save_dir}/dataset_correlations.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()
    
    # ========== PLOT 3: Scatter parametri chiave ==========
    key_params = ['A', 'Kr', 'f1', 'Kb']
    key_indices = [param_names.index(p) for p in key_params]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for i, (param_idx, param_name) in enumerate(zip(key_indices, key_params)):
        ax = axes[i]
        
        scatter = ax.scatter(
            X[:, param_idx],
            y,
            c=y,
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Outreach [m]', fontsize=12)
        ax.set_title(f'{param_name} vs Peak Outreach', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Peak Outreach [m]', fontsize=10)
    
    plt.suptitle('Scatter Plots: Parametri Chiave vs Outreach', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filepath = f'{save_dir}/dataset_scatter.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()
    
    print(f"\n✓ Plot dataset completati in: {save_dir}/")


# ============================================================================
# MAIN: Test completo del modulo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: src/dataset.py")
    print("="*70 + "\n")
    
    # ========== CONFIGURAZIONE ==========
    # Per test veloce: usa pochi campioni
    # Per dataset finale: usa 220+
    N_SAMPLES_TEST = 220  # Cambia a 220 per dataset finale
    
    # ========== GENERAZIONE DATASET ==========
    param_ranges = ParameterRanges()
    
    X, y = generate_dataset(
        n_samples=N_SAMPLES_TEST,
        param_ranges=param_ranges,
        T_sim=60.0,
        dt=0.001,
        n_jobs=-1,  # Usa tutti i core
        seed=42,
        verbose=True
    )
    
    # ========== STATISTICHE ==========
    print("\n" + "="*70)
    print("STATISTICHE DATASET")
    print("="*70)
    print(f"Numero campioni:  {len(y)}")
    print(f"Shape X:          {X.shape}")
    print(f"Shape y:          {y.shape}")
    print(f"\nPeak outreach [m]:")
    print(f"  Mean:   {y.mean():.6f}")
    print(f"  Std:    {y.std():.6f}")
    print(f"  Min:    {y.min():.6f}")
    print(f"  Max:    {y.max():.6f}")
    print(f"  Median: {np.median(y):.6f}")
    print(f"  Q1:     {np.percentile(y, 25):.6f}")
    print(f"  Q3:     {np.percentile(y, 75):.6f}")
    
    # ========== SALVATAGGIO ==========
    df = save_dataset(X, y, 'data/dataset_outreach.csv', param_ranges)
    
    # ========== PLOT ==========
    print("\nGenerazione plot...")
    plot_dataset_stats(X, y, save_dir='figures')
    
    # ========== TEST RICARICAMENTO ==========
    print("\nTest ricaricamento dataset...")
    X_loaded, y_loaded = load_dataset('data/dataset_outreach.csv')
    assert np.allclose(X, X_loaded), "❌ Errore: X non corrisponde!"
    assert np.allclose(y, y_loaded), "❌ Errore: y non corrisponde!"
    print("✓ Ricaricamento OK: dati identici")
    
    print("\n" + "="*70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("="*70)
    print("\nProssimi step:")
    print("  1. Controlla: data/dataset_outreach.csv")
    print("  2. Controlla: figures/dataset_*.png")
    print("  3. Per dataset finale: modifica N_SAMPLES_TEST = 220")
    print("  4. Procedi con: src/models.py (GP training)")
    print("="*70 + "\n")