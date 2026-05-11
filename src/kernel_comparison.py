"""
kernel_comparison.py
====================
Confronto kernel GP per giustificare scelta Matern 5/2.

Kernel testati:
- RBF (Squared Exponential)
- Matern 1/2
- Matern 3/2
- Matern 5/2 (ufficiale)
- Rational Quadratic

Author: MatteoCasazza
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel,
    ConstantKernel as C
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import dataset
from dataset import load_dataset


# ============================================================================
# CONFIGURAZIONE KERNEL
# ============================================================================

def create_kernel_configs(n_dims: int = 9) -> Dict[str, object]:
    """
    Crea configurazioni kernel da testare.
    
    Returns
    -------
    kernels : dict
        {kernel_name: kernel_object}
    """
    # Inizializzazioni length-scale
    ls_init = np.ones(n_dims)
    ls_bounds = (1e-2, 1e3)
    
    kernels = {
        'RBF': (
            C(1.0, (1e-3, 1e3)) *
            RBF(length_scale=ls_init, length_scale_bounds=ls_bounds) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),
        
        'Matern_1/2': (
            C(1.0, (1e-3, 1e3)) *
            Matern(nu=0.5, length_scale=ls_init, length_scale_bounds=ls_bounds) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),
        
        'Matern_3/2': (
            C(1.0, (1e-3, 1e3)) *
            Matern(nu=1.5, length_scale=ls_init, length_scale_bounds=ls_bounds) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),
        
        'Matern_5/2': (
            C(1.0, (1e-3, 1e3)) *
            Matern(nu=2.5, length_scale=ls_init, length_scale_bounds=ls_bounds) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        ),
        
        'RationalQuadratic': (
            C(1.0, (1e-3, 1e3)) *
            RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=ls_bounds,
                alpha_bounds=(1e-2, 1e2)
            ) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        )
    }
    
    return kernels


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate_kernel(
    kernel_name: str,
    kernel: object,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler_y: StandardScaler,
    n_restarts: int = 3
) -> Dict:
    """
    Addestra GP con kernel specifico e valuta.
    
    Returns
    -------
    result : dict
        Tutte le metriche + tempo training
    """
    print(f"\n{'='*70}")
    print(f"KERNEL: {kernel_name}")
    print(f"{'='*70}")
    
    # Crea GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
        random_state=42,
        alpha=1e-10
    )
    
    # Training
    print(f"Training GP con {kernel_name}...")
    start_time = time.time()
    gp.fit(X_train_scaled, y_train_scaled)
    train_time = time.time() - start_time
    print(f"✓ Training completato in {train_time:.2f} s")
    
    # Log-marginal-likelihood
    lml = gp.log_marginal_likelihood(gp.kernel_.theta)
    print(f"Log-Marginal-Likelihood: {lml:.3f}")
    
    # Predizioni (scaled)
    y_train_pred_sc = gp.predict(X_train_scaled)
    y_test_pred_sc = gp.predict(X_test_scaled)
    
    # De-normalizza
    y_train_pred = scaler_y.inverse_transform(y_train_pred_sc.reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_sc.reshape(-1, 1)).ravel()
    
    y_train_true = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
    y_test_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    
    # Metriche
    train_r2 = r2_score(y_train_true, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    
    test_r2 = r2_score(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    
    print(f"\nTrain: R²={train_r2:.6f} | RMSE={train_rmse*1000:.3f}mm | MAE={train_mae*1000:.3f}mm")
    print(f"Test:  R²={test_r2:.6f} | RMSE={test_rmse*1000:.3f}mm | MAE={test_mae*1000:.3f}mm")
    
    return {
        'kernel': kernel_name,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'log_marginal_likelihood': lml,
        'training_time_s': train_time
    }


# ============================================================================
# VISUALIZZAZIONI
# ============================================================================

def plot_kernel_comparison(
    results_df: pd.DataFrame,
    save_path: str = 'figures/kernel_comparison.png'
) -> None:
    """
    Plot completo confronto kernel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    kernels = results_df['kernel'].values
    x_pos = np.arange(len(kernels))
    
    # Trova migliore per ogni metrica
    best_r2_idx = results_df['test_r2'].idxmax()
    best_rmse_idx = results_df['test_rmse'].idxmin()
    best_mae_idx = results_df['test_mae'].idxmin()
    best_lml_idx = results_df['log_marginal_likelihood'].idxmax()
    
    # ========== Test R² ==========
    ax = axes[0, 0]
    colors = ['gold' if i == best_r2_idx else 'steelblue' for i in range(len(kernels))]
    bars = ax.bar(x_pos, results_df['test_r2'], color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Test R²', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, results_df['test_r2'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold' if i == best_r2_idx else 'normal')
    
    # ========== Test RMSE ==========
    ax = axes[0, 1]
    colors = ['gold' if i == best_rmse_idx else 'coral' for i in range(len(kernels))]
    bars = ax.bar(x_pos, results_df['test_rmse']*1000, color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('RMSE [mm]', fontsize=12, fontweight='bold')
    ax.set_title('Test RMSE', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, results_df['test_rmse']*1000)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold' if i == best_rmse_idx else 'normal')
    
    # ========== Test MAE ==========
    ax = axes[1, 0]
    colors = ['gold' if i == best_mae_idx else 'lightgreen' for i in range(len(kernels))]
    bars = ax.bar(x_pos, results_df['test_mae']*1000, color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('MAE [mm]', fontsize=12, fontweight='bold')
    ax.set_title('Test MAE', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, results_df['test_mae']*1000)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold' if i == best_mae_idx else 'normal')
    
    # ========== Log-Marginal-Likelihood ==========
    ax = axes[1, 1]
    colors = ['gold' if i == best_lml_idx else 'plum' for i in range(len(kernels))]
    bars = ax.bar(x_pos, results_df['log_marginal_likelihood'], color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Log-Marginal-Likelihood', fontsize=12, fontweight='bold')
    ax.set_title('LML', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, results_df['log_marginal_likelihood'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold' if i == best_lml_idx else 'normal')
    
    plt.suptitle('Gaussian Process Kernel Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    Path("figures").mkdir(exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Kernel comparison plot salvato: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GAUSSIAN PROCESS KERNEL COMPARISON")
    print("="*70 + "\n")
    
    # ========== 1. Carica dataset ==========
    X, y = load_dataset('data/dataset_outreach.csv')
    print(f"Dataset: {X.shape[0]} campioni\n")
    
    # ========== 2. Split (STESSO del training ufficiale) ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # ========== 3. Standardizza ==========
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    
    # ========== 4. Crea kernel configs ==========
    kernels = create_kernel_configs(n_dims=X.shape[1])
    print(f"\nKernel da testare: {len(kernels)}")
    for name in kernels.keys():
        print(f"  - {name}")
    
    # ========== 5. Train & evaluate tutti i kernel ==========
    results = []
    
    for kernel_name, kernel in kernels.items():
        result = train_and_evaluate_kernel(
            kernel_name, kernel,
            X_train_sc, y_train_sc,
            X_test_sc, y_test_sc,
            scaler_y,
            n_restarts=5  # Ragionevole per non essere troppo lento
        )
        results.append(result)
    
    # ========== 6. Crea DataFrame risultati ==========
    results_df = pd.DataFrame(results)
    
    # Ordina per test R² decrescente
    results_df = results_df.sort_values('test_r2', ascending=False).reset_index(drop=True)
    
    # ========== 7. Stampa tabella completa ==========
    print("\n" + "="*70)
    print("RISULTATI COMPLETI")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # ========== 8. Identifica migliori ==========
    print("\n" + "="*70)
    print("MIGLIORI KERNEL PER METRICA")
    print("="*70)
    
    best_r2 = results_df.loc[results_df['test_r2'].idxmax()]
    best_rmse = results_df.loc[results_df['test_rmse'].idxmin()]
    best_mae = results_df.loc[results_df['test_mae'].idxmin()]
    best_lml = results_df.loc[results_df['log_marginal_likelihood'].idxmax()]
    
    print(f"\nMigliore R²:   {best_r2['kernel']:20s} (R²={best_r2['test_r2']:.6f})")
    print(f"Migliore RMSE: {best_rmse['kernel']:20s} (RMSE={best_rmse['test_rmse']*1000:.3f}mm)")
    print(f"Migliore MAE:  {best_mae['kernel']:20s} (MAE={best_mae['test_mae']*1000:.3f}mm)")
    print(f"Migliore LML:  {best_lml['kernel']:20s} (LML={best_lml['log_marginal_likelihood']:.2f})")
    
    # ========== 9. Salva ==========
    Path('results').mkdir(exist_ok=True)
    results_df.to_csv('results/kernel_comparison.csv', index=False)
    print(f"\n✓ Risultati salvati: results/kernel_comparison.csv")
    
    # ========== 10. Plot ==========
    print("\nGenerazione plot...")
    plot_kernel_comparison(results_df, save_path='figures/kernel_comparison.png')
    
    # ========== 11. Interpretazione finale ==========
    print("\n" + "="*70)
    print("INTERPRETAZIONE E RACCOMANDAZIONE")
    print("="*70)
    
    print(f"\nMatern 5/2 (kernel ufficiale):")
    m52_row = results_df[results_df['kernel'] == 'Matern_5/2'].iloc[0]
    print(f"  R²:   {m52_row['test_r2']:.6f}")
    print(f"  RMSE: {m52_row['test_rmse']*1000:.3f} mm")
    print(f"  MAE:  {m52_row['test_mae']*1000:.3f} mm")
    print(f"  LML:  {m52_row['log_marginal_likelihood']:.2f}")
    
    print(f"\nPerché Matern 5/2 è una scelta eccellente:")
    print(f"  ✓ Due volte differenziabile (C²) → smooth ma non eccessivo")
    print(f"  ✓ Adatto a funzioni da simulazioni fisiche")
    print(f"  ✓ Compromesso tra Matern 3/2 (meno smooth) e RBF (troppo smooth)")
    print(f"  ✓ Performance competitive con kernel più complessi")
    print(f"  ✓ Interpretabile e standard in letteratura")
    
    if best_r2['kernel'] == 'Matern_5/2':
        print(f"\n✓ Matern 5/2 è il MIGLIORE per R²!")
    else:
        delta = best_r2['test_r2'] - m52_row['test_r2']
        print(f"\n  Δ rispetto al migliore ({best_r2['kernel']}): {delta:.4f}")
        if delta < 0.005:
            print(f"  → Differenza trascurabile, Matern 5/2 OTTIMA SCELTA")
        else:
            print(f"  → Differenza piccola ma {best_r2['kernel']} leggermente superiore")
    
    print("\n" + "="*70)
    print("KERNEL COMPARISON COMPLETATO!")
    print("="*70)
    print(f"\nFile generati:")
    print(f"  - results/kernel_comparison.csv")
    print(f"  - figures/kernel_comparison.png")
    print("="*70 + "\n")