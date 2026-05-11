"""
models.py
=========
Gaussian Process Regression per predire peak outreach.

Workflow:
1. Carica dataset
2. Train/test split (80/20)
3. Standardizzazione (X e y)
4. Training GP con kernel Matern
5. Metriche: RMSE, MAE, R²
6. ARD: importanza parametri
7. Plot: parity, residui, uncertainty

Author: [Il tuo nome]
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time
from typing import Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel as C
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import dataset utilities
from dataset import load_dataset, ParameterRanges


# ============================================================================
# PREPROCESSING
# ============================================================================

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           StandardScaler, StandardScaler]:
    """
    Prepara dati per training GP: split + standardizzazione.
    
    Parameters
    ----------
    X : array (n, 9)
        Parametri input
    y : array (n,)
        Peak outreach
    test_size : float
        Frazione test set
    random_state : int
        Random seed
    
    Returns
    -------
    X_train_scaled : array (n_train, 9)
    X_test_scaled : array (n_test, 9)
    y_train_scaled : array (n_train,)
    y_test_scaled : array (n_test,)
    scaler_X : StandardScaler
    scaler_y : StandardScaler
    
    Notes
    -----
    IMPORTANTE: Standardizzazione è OBBLIGATORIA per GP!
    - Migliora convergenza ottimizzatore
    - Rende kernel isotropico interpretabile
    - Evita problemi numerici con length-scale
    """
    print("\n" + "="*70)
    print("PREPROCESSING DATASET")
    print("="*70)
    
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Train set: {len(y_train)} campioni ({100*(1-test_size):.0f}%)")
    print(f"Test set:  {len(y_test)} campioni ({100*test_size:.0f}%)")
    
    # 2. Standardizzazione X
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nStandardizzazione X:")
    print(f"  Mean X_train: {X_train_scaled.mean(axis=0)[:3]} ... (primi 3)")
    print(f"  Std X_train:  {X_train_scaled.std(axis=0)[:3]} ...")
    
    # 3. Standardizzazione y
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"\nStandardizzazione y:")
    print(f"  Mean y_train: {y_train_scaled.mean():.6f}")
    print(f"  Std y_train:  {y_train_scaled.std():.6f}")
    
    # 4. Verifica
    assert np.abs(X_train_scaled.mean()) < 1e-10, "X_train non centrato!"
    assert np.abs(X_train_scaled.std() - 1.0) < 0.1, "X_train non normalizzato!"
    assert np.abs(y_train_scaled.mean()) < 1e-10, "y_train non centrato!"
    
    print("\n✓ Preprocessing completato")
    
    return (X_train_scaled, X_test_scaled, 
            y_train_scaled, y_test_scaled,
            scaler_X, scaler_y)


# ============================================================================
# GAUSSIAN PROCESS TRAINING
# ============================================================================

def create_gp_model(
    kernel_type: str = 'matern52',
    n_dims: int = 9,
    length_scale_init: Optional[np.ndarray] = None,
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e3),
    noise_level: float = 1e-5,
    n_restarts: int = 10
) -> GaussianProcessRegressor:
    """
    Crea modello Gaussian Process con kernel configurabile.
    
    Parameters
    ----------
    kernel_type : str
        Tipo kernel: 'matern32', 'matern52', 'rbf'
    n_dims : int
        Dimensionalità input
    length_scale_init : array, optional
        Valori iniziali length-scale (default: ones)
    length_scale_bounds : tuple
        Bounds per ottimizzazione length-scale
    noise_level : float
        Livello rumore iniziale
    n_restarts : int
        Restart ottimizzatore (importante per GP!)
    
    Returns
    -------
    gp : GaussianProcessRegressor
        Modello non addestrato
    
    Notes
    -----
    Kernel formula:
        k(x, x') = σ² · Matern(x, x', ℓ, ν) + σ_noise² · δ(x, x')
    
    dove:
        σ² = constant_value (signal variance)
        ℓ = length_scale (per ogni dimensione, ARD)
        ν = nu (smoothness: 1.5 per Matern32, 2.5 per Matern52)
        σ_noise² = noise_level
    
    ARD (Automatic Relevance Determination):
        - Ogni dimensione ha la sua length-scale
        - Length-scale piccola → dimensione importante
        - Length-scale grande → dimensione irrilevante
    """
    if length_scale_init is None:
        length_scale_init = np.ones(n_dims)
    
    # Kernel base
    if kernel_type == 'matern32':
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=1.5  # C¹ continuity
        )
    elif kernel_type == 'matern52':
        base_kernel = Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=2.5  # C² continuity (smoother)
        )
    elif kernel_type == 'rbf':
        from sklearn.gaussian_process.kernels import RBF
        base_kernel = RBF(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds
        )
    else:
        raise ValueError(f"Kernel sconosciuto: {kernel_type}")
    
    # Kernel completo: ConstantKernel * BaseKernel + WhiteKernel
    kernel = (
        C(1.0, constant_value_bounds=(1e-3, 1e3)) * base_kernel +
        WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-1))
    )
    
    # Crea GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,  # Già normalizzato esternamente
        random_state=42,
        alpha=1e-10  # Regularizzazione numerica
    )
    
    print(f"\n✓ Creato GP con kernel: {kernel_type.upper()}")
    print(f"  Kernel formula: {kernel}")
    print(f"  N. restart ottimizzatore: {n_restarts}")
    
    return gp


def train_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel_type: str = 'matern52',
    n_restarts: int = 10,
    verbose: bool = True
) -> GaussianProcessRegressor:
    """
    Addestra Gaussian Process.
    
    Parameters
    ----------
    X_train : array (n, 9)
        Training input (GIÀ standardizzato!)
    y_train : array (n,)
        Training output (GIÀ standardizzato!)
    kernel_type : str
        Tipo kernel
    n_restarts : int
        Restart ottimizzatore
    verbose : bool
        Stampa dettagli
    
    Returns
    -------
    gp : GaussianProcessRegressor
        Modello addestrato
    """
    print("\n" + "="*70)
    print("TRAINING GAUSSIAN PROCESS")
    print("="*70)
    
    # Crea modello
    gp = create_gp_model(
        kernel_type=kernel_type,
        n_dims=X_train.shape[1],
        n_restarts=n_restarts
    )
    
    # Training
    print("\nFitting GP (questo può richiedere 10-60s)...")
    start_time = time.time()
    
    gp.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✓ Training completato in {elapsed:.1f} s")
    
    # Log-marginal-likelihood (bontà fit)
    lml = gp.log_marginal_likelihood(gp.kernel_.theta)
    print(f"\nLog-Marginal-Likelihood: {lml:.3f}")
    print(f"  (Più alto = migliore fit)")
    
    # Hyperparameters ottimizzati
    if verbose:
        print(f"\nKernel ottimizzato:")
        print(f"  {gp.kernel_}")
    
    return gp


# ============================================================================
# METRICHE
# ============================================================================

def evaluate_model(
    gp: GaussianProcessRegressor,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler_y: StandardScaler
) -> Dict[str, float]:
    """
    Valuta performance GP su train e test set.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Modello addestrato
    X_train_scaled : array
        Training input (scaled)
    y_train_scaled : array
        Training output (scaled)
    X_test_scaled : array
        Test input (scaled)
    y_test_scaled : array
        Test output (scaled)
    scaler_y : StandardScaler
        Per de-normalizzare predizioni
    
    Returns
    -------
    metrics : dict
        {
            'train_rmse', 'train_mae', 'train_r2',
            'test_rmse', 'test_mae', 'test_r2',
            'train_rmse_scaled', 'test_rmse_scaled',
            'y_train_pred', 'y_test_pred',
            'y_train_std', 'y_test_std'
        }
    
    Notes
    -----
    IMPORTANTE: Predizioni vengono de-normalizzate per calcolare
    metriche in unità fisiche (metri).
    """
    print("\n" + "="*70)
    print("VALUTAZIONE MODELLO")
    print("="*70)
    
    # Predizioni (scaled)
    y_train_pred_scaled, y_train_std_scaled = gp.predict(
        X_train_scaled, return_std=True
    )
    y_test_pred_scaled, y_test_std_scaled = gp.predict(
        X_test_scaled, return_std=True
    )
    
    # De-normalizza
    y_train_pred = scaler_y.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).ravel()
    
    y_train_true = scaler_y.inverse_transform(
        y_train_scaled.reshape(-1, 1)
    ).ravel()
    y_test_true = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).ravel()
    
    # De-normalizza std (scala con std di y)
    y_train_std = y_train_std_scaled * scaler_y.scale_[0]
    y_test_std = y_test_std_scaled * scaler_y.scale_[0]
    
    # Metriche TRAIN
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_r2 = r2_score(y_train_true, y_train_pred)
    
    # Metriche TEST
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)
    
    # Metriche scaled (per confronto)
    train_rmse_scaled = np.sqrt(mean_squared_error(
        y_train_scaled, y_train_pred_scaled
    ))
    test_rmse_scaled = np.sqrt(mean_squared_error(
        y_test_scaled, y_test_pred_scaled
    ))
    
    # Stampa
    print("\n--- TRAINING SET ---")
    print(f"  RMSE:  {train_rmse:.6f} m  (scaled: {train_rmse_scaled:.6f})")
    print(f"  MAE:   {train_mae:.6f} m")
    print(f"  R²:    {train_r2:.6f}")
    
    print("\n--- TEST SET ---")
    print(f"  RMSE:  {test_rmse:.6f} m  (scaled: {test_rmse_scaled:.6f})")
    print(f"  MAE:   {test_mae:.6f} m")
    print(f"  R²:    {test_r2:.6f}")
    
    print("\n--- UNCERTAINTY (Test Set) ---")
    print(f"  Mean std: {y_test_std.mean():.6f} m")
    print(f"  Max std:  {y_test_std.max():.6f} m")
    
    # Interpretazione
    if test_r2 > 0.95:
        print("\n✓ Eccellente! R² > 0.95")
    elif test_r2 > 0.85:
        print("\n✓ Molto buono! R² > 0.85")
    elif test_r2 > 0.70:
        print("\n⚠️  Accettabile, ma si può migliorare (R² < 0.85)")
    else:
        print("\n⚠️  Attenzione: R² < 0.70, considera:")
        print("     - Aumentare n_samples")
        print("     - Provare kernel diverso")
        print("     - Verificare range parametri")
    
    return {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_rmse_scaled': train_rmse_scaled,
        'test_rmse_scaled': test_rmse_scaled,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_std': y_train_std,
        'y_test_std': y_test_std
    }


# ============================================================================
# ARD: AUTOMATIC RELEVANCE DETERMINATION
# ============================================================================

def analyze_ard_relevance(
    gp: GaussianProcessRegressor,
    param_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Analizza importanza parametri tramite ARD length-scales.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Modello addestrato con kernel ARD
    param_names : list, optional
        Nomi parametri (default: usa ParameterRanges)
    
    Returns
    -------
    df : DataFrame
        Colonne: ['Parameter', 'LengthScale', 'Relevance']
        Ordinato per rilevanza decrescente
    
    Notes
    -----
    Relevance formula:
        r_i = (1 / ℓ_i) / Σ_j (1 / ℓ_j)
    
    Interpretazione:
        - Relevance alta → parametro importante
        - Relevance bassa → parametro poco influente
    """
    print("\n" + "="*70)
    print("ARD: IMPORTANZA PARAMETRI")
    print("="*70)
    
    if param_names is None:
        param_names = ParameterRanges.get_param_names()
    
    # Estrai length-scales dal kernel ottimizzato
    kernel = gp.kernel_
    
    # Kernel è: ConstantKernel * Matern + WhiteKernel
    # Length-scales sono in: kernel.k1.k2.length_scale
    try:
        length_scales = kernel.k1.k2.length_scale
    except AttributeError:
        # Fallback per struttura kernel diversa
        print("⚠️  Struttura kernel non standard, provo alternative...")
        # Prova altre posizioni comuni
        try:
            length_scales = kernel.k2.length_scale
        except:
            length_scales = gp.kernel_.theta[1:10]  # Estrazione diretta
    
    print(f"\nLength-scales estratti: {len(length_scales)}")
    
    # Calcola relevance
    relevance_raw = 1.0 / length_scales
    relevance_normalized = relevance_raw / relevance_raw.sum()
    
    # Crea DataFrame
    df = pd.DataFrame({
        'Parameter': param_names,
        'LengthScale': length_scales,
        'Relevance': relevance_normalized
    })
    
    # Ordina per rilevanza
    df = df.sort_values('Relevance', ascending=False).reset_index(drop=True)
    
    # Stampa
    print("\n" + "-"*70)
    print(f"{'Rank':<6}{'Parameter':<15}{'Length-Scale':<18}{'Relevance':<15}")
    print("-"*70)
    for i, row in df.iterrows():
        print(f"{i+1:<6}{row['Parameter']:<15}"
              f"{row['LengthScale']:<18.6f}{row['Relevance']:<15.4f}")
    print("-"*70)
    
    # Interpretazione top 3
    print(f"\n✓ Top 3 parametri più influenti:")
    for i in range(min(3, len(df))):
        print(f"  {i+1}. {df.iloc[i]['Parameter']:12s} "
              f"(relevance: {df.iloc[i]['Relevance']:.3f})")
    
    return df


# ============================================================================
# VISUALIZZAZIONI
# ============================================================================

def plot_training_results(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_train_std: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    metrics: dict,
    save_dir: str = 'figures'
) -> None:
    """
    Plot completi risultati training GP.
    
    Generates
    ---------
    - gp_parity_plot.png: y_true vs y_pred
    - gp_residuals.png: residui
    - gp_uncertainty.png: predizioni con bande incertezza
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # ========== PLOT 1: Parity Plot ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train
    ax1.scatter(y_train_true, y_train_pred, alpha=0.6, s=50, 
                edgecolors='black', linewidth=0.5, label='Train')
    ax1.plot([y_train_true.min(), y_train_true.max()],
             [y_train_true.min(), y_train_true.max()],
             'r--', linewidth=2, label='Identità')
    ax1.set_xlabel('True Peak Outreach [m]', fontsize=12)
    ax1.set_ylabel('Predicted Peak Outreach [m]', fontsize=12)
    ax1.set_title(f'Training Set (R² = {metrics["train_r2"]:.4f})', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test
    ax2.scatter(y_test_true, y_test_pred, alpha=0.6, s=50, 
                c='orange', edgecolors='black', linewidth=0.5, label='Test')
    ax2.plot([y_test_true.min(), y_test_true.max()],
             [y_test_true.min(), y_test_true.max()],
             'r--', linewidth=2, label='Identità')
    ax2.set_xlabel('True Peak Outreach [m]', fontsize=12)
    ax2.set_ylabel('Predicted Peak Outreach [m]', fontsize=12)
    ax2.set_title(f'Test Set (R² = {metrics["test_r2"]:.4f})', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Parity Plot: GP Performance', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    filepath = f'{save_dir}/gp_parity_plot.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()
    
    # ========== PLOT 2: Residuals ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    residuals_train = y_train_true - y_train_pred
    residuals_test = y_test_true - y_test_pred
    
    # Train residuals
    ax1.scatter(y_train_pred, residuals_train, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Peak Outreach [m]', fontsize=12)
    ax1.set_ylabel('Residuals [m]', fontsize=12)
    ax1.set_title(f'Training Residuals (RMSE = {metrics["train_rmse"]:.6f} m)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Test residuals
    ax2.scatter(y_test_pred, residuals_test, alpha=0.6, s=50, c='orange',
                edgecolors='black', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Peak Outreach [m]', fontsize=12)
    ax2.set_ylabel('Residuals [m]', fontsize=12)
    ax2.set_title(f'Test Residuals (RMSE = {metrics["test_rmse"]:.6f} m)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    filepath = f'{save_dir}/gp_residuals.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()
    
    # ========== PLOT 3: Uncertainty ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Ordina per true value
    idx_sorted = np.argsort(y_test_true)
    y_true_sorted = y_test_true[idx_sorted]
    y_pred_sorted = y_test_pred[idx_sorted]
    y_std_sorted = y_test_std[idx_sorted]
    
    x = np.arange(len(y_test_true))
    
    # Plot
    ax.plot(x, y_true_sorted, 'k-', linewidth=2, label='True', alpha=0.8)
    ax.plot(x, y_pred_sorted, 'b-', linewidth=2, label='Predicted')
    ax.fill_between(
        x,
        y_pred_sorted - 1.96 * y_std_sorted,
        y_pred_sorted + 1.96 * y_std_sorted,
        alpha=0.3,
        color='blue',
        label='95% Confidence'
    )
    
    ax.set_xlabel('Test Sample (sorted by true value)', fontsize=12)
    ax.set_ylabel('Peak Outreach [m]', fontsize=12)
    ax.set_title('GP Predictions with Uncertainty Quantification',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = f'{save_dir}/gp_uncertainty.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()


def plot_ard_relevance(
    ard_df: pd.DataFrame,
    save_dir: str = 'figures'
) -> None:
    """
    Bar plot importanza parametri (ARD).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ard_df)))
    
    bars = ax.bar(
        ard_df['Parameter'],
        ard_df['Relevance'],
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_ylabel('Normalized Relevance', fontsize=13, fontweight='bold')
    ax.set_title('ARD: Parameter Importance Ranking',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, ard_df['Relevance'].max() * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    plt.tight_layout()
    filepath = f'{save_dir}/gp_ard_relevance.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {filepath}")
    plt.close()


# ============================================================================
# SALVATAGGIO MODELLO
# ============================================================================

def save_model(
    gp: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    metrics: dict,
    ard_df: pd.DataFrame,
    save_dir: str = 'results'
) -> None:
    """
    Salva modello + scaler + metriche.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Salva GP
    joblib.dump(gp, f'{save_dir}/gp_model.pkl')
    print(f"✓ GP salvato: {save_dir}/gp_model.pkl")
    
    # Salva scalers
    joblib.dump(scaler_X, f'{save_dir}/scaler_X.pkl')
    joblib.dump(scaler_y, f'{save_dir}/scaler_y.pkl')
    print(f"✓ Scalers salvati: {save_dir}/scaler_*.pkl")
    
    # Salva metriche
    metrics_clean = {k: v for k, v in metrics.items() 
                     if not k.startswith('y_')}  # Rimuovi array grandi
    pd.DataFrame([metrics_clean]).to_csv(
        f'{save_dir}/metrics.csv', index=False
    )
    print(f"✓ Metriche salvate: {save_dir}/metrics.csv")
    
    # Salva ARD
    ard_df.to_csv(f'{save_dir}/ard_relevance.csv', index=False)
    print(f"✓ ARD salvato: {save_dir}/ard_relevance.csv")


def load_model(save_dir: str = 'results') -> Tuple:
    """
    Carica modello + scalers.
    """
    gp = joblib.load(f'{save_dir}/gp_model.pkl')
    scaler_X = joblib.load(f'{save_dir}/scaler_X.pkl')
    scaler_y = joblib.load(f'{save_dir}/scaler_y.pkl')
    
    print(f"✓ Modello caricato da: {save_dir}/")
    
    return gp, scaler_X, scaler_y


# ============================================================================
# MAIN: Pipeline completa training
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GAUSSIAN PROCESS REGRESSION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # ========== 1. Carica dataset ==========
    X, y = load_dataset('data/dataset_outreach.csv')
    
    # ========== 2. Preprocessing ==========
    (X_train_sc, X_test_sc, 
     y_train_sc, y_test_sc,
     scaler_X, scaler_y) = prepare_data(X, y, test_size=0.2, random_state=42)
    
    # ========== 3. Training GP ==========
    gp = train_gp(
        X_train_sc, 
        y_train_sc,
        kernel_type='matern52',  # final kernel for inverse optimization stability
        n_restarts=10,
        verbose=True
    )
    
    # ========== 4. Valutazione ==========
    metrics = evaluate_model(
        gp, 
        X_train_sc, y_train_sc,
        X_test_sc, y_test_sc,
        scaler_y
    )
    
    # ========== 5. ARD Analysis ==========
    ard_df = analyze_ard_relevance(gp)
    
    # ========== 6. Plot ==========
    print("\nGenerazione plot...")
    plot_training_results(
        scaler_y.inverse_transform(y_train_sc.reshape(-1, 1)).ravel(),
        metrics['y_train_pred'],
        metrics['y_train_std'],
        scaler_y.inverse_transform(y_test_sc.reshape(-1, 1)).ravel(),
        metrics['y_test_pred'],
        metrics['y_test_std'],
        metrics,
        save_dir='figures'
    )
    
    plot_ard_relevance(ard_df, save_dir='figures')
    
    # ========== 7. Salvataggio ==========
    save_model(gp, scaler_X, scaler_y, metrics, ard_df, save_dir='results')
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("TRAINING COMPLETATO!")
    print("="*70)
    print(f"\n📊 METRICHE FINALI:")
    print(f"  Test R²:   {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.6f} m")
    print(f"  Test MAE:  {metrics['test_mae']:.6f} m")
    
    print(f"\n🔍 TOP 3 PARAMETRI INFLUENTI:")
    for i in range(3):
        print(f"  {i+1}. {ard_df.iloc[i]['Parameter']:12s} "
              f"(relevance: {ard_df.iloc[i]['Relevance']:.3f})")
    
    print(f"\n📁 OUTPUT:")
    print(f"  Modello:   results/gp_model.pkl")
    print(f"  Scalers:   results/scaler_*.pkl")
    print(f"  Metriche:  results/metrics.csv")
    print(f"  ARD:       results/ard_relevance.csv")
    print(f"  Plot:      figures/gp_*.png")
    
    print("\n✅ Prossimo step: src/optimization.py (problema inverso)")
    print("="*70 + "\n")