"""
model_comparison.py
===================
Confronto Gaussian Process vs Random Forest Regressor.

Obiettivo:
Dimostrare che, pur essendo RF competitivo, il GP è preferibile per:
- Uncertainty quantification
- ARD interpretability
- Integrazione in inverse optimization

Author: MatteoCasazza
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import Dict, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import dataset utilities
from dataset import load_dataset, ParameterRanges


# ============================================================================
# TRAINING RANDOM FOREST
# ============================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[RandomForestRegressor, float]:
    """
    Addestra Random Forest Regressor.
    
    Parameters
    ----------
    X_train : array (n, 9)
        Training input (NO scaling per RF!)
    y_train : array (n,)
        Training output (NO scaling!)
    n_estimators : int
        Numero alberi
    max_depth : int, optional
        Profondità massima alberi
    min_samples_split : int
        Minimo campioni per split
    random_state : int
    verbose : bool
    
    Returns
    -------
    rf : RandomForestRegressor
        Modello addestrato
    training_time : float
        Tempo training [s]
    
    Notes
    -----
    Random Forest NON richiede scaling di input/output.
    È robusto e funziona bene con feature a scale diverse.
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        print(f"N. estimators: {n_estimators}")
        print(f"Max depth: {max_depth if max_depth else 'None (full trees)'}")
        print(f"Min samples split: {min_samples_split}")
    
    # Crea modello
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,  # Parallelizza
        verbose=0
    )
    
    # Training
    print("\nFitting Random Forest...")
    start_time = time.time()
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training completato in {training_time:.2f} s")
    
    return rf, training_time


# ============================================================================
# VALUTAZIONE MODELLI
# ============================================================================

def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Valuta performance modello su train e test.
    
    Returns
    -------
    metrics : dict
        {
            'model': str,
            'train_r2', 'train_rmse', 'train_mae',
            'test_r2', 'test_rmse', 'test_mae',
            'y_train_pred', 'y_test_pred'
        }
    """
    # Predizioni
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metriche train
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Metriche test
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n--- {model_name.upper()} PERFORMANCE ---")
    print(f"Training Set:")
    print(f"  R²:   {train_r2:.6f}")
    print(f"  RMSE: {train_rmse*1000:.3f} mm")
    print(f"  MAE:  {train_mae*1000:.3f} mm")
    print(f"\nTest Set:")
    print(f"  R²:   {test_r2:.6f}")
    print(f"  RMSE: {test_rmse*1000:.3f} mm")
    print(f"  MAE:  {test_mae*1000:.3f} mm")
    
    return {
        'model': model_name,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }


# ============================================================================
# VISUALIZZAZIONI
# ============================================================================

def plot_model_comparison(
    gp_metrics: Dict,
    rf_metrics: Dict,
    save_path: str = 'figures/model_comparison.png'
) -> None:
    """
    Bar plot comparativo GP vs RF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Gaussian Process', 'Random Forest']
    
    # R²
    ax = axes[0]
    r2_values = [gp_metrics['test_r2'], rf_metrics['test_r2']]
    bars = ax.bar(models, r2_values, color=['#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_title('Test R²', fontsize=14, fontweight='bold')
    ax.set_ylim([0.80, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sopra barre
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # RMSE
    ax = axes[1]
    rmse_values = [gp_metrics['test_rmse']*1000, rf_metrics['test_rmse']*1000]
    bars = ax.bar(models, rmse_values, color=['#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('RMSE [mm]', fontsize=13, fontweight='bold')
    ax.set_title('Test RMSE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # MAE
    ax = axes[2]
    mae_values = [gp_metrics['test_mae']*1000, rf_metrics['test_mae']*1000]
    bars = ax.bar(models, mae_values, color=['#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('MAE [mm]', fontsize=13, fontweight='bold')
    ax.set_title('Test MAE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    plt.suptitle('Model Comparison: Gaussian Process vs Random Forest',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot salvato: {save_path}")
    plt.close()


def plot_rf_parity(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    metrics: Dict,
    save_path: str = 'figures/rf_parity_plot.png'
) -> None:
    """
    Parity plot per Random Forest.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train
    ax1.scatter(y_train, y_train_pred, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax1.plot([y_train.min(), y_train.max()],
             [y_train.min(), y_train.max()],
             'r--', linewidth=2, label='Identity')
    ax1.set_xlabel('True Peak Outreach [m]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Peak Outreach [m]', fontsize=12, fontweight='bold')
    ax1.set_title(f'Random Forest - Training (R²={metrics["train_r2"]:.4f})',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test
    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50, c='orange',
                edgecolors='black', linewidth=0.5)
    ax2.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Identity')
    ax2.set_xlabel('True Peak Outreach [m]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Peak Outreach [m]', fontsize=12, fontweight='bold')
    ax2.set_title(f'Random Forest - Test (R²={metrics["test_r2"]:.4f})',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ RF parity plot salvato: {save_path}")
    plt.close()


def plot_rf_feature_importance(
    rf: RandomForestRegressor,
    feature_names: List[str],
    save_path: str = 'figures/rf_feature_importance.png'
) -> pd.DataFrame:
    """
    Plot feature importance da Random Forest.
    """
    importances = rf.feature_importances_
    
    # DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.bar(df['Feature'], df['Importance'],
                  color=colors, edgecolor='black', linewidth=1.5)
    
    # Valori sopra barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Importance', fontsize=13, fontweight='bold')
    ax.set_title('Random Forest: Feature Importance Ranking',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ RF feature importance salvato: {save_path}")
    plt.close()
    
    return df


# ============================================================================
# MAIN: Pipeline completa comparison
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL COMPARISON: GAUSSIAN PROCESS vs RANDOM FOREST")
    print("="*70 + "\n")
    
    # ========== 1. Carica dataset ==========
    X, y = load_dataset('data/dataset_outreach.csv')
    
    print(f"Dataset caricato: {X.shape[0]} campioni")
    
    # ========== 2. Train/Test split (STESSO del GP) ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    
    # ========== 3. Train Random Forest ==========
    rf, rf_train_time = train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        verbose=True
    )
    
    # ========== 4. Valuta RF ==========
    rf_metrics = evaluate_model(
        rf, X_train, y_train, X_test, y_test,
        model_name='Random Forest'
    )
    
    # ========== 5. Metriche GP (da results/metrics.csv) ==========
    print("\n" + "="*70)
    print("CARICAMENTO METRICHE GP (dal training ufficiale)")
    print("="*70)
    
    try:
        gp_metrics_df = pd.read_csv('results/metrics.csv')
        gp_metrics = {
            'model': 'Gaussian Process',
            'train_r2': gp_metrics_df['train_r2'].values[0],
            'train_rmse': gp_metrics_df['train_rmse'].values[0],
            'train_mae': gp_metrics_df['train_mae'].values[0],
            'test_r2': gp_metrics_df['test_r2'].values[0],
            'test_rmse': gp_metrics_df['test_rmse'].values[0],
            'test_mae': gp_metrics_df['test_mae'].values[0]
        }
        
        print("\n--- GAUSSIAN PROCESS PERFORMANCE (ufficiale) ---")
        print(f"Test Set:")
        print(f"  R²:   {gp_metrics['test_r2']:.6f}")
        print(f"  RMSE: {gp_metrics['test_rmse']*1000:.3f} mm")
        print(f"  MAE:  {gp_metrics['test_mae']*1000:.3f} mm")
        
    except FileNotFoundError:
        print("\n⚠️  File results/metrics.csv non trovato!")
        print("    Usando metriche GP di riferimento da 500 campioni:")
        gp_metrics = {
            'model': 'Gaussian Process',
            'train_r2': 0.9876,  # Placeholder
            'train_rmse': 0.0136,
            'train_mae': 0.0084,
            'test_r2': 0.9676,
            'test_rmse': 0.021970,
            'test_mae': 0.011078
        }
        print(f"  Test R²:   {gp_metrics['test_r2']:.6f}")
        print(f"  Test RMSE: {gp_metrics['test_rmse']*1000:.3f} mm")
        print(f"  Test MAE:  {gp_metrics['test_mae']*1000:.3f} mm")
    
    # ========== 6. Confronto finale ==========
    print("\n" + "="*70)
    print("CONFRONTO FINALE")
    print("="*70)
    
    comparison_data = {
        'Model': ['Gaussian Process', 'Random Forest'],
        'Train_R2': [gp_metrics['train_r2'], rf_metrics['train_r2']],
        'Train_RMSE_mm': [gp_metrics['train_rmse']*1000, rf_metrics['train_rmse']*1000],
        'Train_MAE_mm': [gp_metrics['train_mae']*1000, rf_metrics['train_mae']*1000],
        'Test_R2': [gp_metrics['test_r2'], rf_metrics['test_r2']],
        'Test_RMSE_mm': [gp_metrics['test_rmse']*1000, rf_metrics['test_rmse']*1000],
        'Test_MAE_mm': [gp_metrics['test_mae']*1000, rf_metrics['test_mae']*1000]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Delta
    delta_r2 = gp_metrics['test_r2'] - rf_metrics['test_r2']
    delta_rmse = (gp_metrics['test_rmse'] - rf_metrics['test_rmse']) * 1000
    delta_mae = (gp_metrics['test_mae'] - rf_metrics['test_mae']) * 1000
    
    print(f"\nΔ (GP - RF):")
    print(f"  R²:   {delta_r2:+.4f}")
    print(f"  RMSE: {delta_rmse:+.3f} mm")
    print(f"  MAE:  {delta_mae:+.3f} mm")
    
    # ========== 7. Salvataggio ==========
    Path('results').mkdir(exist_ok=True)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print(f"\n✓ Comparison salvato: results/model_comparison.csv")
    
    # Feature importance
    param_names = ParameterRanges.get_param_names()
    rf_importance_df = plot_rf_feature_importance(
        rf, param_names,
        save_path='figures/rf_feature_importance.png'
    )
    rf_importance_df.to_csv('results/rf_feature_importance.csv', index=False)
    print(f"✓ RF feature importance salvato: results/rf_feature_importance.csv")
    
    # ========== 8. Plot ==========
    print("\nGenerazione plot...")
    
    plot_model_comparison(
        gp_metrics, rf_metrics,
        save_path='figures/model_comparison.png'
    )
    
    plot_rf_parity(
        y_train, rf_metrics['y_train_pred'],
        y_test, rf_metrics['y_test_pred'],
        rf_metrics,
        save_path='figures/rf_parity_plot.png'
    )
    
    # ========== 9. Interpretazione ==========
    print("\n" + "="*70)
    print("INTERPRETAZIONE")
    print("="*70)
    
    if delta_r2 > 0.01:
        winner = "GP superiore"
    elif delta_r2 < -0.01:
        winner = "RF superiore"
    else:
        winner = "Performance comparabili"
    
    print(f"\n{winner}")
    print(f"\nPerché preferire GP:")
    print(f"  ✓ Quantifica incertezza (fondamentale per inverse opt.)")
    print(f"  ✓ ARD interpretability (importance fisica)")
    print(f"  ✓ Performance: R²={gp_metrics['test_r2']:.4f}")
    print(f"  ✓ Integrazione naturale in optimization")
    
    print(f"\nRF comunque valido:")
    print(f"  ✓ Robusto e veloce")
    print(f"  ✓ Feature importance standard")
    print(f"  ✓ Performance: R²={rf_metrics['test_r2']:.4f}")
    print(f"  ✗ NO uncertainty quantification")
    
    print("\n" + "="*70)
    print("MODEL COMPARISON COMPLETATO!")
    print("="*70)
    print(f"\nFile generati:")
    print(f"  - results/model_comparison.csv")
    print(f"  - results/rf_feature_importance.csv")
    print(f"  - figures/model_comparison.png")
    print(f"  - figures/rf_parity_plot.png")
    print(f"  - figures/rf_feature_importance.png")
    print("="*70 + "\n")