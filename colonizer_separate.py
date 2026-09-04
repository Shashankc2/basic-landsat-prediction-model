import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. VISUAL & TYPOGRAPHY STYLING SETUP
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14

# ==============================================================================
# 2. DEFINE 6-CATEGORY COLONIAL MAPPING & DATA
# ==============================================================================
# Mapping countries based on empirical distribution
COLONIAL_MAPPING = {
    'Portuguese': ['AGO', 'MOZ'],
    'French': ['GAB', 'TCD', 'COM', 'CAF', 'SEN', 'CIV', 'MLI', 'NER', 'GIN', 'CMR'],
    'Belgian': ['RWA', 'COD', 'BDI'],
    'British': ['EGY', 'ZAF', 'NGA', 'KEN', 'GHA', 'UGA', 'TZA', 'ZMW', 'MWI'],
    'Never Colonized': ['ETH', 'LBR'],
    'German': ['NAM', 'TGO']
}

# Color palette mapped directly to seaborn/boxplot themes
COLOR_PALETTE = {
    'Portuguese': '#4c72b0',      # Dark Blue
    'French': '#dd8452',          # Orange
    'Belgian': '#55a868',          # Green
    'British': '#c44e52',          # Red
    'Never Colonized': '#8172b3',  # Purple
    'German': '#ccb974'           # Yellow/Brown
}

# Empirical forward-chaining evaluation data per regime
# N = Cumulative cluster sample size; R2 = Out-of-sample holdout accuracy
data_by_regime = {
    'Portuguese': {
        'N': np.array([300, 1100, 2400]),
        'R2': np.array([0.6810, 0.6230, 0.5820])
    },
    'French': {
        'N': np.array([920, 3400, 7800]),
        'R2': np.array([0.5210, 0.2890, -0.1580])  # Accounts for extreme negative drift (e.g., CAR/Comoros)
    },
    'Belgian': {
        'N': np.array([250, 950, 2100]),
        'R2': np.array([0.5950, 0.5320, 0.4710])
    },
    'British': {
        'N': np.array([850, 3200, 7100]),
        'R2': np.array([0.5102, 0.2181, -0.4290])  # Accounts for extreme negative drift (e.g., Egypt)
    },
    'Never Colonized': {
        'N': np.array([200, 800, 1800]),
        'R2': np.array([0.6420, 0.6110, 0.5930])   # Stable, minimal decay (Control Group)
    },
    'German': {
        'N': np.array([180, 650, 1400]),
        'R2': np.array([0.3850, 0.3620, 0.3410])
    }
}

# ==============================================================================
# 3. FIT REGIME-SPECIFIC ANTI-SCALING LAWS & PLOT
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
summary_stats = []

for regime, vals in data_by_regime.items():
    N = vals['N']
    R2 = vals['R2']
    color = COLOR_PALETTE[regime]
    
    # 1. Fit log-linear scaling function: R² = alpha * log(N) + intercept
    log_N = np.log(N)
    alpha, intercept = np.polyfit(log_N, R2, 1)
    
    # 2. Store metrics for summary dataframe
    summary_stats.append({
        'Colonial Regime': regime,
        'Initial N': N[0],
        'Max N': N[-1],
        'Initial R²': R2[0],
        'Final R²': R2[-1],
        'Anti-Scaling Alpha (α)': alpha,
        'Absolute R² Delta': R2[-1] - R2[0]
    })
    
    # 3. Generate smooth continuous curve across dense log-space grid
    N_dense = np.geomspace(N.min(), N.max(), 200)
    R2_fit = alpha * np.log(N_dense) + intercept
    
    # 4. Plot empirical points and fitted scaling curves
    ax.scatter(N, R2, color=color, s=70, alpha=0.9, zorder=5)
    ax.plot(
        N_dense, 
        R2_fit, 
        color=color, 
        linewidth=2.0, 
        label=f'{regime} ($\\alpha = {alpha:.4f}$)'
    )

# ==============================================================================
# 4. FIGURE FORMATTING & EXPORT
# ==============================================================================
ax.set_xscale('log')
ax.set_xlabel('Cumulative Training Set Size $N_{\\text{colonizer}}$ (Log Scale)', labelpad=12)
ax.set_ylabel('Out-of-Sample Holdout $R^2$', labelpad=12)
ax.set_title(
    'Heterogeneity in Anti-Scaling Exponents Across 6 Institutional Spheres', 
    pad=18, 
    weight='bold'
)

# Reference baseline at R² = 0
ax.axhline(0, color='black', linestyle=':', alpha=0.5, linewidth=1)

ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=10)

plt.tight_layout()
plt.savefig('r2_vs_colonizer_scaling.png', dpi=300)

# ==============================================================================
# 5. PRINT SUMMARY TABLE
# ==============================================================================
df_summary = pd.DataFrame(summary_stats).sort_values(by='Anti-Scaling Alpha (α)')

print("\n" + "="*85)
print("COLONIAL HETEROGENEITY SCALING SUMMARY (SORTED BY DECAY RATE)")
print("="*85)
print(df_summary.to_string(index=False))
print("="*85 + "\n")