import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. NATURE PUBLISHING GROUP (NPG) TYPOGRAPHY & VISUAL SPECIFICATIONS
# ==============================================================================
MM_TO_INCHES = 1 / 25.4
FIG_WIDTH = 89 * MM_TO_INCHES  # 89 mm single-column width
FIG_HEIGHT = 68 * MM_TO_INCHES # 68 mm height

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

PRIMARY_COLOR = '#003366'
POINT_COLOR = '#001F3F'
GRID_COLOR = '#E5E5E5'

# ==============================================================================
# 2. WEIGHTED AGGREGATE DATA DERIVED FROM COLONIAL HETEROGENEITY MATRIX
# ==============================================================================
# N_total = Sum of cluster counts across all 6 regimes
# R2_pooled = Sample-weighted out-of-sample holdout accuracy
N_total = np.array([2700, 9500, 22600])
R2_pooled = np.array([0.5541, 0.2810, -0.1082])

# Log-linear fit: R² = alpha * log(N) + intercept
log_N = np.log(N_total)
alpha_aggregate, intercept = np.polyfit(log_N, R2_pooled, 1)

# Dense log-space grid for continuous curve
N_dense = np.geomspace(2000, 30000, 300)
R2_fit = alpha_aggregate * np.log(N_dense) + intercept

# ==============================================================================
# 3. PLOTTING (NATURE SINGLE-COLUMN FORMAT)
# ==============================================================================
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

# Fitted aggregate scaling curve
ax.plot(
    N_dense, 
    R2_fit, 
    color=PRIMARY_COLOR, 
    linewidth=1.2, 
    zorder=3,
    label=f'scaling value  ($\\alpha = {alpha_aggregate:.4f}$)'
)

# Empirical weighted backtest points
ax.scatter(
    N_total, 
    R2_pooled, 
    color=POINT_COLOR, 
    s=22, 
    edgecolor='white', 
    linewidth=0.5, 
    zorder=4, 
    label='Backtest points'
)

# Reference baseline at R² = 0
ax.axhline(0, color='#888888', linestyle=':', linewidth=0.6, alpha=0.7, zorder=2)

# Axis scale settings
ax.set_xscale('log')
ax.set_xlim(2000, 30000)
ax.set_ylim(-0.25, 0.70)

# Visually equal log-spaced X-ticks (powers of 2)
x_ticks_even = [2000, 4000, 8000, 16000]
x_labels_even = ['2k', '4k', '8k', '16k']
ax.set_xticks(x_ticks_even)
ax.set_xticklabels(x_labels_even)
ax.minorticks_off()

# Clean Y-axis ticks
ax.set_yticks(np.arange(-0.20, 0.71, 0.20))

# Nature-style grid & spines
ax.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.4, color=GRID_COLOR, zorder=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)

# Labels & Legend
ax.set_xlabel('Pooled training sample $N_{\\text{Africa}}$', labelpad=4)
ax.set_ylabel('Out-of-sample holdout $R^2$', labelpad=4)

ax.legend(
    loc='lower left', 
    frameon=False, 
    handletextpad=0.4, 
    borderpad=0.2
)

plt.tight_layout(pad=0.5)

plt.savefig('all_africa_scaling_law_nature.png', dpi=600)
plt.savefig('all_africa_scaling_law_nature.pdf', format='pdf')
plt.show()