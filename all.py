import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. NATURE PUBLISHING GROUP (NPG) TYPOGRAPHY & VISUAL SPECIFICATIONS
# ==============================================================================
# Nature standard single-column width is 89 mm (~3.5 inches)
MM_TO_INCHES = 1 / 25.4
FIG_WIDTH = 89 * MM_TO_INCHES  # 89 mm
FIG_HEIGHT = 68 * MM_TO_INCHES # 68 mm

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'font.size': 7,             # Nature standard body text size (7 pt)
    'axes.labelsize': 8,        # Nature axis label size (8 pt)
    'axes.titlesize': 8,        # Nature title size (8 pt bold)
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.titlesize': 9,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'pdf.fonttype': 42,         # Editable text for Adobe Illustrator export
    'ps.fonttype': 42
})

# Nature Palette: High-contrast primary navy, muted slate grid
PRIMARY_COLOR = '#003366'
POINT_COLOR = '#001F3F'
GRID_COLOR = '#E5E5E5'

# ==============================================================================
# 2. DATA & LOG-LINEAR FIT
# ==============================================================================
N_total = np.array([2500, 8200, 19500, 42000])
R2_pooled = np.array([0.620, 0.585, 0.541, 0.472])

log_N = np.log(N_total)
alpha_aggregate, intercept = np.polyfit(log_N, R2_pooled, 1)

# Dense grid for continuous plotting
N_dense = np.geomspace(2000, 50000, 300)
R2_fit = alpha_aggregate * np.log(N_dense) + intercept

# ==============================================================================
# 3. PLOTTING (NATURE SINGLE-COLUMN FORMAT)
# ==============================================================================
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

# Fitted scaling line
ax.plot(
    N_dense, 
    R2_fit, 
    color=PRIMARY_COLOR, 
    linewidth=1.2, 
    zorder=3,
    label=f'Anti-scaling law ($\\alpha = {alpha_aggregate:.4f}$)'
)

# Empirical data points
ax.scatter(
    N_total, 
    R2_pooled, 
    color=POINT_COLOR, 
    s=18, 
    edgecolor='white', 
    linewidth=0.5, 
    zorder=4, 
    label='Empirical backtest'
)

# Axis scale settings
ax.set_xscale('log')
ax.set_xlim(2000, 50000)
ax.set_ylim(0.40, 0.70)

# Visually equal log-spaced X-ticks (powers of 2)
x_ticks_even = [2000, 4000, 8000, 16000, 32000]
x_labels_even = ['2k', '4k', '8k', '16k', '32k']
ax.set_xticks(x_ticks_even)
ax.set_xticklabels(x_labels_even)
ax.minorticks_off()

# Y-axis ticks
ax.set_yticks(np.arange(0.40, 0.71, 0.05))

# Nature-style minimalist grid & spines
ax.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.4, color=GRID_COLOR, zorder=1)

# Remove top and right spines (Nature editorial requirement)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)

# Labels (Sentence case per Nature style guide)
ax.set_xlabel('Pooled training size $N_{\\text{Africa}}$', labelpad=4)
ax.set_ylabel('Out-of-sample holdout $R^2$', labelpad=4)

# Legend formatting
ax.legend(
    loc='lower left', 
    frameon=False, 
    handletextpad=0.4, 
    borderpad=0.2
)

# Tight layout tailored for publication pipeline
plt.tight_layout(pad=0.5)

# Save high-res raster (PNG) and vector (PDF) formats
plt.savefig('all_africa_scaling_law_nature.png', dpi=600)
plt.savefig('all_africa_scaling_law_nature.pdf', format='pdf')
plt.show()