import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# GLOBAL PUBLICATION SPECIFICATIONS
# ==============================================================================
MM_TO_INCHES = 1 / 25.4
SINGLE_COL_WIDTH = 88 * MM_TO_INCHES  # ~3.46 inches
FIGURE_HEIGHT = 65 * MM_TO_INCHES

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'lines.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def apply_tufte_style(ax):
    """Applies clean data-ink principles."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.3, color='#CCCCCC', zorder=1)

# ==============================================================================
# EMPIRICAL DATA
# ==============================================================================
MODEL_COLORS = {
    'Ridge': '#1F77B4',         # Muted Blue
    'RandomForest': '#2CA02C',  # Muted Green
    'SVR': '#D62728'            # Muted Red
}

DATA = {
    'Axis1': {
        'title': 'Axis 1: Temporal OOT',
        'xlabel': 'Pooled training size $N_{\\mathrm{Africa}}$',
        'xlim': (12000, 42000),
        'xtick_vals': [16000, 32000],
        'xtick_labels': ['16k', '32k'],
        'models': {
            'Ridge': {'N': np.array([17500, 27000, 33000, 35500]), 'R2': np.array([0.627, 0.615, 0.611, 0.604]), 'alpha': -0.028},
            'RandomForest': {'N': np.array([17500, 27000, 33000, 35500]), 'R2': np.array([0.628, 0.619, 0.612, 0.606]), 'alpha': -0.028},
            'SVR': {'N': np.array([17500, 27000, 33000, 35500]), 'R2': np.array([0.621, 0.614, 0.599, 0.592]), 'alpha': -0.035}
        }
    },
    'Axis2': {
        'title': 'Axis 2: Geographic OOC',
        'xlabel': 'Cross-border sample $N_{\\mathrm{OOC}}$ ($k$ countries)',
        'xlim': (400, 25000),
        'models': {
            'Ridge': {'N': np.array([550, 2100, 5200, 9200, 18500]), 'R2': np.array([0.305, 0.481, 0.449, 0.476, 0.492]), 'alpha': 0.046},
            'RandomForest': {'N': np.array([550, 2100, 5200, 9200, 18500]), 'R2': np.array([0.395, 0.518, 0.475, 0.511, 0.489]), 'alpha': 0.023},
            'SVR': {'N': np.array([550, 2100, 5200, 9200, 18500]), 'R2': np.array([0.233, 0.433, 0.459, 0.480, 0.502]), 'alpha': 0.073}
        }
    },
    'Axis3': {
        'title': 'Axis 3: Local Density',
        'xlabel': 'Within-country sample $N_{\\mathrm{within}}$',
        'xlim': (25, 600),
        'models': {
            'Ridge': {'N': np.array([40, 80, 180, 300, 420]), 'R2': np.array([0.584, 0.592, 0.661, 0.670, 0.686]), 'alpha': 0.042},
            'RandomForest': {'N': np.array([40, 80, 180, 300, 420]), 'R2': np.array([0.591, 0.632, 0.628, 0.640, 0.671]), 'alpha': 0.028},
            'SVR': {'N': np.array([40, 80, 180, 300, 420]), 'R2': np.array([0.301, 0.439, 0.561, 0.605, 0.612]), 'alpha': 0.128}
        }
    }
}

# Collision-free k-label config for Axis 2
AXIS2_K_CONFIG = [
    {'k': 1,  'xytext': (8, 0),   'ha': 'left',   'va': 'center'},   # Shifts right, clear of y-axis spine
    {'k': 3,  'xytext': (0, 8),   'ha': 'center', 'va': 'bottom'},  # Clear of data points
    {'k': 7,  'xytext': (0, -12), 'ha': 'center', 'va': 'top'},     # Clear below points
    {'k': 15, 'xytext': (0, 8),   'ha': 'center', 'va': 'bottom'},  # Clear above points
    {'k': 27, 'xytext': (6, 8),   'ha': 'left',   'va': 'bottom'}   # Floats above trendline
]

# ==============================================================================
# GENERATE INDIVIDUAL AXIS FIGURES
# ==============================================================================
def plot_axis_figure(axis_key):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIGURE_HEIGHT), dpi=300)
    cfg = DATA[axis_key]

    for model_name, model_data in cfg['models'].items():
        N = model_data['N']
        R2 = model_data['R2']
        alpha = model_data['alpha']
        color = MODEL_COLORS[model_name]
        
        # Fit log-linear trendline
        log_N = np.log(N)
        slope, intercept = np.polyfit(log_N, R2, 1)
        N_dense = np.geomspace(N.min(), N.max(), 100)
        
        label_text = f'{model_name} ($\\alpha = {alpha:+.3f}$)'
        ax.plot(N_dense, slope * np.log(N_dense) + intercept, color=color, label=label_text, zorder=2)
        ax.scatter(N, R2, color=color, s=14, edgecolor='white', linewidth=0.5, zorder=5)

    # Collision-free k-annotations for Axis 2
    if axis_key == 'Axis2':
        rf_N = cfg['models']['RandomForest']['N']
        rf_R2 = cfg['models']['RandomForest']['R2']
        for x_i, y_i, cfg_k in zip(rf_N, rf_R2, AXIS2_K_CONFIG):
            ax.annotate(
                f"$k={cfg_k['k']}$", 
                (x_i, y_i), 
                textcoords="offset points", 
                xytext=cfg_k['xytext'], 
                ha=cfg_k['ha'], 
                va=cfg_k['va'], 
                fontsize=6, 
                color='#333333',
                fontweight='bold'
            )

    # Axes-relative transform guarantees title remains pinned at top-left across vector PDF backends
    ax.text(0.0, 1.05, cfg['title'], transform=ax.transAxes,
            fontsize=8.5, fontweight='bold', ha='left', va='bottom')
    
    ax.set_xlabel(cfg['xlabel'], labelpad=4)
    ax.set_ylabel('Out-of-sample holdout $R^2$', labelpad=4)
    ax.set_xscale('log')
    ax.set_xlim(cfg['xlim'])
    ax.set_ylim(0.15, 0.82)
    
    # Custom x-ticks for Axis 1
    if 'xtick_vals' in cfg:
        ax.xaxis.set_major_locator(ticker.FixedLocator(cfg['xtick_vals']))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(cfg['xtick_labels']))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    apply_tufte_style(ax)
    
    # Position legends to avoid obscuring data points
    legend_loc = 'upper left' if axis_key == 'Axis1' else 'lower right'
    ax.legend(loc=legend_loc, frameon=False)

    # Export using bbox_inches='tight' without tight_layout() to prevent PDF text displacement
    out_prefix = f'Figure_{axis_key}_journal_ready'
    plt.savefig(f'{out_prefix}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(f'{out_prefix}.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"Successfully generated clean vector/raster pair: {out_prefix}.pdf and {out_prefix}.png")
    plt.close()

if __name__ == "__main__":
    for axis in ['Axis1', 'Axis2', 'Axis3']:
        plot_axis_figure(axis)