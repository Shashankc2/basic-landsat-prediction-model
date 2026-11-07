import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

MM_TO_INCHES = 1 / 25.4
DOUBLE_COL_WIDTH = 180 * MM_TO_INCHES
FIG_HEIGHT = 70 * MM_TO_INCHES

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42
})

cache_path = "cache/rolling_window_metrics.json"

# Real results written by rolling_window_cache.py; no synthetic fallback.
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"{cache_path} not found. Run rolling_window_cache.py first.")
with open(cache_path) as f:
    rw_data = json.load(f)
windows = rw_data['windows']
models_perf = rw_data['models']
models_sd = rw_data['sd']
x_idx = np.arange(len(windows))

fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, FIG_HEIGHT), dpi=300)

style_map = {
    'Ridge': {'color': '#1A365D', 'ls': '-', 'marker': 'o'},
    'RandomForest': {'color': '#2B5C8F', 'ls': '--', 'marker': 's'},
    'SVR': {'color': '#E53E3E', 'ls': ':', 'marker': '^'}
}

for mname in ['Ridge', 'RandomForest', 'SVR']:
    r2_series = models_perf[mname]
    st = style_map[mname]
    ax.plot(x_idx, r2_series, color=st['color'], linestyle=st['ls'], linewidth=1.3,
            marker=st['marker'], markersize=4, label=mname, zorder=3)
    sd = np.array(models_sd[mname])
    ax.fill_between(x_idx, np.array(r2_series) - sd, np.array(r2_series) + sd,
                    color=st['color'], alpha=0.15, linewidth=0, zorder=2)

ax.set_title("5-Year Rolling Window Model Performance Trajectories (1995–2015)", fontsize=9, fontweight='bold', pad=10, loc='left')
ax.set_xticks(x_idx)
ax.set_xticklabels(windows, fontsize=7.5)
ax.set_xlabel("5-Year Rolling Window Span", labelpad=4)
ax.set_ylabel(r'Holdout $R^2$', labelpad=6)
ax.set_ylim(0.45, 0.85)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CBD5E0')
ax.spines['left'].set_color('#CBD5E0')
ax.grid(True, linestyle=':', alpha=0.35, color='#A0AEC0', linewidth=0.4)

ax.legend(title="Model Family", fontsize=6.5, title_fontsize=7, frameon=False, loc='lower left')

plt.subplots_adjust(left=0.08, bottom=0.18, right=0.95, top=0.88)

plt.savefig('Figure_Rolling_Window.pdf', format='pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('Figure_Rolling_Window.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close()

print("Rendered Figure_Rolling_Window.pdf and .png successfully.")