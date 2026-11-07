import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

MM_TO_INCHES = 1 / 25.4
DOUBLE_COL_WIDTH = 180 * MM_TO_INCHES
FIG_HEIGHT = 100 * MM_TO_INCHES

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42
})

fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, FIG_HEIGHT), dpi=300)

groups = ['Portuguese', 'French', 'Belgian', 'British', 'Never Colonized', 'German']
models = ['Ridge', 'RandomForest', 'SVR']

colors = {
    'Ridge': '#1A365D',        # Deep Navy
    'RandomForest': '#2B5C8F', # Medium Slate Blue
    'SVR': '#C53030'           # Deep Crimson
}

KEY_MAP = {
    'Uncolonized': 'Never Colonized',
    'Never Colonized': 'Never Colonized'
}

data = {m: {} for m in models}

cache_path = "cache/colonizer_metrics.json"

# Real per-model results written by build_figure_caches.py; no synthetic fallback.
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"{cache_path} not found. Run build_figure_caches.py first.")
with open(cache_path) as f:
    raw = json.load(f)
if 'by_model' not in raw:
    raise KeyError("colonizer_metrics.json has no 'by_model' key. Run build_figure_caches.py.")
for m in models:
    for g, val in raw['by_model'][m].items():
        data[m][KEY_MAP.get(g, g)] = val['r2_scores'] if isinstance(val, dict) else val

n_groups = len(groups)
n_models = len(models)
group_positions = np.arange(n_groups)
width = 0.22

for m_idx, mname in enumerate(models):
    pos = group_positions + (m_idx - 1) * width
    model_scores = [data[mname][g] for g in groups]
    
    bp = ax.boxplot(model_scores, positions=pos, patch_artist=True,
                    widths=0.18, showfliers=False,
                    boxprops=dict(linewidth=0.7, color='#2D3748', facecolor=colors[mname], alpha=0.65),
                    whiskerprops=dict(linewidth=0.7, color='#2D3748'),
                    capprops=dict(linewidth=0.7, color='#2D3748'),
                    medianprops=dict(linewidth=1.0, color='#1A202C'))
    
    # Scatter points per model group
    for i, g in enumerate(groups):
        vals = np.array(data[mname][g])
        np.random.seed(42 + i)
        jitter = np.random.normal(0, 0.02, size=len(vals))
        ax.scatter(pos[i] + jitter, vals, color=colors[mname], alpha=0.5, s=8, edgecolor='none', zorder=3)

# Dummy handles for clear model legend
for mname in models:
    ax.plot([], [], color=colors[mname], label=mname, linewidth=3, alpha=0.85)

ax.set_title("Model Accuracy Across Institutional History (Ridge vs. RF vs. SVR)", fontsize=10, fontweight='bold', pad=12, loc='center')
ax.set_xticks(group_positions)
ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
ax.set_ylabel(r'Holdout $R^2$', labelpad=8, fontsize=8.5)
ax.set_ylim(-7.5, 1.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CBD5E0')
ax.spines['left'].set_color('#CBD5E0')
ax.grid(True, axis='y', linestyle=':', alpha=0.4, color='#A0AEC0', linewidth=0.5)

ax.legend(title="Model Family", fontsize=7.5, title_fontsize=8, frameon=False, loc='lower left')

plt.subplots_adjust(left=0.12, bottom=0.22, right=0.95, top=0.90)

plt.savefig('Figure_Colonizer_R2.pdf', format='pdf', bbox_inches='tight', pad_inches=0.04)
plt.savefig('Figure_Colonizer_R2.png', dpi=600, bbox_inches='tight', pad_inches=0.04)
plt.close()

print("Rendered Figure_Colonizer_R2.pdf and .png successfully.")