import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

MM_TO_INCHES = 1 / 25.4
SINGLE_COL_WIDTH = 90 * MM_TO_INCHES
FIG_HEIGHT = 85 * MM_TO_INCHES

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42
})

cache_path = "cache/axis1_robust.json"

# Real results + holdout-country bootstrap CIs from axis1_robustness.py; no synthetic fallback.
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"{cache_path} not found. Run axis1_robustness.py first.")
with open(cache_path) as f:
    robust = json.load(f)['baseline']
model_data = robust['models']
N_windows = np.array(robust['N'])

# Shared y-range across Figs 1-3 (set to None to autoscale / zoom).
Y_LIM = (0.2, 0.8)
# Per-window CIs share the same holdout-country noise, so they overlap even when the paired
# slope is clearly negative; off by default (slope CI is in the legend, paired diffs in the controls figure).
SHOW_POINT_CI = False

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT), dpi=300)

style_map = {
    'Ridge': {'color': '#1A365D', 'ls': '-', 'marker': 'o'},
    'RandomForest': {'color': '#2B5C8F', 'ls': '--', 'marker': 's'},
    'SVR': {'color': '#E53E3E', 'ls': ':', 'marker': '^'}
}

for mname in ['Ridge', 'RandomForest', 'SVR']:
    d = model_data[mname]
    N = N_windows
    R2 = np.array(d['R2'])
    lo = np.array([c[0] for c in d['ci']])
    hi = np.array([c[1] for c in d['ci']])

    slope = d['slope']
    s_lo, s_hi = d['slope_ci']
    intercept = np.polyfit(np.log(N), R2, 1)[1]

    N_dense = np.geomspace(N.min(), N.max(), 100)
    R2_fit = slope * np.log(N_dense) + intercept

    st = style_map[mname]
    # t and p from an OLS fit of R^2 on log N through the plotted points (as in the original figures)
    ols = stats.linregress(np.log(N), R2)
    t_stat = ols.slope / ols.stderr if ols.stderr else float('nan')
    label_str = (f"{mname} ($\\alpha={slope:+.3f}$ [{s_lo:+.3f}, {s_hi:+.3f}], "
                 f"$t={t_stat:+.2f}$, $p={ols.pvalue:.3f}$)")

    ax.plot(N_dense, R2_fit, color=st['color'], linestyle=st['ls'], linewidth=1.2,
            alpha=0.85, label=label_str, zorder=3)
    if SHOW_POINT_CI:
        ax.errorbar(N, R2, yerr=[R2 - lo, hi - R2], fmt='none', ecolor=st['color'],
                    elinewidth=0.6, capsize=1.5, capthick=0.6, alpha=0.45, zorder=2)
    ax.scatter(N, R2, color=st['color'], marker=st['marker'], s=14,
               edgecolor='white', linewidth=0.5, zorder=4)

ax.set_title("Axis 1: Temporal Horizon Scaling", fontsize=8.5, fontweight='bold', loc='left', pad=6)
ax.set_xlabel(r"Pooled training size $N_{\mathrm{Africa}}$", labelpad=4)
ax.set_ylabel(r'Holdout $R^2$', labelpad=4)
ax.set_xscale('log')
if Y_LIM is not None:
    ax.set_ylim(*Y_LIM)
ax.set_xticks([20000, 25000, 30000, 35000])
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())

# Label each point with its training window (labels come from the robustness cache)
windows = [w.replace('\u20132010', '\u201310') for w in robust['windows']]
ref = list(N_windows)
top = ax.secondary_xaxis('top')
top.set_xticks(ref)
top.set_xticklabels(windows, rotation=30, ha='left', rotation_mode='anchor', fontsize=6, color='#4A5568')
top.xaxis.set_minor_locator(plt.NullLocator())
top.tick_params(axis='x', length=2.5, width=0.5, color='#A0AEC0', pad=2)
top.set_xlabel("Training window", fontsize=6.5, color='#4A5568', labelpad=1)
top.spines['top'].set_visible(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CBD5E0')
ax.spines['left'].set_color('#CBD5E0')
ax.grid(True, linestyle=':', alpha=0.35, color='#A0AEC0', linewidth=0.4)

ax.legend(fontsize=5.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)

plt.subplots_adjust(left=0.18, bottom=0.30, right=0.95, top=0.90)
plt.savefig('Figure_Axis1_Temporal.pdf', format='pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('Figure_Axis1_Temporal.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close()

print("Rendered Figure_Axis1_Temporal.pdf and .png successfully.")