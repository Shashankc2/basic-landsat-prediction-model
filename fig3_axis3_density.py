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

cache_path = "cache/axis3_robust.json"

# Real results + country-bootstrap CIs from axes23_robustness.py; no synthetic fallback.
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"{cache_path} not found. Run axes23_robustness.py first.")
with open(cache_path) as f:
    robust = json.load(f)
model_data = robust['models']

# Shared y-range across Figs 1-3 (set to None to autoscale / zoom).
Y_LIM = (0.2, 0.8)
# Light per-point 95% CIs (country bootstrap); turn off to match Fig 1 exactly.
SHOW_POINT_CI = True

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, FIG_HEIGHT), dpi=300)

style_map = {
    'Ridge': {'color': '#1A365D', 'ls': '-', 'marker': 'o'},
    'RandomForest': {'color': '#2B5C8F', 'ls': '--', 'marker': 's'},
    'SVR': {'color': '#E53E3E', 'ls': ':', 'marker': '^'}
}

for mname in ['Ridge', 'RandomForest', 'SVR']:
    d = model_data[mname]
    N = np.array(d['N'])
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

ax.set_title("Axis 3: Point Density Scaling", fontsize=8.5, fontweight='bold', loc='left', pad=8)
ax.set_xlabel(r"Within-country sample $N_{\mathrm{within}}$", labelpad=4)
ax.set_ylabel(r'Holdout $R^2$', labelpad=4)
ax.set_xscale('log')
if Y_LIM is not None:
    ax.set_ylim(*Y_LIM)
ax.set_xticks([50, 100, 200, 500])
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CBD5E0')
ax.spines['left'].set_color('#CBD5E0')
ax.grid(True, linestyle=':', alpha=0.35, color='#A0AEC0', linewidth=0.4)

ax.legend(fontsize=5.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)

plt.subplots_adjust(left=0.18, bottom=0.30, right=0.95, top=0.90)
plt.savefig('Figure_Axis3_Density.pdf', format='pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('Figure_Axis3_Density.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close()

print("Rendered Figure_Axis3_Density.pdf and .png successfully.")