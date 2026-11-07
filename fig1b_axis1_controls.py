import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

MM_TO_INCHES = 1 / 25.4
DOUBLE_COL_WIDTH = 180 * MM_TO_INCHES
FIG_HEIGHT = 72 * MM_TO_INCHES

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42
})

# Real results from axis1_robustness.py, axis1_multi_holdout.py (no synthetic fallback)
for p in ("cache/axis1_robust.json", "cache/axis1_multi_holdout.json", "cache/axis1_multi_holdout_preds.npz"):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found. Run axis1_robustness.py and axis1_multi_holdout.py first.")
with open("cache/axis1_robust.json") as f:
    robust = json.load(f)
with open("cache/axis1_multi_holdout.json") as f:
    multi = json.load(f)
preds = np.load("cache/axis1_multi_holdout_preds.npz", allow_pickle=True)

MODELS = ['Ridge', 'RandomForest', 'SVR']
style_map = {
    'Ridge': {'color': '#1A365D', 'marker': 'o'},
    'RandomForest': {'color': '#2B5C8F', 'marker': 's'},
    'SVR': {'color': '#E53E3E', 'marker': '^'}
}

# ---- baseline paired difference (oldest minus newest window, full 2015+ holdout) ----
N_BOOT, SEED = 2000, 20260103
y, country = preds['y'], preds['country']
countries = np.unique(country)
k = len(countries)
c_idx = np.searchsorted(countries, country)
rng = np.random.default_rng(SEED)
W = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(N_BOOT)]).astype(float)
first, last = robust['baseline']['windows'][0], robust['baseline']['windows'][-1]


def r2_boot(pred):
    err2 = (y - pred) ** 2
    st = [np.bincount(c_idx, minlength=k).astype(float),
          np.bincount(c_idx, weights=y, minlength=k),
          np.bincount(c_idx, weights=y ** 2, minlength=k),
          np.bincount(c_idx, weights=err2, minlength=k)]
    n, sy, syy, sse = (W @ s for s in st)
    pt = [s.sum() for s in st]
    return 1 - sse / (syy - sy ** 2 / n), 1 - pt[3] / (pt[2] - pt[1] ** 2 / pt[0])


forest = {}   # model -> list of (label, diff, lo, hi)
for m in MODELS:
    b_new, p_new = r2_boot(preds[f"{first}|{m}"])
    b_old, p_old = r2_boot(preds[f"{last}|{m}"])
    d = b_old - b_new
    rows = [("Cumulative windows\n(more rows + older data)", p_old - p_new, *np.percentile(d, [2.5, 97.5]))]
    e = robust['equalN']['models'][m]
    rows.append(("Equal N = %s rows\n(older data only)" % f"{robust['equalN']['n']:,}",
                 e['diff_oldest_minus_newest'], *e['diff_ci']))
    a = robust['ageBlocks']['models'][m]
    rows.append(("Single 5-yr blocks, N = %s\n(1992–99 vs 2010–14)" % f"{robust['ageBlocks']['n']:,}",
                 a['diff_oldest_minus_newest'], *a['diff_ci']))
    forest[m] = rows

fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, FIG_HEIGHT), dpi=300,
                               gridspec_kw={'width_ratios': [1.25, 1]})

# ---- Panel A: forest plot of paired R2 differences ----
labels = [r[0] for r in forest['Ridge']]
ypos = np.arange(len(labels))[::-1]
offsets = {'Ridge': 0.2, 'RandomForest': 0.0, 'SVR': -0.2}
for m in MODELS:
    st = style_map[m]
    for (lab, dv, lo, hi), yp in zip(forest[m], ypos):
        axA.errorbar(dv, yp + offsets[m], xerr=[[dv - lo], [hi - dv]], fmt=st['marker'],
                     color=st['color'], markersize=3.5, elinewidth=0.8, capsize=1.5, capthick=0.6,
                     markeredgecolor='white', markeredgewidth=0.4, label=m if yp == ypos[0] else None)
axA.axvline(0, color='#718096', linewidth=0.6, linestyle='--', zorder=0)
axA.set_yticks(ypos)
axA.set_yticklabels(labels, fontsize=6)
axA.set_xlabel(r"$\Delta R^2$ (oldest $-$ newest training data), 95% CI")
axA.set_title("(a) Drop persists at equal sample size", fontsize=8, fontweight='bold', loc='left', pad=6)
axA.legend(fontsize=6, frameon=False, loc='lower right')

# ---- Panel B: slope across separate future test periods ----
periods = list(multi['periods'].keys())
xpos = np.arange(len(periods))
for m in MODELS:
    st = style_map[m]
    vals = [multi['periods'][p]['models'][m] for p in periods]
    sl = np.array([v['slope'] for v in vals])
    lo = np.array([v['slope_ci'][0] for v in vals])
    hi = np.array([v['slope_ci'][1] for v in vals])
    axB.errorbar(xpos + offsets[m] * 0.8, sl, yerr=[sl - lo, hi - sl], fmt=st['marker'] + '-',
                 color=st['color'], markersize=3.5, linewidth=0.8, elinewidth=0.8, capsize=1.5,
                 capthick=0.6, markeredgecolor='white', markeredgewidth=0.4, alpha=0.9)
axB.axhline(0, color='#718096', linewidth=0.6, linestyle='--', zorder=0)
axB.set_xticks(xpos)
axB.set_xticklabels([f"{p}\n({multi['periods'][p]['n_countries']} countries)" for p in periods], fontsize=6)
axB.set_xlim(-0.5, len(periods) - 0.5)
axB.set_xlabel("Test period (training data ends 2014)")
axB.set_ylabel(r"Scaling slope $\alpha$, 95% CI")
axB.set_title("(b) Same pattern in every test period", fontsize=8, fontweight='bold', loc='left', pad=6)

for ax in (axA, axB):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.spines['left'].set_color('#CBD5E0')
    ax.grid(True, linestyle=':', alpha=0.35, color='#A0AEC0', linewidth=0.4)

plt.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.86, wspace=0.42)
plt.savefig('Figure_Axis1_Controls.pdf', format='pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('Figure_Axis1_Controls.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close()

print("Rendered Figure_Axis1_Controls.pdf and .png successfully.")
