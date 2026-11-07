import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

MM_TO_INCHES = 1 / 25.4
DOUBLE_COL_WIDTH = 180 * MM_TO_INCHES
FIG_HEIGHT = 65 * MM_TO_INCHES

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7.5,
    'axes.linewidth': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42
})

cache_path = "cache/gdp_metrics.json"

# Real results written by build_figure_caches.py (RandomForest LOCO run); no synthetic fallback.
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"{cache_path} not found. Run compCache.py / build_figure_caches.py first.")
with open(cache_path) as f:
    gdp_data = json.load(f)
years = gdp_data['years']
data_by_year = gdp_data['data_by_year']

fig, axes = plt.subplots(
    1, len(years),
    figsize=(DOUBLE_COL_WIDTH, FIG_HEIGHT),
    sharey=True,
    dpi=300
)

for idx, year in enumerate(years):
    ax = axes[idx]
    y_data = data_by_year[str(year)]

    x = np.array(y_data['log_gdp'])
    y = np.array(y_data['holdout_r2'])

    inliers = y >= -0.50
    outliers = y < -0.50

    x_in, y_in = x[inliers], y[inliers]
    n = len(x_in)
    n_outliers = int(np.sum(outliers))

    # Linear regression on values within the displayed range
    res = stats.linregress(x_in, y_in)
    t_stat = res.slope / res.stderr if res.stderr not in (None, 0) else 0.0
    p_val = res.pvalue

    # 95% confidence interval for the estimated mean trend
    x_dense = np.linspace(x_in.min(), x_in.max(), 100)
    y_dense = res.slope * x_dense + res.intercept

    df = max(n - 2, 1)
    x_mean = np.mean(x_in)
    ss_x = np.sum((x_in - x_mean) ** 2)
    residuals = y_in - (res.slope * x_in + res.intercept)
    s_err = np.sqrt(np.sum(residuals ** 2) / df)

    se_fit = s_err * np.sqrt(
        1.0 / n + (x_dense - x_mean) ** 2 / ss_x
    )
    t_crit = stats.t.ppf(0.975, df=df)

    ci_lower = y_dense - t_crit * se_fit
    ci_upper = y_dense + t_crit * se_fit

    # In-range observations
    ax.scatter(
        x_in, y_in,
        s=10, color='#2B5C8F', alpha=0.65,
        edgecolor='none', zorder=3
    )

    # Out-of-range observations, marked at the lower plot boundary
    if n_outliers:
        ax.scatter(
            x[outliers], np.full(n_outliers, -0.485),
            marker='v', color='#C53030', s=20,
            edgecolor='white', linewidth=0.4,
            zorder=4, clip_on=True
        )

    # Trend and confidence band; axis limits clip the band naturally
    ax.fill_between(
        x_dense, ci_lower, ci_upper,
        color='#C53030', alpha=0.14,
        linewidth=0, zorder=1
    )
    ax.plot(
        x_dense, ci_lower,
        color='#C53030', linewidth=0.5,
        linestyle=':', alpha=0.75, zorder=2
    )
    ax.plot(
        x_dense, ci_upper,
        color='#C53030', linewidth=0.5,
        linestyle=':', alpha=0.75, zorder=2
    )
    ax.plot(
        x_dense, y_dense,
        color='#C53030', linewidth=1.1, zorder=3
    )

    title_str = (
        f"{year}\n"
        f"$t = {t_stat:+.2f}$, $p = {p_val:.3f}$\n"
        f"Excluded: {n_outliers}"
    )
    ax.set_title(
        title_str, fontsize=6.5, fontweight='bold',
        pad=5, color='#1A202C'
    )

    if idx == 0:
        ax.text(
            0.04, 0.96, 'Shaded area: 95% CI',
            transform=ax.transAxes,
            fontsize=5.2, color='#8B3030',
            va='top',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='white',
                edgecolor='#E8CACA',
                alpha=0.9,
                linewidth=0.4
            )
        )

    ax.set_xlabel(r'$\log_{10}(\mathrm{GDP\ p.c.})$', labelpad=3)
    if idx == 0:
        ax.set_ylabel(r'Holdout $R^2$', labelpad=4)

    ax.set_ylim(-0.50, 0.85)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.spines['left'].set_color('#CBD5E0')
    ax.grid(
        True, linestyle=':', alpha=0.4,
        color='#A0AEC0', linewidth=0.4
    )

fig.suptitle(
    'Holdout Model Accuracy vs. GDP per Capita (1995–2015), RandomForest',
    fontsize=9, fontweight='bold', x=0.5, y=0.98
)

plt.subplots_adjust(
    wspace=0.12, left=0.08, right=0.98,
    top=0.74, bottom=0.27
)

fig.text(
    0.5, 0.07,
    'Red triangles mark observations below −0.50 (placed at the plot boundary).',
    ha='center', va='center', fontsize=5.5, color='#4A5568'
)

plt.savefig(
    'Figure_GDP_Trend.pdf',
    format='pdf', bbox_inches='tight', pad_inches=0.03
)
plt.savefig(
    'Figure_GDP_Trend.png',
    dpi=600, bbox_inches='tight', pad_inches=0.03
)
plt.close()

print("Rendered Figure_GDP_Trend.pdf with 95% CIs and t-stats successfully.")