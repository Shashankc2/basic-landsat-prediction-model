import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def find_column_name(df, candidates, description):
    """Utility to resolve schema variations in DHS datasets."""
    for col in candidates:
        if col in df.columns:
            print(f"Mapped '{description}' to column: '{col}'")
            return col
    raise KeyError(f"Could not find a valid '{description}' column.")

# ==============================================================================
# CORE EXPERIMENTAL RUNNER FOR A SINGLE TIME WINDOW
# ==============================================================================
def run_within_country_window(df_master, feature_cols, year_col, country_col, target_col, time_window):
    """
    Axis 3 Within-Country Scaling for a specific historical time window.
    """
    time_mask = (df_master[year_col] >= time_window[0]) & (df_master[year_col] <= time_window[1])
    df_window = df_master[time_mask].dropna(subset=[target_col, country_col] + feature_cols).copy()
    
    country_counts = df_window[country_col].value_counts()
    min_total_required = int(50 / 0.80)  # Requires at least ~62 clusters
    eligible_countries = country_counts[country_counts >= min_total_required].index.tolist()
    
    if len(eligible_countries) == 0:
        print(f"Skipping window {time_window}: No countries with >= {min_total_required} clusters.")
        return None, None, None

    fractions = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    country_alpha_list = []
    fractional_records = []

    for c in eligible_countries:
        df_c = df_window[df_window[country_col] == c]
        X_c = df_c[feature_cols].values
        y_c = df_c[target_col].values

        # Fix 20% local test set strictly held out
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_c, y_c, test_size=0.20, random_state=42, shuffle=True
        )

        n_train_total = len(X_train_full)
        c_n_list, c_r2_list = [], []

        # Scale local training set from 10% to 100%
        for frac in fractions:
            n_sub = int(n_train_total * frac)
            if n_sub < 15:
                continue

            r2_runs = []
            for run_idx in range(3):
                np.random.seed(42 + run_idx * 100 + int(frac * 1000))
                idx = np.random.choice(n_train_total, size=n_sub, replace=False)
                
                model = Ridge(alpha=1.0)
                model.fit(X_train_full[idx], y_train_full[idx])
                preds = model.predict(X_test)
                r2_runs.append(r2_score(y_test, preds))

            mean_r2 = np.mean(r2_runs)
            c_n_list.append(n_sub)
            c_r2_list.append(mean_r2)

            fractional_records.append({
                'country': c, 'fraction': frac, 'N_train': n_sub, 'R2': mean_r2
            })

        # Calculate local scaling exponent
        if len(c_n_list) >= 3:
            slope_c, _ = np.polyfit(np.log(c_n_list), c_r2_list, 1)
            country_alpha_list.append({'country': c, 'alpha': slope_c})

    df_raw = pd.DataFrame(fractional_records)
    df_alphas = pd.DataFrame(country_alpha_list)

    df_summary = df_raw.groupby('fraction').agg(
        mean_N=('N_train', 'mean'),
        mean_R2=('R2', 'mean'),
        std_R2=('R2', 'std')
    ).reset_index()

    mean_country_alpha = df_alphas['alpha'].mean()
    return df_summary, mean_country_alpha, len(eligible_countries)

# ==============================================================================
# MULTI-ERA PLOTTING ROUTINE (CLEANED LAYOUT)
# ==============================================================================
def plot_multi_era_within_country(results_dict):
    """
    Renders Nature-style multi-era Within-Country scaling plot.
    Upper-left legend and elevated text offsets prevent all overlaps.
    """
    MM_TO_INCHES = 1 / 25.4
    FIG_WIDTH = 95 * MM_TO_INCHES
    FIG_HEIGHT = 75 * MM_TO_INCHES

    plt.rcParams.update({
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.0,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

    # Configurable offsets per curve to prevent line/text cutoffs
    style_map = {
        '2015–2021': {'color': '#1F77B4', 'marker': 'o', 'text_offset': (5, 0)},   # Blue
        '2010–2014': {'color': '#2CA02C', 'marker': 's', 'text_offset': (5, 0)},   # Green
        '2005–2009': {'color': '#FF7F0E', 'marker': '^', 'text_offset': (5, 10)}   # Orange (Shifted UP)
    }

    for label, res in results_dict.items():
        if res['summary'] is None:
            continue

        df_summary = res['summary']
        mean_alpha = res['alpha']
        n_countries = res['n_countries']
        cfg = style_map.get(label, {'color': '#333333', 'marker': 'o', 'text_offset': (5, 0)})

        N_arr = df_summary['mean_N'].values
        R2_arr = df_summary['mean_R2'].values
        std_arr = df_summary['std_R2'].values

        # Log-linear fit
        log_N = np.log(N_arr)
        slope, intercept = np.polyfit(log_N, R2_arr, 1)
        N_dense = np.geomspace(N_arr.min(), N_arr.max(), 200)
        R2_fit = slope * np.log(N_dense) + intercept

        # 1. Fitted curve
        ax.plot(
            N_dense, R2_fit, 
            color=cfg['color'], linewidth=1.3, zorder=4,
            label=f'{label} ($\\bar{{\\alpha}}={mean_alpha:.3f}$, $N={n_countries}$)'
        )

        # 2. Scatter points
        ax.scatter(
            N_arr, R2_arr, 
            color=cfg['color'], marker=cfg['marker'], s=14, edgecolor='white', linewidth=0.5, zorder=5
        )

        # 3. Lightened error shading (alpha=0.08)
        y_low = np.maximum(0.0, R2_arr - 0.5 * std_arr)
        y_high = R2_arr + 0.5 * std_arr
        ax.fill_between(
            N_arr, y_low, y_high,
            color=cfg['color'], alpha=0.08, zorder=3
        )

        # 4. Direct text annotation
        end_x = N_dense[-1]
        end_y = R2_fit[-1]
        ax.annotate(
            label,
            (end_x, end_y),
            textcoords="offset points",
            xytext=cfg['text_offset'],
            ha='left',
            va='center',
            fontsize=6.0,
            fontweight='bold',
            color=cfg['color'],
            zorder=6
        )

    # Set axes limits with padding
    ax.set_xscale('log')
    ax.set_ylim(0.08, 0.88)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)

    ax.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=0.4, color='#E0E0E0', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.set_xlabel('Training sample size $N_{\\text{within}}$', labelpad=4)
    ax.set_ylabel('In-country holdout asset index $R^2$', labelpad=4)
    ax.set_title('Within-Country Scaling Across Temporal Eras', loc='left', pad=4, fontsize=8, fontweight='bold')

    # Legend placed safely in upper left
    ax.legend(
        loc='upper left', 
        frameon=True, 
        facecolor='white', 
        edgecolor='none',
        framealpha=0.90,
        handletextpad=0.4, 
        borderpad=0.3
    )

    plt.tight_layout(pad=0.4)
    plt.savefig('axis3_multi_era_scaling.png', dpi=600)
    plt.savefig('axis3_multi_era_scaling.pdf', format='pdf')
    print("Saved clean multi-era plot to axis3_multi_era_scaling.png and .pdf")
    plt.close()

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parquet_path = 'master_merged_data.parquet'
    print(f"Loading real DHS Master Matrix from {parquet_path}...")
    df_master = pd.read_parquet(parquet_path)

    year_col = find_column_name(df_master, ['year', 'survey_year', 'DHSYEAR', 'Year'], "survey year")
    country_col = find_column_name(df_master, ['country', 'country_iso', 'ISO', 'Country'], "country ISO")
    target_col = find_column_name(df_master, ['iwi', 'asset_index', 'wealth_index', 'y'], "target asset index")

    feature_cols = [col for col in df_master.columns if col.startswith('F_')]
    if len(feature_cols) == 0:
        metadata_cols = {'Unnamed: 0', 'cluster_id', 'lon', 'lat', 'rural', 'region_id', country_col, year_col, target_col}
        feature_cols = [col for col in df_master.select_dtypes(include=[np.number]).columns if col not in metadata_cols]

    print(f"Isolated {len(feature_cols)} satellite embedding feature columns.")

    # Historical survey windows
    time_windows = {
        '2015–2021': (2015, 2021),
        '2010–2014': (2010, 2014),
        '2005–2009': (2005, 2009)
    }

    results = {}
    for label, window in time_windows.items():
        print(f"\n=================================================================")
        print(f"RUNNING WITHIN-COUNTRY SCALING FOR ERA: {label}")
        print(f"=================================================================")
        summary, mean_alpha, n_countries = run_within_country_window(
            df_master, feature_cols, year_col, country_col, target_col, window
        )
        results[label] = {'summary': summary, 'alpha': mean_alpha, 'n_countries': n_countries}

    # Render updated multi-era plot
    plot_multi_era_within_country(results)