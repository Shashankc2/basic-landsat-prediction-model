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
# EXPERIMENTAL PIPELINE: AXIS 3 (WITHIN-COUNTRY SCALING)
# ==============================================================================
def run_axis3_within_country(
    df_master, 
    feature_cols, 
    country_col='country',
    year_col='year',
    target_col='iwi', 
    fractions=[0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
    time_window=(2015, 2021),
    test_size=0.20,
    min_train_samples=50,
    random_state=42
):
    """
    Axis 3: Within-Country Out-of-Sample Scaling.
    Maintains a fixed local test set per country and scales training data from 10% up to 100%.
    """
    print(f"\nFiltering dataset for time window {time_window}...")
    time_mask = (df_master[year_col] >= time_window[0]) & (df_master[year_col] <= time_window[1])
    df_window = df_master[time_mask].dropna(subset=[target_col, country_col] + feature_cols).copy()
    
    country_counts = df_window[country_col].value_counts()
    min_total_required = int(min_train_samples / (1.0 - test_size))
    eligible_countries = country_counts[country_counts >= min_total_required].index.tolist()
    
    print("="*65)
    print(f"AXIS 3 WITHIN-COUNTRY SCALING ({time_window[0]}-{time_window[1]})")
    print(f"Total rows in window: {len(df_window)}")
    print(f"Eligible Countries (N >= {min_total_required}): {len(eligible_countries)}")
    print("="*65 + "\n")

    if len(eligible_countries) == 0:
        raise ValueError(f"No countries met the minimum threshold of {min_total_required} samples in window {time_window}.")

    country_alpha_list = []
    fractional_records = []

    for c in eligible_countries:
        df_c = df_window[df_window[country_col] == c]
        X_c = df_c[feature_cols].values
        y_c = df_c[target_col].values

        # Fix 20% local test set strictly held out for this country
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_c, y_c, test_size=test_size, random_state=random_state, shuffle=True
        )

        n_train_total = len(X_train_full)
        c_n_list = []
        c_r2_list = []

        # Scale 10% to 100% of local training pool
        for frac in fractions:
            n_sub = int(n_train_total * frac)
            if n_sub < 15:
                continue

            r2_runs = []
            for run_idx in range(5):
                seed = random_state + run_idx * 100 + int(frac * 1000)
                np.random.seed(seed)

                idx = np.random.choice(n_train_total, size=n_sub, replace=False)
                X_sub = X_train_full[idx]
                y_sub = y_train_full[idx]

                model = Ridge(alpha=1.0)
                model.fit(X_sub, y_sub)

                preds = model.predict(X_test)
                r2_runs.append(r2_score(y_test, preds))

            mean_r2 = np.mean(r2_runs)
            c_n_list.append(n_sub)
            c_r2_list.append(mean_r2)

            fractional_records.append({
                'country': c,
                'fraction': frac,
                'N_train': n_sub,
                'N_test': len(y_test),
                'R2': mean_r2
            })

        # Calculate per-country scaling exponent (alpha_c)
        if len(c_n_list) >= 3:
            slope_c, _ = np.polyfit(np.log(c_n_list), c_r2_list, 1)
            country_alpha_list.append({'country': c, 'alpha': slope_c, 'max_N': n_train_total})

    df_raw = pd.DataFrame(fractional_records)
    df_alphas = pd.DataFrame(country_alpha_list)

    # Aggregation across all countries at matching fractional steps
    df_summary = df_raw.groupby('fraction').agg(
        mean_N=('N_train', 'mean'),
        mean_R2=('R2', 'mean'),
        std_R2=('R2', 'std'),
        num_countries=('country', 'nunique')
    ).reset_index()

    alpha_global, _ = np.polyfit(np.log(df_summary['mean_N']), df_summary['mean_R2'], 1)
    mean_country_alpha = df_alphas['alpha'].mean()

    print(df_summary.to_string(index=False))
    print("\n" + "="*65)
    print(f"Mean Country Scaling Exponent (Mean α_within): {mean_country_alpha:.4f}")
    print(f"Global Aggregated Scaling Exponent (α_global):   {alpha_global:.4f}")
    print("="*65 + "\n")

    return df_summary, df_alphas, df_raw, alpha_global, mean_country_alpha

# ==============================================================================
# NATURE-STYLE PLOTTING WITH DATA POINT LABELS
# ==============================================================================
def plot_axis3_publication(df_summary, df_raw, alpha_global, mean_country_alpha, time_window=(2015, 2021)):
    """
    Renders Nature-style Figure for Axis 3 (Within-Country Scaling).
    Features individual country trajectories, annotated aggregated mean curve, and clean legend.
    """
    MM_TO_INCHES = 1 / 25.4
    FIG_WIDTH = 89 * MM_TO_INCHES
    FIG_HEIGHT = 70 * MM_TO_INCHES

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
        'pdf.fonttype': 42
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

    # 1. Background thin grey lines for individual countries
    countries = df_raw['country'].unique()
    for idx, c in enumerate(countries):
        df_c = df_raw[df_raw['country'] == c].sort_values('N_train')
        ax.plot(
            df_c['N_train'], df_c['R2'], 
            color='#B0BEC5', alpha=0.35, linewidth=0.6, zorder=2,
            label='Individual country' if idx == 0 else ""
        )

    # 2. Extract mean aggregated points
    N_arr = df_summary['mean_N'].values
    R2_arr = df_summary['mean_R2'].values
    std_arr = df_summary['std_R2'].values

    # Fit log-linear curve
    log_N = np.log(N_arr)
    slope, intercept = np.polyfit(log_N, R2_arr, 1)
    N_dense = np.geomspace(N_arr.min(), N_arr.max(), 200)
    R2_fit = slope * np.log(N_dense) + intercept

    # 3. Main fitted scaling line
    ax.plot(
        N_dense, R2_fit, 
        color='#1F77B4', linewidth=1.4, zorder=4, 
        label=f'Mean scaling ($\\bar{{\\alpha}} = {mean_country_alpha:.4f}$)'
    )

    # 4. Empirical aggregated mean points
    ax.scatter(
        N_arr, R2_arr, 
        color='#1F77B4', s=16, edgecolor='white', linewidth=0.5, zorder=5
    )

    # 5. ANNOTATE POINTS WITH FRACTION % LABELS
    frac_labels = ['10%', '20%', '40%', '60%', '80%', '100%']
    for x_i, y_i, label in zip(N_arr, R2_arr, frac_labels):
        ax.annotate(
            label,
            (x_i, y_i),
            textcoords="offset points",
            xytext=(0, 4),
            ha='center',
            fontsize=5.5,
            color='#1F77B4',
            fontweight='bold',
            zorder=6
        )

    # 6. Standard deviation error band
    ax.fill_between(
        N_arr, R2_arr - 0.5 * std_arr, R2_arr + 0.5 * std_arr,
        color='#1F77B4', alpha=0.15, zorder=3
    )

    # Formatting
    ax.set_xscale('log')
    ax.set_ylim(0.2, 0.85)

    ax.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=0.4, color='#E0E0E0', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.set_xlabel('Single-country training sample size $N_{\\text{within}}$', labelpad=4)
    ax.set_ylabel('In-country holdout asset index $R^2$', labelpad=4)

    # Title indicating the exact time window analyzed
    ax.set_title(f'Within-Country Scaling ({time_window[0]}–{time_window[1]})', loc='left', pad=4, fontsize=7.5, fontweight='bold')

    # Semi-transparent legend box to eliminate line-text collisions
    ax.legend(
        loc='lower right', 
        frameon=True, 
        facecolor='white', 
        edgecolor='none',
        framealpha=0.85,
        handletextpad=0.4, 
        borderpad=0.3
    )

    plt.tight_layout(pad=0.4)
    plt.savefig('axis3_within_country_scaling.png', dpi=600)
    plt.savefig('axis3_within_country_scaling.pdf', format='pdf')
    print("Saved updated labeled figure to axis3_within_country_scaling.png and .pdf")
    plt.close()

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parquet_path = 'master_merged_data.parquet'
    TIME_WINDOW = (2015, 2021)
    
    print(f"Loading real DHS Master Matrix from {parquet_path}...")
    df_master = pd.read_parquet(parquet_path)

    year_candidates = ['year', 'survey_year', 'DHSYEAR', 'survey_year_dhs', 'Year']
    year_col = find_column_name(df_master, year_candidates, "survey year")

    country_candidates = ['country', 'country_iso', 'ISO', 'country_code', 'Country']
    country_col = find_column_name(df_master, country_candidates, "country ISO")

    target_candidates = ['iwi', 'asset_index', 'wealth_index', 'wealthindex', 'DHSWALTH', 'target', 'y']
    target_col = find_column_name(df_master, target_candidates, "target asset index (IWI)")

    feature_cols = [col for col in df_master.columns if col.startswith('F_')]
    if len(feature_cols) == 0:
        metadata_cols = {
            'Unnamed: 0', 'cluster_id', 'lon', 'lat', 'rural', 'region_id', 
            country_col, 'survey', 'month', year_col, target_col, 
            'address_osm', 'address_google', 'Lon', 'Lat', 'Invalid',
            'time_block', 'colonial_sphere', 'geometry'
        }
        numeric_cols = df_master.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in metadata_cols]

    print(f"Isolated {len(feature_cols)} satellite embedding feature columns.")

    # 1. Run Axis 3 Experiment
    summary, df_alphas, df_raw, alpha_global, mean_country_alpha = run_axis3_within_country(
        df_master=df_master,
        feature_cols=feature_cols,
        country_col=country_col,
        year_col=year_col,
        target_col=target_col,
        fractions=[0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
        time_window=TIME_WINDOW
    )

    # 2. Render and save updated labeled figure
    plot_axis3_publication(summary, df_raw, alpha_global, mean_country_alpha, time_window=TIME_WINDOW)