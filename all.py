import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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
# CORE EXPERIMENTAL RUNNER (TEMPORAL FORWARD-CHAINING WITH MEDIANS & T-STATS)
# ==============================================================================
def run_axis1_out_of_time(df_master, feature_cols, year_col, country_col, target_col, model_name='Ridge', model_obj=None):
    """
    Axis 1 Out-of-Time Scaling using MEDIANS and OLS t-statistics.
    Evaluates how accumulating historical survey data affects holdout performance on modern surveys (>= 2015).
    """
    # Holdout set: Modern surveys (2015-2021)
    df_holdout = df_master[df_master[year_col] >= 2015].dropna(subset=[target_col] + feature_cols).copy()
    X_test = df_holdout[feature_cols].values
    y_test = df_holdout[target_col].values

    if len(df_holdout) == 0:
        raise ValueError("Holdout set (2015+) is empty! Check year column filtering.")

    # Historical cumulative training pools (expanding backwards in time)
    # 1. Modern only (2010-2014)
    # 2. Add 2005-2009
    # 3. Add 2000-2004
    # 4. Full historical pool (1990-2014)
    historical_cutoffs = [
        ('2010–2014', 2010, 2014),
        ('+ 2005–2009', 2005, 2014),
        ('+ 2000–2004', 2000, 2014),
        ('Full History (1990–2014)', 1990, 2014)
    ]

    records = []
    
    for label, min_yr, max_yr in historical_cutoffs:
        df_train_pool = df_master[(df_master[year_col] >= min_yr) & (df_master[year_col] <= max_yr)].dropna(subset=[target_col] + feature_cols).copy()
        
        N_train = len(df_train_pool)
        if N_train < 100:
            continue

        X_train = df_train_pool[feature_cols].values
        y_train = df_train_pool[target_col].values

        # Subsampling runs for median estimation
        r2_runs = []
        for run_idx in range(5):
            np.random.seed(42 + run_idx * 100)
            
            # Subsample 80% to generate distribution variance
            sub_idx = np.random.choice(N_train, size=int(N_train * 0.80), replace=False)
            
            model = model_obj if model_obj is not None else Ridge(alpha=1.0)
            model.fit(X_train[sub_idx], y_train[sub_idx])
            preds = model.predict(X_test)
            r2_runs.append(r2_score(y_test, preds))

        median_r2 = np.median(r2_runs)
        std_r2 = np.std(r2_runs)

        records.append({
            'label': label,
            'N_train': N_train,
            'R2': median_r2,
            'std_R2': std_r2
        })

    df_results = pd.DataFrame(records)

    # OLS Scaling Exponent (alpha) and t-statistic calculation
    log_N = np.log(df_results['N_train'].values)
    X_ols = sm.add_constant(log_N)
    ols_model = sm.OLS(df_results['R2'].values, X_ols).fit()

    alpha = ols_model.params[1]
    t_stat = ols_model.tvalues[1]
    p_val = ols_model.pvalues[1]

    print(f"[{model_name} Axis 1] Alpha: {alpha:.4f} | t-stat: {t_stat:.2f} | p-val: {p_val:.4f}")

    return df_results, alpha, t_stat, p_val

# ==============================================================================
# UNIFIED PLOTTING ROUTINE WITH LOCKED SCALES
# ==============================================================================
def plot_axis1_out_of_time(df_results, alpha, model_name='Ridge'):
    """
    Renders Nature-style Axis 1 Out-of-Time anti-scaling plot.
    Shares locked limits (0.0 to 0.85) to align directly with Axis 3 plots.
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

    N_arr = df_results['N_train'].values
    R2_arr = df_results['R2'].values
    std_arr = df_results['std_R2'].values

    # Fit log-linear curve
    log_N = np.log(N_arr)
    slope, intercept = np.polyfit(log_N, R2_arr, 1)
    N_dense = np.geomspace(N_arr.min(), N_arr.max(), 200)
    R2_fit = slope * np.log(N_dense) + intercept

    line_color = '#003366'  # Deep Navy blue matching all_africa_scaling_law.png

    # 1. Anti-scaling fitted line
    ax.plot(
        N_dense, R2_fit, 
        color=line_color, linewidth=1.4, zorder=4,
        label=f'Anti-scaling law ($\\alpha = {alpha:.4f}$)'
    )

    # 2. Empirical Backtest Data Points
    ax.scatter(
        N_arr, R2_arr, 
        color=line_color, s=20, zorder=5, edgecolor='white', linewidth=0.5,
        label='Empirical backtest'
    )

    # 3. Smooth error shading clipped cleanly
    y_low = np.clip(R2_arr - 0.5 * std_arr, a_min=0.0, a_max=None)
    y_high = R2_arr + 0.5 * std_arr
    ax.fill_between(N_arr, y_low, y_high, color=line_color, alpha=0.10, zorder=3)

    ax.set_xscale('log')
    
    # LOCKED SCALES MATCHING AXIS 3
    ax.set_ylim(0.0, 0.85)
    ax.set_xlim(2000, 45000)

    # Format X ticks in thousands (2k, 4k, 8k, 16k, 32k)
    ax.set_xticks([2000, 4000, 8000, 16000, 32000])
    ax.set_xticklabels(['2k', '4k', '8k', '16k', '32k'])

    ax.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=0.4, color='#E0E0E0', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.set_xlabel('Pooled training size $N_{\\text{Africa}}$', labelpad=4)
    ax.set_ylabel('Out-of-sample holdout $R^2$', labelpad=4)
    ax.set_title(f'Out-of-Time Scaling ({model_name})', loc='left', pad=4, fontsize=8, fontweight='bold')

    ax.legend(
        loc='lower left', 
        frameon=True, 
        facecolor='white', 
        edgecolor='none',
        framealpha=0.90,
        handletextpad=0.4, 
        borderpad=0.3
    )

    plt.tight_layout(pad=0.4)
    filename = f'axis1_out_of_time_{model_name.lower()}'
    plt.savefig(f'{filename}.png', dpi=600)
    plt.savefig(f'{filename}.pdf', format='pdf')
    print(f"Saved clean Axis 1 plot to {filename}.png and .pdf")
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

    # MODEL FAMILY LOOP FOR ROBUSTNESS CHECKS
    model_families = {
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(C=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }

    for model_name, model_obj in model_families.items():
        print(f"\n=================================================================")
        print(f"RUNNING AXIS 1 (OUT-OF-TIME) FOR MODEL FAMILY: {model_name}")
        print(f"=================================================================")
        
        df_results, alpha, t_stat, p_val = run_axis1_out_of_time(
            df_master, feature_cols, year_col, country_col, target_col,
            model_name=model_name, model_obj=model_obj
        )

        # Render updated plot
        plot_axis1_out_of_time(df_results, alpha, model_name=model_name)