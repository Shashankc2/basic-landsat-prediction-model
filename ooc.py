import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
# CORE EXPERIMENTAL RUNNER (MULTI-MODEL OUT-OF-COUNTRY SCALING)
# ==============================================================================
def run_axis2_ooc_experiment(
    df_master, 
    feature_cols, 
    year_col='year',
    country_col='country',
    target_col='iwi', 
    modern_window=(2015, 2021),
    country_steps=[1, 3, 7, 15, 27],
    model_name='Ridge',
    model_obj=None
):
    """
    Executes Leave-One-Country-Out (LOCO) evaluation scaling on real DHS Master Data.
    Holds time fixed to a modern window while scaling the number of donor training countries.
    """
    time_mask = (df_master[year_col] >= modern_window[0]) & (df_master[year_col] <= modern_window[1])
    df_modern = df_master[time_mask].dropna(subset=[target_col, country_col] + feature_cols).copy()
    
    available_countries = list(df_modern[country_col].dropna().unique())
    total_countries = len(available_countries)
    donor_pool_size = total_countries - 1
    
    if total_countries < 5:
        raise ValueError(f"Need at least 5 unique countries for Axis 2 evaluation. Found: {total_countries}")

    valid_steps = [k for k in country_steps if k <= donor_pool_size]
    
    ooc_results = []
    
    # Leave-One-Country-Out (LOCO) Loop
    for target_country in available_countries:
        df_target = df_modern[df_modern[country_col] == target_country]
        
        # Skip target countries with small evaluation sample sizes
        if len(df_target) < 30:
            continue
            
        X_test = df_target[feature_cols].values
        y_test = df_target[target_col].values
        
        donor_pool = [c for c in available_countries if c != target_country]
        
        for k in valid_steps:
            seed = int(abs(hash(str(target_country)))) % (2**31) + k
            np.random.seed(seed)
            
            selected_donors = np.random.choice(donor_pool, size=k, replace=False)
            
            df_train = df_modern[df_modern[country_col].isin(selected_donors)]
            X_train = df_train[feature_cols].values
            y_train = df_train[target_col].values
            
            # Instantiate model family object
            model = model_obj if model_obj is not None else Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            
            ooc_results.append({
                'target_country': target_country,
                'num_donors': k,
                'N_train': len(X_train),
                'N_test': len(y_test),
                'R2_ooc': r2
            })
            
    df_ooc_raw = pd.DataFrame(ooc_results)
    
    # Aggregate median performance across LOCO targets
    df_ooc_summary = df_ooc_raw.groupby('num_donors').agg(
        mean_N=('N_train', 'median'),
        mean_R2=('R2_ooc', 'median'),
        std_R2=('R2_ooc', 'std'),
        target_count=('R2_ooc', 'count')
    ).reset_index()
    
    # Compute OLS Log-Linear Scaling Exponent (alpha), t-statistic, and p-value
    log_N = np.log(df_ooc_summary['mean_N'].values)
    X_ols = sm.add_constant(log_N)
    ols_model = sm.OLS(df_ooc_summary['mean_R2'].values, X_ols).fit()
    
    alpha_ooc = ols_model.params[1]
    t_stat = ols_model.tvalues[1]
    p_val = ols_model.pvalues[1]
    
    print(f"[{model_name} Axis 2] Alpha: {alpha_ooc:.4f} | t-stat: {t_stat:.2f} | p-val: {p_val:.4f}")
    
    return df_ooc_summary, alpha_ooc, t_stat, p_val


# ==============================================================================
# UNIFIED PLOTTING ROUTINE
# ==============================================================================
def plot_axis2(df_ooc_summary, alpha_ooc, model_name='Ridge'):
    """
    Renders Nature-style Axis 2 Out-of-Country scaling plot.
    Shares locked limits (0.0 to 0.85) to align directly with Axis 1 and 3 plots.
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

    N_arr = df_ooc_summary['mean_N'].values
    R2_arr = df_ooc_summary['mean_R2'].values
    std_arr = df_ooc_summary['std_R2'].values
    k_arr = df_ooc_summary['num_donors'].values

    log_N = np.log(N_arr)
    slope, intercept = np.polyfit(log_N, R2_arr, 1)

    N_dense = np.geomspace(N_arr.min(), N_arr.max(), 200)
    R2_fit = slope * np.log(N_dense) + intercept

    line_color = '#1B5E20'  # Deep Forest Green for Axis 2

    # Fitted scaling curve
    ax.plot(
        N_dense, R2_fit, 
        color=line_color, linewidth=1.4, zorder=4,
        label=f'Cross-country law ($\\alpha = {alpha_ooc:.4f}$)'
    )

    # Empirical data points
    ax.scatter(
        N_arr, R2_arr, 
        color=line_color, s=20, zorder=5, edgecolor='white', linewidth=0.5,
        label='LOCO holdout median'
    )

    # Shaded median error bounds
    y_low = np.clip(R2_arr - 0.5 * std_arr, a_min=0.0, a_max=None)
    y_high = R2_arr + 0.5 * std_arr
    ax.fill_between(N_arr, y_low, y_high, color=line_color, alpha=0.10, zorder=3)

    # Annotate donor count (k)
    for x_i, y_i, k_i in zip(N_arr, R2_arr, k_arr):
        ax.annotate(
            f'$k={k_i}$',
            (x_i, y_i),
            textcoords="offset points",
            xytext=(0, 6),
            ha='center',
            fontsize=6,
            color='#333333'
        )

    ax.set_xscale('log')
    
    # LOCKED SCALES MATCHING AXIS 1 & 3
    ax.set_ylim(0.0, 0.85)

    ax.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=0.4, color='#E0E0E0', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.set_xlabel('Cross-border sample size $N_{\\text{OOC}}$ (across $k$ countries)', labelpad=4)
    ax.set_ylabel('Unobserved country holdout $R^2$', labelpad=4)
    ax.set_title(f'Out-of-Country Scaling ({model_name})', loc='left', pad=4, fontsize=8, fontweight='bold')

    ax.legend(
        loc='lower right', 
        frameon=True, 
        facecolor='white', 
        edgecolor='none',
        framealpha=0.90,
        handletextpad=0.4, 
        borderpad=0.3
    )

    plt.tight_layout(pad=0.4)
    filename = f'axis2_out_of_country_{model_name.lower()}'
    plt.savefig(f'{filename}.png', dpi=600)
    plt.savefig(f'{filename}.pdf', format='pdf')
    print(f"Saved clean Axis 2 plot to {filename}.png and .pdf")
    plt.close()


# ==============================================================================
# EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parquet_path = 'master_merged_data.parquet'
    
    print(f"Loading real DHS Master Matrix from {parquet_path}...")
    df_master = pd.read_parquet(parquet_path)
    
    year_col = find_column_name(df_master, ['year', 'survey_year', 'DHSYEAR', 'Year'], "survey year")
    country_col = find_column_name(df_master, ['country', 'country_iso', 'ISO', 'Country'], "country ISO")
    target_col = find_column_name(df_master, ['iwi', 'asset_index', 'wealth_index', 'y'], "target asset index (IWI)")

    feature_cols = [col for col in df_master.columns if col.startswith('F_')]
    if len(feature_cols) == 0:
        metadata_cols = {'Unnamed: 0', 'cluster_id', 'lon', 'lat', 'rural', 'region_id', country_col, year_col, target_col}
        feature_cols = [col for col in df_master.select_dtypes(include=[np.number]).columns if col not in metadata_cols]

    print(f"Isolated {len(feature_cols)} clean satellite embedding feature columns.")

    # MULTI-MODEL SUITE DEFINITION (SVR updated with loss='squared_epsilon_insensitive')
    model_families = {
        'Ridge': Ridge(alpha=1.0),
        'SVR': make_pipeline(
            StandardScaler(), 
            LinearSVR(
                C=1.0, 
                loss='squared_epsilon_insensitive', 
                dual=False, 
                tol=1e-3, 
                max_iter=10000, 
                random_state=42
            )
        ),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }

    for model_name, model_obj in model_families.items():
        print(f"\n=================================================================")
        print(f"RUNNING AXIS 2 (OUT-OF-COUNTRY) FOR MODEL FAMILY: {model_name}")
        print(f"=================================================================")

        summary, alpha, t_stat, p_val = run_axis2_ooc_experiment(
            df_master=df_master,
            feature_cols=feature_cols,
            year_col=year_col,
            country_col=country_col,
            target_col=target_col,
            modern_window=(2015, 2021),
            country_steps=[1, 3, 7, 15, 27],
            model_name=model_name,
            model_obj=model_obj
        )
        
        plot_axis2(summary, alpha, model_name=model_name)