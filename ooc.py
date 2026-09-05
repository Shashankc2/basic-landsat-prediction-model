import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def find_column_name(df, candidates, description):
    """Utility to resolve schema variations in DHS datasets."""
    for col in candidates:
        if col in df.columns:
            print(f"Mapped '{description}' to column: '{col}'")
            return col
    raise KeyError(f"Could not find a valid '{description}' column.")

# ==============================================================================
# AXIS 2 (OOC) EXPERIMENTAL PIPELINE
# ==============================================================================
def run_axis2_ooc_experiment(
    df_master, 
    feature_cols, 
    year_col='year',
    country_col='country',
    target_col='iwi', 
    modern_window=(2015, 2021),
    country_steps=[1, 3, 7, 15, 27]
):
    """
    Executes Out-of-Country (OOC) evaluation scaling on real DHS Master Data.
    Holds time fixed to a modern window while scaling the number of donor training countries.
    """
    # 1. Restrict strictly to the fixed modern time window (Temporal Isolation)
    time_mask = (df_master[year_col] >= modern_window[0]) & (df_master[year_col] <= modern_window[1])
    df_modern = df_master[time_mask].copy()
    
    # Extract all available countries present in the dataset within the time window
    available_countries = list(df_modern[country_col].dropna().unique())
    total_countries = len(available_countries)
    donor_pool_size = total_countries - 1
    
    print(f"\nExecuting Axis 2 (OOC) across {total_countries} DHS countries in modern window {modern_window[0]}-{modern_window[1]}...")
    print(f"Countries found: {available_countries}")
    print(f"Max available cross-border donor pool size: {donor_pool_size} countries")
    
    if total_countries == 0:
        raise ValueError(f"No countries found matching modern window {modern_window}. Unique years in dataset: {sorted(df_master[year_col].unique())}")

    # Adjust country steps so they do not exceed available donors
    valid_steps = [k for k in country_steps if k <= donor_pool_size]
    if len(valid_steps) < len(country_steps):
        print(f"Note: Steps {[k for k in country_steps if k > donor_pool_size]} exceeded donor pool size ({donor_pool_size}) and were adjusted to {valid_steps}.")
    
    ooc_results = []
    
    # 2. Leave-One-Country-Out (LOCO) Cross-Validation Loop
    for target_country in available_countries:
        # Unobserved target evaluation set
        df_target = df_modern[df_modern[country_col] == target_country]
        
        # Skip target countries with insufficient cluster evaluation size
        if len(df_target) < 30:
            continue
            
        X_test = df_target[feature_cols].values
        y_test = df_target[target_col].values
        
        # Candidate donor pool (all countries except current target)
        donor_pool = [c for c in available_countries if c != target_country]
        
        for k in valid_steps:
            # Deterministic seed per target-step pair for exact reproducibility
            seed = int(abs(hash(str(target_country)))) % (2**31) + k
            np.random.seed(seed)
            
            selected_donors = np.random.choice(donor_pool, size=k, replace=False)
            
            df_train = df_modern[df_modern[country_col].isin(selected_donors)]
            X_train = df_train[feature_cols].values
            y_train = df_train[target_col].values
            
            # Fit model on cross-border donor pool
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            # Evaluate on unobserved target country
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
    
    # 3. Aggregate across all LOCO target evaluation runs
    df_ooc_summary = df_ooc_raw.groupby('num_donors').agg(
        mean_N=('N_train', 'mean'),
        mean_R2=('R2_ooc', 'mean'),
        std_R2=('R2_ooc', 'std'),
        target_count=('R2_ooc', 'count')
    ).reset_index()
    
    # Calculate log-linear geographic scaling exponent (alpha_OOC)
    log_N = np.log(df_ooc_summary['mean_N'])
    alpha_ooc, intercept = np.polyfit(log_N, df_ooc_summary['mean_R2'], 1)
    
    print("\n" + "="*65)
    print("AXIS 2 (OUT-OF-COUNTRY) SCALING RESULTS")
    print("="*65)
    print(df_ooc_summary.to_string(index=False))
    print(f"\nGeographic Scaling Exponent (α_OOC): {alpha_ooc:.4f}")
    print("="*65 + "\n")
    
    return df_ooc_summary, alpha_ooc


# ==============================================================================
# NATURE-STYLE PLOTTING FOR AXIS 2
# ==============================================================================
def plot_axis2(df_ooc_summary, alpha_ooc):
    """
    Plots Axis 2 (OOC) results adhering to publication formatting.
    """
    MM_TO_INCHES = 1 / 25.4
    FIG_WIDTH = 89 * MM_TO_INCHES
    FIG_HEIGHT = 68 * MM_TO_INCHES

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

    N_arr = df_ooc_summary['mean_N'].values
    R2_arr = df_ooc_summary['mean_R2'].values
    k_arr = df_ooc_summary['num_donors'].values

    log_N = np.log(N_arr)
    slope, intercept = np.polyfit(log_N, R2_arr, 1)

    # Restrict continuous line strictly to empirical N range
    N_dense = np.geomspace(N_arr.min(), N_arr.max(), 200)
    R2_fit = slope * np.log(N_dense) + intercept

    # Plot scaling curve and empirical LOCO points
    ax.plot(N_dense, R2_fit, color='#D95F02', linewidth=1.2, zorder=3, label=f'OOC scaling ($\\alpha = {alpha_ooc:.4f}$)')
    ax.scatter(N_arr, R2_arr, color='#7570B3', s=18, edgecolor='white', linewidth=0.5, zorder=4, label='LOCO holdout mean')

    # Annotate each point with the donor country count (k)
    y_range = R2_arr.max() - R2_arr.min() if R2_arr.max() != R2_arr.min() else 0.01
    for x_i, y_i, k_i in zip(N_arr, R2_arr, k_arr):
        ax.annotate(
            f'$k={k_i}$',
            (x_i, y_i),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=6,
            color='#333333'
        )

    ax.set_xscale('log')
    ax.set_ylim(R2_arr.min() - 0.25 * y_range, R2_arr.max() + 0.55 * y_range)

    ax.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.4, color='#E5E5E5', zorder=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.set_xlabel('Cross-border sample size $N_{\\text{OOC}}$ (across $k$ countries)', labelpad=4)
    ax.set_ylabel('Unobserved country holdout $R^2$', labelpad=4)

    ax.legend(
        loc='upper left', 
        bbox_to_anchor=(0.02, 0.98), 
        frameon=False, 
        handletextpad=0.4, 
        borderpad=0.2
    )

    plt.tight_layout(pad=0.5)
    plt.savefig('axis2_out_of_country_scaling.png', dpi=600)
    plt.savefig('axis2_out_of_country_scaling.pdf', format='pdf')
    print("Saved figure to axis2_out_of_country_scaling.png and .pdf")
    plt.show()

# ==============================================================================
# EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parquet_path = 'master_merged_data.parquet'
    
    print(f"Loading real DHS Master Matrix from {parquet_path}...")
    df_master = pd.read_parquet(parquet_path)
    
    # 1. Resolve Year Column
    year_candidates = ['year', 'survey_year', 'DHSYEAR', 'survey_year_dhs', 'Year']
    year_col = find_column_name(df_master, year_candidates, "survey year")

    # 2. Resolve Country Column
    country_candidates = ['country', 'country_iso', 'ISO', 'country_code', 'Country']
    country_col = find_column_name(df_master, country_candidates, "country ISO")

    # 3. Resolve Target Column (iwi)
    target_candidates = ['iwi', 'asset_index', 'wealth_index', 'wealthindex', 'DHSWALTH', 'target', 'y']
    target_col = find_column_name(df_master, target_candidates, "target asset index (IWI)")

    # 4. Detect Feature Columns starting with 'F_'
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

    print(f"Isolated {len(feature_cols)} clean satellite embedding feature columns (F_0 to F_{len(feature_cols)-1}).")

    # Run Axis 2 pipeline
    summary, alpha = run_axis2_ooc_experiment(
        df_master=df_master,
        feature_cols=feature_cols,
        year_col=year_col,
        country_col=country_col,
        target_col=target_col,
        modern_window=(2015, 2021),
        country_steps=[1, 3, 7, 15, 27]
    )
    
    # Render figure
    plot_axis2(summary, alpha)