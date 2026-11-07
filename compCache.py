import json
import os
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

os.makedirs("cache", exist_ok=True)

def find_column_name(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find a valid '{description}' column.")

# ==============================================================================
# 1. ALL SCALING AXES COMPUTATION (AXIS 1, AXIS 2, AXIS 3)
# ==============================================================================
parquet_path = 'master_merged_data.parquet'
scaling_cache = {}

if os.path.exists(parquet_path):
    print(f"Loading DHS Master Matrix from {parquet_path}...")
    df_master = pd.read_parquet(parquet_path)

    year_col = find_column_name(df_master, ['year', 'survey_year', 'DHSYEAR', 'Year'], "survey year")
    country_col = find_column_name(df_master, ['country', 'country_iso', 'ISO', 'Country'], "country ISO")
    target_col = find_column_name(df_master, ['iwi', 'asset_index', 'wealth_index', 'y'], "target asset index")

    feature_cols = [col for col in df_master.columns if col.startswith('F_')]
    if len(feature_cols) == 0:
        metadata_cols = {'Unnamed: 0', 'cluster_id', 'lon', 'lat', 'rural', 'region_id', country_col, year_col, target_col}
        feature_cols = [col for col in df_master.select_dtypes(include=[np.number]).columns if col not in metadata_cols]

    # --------------------------------------------------------------------------
    # AXIS 1: TEMPORAL OUT-OF-TIME
    # --------------------------------------------------------------------------
    df_holdout = df_master[df_master[year_col] >= 2015].dropna(subset=[target_col] + feature_cols)
    X_test_a1, y_test_a1 = df_holdout[feature_cols].values, df_holdout[target_col].values

    historical_cutoffs = [
        ('2010–2014', 2010, 2014),
        ('+ 2005–2009', 2005, 2014),
        ('+ 2000–2004', 2000, 2014),
        ('Full History (1990–2014)', 1990, 2014)
    ]

    axis1_cache = {'title': 'Axis 1: Temporal OOT', 'xlabel': r'Pooled training size $N_{\mathrm{Africa}}$', 'models': {}}
    model_suite = {
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(C=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }

    for mname, mobj in model_suite.items():
        N_list, R2_list = [], []
        for label, min_yr, max_yr in historical_cutoffs:
            df_tr = df_master[(df_master[year_col] >= min_yr) & (df_master[year_col] <= max_yr)].dropna(subset=[target_col] + feature_cols)
            if len(df_tr) < 100: continue
            X_tr, y_tr = df_tr[feature_cols].values, df_tr[target_col].values
            mobj.fit(X_tr, y_tr)
            preds = mobj.predict(X_test_a1)
            N_list.append(len(df_tr))
            R2_list.append(float(r2_score(y_test_a1, preds)))
        
        slope, intercept = np.polyfit(np.log(N_list), R2_list, 1)
        axis1_cache['models'][mname] = {'N': N_list, 'R2': R2_list, 'alpha': float(slope), 'intercept': float(intercept)}

    scaling_cache['Axis1'] = axis1_cache

    # --------------------------------------------------------------------------
    # AXIS 2: GEOGRAPHIC OUT-OF-COUNTRY (LOCO LOOP)
    # --------------------------------------------------------------------------
    df_modern = df_master[(df_master[year_col] >= 2015) & (df_master[year_col] <= 2021)].dropna(subset=[target_col, country_col] + feature_cols)
    available_countries = list(df_modern[country_col].dropna().unique())
    k_steps = [1, 3, 7, 15, 27]
    valid_steps = [k for k in k_steps if k <= len(available_countries) - 1]

    axis2_cache = {'title': 'Axis 2: Geographic OOC', 'xlabel': r'Cross-border sample $N_{\mathrm{OOC}}$ ($k$ countries)', 'models': {}}
    
    for mname, mobj in model_suite.items():
        ooc_records = []
        for target_c in available_countries:
            df_target = df_modern[df_modern[country_col] == target_c]
            if len(df_target) < 30: continue
            X_t, y_t = df_target[feature_cols].values, df_target[target_col].values
            donor_pool = [c for c in available_countries if c != target_c]

            for k in valid_steps:
                np.random.seed(int(abs(hash(str(target_c)))) % (2**31) + k)
                selected_donors = np.random.choice(donor_pool, size=k, replace=False)
                df_tr = df_modern[df_modern[country_col].isin(selected_donors)]
                mobj.fit(df_tr[feature_cols].values, df_tr[target_col].values)
                r2 = r2_score(y_t, mobj.predict(X_t))
                ooc_records.append({'k': k, 'N_train': len(df_tr), 'R2': r2})

        df_ooc = pd.DataFrame(ooc_records).groupby('k').agg({'N_train': 'median', 'R2': 'median'}).reset_index()
        slope, intercept = np.polyfit(np.log(df_ooc['N_train'].values), df_ooc['R2'].values, 1)
        axis2_cache['models'][mname] = {
            'N': df_ooc['N_train'].tolist(),
            'R2': df_ooc['R2'].tolist(),
            'k': df_ooc['k'].tolist(),
            'alpha': float(slope),
            'intercept': float(intercept)
        }

    scaling_cache['Axis2'] = axis2_cache

    # --------------------------------------------------------------------------
    # AXIS 3: WITHIN-COUNTRY LOCAL DENSITY SCALING
    # --------------------------------------------------------------------------
    fractions = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    axis3_cache = {'title': 'Axis 3: Local Density', 'xlabel': r'Within-country sample $N_{\mathrm{within}}$', 'models': {}}

    for mname, mobj in model_suite.items():
        a3_records = []
        for c in available_countries:
            df_c = df_modern[df_modern[country_col] == c]
            if len(df_c) < 60: continue
            X_tr_full, X_te, y_tr_full, y_te = train_test_split(df_c[feature_cols].values, df_c[target_col].values, test_size=0.20, random_state=42)
            
            for frac in fractions:
                n_sub = int(len(X_tr_full) * frac)
                if n_sub < 15: continue
                mobj.fit(X_tr_full[:n_sub], y_tr_full[:n_sub])
                r2 = r2_score(y_te, mobj.predict(X_te))
                a3_records.append({'frac': frac, 'N_train': n_sub, 'R2': r2})

        df_a3 = pd.DataFrame(a3_records).groupby('frac').agg({'N_train': 'mean', 'R2': 'mean'}).reset_index()
        slope, intercept = np.polyfit(np.log(df_a3['N_train'].values), df_a3['R2'].values, 1)
        axis3_cache['models'][mname] = {
            'N': df_a3['N_train'].tolist(),
            'R2': df_a3['R2'].tolist(),
            'alpha': float(slope),
            'intercept': float(intercept)
        }

    scaling_cache['Axis3'] = axis3_cache

with open("cache/scaling_metrics.json", "w") as f:
    json.dump(scaling_cache, f, indent=2)

# ==============================================================================
# 2. SPATIAL MAP METRICS
# ==============================================================================
csv_path = 'landsat_results_final.csv'
if os.path.exists(csv_path):
    print(f"Processing spatial & metadata results from {csv_path}...")
    df_results = pd.read_csv(csv_path)
    
    name_map = {
        "Congo - Kinshasa": "Democratic Republic of the Congo",
        "Côte d’Ivoire": "Ivory Coast",
        "Eswatini": "Swaziland",
        "Tanzania": "United Republic of Tanzania"
    }
    df_results['Holdout Country'] = df_results['Holdout Country'].replace(name_map)

    map_cache = {'years': [1995, 2000, 2005, 2010, 2015], 'sample_sizes': {}, 'country_r2': {}}

    for y in map_cache['years']:
        sub = df_results[df_results['Benchmark Year'] == y]
        map_cache['sample_sizes'][str(y)] = int(sub['Train Size'].iloc[0]) if not sub.empty else 0
        for _, row in sub.iterrows():
            c_name = row['Holdout Country']
            if c_name not in map_cache['country_r2']:
                map_cache['country_r2'][c_name] = {}
            map_cache['country_r2'][c_name][str(y)] = float(row['Holdout R^2'])

    with open("cache/map_metrics.json", "w") as f:
        json.dump(map_cache, f, indent=2)

    # ==========================================================================
    # 3. GDP TREND METRICS
    # ==========================================================================
    zip_path = 'API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.zip'
    if os.path.exists(zip_path):
        print("Extracting World Bank GDP per capita metrics...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv') as f:
                gdp_df = pd.read_csv(f, skiprows=4)

        wb_name_map = {
            'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
            'Cote d\'Ivoire': 'Ivory Coast',
            'Egypt, Arab Rep.': 'Egypt',
            'Gambia, The': 'Gambia',
            'Tanzania': 'United Republic of Tanzania', 
        }
        gdp_df['Country Name'] = gdp_df['Country Name'].replace(wb_name_map)

        gdp_cache = {'years': [1995, 2000, 2005, 2010, 2015], 'data_by_year': {}}

        for y in gdp_cache['years']:
            sub_res = df_results[df_results['Benchmark Year'] == y].copy()
            sub_gdp = gdp_df[['Country Name', str(y)]].rename(columns={str(y): f'GDP_PPP_{y}'})
            merged = sub_res.merge(sub_gdp, left_on='Holdout Country', right_on='Country Name', how='left')
            merged['Log_GDP'] = np.log10(merged[f'GDP_PPP_{y}'])

            valid = merged.dropna(subset=['Log_GDP', 'Holdout R^2'])
            ols = sm.OLS(valid['Holdout R^2'], sm.add_constant(valid['Log_GDP'])).fit()
            
            gdp_cache['data_by_year'][str(y)] = {
                'countries': valid['Holdout Country'].tolist(),
                'log_gdp': valid['Log_GDP'].tolist(),
                'holdout_r2': valid['Holdout R^2'].tolist(),
                'slope': float(ols.params[1]),
                'intercept': float(ols.params[0])
            }

        with open("cache/gdp_metrics.json", "w") as f:
            json.dump(gdp_cache, f, indent=2)

    # ==========================================================================
    # 4. COLONIZER / INSTITUTIONAL HISTORY METRICS
    # ==========================================================================
    colonial_map = {
        'Angola': 'Portuguese', 'Benin': 'French', 'Burkina Faso': 'French',
        'Burundi': 'Belgian', 'Cameroon': 'French', 'Central African Republic': 'French',
        'Chad': 'French', 'Comoros': 'French', 'Democratic Republic of the Congo': 'Belgian',
        'Ivory Coast': 'French', 'Egypt': 'British', 'Swaziland': 'British',
        'Ethiopia': 'Uncolonized', 'Gabon': 'French', 'Gambia': 'British',
        'Ghana': 'British', 'Guinea': 'French', 'Kenya': 'British',
        'Lesotho': 'British', 'Liberia': 'Uncolonized', 'Madagascar': 'French',
        'Malawi': 'British', 'Mali': 'French', 'Mauritania': 'French',
        'Morocco': 'French', 'Mozambique': 'Portuguese', 'Namibia': 'German',
        'Niger': 'French', 'Nigeria': 'British', 'Rwanda': 'Belgian',
        'Senegal': 'French', 'Sierra Leone': 'British', 'South Africa': 'British',
        'United Republic of Tanzania': 'British', 'Togo': 'French', 'Uganda': 'British',
        'Zambia': 'British', 'Zimbabwe': 'British'
    }

    df_col = df_results.copy()
    df_col['Colonizer'] = df_col['Holdout Country'].map(colonial_map).fillna('Other')

    colonizer_cache = {'groups': {}}
    for group, group_df in df_col.groupby('Colonizer'):
        colonizer_cache['groups'][group] = {
            'countries': group_df['Holdout Country'].tolist(),
            'r2_scores': group_df['Holdout R^2'].tolist(),
            'mean_r2': float(group_df['Holdout R^2'].mean()),
            'median_r2': float(group_df['Holdout R^2'].median()),
            'q25': float(group_df['Holdout R^2'].quantile(0.25)),
            'q75': float(group_df['Holdout R^2'].quantile(0.75))
        }

    with open("cache/colonizer_metrics.json", "w") as f:
        json.dump(colonizer_cache, f, indent=2)

print("Computation engine execution finished. All metrics written to ./cache/")