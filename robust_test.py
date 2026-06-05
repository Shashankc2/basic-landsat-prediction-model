import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load the data
data = pd.read_parquet('master_merged_data.parquet') 
features = [col for col in data.columns if col.startswith('F_')]
years = sorted(data['year'].unique())
window_size = 5  
results = []

# Extract country code once at the beginning (e.g., 'RW' from 'RW.East.5A.226')
data['country_code'] = data['cluster_id'].str.split('.').str[0]

print(f"Original dataset size: {len(data)}")
print("\n============ ROBUSTNESS LOOP & DIAGNOSTIC ============")

for i in range(len(years) - window_size + 1):
    window_years = years[i : i + window_size]
    
    # Get all data for this specific 5-year window
    window_data = data[data['year'].isin(window_years)].copy()
    
    # Count how many unique years each country has WITHIN THIS SPECIFIC WINDOW
    country_year_counts = window_data.groupby('country_code')['year'].nunique()
    
    # Only keep countries that appear in all 5 years of this window
    stable_countries = country_year_counts[country_year_counts == window_size].index
    train_data = window_data[window_data['country_code'].isin(stable_countries)].copy()
    
    # --- DIAGNOSTIC PRINTS ---
    surviving_countries = sorted(list(stable_countries))
    print(f"\nWindow: {window_years[0]} to {window_years[-1]}")
    print(f"  Countries Kept ({len(surviving_countries)}): {surviving_countries}")
    print(f"  Rows Kept: {len(train_data)}")
    # -------------------------
    
    if len(train_data) == 0:
        print(f"  ⚠️ Warning: No balanced countries found for window ending {window_years[-1]}. Skipping.")
        continue
        
    # Standardize window-by-window
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data[features])
    
    model = Ridge(alpha=1.0) 
    model.fit(X_scaled, train_data['iwi'])
    
    results.append({
        'end_year': window_years[-1],
        'coefs': model.coef_
    })

print("\n=======================================================\n")

# 2. Process and Save Coefficients
coef_df = pd.DataFrame([r['coefs'] for r in results], 
                       index=[r['end_year'] for r in results], 
                       columns=features)

coef_df.to_csv('coefficients_robust.csv')
print("Saved: coefficients_robust.csv")

# 3. Dynamic Extraction of the TRUE Top 5 Robust Features
top_5_robust = coef_df.abs().mean().sort_values(ascending=False).head(5).index
print(f"True Top 5 Robust Features detected: {list(top_5_robust)}")

# 4. Plotting
plt.figure(figsize=(12, 6))
for feature in top_5_robust:
    if feature in coef_df.columns:
        plt.plot(coef_df.index, coef_df[feature], marker='o', label=f"Feature {feature}")

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title("Robustness Check: Balanced Country Panel (True Top 5 Features)", fontsize=14)
plt.xlabel("Window End Year", fontsize=12)
plt.ylabel("Regression Coefficient Weight", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('feature_evolution_robust.png', dpi=300)
print("Saved: feature_evolution_robust.png")
plt.show()