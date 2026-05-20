import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

data = pd.read_parquet('master_merged_data.parquet') 

features = [col for col in data.columns if col.startswith('F_')]
years = sorted(data['year'].unique())
window_size = 5  
results = []

print("Calculating rolling windows with standardized embeddings...")
for i in range(len(years) - window_size + 1):
    window_years = years[i : i + window_size]
    train_data = data[data['year'].isin(window_years)].copy()
    
    # Fit and transform ONLY the features within this specific 5-year window
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data[features])
    
    model = Ridge(alpha=1.0) 
    model.fit(X_scaled, train_data['iwi']) 
    
    results.append({
        'end_year': window_years[-1],
        'coefs': model.coef_
    })

coef_df = pd.DataFrame([r['coefs'] for r in results], 
                       index=[r['end_year'] for r in results], 
                       columns=features)

coef_df.to_csv('coefficients_master.csv')
print("Saved: coefficients_master.csv")

top_5 = coef_df.abs().mean().sort_values(ascending=False).head(5).index

plt.figure(figsize=(12, 6))
for feature in top_5:
    plt.plot(coef_df.index, coef_df[feature], marker='o', label=f"Feature {feature}")

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title("Evolution of Economic Predictors (Standardized 5-Year Rolling Windows)", fontsize=14)
plt.xlabel("Window End Year", fontsize=12)
plt.ylabel("Regression Coefficient Weight", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('feature_evolution_plot_std.png', dpi=300)
print("Saved: feature_evolution_plot_std.png")
plt.show()