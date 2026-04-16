import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from scipy import stats

years = {1995, 2000, 2005, 2010, 2015}

slopes = []

results = pd.read_csv('landsat_results_final.csv')



zip_path = 'API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.zip'

# Read the GDP data from the zip file
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv') as f:
        gdp_df = pd.read_csv(f, skiprows=4)


# Standardize country names between the results and GDP datasets
name_map = {
    'Congo, Dem. Rep.': 'Congo - Kinshasa',
    'Cote d\'Ivoire': 'Côte d’Ivoire',
    'Egypt, Arab Rep.': 'Egypt',
    'Gambia, The': 'Gambia',
    'Tanzania': 'Tanzania', 
    }

# Apply the name mapping to the GDP DataFrame
gdp_df['Country Name'] = gdp_df['Country Name'].replace(name_map)


# Getting lowest and highest GDP values across all years for setting consistent x-axis limits
all_years = [str(y) for y in years]
all_values = gdp_df[all_years].values.flatten()
all_values = all_values[~np.isnan(all_values)] 

global_x_min = np.log10(all_values.min()) * 0.98  
global_x_max = np.log10(all_values.max()) * 1.02  

fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

for i, year in enumerate(sorted(years)):
    
    results_year = results[results['Benchmark Year'] == year].copy()
    gdp_year = gdp_df[['Country Name', str(year)]].rename(columns={str(year): f'GDP_PPP_{year}'})
    merged = results_year.merge(gdp_year, left_on='Holdout Country', right_on='Country Name', how='left')
    merged = merged[merged['Holdout Country'] != 'Egypt']
    merged['Log_GDP'] = np.log10(merged[f'GDP_PPP_{year}'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged['Log_GDP'], 
        merged['Holdout R^2']
    )
    slopes.append(slope)

    Q1 = merged['Holdout R^2'].quantile(0.25)
    Q3 = merged['Holdout R^2'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    sns.regplot(data=merged, x='Log_GDP', y='Holdout R^2', ax=axes[i],
                scatter_kws={'alpha':0.4, 's':40}, 
                line_kws={'color':'red', 'zorder': 1})
    
    # Set consistent x-axis limits across all subplots
    axes[i].set_xlim(global_x_min, global_x_max)

    for x, y, name in zip(merged['Log_GDP'], merged['Holdout R^2'], merged['Holdout Country']):
        # If the country is outside the 'normal' range (the outliers)
        if (y < lower_bound) or (y > upper_bound):
            axes[i].text(x, y + 0.02, name, 
                         fontsize=9, fontweight='bold', 
                         ha='center', va='bottom', color='black', zorder=5)

    axes[i].set_title(f'Year: {year}', fontsize=15)
    axes[i].set_xlabel('Log10 GDP')
    if i == 0: axes[i].set_ylabel('Model Accuracy ($R^2$)') 


print("\n--- SLOPE OF ACCURACY VS WEALTH ---")
slope_df = pd.DataFrame(slopes)
print(slope_df.to_string(index=False))

plt.tight_layout()
plt.savefig('gdp_trend_panel_1995_2015_no_Egypt.png', dpi=300)
print("Panel plot saved.")
plt.show()
