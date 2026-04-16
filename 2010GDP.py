import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile

results = pd.read_csv('landsat_results_final.csv')

results_2010 = results[results['Benchmark Year'] == 2010].copy()

zip_path = 'API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.zip'
csv_inside_zip = 'API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_inside_zip) as f:
        gdp_df = pd.read_csv(f, skiprows=4)

name_map = {
    'Congo, Dem. Rep.': 'Congo - Kinshasa',
    'Cote d\'Ivoire': 'Côte d’Ivoire',
    'Egypt, Arab Rep.': 'Egypt',
    'Gambia, The': 'Gambia',
    'Tanzania': 'Tanzania', 
    }
gdp_df['Country Name'] = gdp_df['Country Name'].replace(name_map)


gdp_2010 = gdp_df[['Country Name', '2010']].rename(columns={'2010': 'GDP_PPP_2010'})
merged = results_2010.merge(gdp_2010, left_on='Holdout Country', right_on='Country Name', how='left')


merged['Log_GDP'] = np.log10(merged['GDP_PPP_2010'])

plt.figure(figsize=(10, 6))
sns.regplot(data=merged, x='Log_GDP', y='Holdout R^2', # regression plot with log GDP on x-axis and R^2 on y-axis
            scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'color': 'red'}) # adds a regression line in red and makes points larger and semi-transparent

for i in range(merged.shape[0]):
    if pd.notnull(merged.Log_GDP.iloc[i]):
        plt.text(merged.Log_GDP.iloc[i]+0.02, merged['Holdout R^2'].iloc[i], #labels country names next to points
                 merged['Holdout Country'].iloc[i], fontsize=9)

plt.title('Relationship Between Economic Wealth and Model Accuracy (2010)', fontsize=14)
plt.xlabel('Log10(GDP per capita, PPP)', fontsize=12)
plt.ylabel('Model Accuracy ($R^2$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('gdp_vs_accuracy_log2010.png', dpi=300)
plt.show()

missing = merged[merged['GDP_PPP_2010'].isna()]['Holdout Country'].unique()
if len(missing) > 0:
    print(f"Warning: No GDP data found for: {missing}")