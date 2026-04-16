import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_raw = pd.read_csv('landsat_results_final.csv') 

df = df_raw.groupby('Holdout Country')['Holdout R^2'].mean().reset_index()

colonial_map = {
    'Angola': 'Portuguese', 'Benin': 'French', 'Burkina Faso': 'French',
    'Burundi': 'Belgian', 'Cameroon': 'French', 'Central African Republic': 'French',
    'Chad': 'French', 'Comoros': 'French', 'Congo - Kinshasa': 'Belgian',
    'Côte d’Ivoire': 'French', 'Egypt': 'British', 'Eswatini': 'British',
    'Ethiopia': 'Never Colonized', 'Gabon': 'French', 'Gambia': 'British',
    'Ghana': 'British', 'Guinea': 'French', 'Kenya': 'British',
    'Lesotho': 'British', 'Liberia': 'Never Colonized', 'Madagascar': 'French',
    'Malawi': 'British', 'Mali': 'French', 'Mauritania': 'French',
    'Morocco': 'French', 'Mozambique': 'Portuguese', 'Namibia': 'German',
    'Niger': 'French', 'Nigeria': 'British', 'Rwanda': 'Belgian',
    'Senegal': 'French', 'Sierra Leone': 'British', 'South Africa': 'British',
    'Tanzania': 'British', 'Togo': 'French', 'Uganda': 'British',
    'Zambia': 'British', 'Zimbabwe': 'British'
}

# Map the colonial history to each country
df['Colonizer'] = df['Holdout Country'].map(colonial_map)

# 
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create a boxplot to show the distribution of R^2 values for each colonizer group
sns.boxplot(data=df, x='Colonizer', y='Holdout R^2', palette='Set2', showfliers=False)

# Overlay the individual country dots so we can see the spread
sns.stripplot(data=df, x='Colonizer', y='Holdout R^2', color='black', alpha=0.5, jitter=True)

for colonizer in df['Colonizer'].unique():
    group = df[df['Colonizer'] == colonizer]
    
    Q1 = group['Holdout R^2'].quantile(0.25)
    Q3 = group['Holdout R^2'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = group[(group['Holdout R^2'] < lower_bound) | (group['Holdout R^2'] > upper_bound)]

    
    
    for i in range(outliers.shape[0]):
        plt.text(
            x = list(df['Colonizer'].unique()).index(colonizer) + 0.1, 
            y = outliers['Holdout R^2'].iloc[i],
            s = outliers['Holdout Country'].iloc[i],
            fontsize=9, fontweight='bold', color='red' 
        )

plt.title('Is Model Accuracy ($R^2$) Linked to Colonial History?', fontsize=16)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('r2_vs_colonizerAvg.png', dpi=300)
plt.show()