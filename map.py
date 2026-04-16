import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('landsat_results_final.csv')

url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
world = gpd.read_file(url)

name_col = next((c for c in ['name', 'name_en', 'NAME'] if c in world.columns), 'name')
africa = world[world['continent'] == 'Africa'].copy().rename(columns={name_col: 'NAME'})

name_map = {
    "Congo - Kinshasa": "Democratic Republic of the Congo",
    "Côte d’Ivoire": "Ivory Coast",
    "Central African Republic": "Central African Republic",
    "Eswatini": "Swaziland",
    "Gambia": "Gambia",
    "Tanzania": "United Republic of Tanzania"
}
df['Holdout Country'] = df['Holdout Country'].replace(name_map)

years = [1995, 2000, 2005, 2010, 2015]
fig, axes = plt.subplots(1, 5, figsize=(25, 8), dpi=200)

vmin, vmax = -0.5, 0.7


for i, year in enumerate(years):
    ax = axes[i]
    year_df = df[df['Benchmark Year'] == year]
    
    merged = africa.merge(year_df, left_on='NAME', right_on='Holdout Country', how='left')
    
    africa.plot(ax=ax, color='#f2f2f2', edgecolor='#ffffff', linewidth=0.5)
    
    merged.dropna(subset=['Holdout R^2']).plot(
        column='Holdout R^2',
        ax=ax,
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax
    )
    
    train_n = year_df['Train Size'].iloc[0] if not year_df.empty else 0
    ax.set_title(f"{year}\n(N={train_n})", fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

plt.suptitle("Model Evaluation Over Time (1995–2015)", 
             fontsize=24, y=1.05, fontweight='bold')

sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.02]) 
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Holdout R² Accuracy', fontsize=12, labelpad=10)

plt.tight_layout()
plt.savefig('model_map.png', bbox_inches='tight')
print("5-map sequence saved as 'model_map.png'")
plt.show()