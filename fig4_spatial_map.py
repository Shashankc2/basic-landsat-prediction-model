import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

MM_TO_INCHES = 1 / 25.4
FIG_WIDTH = 180 * MM_TO_INCHES
FIG_HEIGHT = 68 * MM_TO_INCHES

fig, axes = plt.subplots(1, 5, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.82, bottom=0.22)

with open("cache/map_metrics.json") as f:
    MAP_DATA = json.load(f)

url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
world = gpd.read_file(url)
name_col = next((c for c in ['name', 'name_en', 'NAME'] if c in world.columns), 'name')

# Filter Africa and use standard ESRI Africa Albers projection (or fallback cleanly)
africa = world[world['continent'] == 'Africa'].copy().rename(columns={name_col: 'NAME'})

try:
    africa = africa.to_crs("ESRI:102022")
except Exception:
    africa = africa.to_crs("EPSG:4326")

CMAP = plt.cm.RdYlGn
NORM = mcolors.Normalize(vmin=-0.5, vmax=0.7)

for idx, year in enumerate(MAP_DATA['years']):
    ax = axes[idx]
    year_str = str(year)
    africa.plot(ax=ax, color='#EEEEEE', edgecolor='#FFFFFF', linewidth=0.35, zorder=1)
    
    for country, r2_dict in MAP_DATA['country_r2'].items():
        if year_str in r2_dict:
            color = CMAP(NORM(r2_dict[year_str]))
            c_geom = africa[africa['NAME'] == country]
            if not c_geom.empty:
                c_geom.plot(ax=ax, color=color, edgecolor='#FFFFFF', linewidth=0.35, zorder=2)

    ax.axis('off')
    n_size = MAP_DATA['sample_sizes'][year_str]
    ax.text(0.5, 1.05, f"{year}\n(N = {n_size:,})", transform=ax.transAxes, fontsize=7.5, fontweight='bold', ha='center', va='bottom')

fig.suptitle('Model Evaluation Over Time (1995–2015)', fontsize=10, fontweight='bold', x=0.5, y=0.96)
cax = fig.add_axes([0.30, 0.12, 0.40, 0.025])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm._A = []
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=6.5)
cbar.set_label(r'Holdout $R^2$ Accuracy', fontsize=7.5, labelpad=3)

plt.savefig('Figure_Spatial_Map.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('Figure_Spatial_Map.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.close()
print("Rendered Figure_Spatial_Map successfully.")