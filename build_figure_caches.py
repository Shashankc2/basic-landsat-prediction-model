import os
import json
import pandas as pd

# Builds the per-model colonizer cache for Fig 6 from landsat_results_per_model.csv
# (written by per_model_loco.py). Keeps the original single-model 'groups' block
# (RandomForest) for backward compatibility with compCache.py output.

name_map = {
    "Congo - Kinshasa": "Democratic Republic of the Congo",
    "Côte d’Ivoire": "Ivory Coast",
    "Eswatini": "Swaziland",
    "Tanzania": "United Republic of Tanzania"
}

# Same mapping as compCache.py
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

df = pd.read_csv('landsat_results_per_model.csv')
df['Holdout Country'] = df['Holdout Country'].replace(name_map)
df['Colonizer'] = df['Holdout Country'].map(colonial_map).fillna('Other')

path = 'cache/colonizer_metrics.json'
with open(path) as f:
    cache = json.load(f)

by_model = {}
for model, mdf in df.groupby('Model'):
    by_model[model] = {}
    for group, gdf in mdf.groupby('Colonizer'):
        by_model[model][group] = {
            'countries': gdf['Holdout Country'].tolist(),
            'r2_scores': gdf['Holdout R^2'].tolist(),
            'median_r2': float(gdf['Holdout R^2'].median()),
        }
cache['by_model'] = by_model
with open(path, 'w') as f:
    json.dump(cache, f, indent=2)
print({m: {g: len(v['r2_scores']) for g, v in gs.items()} for m, gs in by_model.items()})
