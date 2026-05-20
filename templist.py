import pandas as pd

df = pd.read_parquet('master_merged_data.parquet')

top_features = ['F_224', 'F_279', 'F_36', 'F_150', 'F_77'] 

results = []

for feat in top_features:
    high = df.nlargest(5, feat).copy()
    low = df.nsmallest(5, feat).copy()
    
    high['type'] = 'High'
    low['type'] = 'Low'
    high['feature_name'] = feat
    low['feature_name'] = feat
    
    results.extend([high, low])

final_list = pd.concat(results)[['feature_name', 'type', 'cluster_id', 'year', 'lat', 'lon', feat]]
final_list.to_csv('target_locations.csv', index=False)

print("Checklist created: target_locations.csv")
print(final_list.head(50))