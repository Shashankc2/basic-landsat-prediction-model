import pandas as pd

outcomes = pd.read_csv('dhs_with_imgs_revgeocoded.csv') 
embeddings = pd.read_parquet('landsat_embeddings.parquet')

# Instead of int, we convert both to string
outcomes['cluster_id'] = outcomes['cluster_id'].astype(str)
embeddings['cluster_id'] = embeddings['cluster_id'].astype(str)

# Standardize the year column name
if 'Year' in embeddings.columns:
    embeddings = embeddings.rename(columns={'Year': 'year'})

# Make sure year is also the same type (int is usually fine for years)
outcomes['year'] = outcomes['year'].astype(int)
embeddings['year'] = embeddings['year'].astype(int)

print("Merging datasets...")
merged_df = outcomes.merge(
    embeddings, 
    on=['cluster_id', 'year'], 
    how='inner'
)

print(f"Successfully merged! Final shape: {merged_df.shape}")
if merged_df.empty:
    print("WARNING: Merged dataframe is empty. Check if cluster_id formats match exactly.")
    print(f"Outcome ID sample: {outcomes['cluster_id'].iloc[0]}")
    print(f"Embedding ID sample: {embeddings['cluster_id'].iloc[0]}")
else:
    merged_df.to_parquet('master_merged_data.parquet')
    print("Master file saved as 'master_merged_data.parquet'")