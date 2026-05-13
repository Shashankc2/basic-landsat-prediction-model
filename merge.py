import pandas as pd

# 1. LOAD THE DATA
print("Loading data...")
outcomes = pd.read_csv('dhs_with_imgs_revgeocoded.csv') 
embeddings = pd.read_parquet('landsat_embeddings.parquet')

# 2. FIX THE IDENTIFIERS (The "String" Fix)
# Instead of int, we convert both to string (str) to handle IDs like 'AO.Bengo.71.135'
outcomes['cluster_id'] = outcomes['cluster_id'].astype(str)
embeddings['cluster_id'] = embeddings['cluster_id'].astype(str)

# Standardize the year column name
if 'Year' in embeddings.columns:
    embeddings = embeddings.rename(columns={'Year': 'year'})

# Make sure year is also the same type (int is usually fine for years)
outcomes['year'] = outcomes['year'].astype(int)
embeddings['year'] = embeddings['year'].astype(int)

# 3. THE MERGE
print("Merging datasets...")
merged_df = outcomes.merge(
    embeddings, 
    on=['cluster_id', 'year'], 
    how='inner'
)

# 4. VALIDATION
print(f"Successfully merged! Final shape: {merged_df.shape}")
if merged_df.empty:
    print("WARNING: Merged dataframe is empty. Check if cluster_id formats match exactly.")
    print(f"Outcome ID sample: {outcomes['cluster_id'].iloc[0]}")
    print(f"Embedding ID sample: {embeddings['cluster_id'].iloc[0]}")
else:
    # 5. SAVE THE MASTER FILE
    merged_df.to_parquet('master_merged_data.parquet')
    print("Master file saved as 'master_merged_data.parquet'")