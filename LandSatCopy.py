import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
outcomes = pd.read_parquet('outcomes.parquet')
landsat_embeddings = pd.read_parquet('landsat_embeddings.parquet')

# 2. Merge Features and Targets
data = outcomes.merge(
    landsat_embeddings[['cluster_id', 'Year'] + [f'F_{i}' for i in range(384)]],
    left_on=['cluster_id', 'year'],   
    right_on=['cluster_id', 'Year'],
    how='inner'
)

# 3. Define Features
features = [col for col in data.columns if col.startswith('F_')]
if 'rural' in data.columns:
    features.append('rural')
    data['rural'] = data['rural'].astype(int)

countries = data['country'].unique()
benchmarks = [1995, 2000, 2005, 2010, 2015]
results = []

# Formatting for the Real-Time Table
header = f"{'Holdout Country':<15} | {'Year':<6} | {'Train Size':<10} | {'Train R^2':<10} | {'Holdout R^2':<12} | {'Other R^2':<10}"
print(header)
print("-" * len(header))

# 4. The Evaluation Loop
for country in countries:
    other_data = data[data['country'] != country]
    holdout_data = data[data['country'] == country]

    for year in benchmarks:
        train = other_data[other_data['year'] < year]
        test_holdout = holdout_data[holdout_data['year'] >= year]
        test_other = other_data[other_data['year'] >= year]

        if train.empty or test_holdout.empty or test_other.empty:
            continue

        # Train Random Forests
        aplhas = [0.1, 1.0, 10.0, 100, 1000]
        model = RidgeCV(alphas=aplhas, cv=5)
        model.fit(train[features], train['iwi'])

        print(f"Best alpha for {country}: {model.alpha_}")

        # Calculate Scores
        score_train = model.score(train[features], train['iwi'])
        score_holdout = model.score(test_holdout[features], test_holdout['iwi'])
        score_other = model.score(test_other[features], test_other['iwi'])

        # Store results
        results.append({
            'holdout_country': country,
            'benchmark_year': year,
            'train_r2': score_train,
            'test_r2_holdout': score_holdout,
            'test_r2_other': score_other,
            'train_size': len(train)
        })

        # Print the row with the newly added Train R^2
        print(f"{country:<15} | {year:<6} | {len(train):<10} | {score_train:>10.4f} | {score_holdout:>12.4f} | {score_other:>10.4f}")

# 5. Final Summary Output
results_df = pd.DataFrame(results)

print("\n" + "="*40)
print("--- OVERALL SUMMARY STATISTICS ---")
# 1. Median Metrics 
print(f"Median Training R^2:      {results_df['train_r2'].median():.4f}")
print(f"Median Holdout R^2:       {results_df['test_r2_holdout'].median():.4f}" )
print(f"Median Other R^2:         {results_df['test_r2_other'].median():.4f}")

print("-" * 50)

# 2. Mean Metrics 
print(f"Mean Training R^2:        {results_df['train_r2'].mean():.4f}")
print(f"Mean Holdout R^2:         {results_df['test_r2_holdout'].mean():.4f}")
print(f"Mean Other R^2:           {results_df['test_r2_other'].mean():.4f}")

print("-" * 50)


gap = results_df['train_r2'].median() - results_df['test_r2_holdout'].median()
print(f"Median Generalization Gap: {gap:.4f}")
print("="*50)