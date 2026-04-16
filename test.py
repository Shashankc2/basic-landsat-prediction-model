import pandas as pd

# 1. Load the big revgeocoded file
file_path = '/Users/shashankchathapuram/Desktop/Basic Prediction Model/dhs_with_imgs_revgeocoded.csv'
df = pd.read_csv(file_path)

# 2. Find which column looks like it holds country names
# We look for common names like 'country', 'country_name', 'ADM0', or 'Holdout Country'
potential_cols = ['country', 'Country', 'country_name', 'Holdout Country', 'ADM0_NAME']
actual_col = next((c for c in potential_cols if c in df.columns), None)

if actual_col:
    print(f"--- Found country data in column: '{actual_col}' ---")
    countries = sorted(df[actual_col].unique())
    print(f"Total countries found: {len(countries)}")
    for c in countries:
        print(f" - {c}")
else:
    print("Could not find a country column. Here are all your columns:")
    print(df.columns.tolist())