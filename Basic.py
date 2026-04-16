import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


def train_evaluate_model(df, split_type = 'random', cutoff_year = 2018) :
    df_work = df.copy()
    df_work['rural'] = df_work['rural'].astype(int)
    
    features = ['lon', 'lat', 'rural']
    target = 'iwi'

    if split_type == 'random':
        train, test = train_test_split(df_work, test_size=0.2, random_state=42)

    elif split_type == 'oot':
        train = df_work[df_work['year'] < cutoff_year]
        test = df_work[df_work['year'] >= cutoff_year]

    elif split_type == 'oop':
        countries = df_work['country'].unique()
        np.random.seed(42)
        test_countries = np.random.choice(countries, size=int(0.2 * len(countries)), replace=False)
        train = df_work[~df_work['country'].isin(test_countries)]
        test = df_work[df_work['country'].isin(test_countries)]

    elif split_type == 'ootp':
        countries = df_work['country'].unique()
        np.random.seed(42)
        test_countries = np.random.choice(countries, size=int(0.3 * len(countries)), replace=False)
        train = df_work[(df_work['year'] < cutoff_year) & (~df_work['country'].isin(test_countries))]
        test = df_work[(df_work['year'] >= cutoff_year) & (df_work['country'].isin(test_countries))]

    
    # Training the Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train[features], train[target])

    # Evaluating the Model
    score = model.score(test[features], test[target])
    print(f"Train Size: {len(train)} | Test Size: {len(test)}")
    print(f"R^2 Score: {score:.4f}\n")
    return model




df = pd.read_csv('dhs_with_imgs_revgeocoded.csv')

model_random = train_evaluate_model(df, split_type='random')
model_oot    = train_evaluate_model(df, split_type='oot')
model_oop    = train_evaluate_model(df, split_type='oop')
model_ootp   = train_evaluate_model(df, split_type='ootp')



    




