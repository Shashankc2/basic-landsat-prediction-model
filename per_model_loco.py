import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Leave-one-country-out x benchmark-year evaluation (same protocol as LandSat.py) for the
# models that landsat_results_final.csv does not contain. RandomForest rows are copied from
# that CSV so Fig 4-6 stay on the original RF run.
#   train = all other countries with year < benchmark
#   test  = held-out country with year >= benchmark
# Output: landsat_results_per_model.csv  (Model, Holdout Country, Benchmark Year, Train Size, Holdout R^2)

N_JOBS = 6
BENCHMARKS = [1995, 2000, 2005, 2010, 2015]

outcomes = pd.read_parquet('outcomes.parquet')
emb = pd.read_parquet('landsat_embeddings.parquet')
data = outcomes.merge(
    emb[['cluster_id', 'Year'] + [f'F_{i}' for i in range(384)]],
    left_on=['cluster_id', 'year'], right_on=['cluster_id', 'Year'], how='inner'
)
data['rural'] = data['rural'].astype(int)
features = [c for c in data.columns if c.startswith('F_')] + ['rural']

X = data[features].values
y = data['iwi'].values
years = data['year'].values
ctry = data['country'].values
countries = data['country'].unique()

MODELS = {'Ridge': lambda: Ridge(alpha=1.0), 'SVR': lambda: SVR(C=1.0)}


def run_one(model_name, country, year):
    tr = (ctry != country) & (years < year)
    te = (ctry == country) & (years >= year)
    te_other = (ctry != country) & (years >= year)
    if tr.sum() == 0 or te.sum() == 0 or te_other.sum() == 0:   # same skip rule as LandSat.py
        return None
    t = time.time()
    m = MODELS[model_name]()
    m.fit(X[tr], y[tr])
    r2 = float(r2_score(y[te], m.predict(X[te])))
    print(f"{model_name:<5} {country:<28} {year} N={tr.sum():>6} R2={r2:7.4f} ({time.time()-t:.0f}s)", flush=True)
    return {'Model': model_name, 'Holdout Country': country, 'Benchmark Year': year,
            'Train Size': int(tr.sum()), 'Holdout R^2': r2}


# Largest benchmarks first so the long SVR fits start early
tasks = [(m, c, yr) for yr in sorted(BENCHMARKS, reverse=True) for m in MODELS for c in countries]
rows = Parallel(n_jobs=N_JOBS)(delayed(run_one)(*t) for t in tasks)
df = pd.DataFrame([r for r in rows if r is not None])

rf = pd.read_csv('landsat_results_final.csv')[['Holdout Country', 'Benchmark Year', 'Train Size', 'Holdout R^2']]
rf.insert(0, 'Model', 'RandomForest')

out = pd.concat([rf, df], ignore_index=True).sort_values(['Model', 'Benchmark Year', 'Holdout Country'])
out.to_csv('landsat_results_per_model.csv', index=False)
print(out.groupby(['Model', 'Benchmark Year']).size().unstack())
print("Wrote landsat_results_per_model.csv")
