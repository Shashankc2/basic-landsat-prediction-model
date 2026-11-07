import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Fig 7 data: within-window holdout R^2 for 5-year rolling windows (same 80/20 within-window
# protocol as temporal_analysis_pipeline.py), for Ridge / RandomForest / SVR.
# Repeated over random splits; writes mean and sd per window to cache/rolling_window_metrics.json.

WINDOWS = [(1995, 1999), (1999, 2003), (2003, 2007), (2007, 2011), (2011, 2015)]
REPS = 5
N_JOBS = 4

df = pd.read_parquet('master_merged_data.parquet')
F = [c for c in df.columns if c.startswith('F_')]
df = df.dropna(subset=['iwi'] + F)

MODELS = {
    'Ridge': lambda s: Ridge(alpha=1.0),
    'RandomForest': lambda s: RandomForestRegressor(n_estimators=50, random_state=s, n_jobs=1),
    'SVR': lambda s: SVR(C=1.0),
}


def run(w_idx, model_name, rep):
    a, b = WINDOWS[w_idx]
    sub = df[(df['year'] >= a) & (df['year'] <= b)]
    X_tr, X_te, y_tr, y_te = train_test_split(sub[F].values, sub['iwi'].values,
                                              test_size=0.20, random_state=42 + rep, shuffle=True)
    m = MODELS[model_name](42 + rep)
    m.fit(X_tr, y_tr)
    r2 = float(r2_score(y_te, m.predict(X_te)))
    print(f"{a}-{b} {model_name:<12} rep {rep} N={len(sub)} R2={r2:.4f}", flush=True)
    return w_idx, model_name, rep, r2, len(sub)


tasks = [(w, m, r) for w in reversed(range(len(WINDOWS))) for m in MODELS for r in range(REPS)]
res = Parallel(n_jobs=N_JOBS)(delayed(run)(*t) for t in tasks)

out = {'windows': [f"{a}–{b}" for a, b in WINDOWS], 'n': [], 'models': {}, 'sd': {}, 'reps': REPS}
for w in range(len(WINDOWS)):
    out['n'].append(next(r[4] for r in res if r[0] == w))
for m in MODELS:
    means, sds = [], []
    for w in range(len(WINDOWS)):
        vals = [r[3] for r in res if r[0] == w and r[1] == m]
        means.append(float(np.mean(vals)))
        sds.append(float(np.std(vals, ddof=1)))
    out['models'][m] = means
    out['sd'][m] = sds

os.makedirs('cache', exist_ok=True)
with open('cache/rolling_window_metrics.json', 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
