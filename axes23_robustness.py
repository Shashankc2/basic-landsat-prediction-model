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

# Repeats + uncertainty for Axis 2 (out-of-country) and Axis 3 (within-country density).
# Same protocol as compCache.py, but:
#   Axis 2: REPS_A2 independent random donor sets per (target country, k); deterministic seeds
#           (compCache used Python hash(), which changes every run).
#   Axis 3: REPS_A3 random 80/20 splits per country (compCache used one fixed split).
# Uncertainty: bootstrap over target countries (the unit of independence), same draws across
# k / fractions so the paired slope CI is meaningful.

REPS_A2, REPS_A3 = 5, 20
N_BOOT, SEED, N_JOBS = 2000, 20260104, 10
MODEL_NAMES = ['Ridge', 'SVR', 'RandomForest']

df = pd.read_parquet('master_merged_data.parquet')
F = [c for c in df.columns if c.startswith('F_')]
d = df[(df['year'] >= 2015) & (df['year'] <= 2021)].dropna(subset=['iwi', 'country'] + F).reset_index(drop=True)
X, y, ctry = d[F].values, d['iwi'].values, d['country'].values
all_countries = sorted(pd.unique(ctry))
print(f"rows={len(d)} countries={len(all_countries)}", flush=True)


def make_model(name, rep):
    return {'Ridge': lambda: Ridge(alpha=1.0), 'SVR': lambda: SVR(C=1.0),
            'RandomForest': lambda: RandomForestRegressor(n_estimators=50, random_state=42 + rep, n_jobs=1)}[name]()


# ---------------------------------------------------------------- Axis 2
K_STEPS = [k for k in [1, 3, 7, 15, 27] if k <= len(all_countries) - 1]
targets = [c for c in all_countries if (ctry == c).sum() >= 30]


def a2_task(t_idx, rep, mname):
    target = targets[t_idx]
    te = ctry == target
    donors = [c for c in all_countries if c != target]
    out = []
    for k in K_STEPS:
        rng = np.random.default_rng([SEED, t_idx, k, rep])          # same donors for every model
        chosen = rng.choice(donors, size=k, replace=False)
        tr = np.isin(ctry, chosen)
        m = make_model(mname, rep)
        m.fit(X[tr], y[tr])
        out.append((k, int(tr.sum()), float(r2_score(y[te], m.predict(X[te])))))
    return t_idx, rep, mname, out


print(f"AXIS 2: {len(targets)} targets x {REPS_A2} reps x 3 models", flush=True)
a2 = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(a2_task)(t, r, m) for t in range(len(targets))
    for r in range(REPS_A2) for m in MODEL_NAMES)

# ---------------------------------------------------------------- Axis 3
FRACTIONS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
a3_countries = [c for c in all_countries if (ctry == c).sum() >= 60]


def a3_task(c_idx, rep, mname):
    c = a3_countries[c_idx]
    Xc, yc = X[ctry == c], y[ctry == c]
    seed = int(np.random.default_rng([SEED, 3, c_idx, rep]).integers(0, 2**31 - 1))
    X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.20, random_state=seed, shuffle=True)
    out = []
    for fr in FRACTIONS:
        n_sub = int(len(X_tr) * fr)
        if n_sub < 15:
            out.append((fr, n_sub, float('nan')))
            continue
        m = make_model(mname, rep)
        m.fit(X_tr[:n_sub], y_tr[:n_sub])
        out.append((fr, n_sub, float(r2_score(y_te, m.predict(X_te)))))
    return c_idx, rep, mname, out


print(f"AXIS 3: {len(a3_countries)} countries x {REPS_A3} reps x 3 models", flush=True)
a3 = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(a3_task)(c, r, m) for c in range(len(a3_countries))
    for r in range(REPS_A3) for m in MODEL_NAMES)


# ---------------------------------------------------------------- aggregate + bootstrap
def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def aggregate(res, n_units, n_steps, reps, stat):
    """res: list of (unit, rep, model, [(step, N, r2)...]). Returns per-model dict."""
    out = {}
    for mname in MODEL_NAMES:
        R = np.full((n_units, reps, n_steps), np.nan)
        N = np.full((n_units, reps, n_steps), np.nan)
        for u, r, m, rows in res:
            if m != mname:
                continue
            for j, (_, n, r2) in enumerate(rows):
                R[u, r, j], N[u, r, j] = r2, n
        R_u = np.nanmean(R, axis=1)                                  # (units, steps): mean over reps
        N_step = (np.nanmedian if stat == 'median' else np.nanmean)(np.nanmean(N, axis=1), axis=0)
        agg = (np.nanmedian if stat == 'median' else np.nanmean)
        point = agg(R_u, axis=0)
        rng = np.random.default_rng(SEED + 7)
        idx = rng.integers(0, n_units, (N_BOOT, n_units))            # same draws for every step
        boot = agg(R_u[idx], axis=1)                                 # (B, steps)
        logN = np.log(N_step)
        slope = float(np.polyfit(logN, point, 1)[0])
        slopes = np.array([np.polyfit(logN, boot[b], 1)[0] for b in range(N_BOOT)])
        out[mname] = {
            'N': N_step.tolist(), 'R2': point.tolist(),
            'ci': [ci(boot[:, j]) for j in range(n_steps)],
            'slope': slope, 'slope_ci': ci(slopes), 'p_slope_le_0': float((slopes <= 0).mean()),
            'rep_sd': np.nanmean(np.nanstd(R, axis=1), axis=0).tolist(),   # mean within-country sd across reps
        }
    return out


os.makedirs('cache', exist_ok=True)
a2_models = aggregate(a2, len(targets), len(K_STEPS), REPS_A2, 'median')
with open('cache/axis2_robust.json', 'w') as f:
    json.dump({'k': K_STEPS, 'n_targets': len(targets), 'reps': REPS_A2, 'n_boot': N_BOOT, 'aggregate': 'median over targets',
               'models': a2_models}, f, indent=2)

a3_models = aggregate(a3, len(a3_countries), len(FRACTIONS), REPS_A3, 'mean')
with open('cache/axis3_robust.json', 'w') as f:
    json.dump({'fractions': FRACTIONS, 'n_countries': len(a3_countries), 'reps': REPS_A3, 'n_boot': N_BOOT,
               'aggregate': 'mean over countries', 'models': a3_models}, f, indent=2)

for name, mods in [('AXIS2', a2_models), ('AXIS3', a3_models)]:
    for m, v in mods.items():
        print(name, m, 'R2', [round(x, 3) for x in v['R2']], 'slope', round(v['slope'], 4),
              [round(x, 4) for x in v['slope_ci']], flush=True)
print("Wrote cache/axis2_robust.json and cache/axis3_robust.json")
