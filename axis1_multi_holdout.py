import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Axis 1 repeated out-of-time evaluation: the same four cumulative training windows
# (all ending in 2014) are each fit once and then scored on three separate future test periods.
# If anti-scaling shows up in every period, it is not an artifact of one particular holdout.
# Uncertainty: country-level bootstrap within each test period (same draws across windows).

N_BOOT = 2000
SEED = 20260102
WINDOWS = [('2010–14', 2010), ('2005–14', 2005), ('2000–14', 2000), ('1990–14', 1990)]
PERIODS = [('2015–17', 2015, 2017), ('2018–20', 2018, 2020), ('2021–23', 2021, 2023)]

df = pd.read_parquet('master_merged_data.parquet')
F = [c for c in df.columns if c.startswith('F_')]
d = df.dropna(subset=['iwi'] + F).reset_index(drop=True)
years, y_all, X_all, ctry = d['year'].values, d['iwi'].values, d[F].values, d['country'].values

ho = years >= 2015
X_ho, y_ho, yr_ho, c_ho = X_all[ho], y_all[ho], years[ho], ctry[ho]

MODELS = {
    'Ridge': lambda: Ridge(alpha=1.0),
    'SVR': lambda: SVR(C=1.0),
    'RandomForest': lambda: RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=4),
}

# 1. Fit each (window, model) once; predict the full 2015+ set
Ns, preds = [], {}
for wl, start in WINDOWS:
    tr = (years >= start) & (years <= 2014)
    Ns.append(int(tr.sum()))
    for m, make in MODELS.items():
        t = time.time()
        model = make()
        model.fit(X_all[tr], y_all[tr])
        preds[(wl, m)] = model.predict(X_ho)
        print(f"window {wl} N={tr.sum()} {m:<12} fit+predict {time.time()-t:.0f}s", flush=True)
logN = np.log(Ns)
np.savez_compressed('cache/axis1_multi_holdout_preds.npz',
                    **{f"{wl}|{m}": p for (wl, m), p in preds.items()}, y=y_ho, year=yr_ho, country=c_ho)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# 2. Score per test period with country bootstrap
out = {'windows': [w[0] for w in WINDOWS], 'N': Ns, 'periods': {}, 'n_boot': N_BOOT}
rng = np.random.default_rng(SEED)
for pl, a, b in PERIODS:
    sel = (yr_ho >= a) & (yr_ho <= b)
    ys, cs = y_ho[sel], c_ho[sel]
    countries = np.unique(cs)
    ci_idx = np.searchsorted(countries, cs)
    k = len(countries)
    W = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(N_BOOT)]).astype(float)

    def stats(pred):
        err2 = (ys - pred) ** 2
        return np.stack([np.bincount(ci_idx, minlength=k).astype(float),
                         np.bincount(ci_idx, weights=ys, minlength=k),
                         np.bincount(ci_idx, weights=ys ** 2, minlength=k),
                         np.bincount(ci_idx, weights=err2, minlength=k)])

    def r2(st, w):
        n, sy, syy, sse = (w @ st[i] for i in range(4))
        return 1.0 - sse / (syy - sy ** 2 / n)

    res = {'n_rows': int(sel.sum()), 'n_countries': int(k), 'models': {}}
    for m in MODELS:
        pt, bt = [], []
        for wl, _ in WINDOWS:
            st = stats(preds[(wl, m)][sel])
            pt.append(float(r2(st, np.ones((1, k)))[0]))
            bt.append(r2(st, W))
        bt = np.stack(bt)                                      # (4, B)
        slope = float(np.polyfit(logN, pt, 1)[0])
        slopes = np.array([np.polyfit(logN, bt[:, i], 1)[0] for i in range(N_BOOT)])
        diff = bt[-1] - bt[0]                                  # oldest window minus newest window
        res['models'][m] = {
            'R2': pt, 'ci': [ci(bt[i]) for i in range(len(WINDOWS))],
            'slope': slope, 'slope_ci': ci(slopes), 'p_slope_ge_0': float((slopes >= 0).mean()),
            'diff_oldest_minus_newest': float(pt[-1] - pt[0]), 'diff_ci': ci(diff),
        }
    out['periods'][pl] = res

with open('cache/axis1_multi_holdout.json', 'w') as f:
    json.dump(out, f, indent=2)
for pl, r in out['periods'].items():
    print(pl, r['n_rows'], 'rows', r['n_countries'], 'countries',
          {m: (round(v['slope'], 4), [round(x, 4) for x in v['slope_ci']]) for m, v in r['models'].items()}, flush=True)
print("Wrote cache/axis1_multi_holdout.json")
