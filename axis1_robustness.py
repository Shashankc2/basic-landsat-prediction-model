import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Robustness checks for Axis 1 (temporal out-of-time scaling).
#   baseline : cumulative windows (as in compCache.py) + holdout-country bootstrap CIs
#   equalN   : every cumulative window subsampled to the smallest window's N
#   ageBlocks: disjoint 5-year blocks at equal (small) N -> isolates training-data age
# Holdout = survey year >= 2015. Bootstrap resamples holdout countries (rows within a
# country are spatially correlated), with the same draw reused across windows (paired).

N_BOOT = 2000
REPS_EQUAL_N = 5
REPS_AGE = 10
HOLDOUT_FROM = 2015
SEED = 20260101

df = pd.read_parquet('master_merged_data.parquet')
F = [c for c in df.columns if c.startswith('F_')]
d = df.dropna(subset=['iwi'] + F).reset_index(drop=True)
years, y_all, X_all, ctry = d['year'].values, d['iwi'].values, d[F].values, d['country'].values

ho = years >= HOLDOUT_FROM
X_ho, y_ho = X_all[ho], y_all[ho]
countries = np.unique(ctry[ho])
c_idx = np.searchsorted(countries, ctry[ho])
n_c = len(countries)
print(f"holdout rows={ho.sum()} countries={n_c}", flush=True)


def make_models(rep):
    return {
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(C=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42 + rep, n_jobs=-1),
    }


def country_stats(pred):
    """Per-country sufficient statistics for R^2: n, sum y, sum y^2, SSE."""
    err2 = (y_ho - pred) ** 2
    return np.stack([
        np.bincount(c_idx, minlength=n_c).astype(float),
        np.bincount(c_idx, weights=y_ho, minlength=n_c),
        np.bincount(c_idx, weights=y_ho ** 2, minlength=n_c),
        np.bincount(c_idx, weights=err2, minlength=n_c),
    ])


def r2_from_stats(st, w):
    """R^2 for country weights w (n_boot, n_c) -> (n_boot,)."""
    n, sy, syy, sse = (w @ st[i] for i in range(4))
    sst = syy - sy ** 2 / n
    return 1.0 - sse / sst


rng = np.random.default_rng(SEED)
W_boot = np.stack([np.bincount(rng.integers(0, n_c, n_c), minlength=n_c) for _ in range(N_BOOT)]).astype(float)
W_full = np.ones((1, n_c))


def evaluate(train_mask_fn, reps, n_sub=None):
    """Fit each model on (optionally subsampled) training rows; return per-model R2 stats.

    Returns {model: (point_r2, boot_r2[N_BOOT])}, with R^2 averaged over reps.
    """
    pts = {m: [] for m in ['Ridge', 'SVR', 'RandomForest']}
    boots = {m: [] for m in pts}
    idx_all = np.where(train_mask_fn)[0]
    for r in range(reps):
        idx = idx_all
        if n_sub is not None and len(idx_all) > n_sub:
            idx = np.random.default_rng(SEED + r).choice(idx_all, size=n_sub, replace=False)
        for mname, model in make_models(r).items():
            t = time.time()
            model.fit(X_all[idx], y_all[idx])
            st = country_stats(model.predict(X_ho))
            pts[mname].append(r2_from_stats(st, W_full)[0])
            boots[mname].append(r2_from_stats(st, W_boot))
            print(f"    rep {r} {mname:<12} N={len(idx):>6} R2={pts[mname][-1]:.4f} ({time.time()-t:.0f}s)", flush=True)
    return {m: (float(np.mean(pts[m])), np.mean(boots[m], axis=0), float(np.std(pts[m]))) for m in pts}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


out = {'n_boot': N_BOOT, 'n_holdout_countries': int(n_c), 'holdout_rows': int(ho.sum())}

# --------------------------------------------------------------------------
# 1. Baseline: cumulative windows, bootstrap CIs on points and on the log-N slope
# --------------------------------------------------------------------------
windows = [('2010–14', 2010), ('2005–14', 2005), ('2000–14', 2000), ('1990–14', 1990)]
print("BASELINE (cumulative windows)", flush=True)
base = {}
Ns = []
for label, start in windows:
    mask = (years >= start) & (years <= 2014)
    Ns.append(int(mask.sum()))
    print(f"  window {label}: N={mask.sum()}", flush=True)
    base[label] = evaluate(mask, reps=1)
logN = np.log(Ns)
out['baseline'] = {'windows': [w[0] for w in windows], 'N': Ns, 'models': {}}
for m in ['Ridge', 'SVR', 'RandomForest']:
    pt = np.array([base[w[0]][m][0] for w in windows])
    bt = np.stack([base[w[0]][m][1] for w in windows])          # (4, B)
    slope = float(np.polyfit(logN, pt, 1)[0])
    slopes = np.array([np.polyfit(logN, bt[:, b], 1)[0] for b in range(N_BOOT)])
    out['baseline']['models'][m] = {
        'R2': pt.tolist(),
        'ci': [ci(bt[i]) for i in range(len(windows))],
        'slope': slope, 'slope_ci': ci(slopes), 'p_slope_ge_0': float((slopes >= 0).mean()),
    }

# --------------------------------------------------------------------------
# 2. Equal-N cumulative windows (subsample every window to the smallest N)
# --------------------------------------------------------------------------
n_eq = min(Ns)
print(f"EQUAL-N CUMULATIVE (n={n_eq}, reps={REPS_EQUAL_N})", flush=True)
eq = {}
for label, start in windows:
    mask = (years >= start) & (years <= 2014)
    print(f"  window {label}", flush=True)
    eq[label] = evaluate(mask, reps=1 if mask.sum() == n_eq else REPS_EQUAL_N, n_sub=n_eq)
out['equalN'] = {'n': int(n_eq), 'windows': [w[0] for w in windows], 'models': {}}
for m in ['Ridge', 'SVR', 'RandomForest']:
    first, last = windows[0][0], windows[-1][0]
    diff = eq[last][m][1] - eq[first][m][1]
    out['equalN']['models'][m] = {
        'R2': [eq[w[0]][m][0] for w in windows],
        'ci': [ci(eq[w[0]][m][1]) for w in windows],
        'rep_sd': [eq[w[0]][m][2] for w in windows],
        'diff_oldest_minus_newest': float(eq[last][m][0] - eq[first][m][0]),
        'diff_ci': ci(diff), 'p_diff_ge_0': float((diff >= 0).mean()),
    }

# --------------------------------------------------------------------------
# 3. Disjoint 5-year blocks at equal N -> pure effect of training-data age
# --------------------------------------------------------------------------
blocks = [('2010–14', 2010, 2014), ('2005–09', 2005, 2009), ('2000–04', 2000, 2004), ('1992–99', 1990, 1999)]
n_blk = min(int(((years >= a) & (years <= b)).sum()) for _, a, b in blocks)
print(f"AGE BLOCKS (n={n_blk}, reps={REPS_AGE})", flush=True)
ab = {}
for label, a, b in blocks:
    print(f"  block {label}", flush=True)
    ab[label] = evaluate((years >= a) & (years <= b), reps=REPS_AGE, n_sub=n_blk)
out['ageBlocks'] = {'n': n_blk, 'blocks': [b[0] for b in blocks], 'models': {}}
for m in ['Ridge', 'SVR', 'RandomForest']:
    first, last = blocks[0][0], blocks[-1][0]
    diff = ab[last][m][1] - ab[first][m][1]
    out['ageBlocks']['models'][m] = {
        'R2': [ab[b[0]][m][0] for b in blocks],
        'ci': [ci(ab[b[0]][m][1]) for b in blocks],
        'rep_sd': [ab[b[0]][m][2] for b in blocks],
        'diff_oldest_minus_newest': float(ab[last][m][0] - ab[first][m][0]),
        'diff_ci': ci(diff), 'p_diff_ge_0': float((diff >= 0).mean()),
    }

os.makedirs('cache', exist_ok=True)
with open('cache/axis1_robust.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Wrote cache/axis1_robust.json", flush=True)
