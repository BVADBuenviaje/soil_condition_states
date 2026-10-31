"""Stability under resampling, recognisability on a withheld node, and the run length
comparison against surrogate records. Writes validation_stability.csv,
validation_prediction_strength.csv, validation_folds.csv, validation_persistence.csv
and validation_tests.json."""
import json, warnings
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from common import load, name_clusters, present, run_lengths, FEAT, K, SEED

warnings.filterwarnings('ignore')
RESAMPLES, SURROGATES, BURN = 500, 100, 200
K_RANGE = [3, 4, 5, 6]
STABLE, DOUBTFUL, DISSOLVED, RECOGNISED = 0.75, 0.6, 0.5, 0.8
TOL = pd.Timedelta(minutes=20)
rng = np.random.default_rng(SEED)

d, X = load()
nodes = sorted(d.node.unique())
ref = KMeans(K, n_init=30, random_state=SEED).fit(X)
ref_names = name_clusters(ref.cluster_centers_)
ref_lab = np.array([ref_names[c] for c in ref.labels_], dtype=object)
ORDER = present(ref_lab)
sizes = {s: int((ref_lab == s).sum()) for s in ORDER}
print('reference states :', ORDER)
print('reference sizes  :', sizes, 'total', len(d))

DAY_BLOCKS = [g.values for _, g in pd.Series(np.arange(len(d))).groupby(
    [d.node.values, d.ts.dt.normalize().values])]
HOUR_PAIRS = {}
for n in nodes:
    pos = np.flatnonzero((d.node == n).values)
    t = d.ts.values[pos]
    gap = (t[1:] - t[:-1]) / np.timedelta64(1, 'h')
    keep = (gap >= 0.75) & (gap <= 1.25)
    HOUR_PAIRS[n] = (pos[:-1][keep], pos[1:][keep])
PREV = {}
for h in (1, 3):
    col = np.full(len(d), -1)
    for n in nodes:
        pos = np.flatnonzero((d.node == n).values)
        s = pd.Series(pos, index=pd.DatetimeIndex(d.ts.values[pos]))
        m = s.reindex(s.index - pd.Timedelta(hours=h), method='nearest', tolerance=TOL)
        col[pos] = np.where(np.isnan(m.values), -1, np.nan_to_num(m.values, nan=-1)).astype(int)
    PREV[h] = col


def verdict(v):
    if v >= STABLE:
        return 'stable'
    if v > DOUBTFUL:
        return 'doubtful'
    if v > DISSOLVED:
        return 'not trustworthy'
    return 'dissolved'


def stability(kind):
    """Mean Jaccard of every reference state over resamples of the given kind. A reading
    drawn more than once counts once, as the reference implementation does by default."""
    per_state = {s: [] for s in ORDER}
    for r in range(RESAMPLES):
        idx = (rng.integers(0, len(X), len(X)) if kind == 'reading' else
               np.concatenate([DAY_BLOCKS[b] for b in
                               rng.integers(0, len(DAY_BLOCKS), len(DAY_BLOCKS))]))
        km = KMeans(K, n_init=30, random_state=r).fit(X[idx])
        nm = name_clusters(km.cluster_centers_)
        lab_r = np.array([nm[c] for c in km.labels_], dtype=object)
        drawn = np.unique(idx)
        in_cluster = {t: np.unique(idx[lab_r == t]) for t in set(lab_r)}
        for s in ORDER:
            a = drawn[ref_lab[drawn] == s]
            if not len(a):
                continue
            best = 0.0
            for b in in_cluster.values():
                inter = len(np.intersect1d(a, b, assume_unique=True))
                union = len(a) + len(b) - inter
                if union and inter / union > best:
                    best = inter / union
            per_state[s].append(best)
    return {s: (dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))),
                     resamples=len(v)) if len(v) > 1 else
                dict(mean=float('nan'), se=float('nan'), resamples=len(v)))
            for s, v in per_state.items()}


rows = []
for kind, label in [('day', 'whole days'), ('reading', 'single readings')]:
    res = stability(kind)
    for s in ORDER:
        v = res[s]['mean']
        rows.append(dict(state=s, readings=sizes[s], resampling=label, mean_jaccard=round(v, 3),
                         standard_error=round(res[s]['se'], 4), resamples=res[s]['resamples'],
                         verdict=verdict(v)))
    print(f'jaccard, {label:15s}: ' + '  '.join(
        f"{s} {res[s]['mean']:.3f} ({res[s]['se']:.3f})" for s in ORDER))
stab = pd.DataFrame(rows)
stab.to_csv('validation_stability.csv', index=False)


def reselected_k(Xtr):
    """The number of states selected again on the training nodes alone: of the candidates
    whose clusters the naming rule can all name, the highest silhouette, with the lower
    Davies and Bouldin index breaking a tie."""
    best = None
    for k in K_RANGE:
        km = KMeans(k, n_init=30, random_state=SEED).fit(Xtr)
        if any(v.startswith('Unnamed') for v in name_clusters(km.cluster_centers_).values()):
            continue
        sil = round(float(silhouette_score(Xtr, km.labels_)), 3)
        db = float(davies_bouldin_score(Xtr, km.labels_))
        if best is None or (sil, -db) > (best[1], -best[2]):
            best = (k, sil, db)
    return best[0] if best else K


def match(centres, reference):
    cost = ((centres[:, None, :] - reference[None, :, :]) ** 2).sum(2)
    r, c = linear_sum_assignment(cost)
    return dict(zip(r.tolist(), c.tolist()))


def fold_ps(Xtr, Xte, reference):
    """Prediction strength of every cluster of the withheld node, keyed by the reference
    state its centre is matched to. Matching assigns the label only; the proportion itself
    depends on the withheld node's own clustering and the training centroids."""
    tr = KMeans(K, n_init=30, random_state=SEED).fit(Xtr)
    te = KMeans(K, n_init=30, random_state=SEED).fit(Xte)
    assigned = tr.predict(Xte)
    to_state = match(te.cluster_centers_, reference)
    out = {}
    for c in range(K):
        m = te.labels_ == c
        n = int(m.sum())
        cnt = np.bincount(assigned[m], minlength=K)
        out[to_state[c]] = ((float((cnt * (cnt - 1)).sum() / (n * (n - 1))) if n > 1
                             else float('nan')), n)
    return out


def ar1_record():
    """One surrogate record. Each of the three level channels of each node is replaced by a
    first order autoregressive series whose lag one correlation is estimated from pairs of
    readings about one hour apart, and the two change features are recomputed under the same
    tolerance rule as the real features."""
    Xs = np.empty_like(X)
    for n in nodes:
        m = (d.node == n).values
        blk = X[m]
        L = len(blk)
        out = np.empty((L, X.shape[1]))
        i0, i1 = HOUR_PAIRS[n]
        for j in range(3):
            x = blk[:, j]
            phi = float(np.clip(np.corrcoef(X[i0, j], X[i1, j])[0, 1], -0.99, 0.99))
            sd = x.std() * np.sqrt(max(1.0 - phi ** 2, 1e-6))
            out[:, j] = lfilter([1.0], [1.0, -phi], rng.normal(0, sd, L + BURN))[BURN:] + x.mean()
        Xs[m] = out
    mo = Xs[:, 0]
    for col, h in ((3, 1), (4, 3)):
        prev = PREV[h]
        Xs[:, col] = np.where(prev >= 0, mo - mo[np.maximum(prev, 0)], 0.0)
    return np.clip(Xs, -5, 5)


print('generating surrogates')
surr, sur_runs = [], {s: [] for s in ORDER}
for _ in range(SURROGATES):
    Xs = ar1_record()
    surr.append((Xs, KMeans(K, n_init=30, random_state=SEED).fit(Xs).cluster_centers_))
    lab = np.array([ref_names[c] for c in ref.predict(Xs)], dtype=object)
    per, _ = run_lengths(lab, d.node.values, d.ts.values)
    for s in ORDER:
        sur_runs[s].append(float(per[s]) if s in per.index else np.nan)

obs_runs, _ = run_lengths(ref_lab, d.node.values, d.ts.values)
prows = []
for s in ORDER:
    v = np.array(sur_runs[s], dtype=float)
    bar = float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else float('nan')
    prows.append(dict(state=s, observed_mean_run_hours=round(float(obs_runs[s]), 2),
                      surrogate_median_hours=round(float(np.nanmedian(v)), 2),
                      surrogate_95th_hours=round(bar, 2),
                      surrogate_spread_hours=round(float(np.nanstd(v, ddof=1)), 2),
                      surrogate_records_containing_state=int(np.isfinite(v).sum()),
                      exceeds_surrogate=bool(float(obs_runs[s]) > bar) if np.isfinite(bar) else None))
pers = pd.DataFrame(prows)
pers.to_csv('validation_persistence.csv', index=False)
print('run length against the surrogate records:')
for r in prows:
    print(f"  {r['state']:20s} observed {r['observed_mean_run_hours']:6.2f} h   "
          f"surrogate 95th {r['surrogate_95th_hours']:6.2f} h   "
          f"spread {r['surrogate_spread_hours']:.2f} h   exceeds {r['exceeds_surrogate']}")

rows, frows = [], []
for n in nodes:
    tr_m = (d.node != n).values
    te_m = ~tr_m
    kk = reselected_k(X[tr_m])
    real = fold_ps(X[tr_m], X[te_m], ref.cluster_centers_)
    obs_min = min(v for v, _ in real.values() if not np.isnan(v))
    pooled, worst = [], []
    for Xs, cen in surr:
        v = [x for x, _ in fold_ps(Xs[tr_m], Xs[te_m], cen).values() if not np.isnan(x)]
        pooled += v
        worst.append(min(v, default=np.nan))
    bar_pooled = float(np.percentile(pooled, 95))
    bar_min = float(np.nanpercentile(worst, 95))
    frows.append(dict(held_out=n, reselected_k=kk, agrees_with_reference_k=bool(kk == K),
                      observed_minimum=round(obs_min, 3),
                      surrogate_95th_minimum=round(bar_min, 3),
                      minimum_beats_surrogate=bool(obs_min > bar_min),
                      surrogate_95th_pooled=round(bar_pooled, 3),
                      surrogate_median_pooled=round(float(np.median(pooled)), 3)))
    for j, (v, nj) in sorted(real.items()):
        rows.append(dict(held_out=n, state=ref_names[j], readings_on_withheld_node=nj,
                         prediction_strength=None if np.isnan(v) else round(v, 3),
                         reaches_threshold=None if np.isnan(v) else bool(v >= RECOGNISED),
                         surrogate_95th_pooled=round(bar_pooled, 3),
                         beats_pooled_surrogate=None if np.isnan(v) else bool(v > bar_pooled)))
    print(f'{n}: k {kk}, minimum {obs_min:.3f} against {bar_min:.3f}, pooled bar {bar_pooled:.3f} | '
          + ', '.join(f'{ref_names[j]} {v:.3f} (n {nj})' for j, (v, nj) in sorted(real.items())))
ps = pd.DataFrame(rows)
ps.to_csv('validation_prediction_strength.csv', index=False)
folds = pd.DataFrame(frows)
folds.to_csv('validation_folds.csv', index=False)

summary = {}
for s in ORDER:
    sub = ps[ps.state == s]
    day = float(stab[(stab.state == s) & (stab.resampling == 'whole days')].mean_jaccard.iloc[0])
    summary[s] = dict(
        readings=sizes[s], jaccard_day=day,
        jaccard_day_se=float(stab[(stab.state == s) &
                                  (stab.resampling == 'whole days')].standard_error.iloc[0]),
        jaccard_reading=float(stab[(stab.state == s) &
                                   (stab.resampling == 'single readings')].mean_jaccard.iloc[0]),
        verdict=verdict(day),
        min_prediction_strength=float(sub.prediction_strength.min()),
        max_prediction_strength=float(sub.prediction_strength.max()),
        folds_reaching_threshold=int(sub.reaches_threshold.sum()), folds=int(len(sub)),
        folds_beating_pooled=int(sub.beats_pooled_surrogate.sum()))
print('\nsummary per state')
for s, v in summary.items():
    print(f"  {s:20s} jaccard {v['jaccard_day']:.3f} ({v['verdict']})   prediction strength "
          f"{v['min_prediction_strength']:.3f} to {v['max_prediction_strength']:.3f}   "
          f"folds at {RECOGNISED} {v['folds_reaching_threshold']}/{v['folds']}")
print('folds whose minimum beats its matched reference level:',
      int(folds.minimum_beats_surrogate.sum()), 'of', len(folds))

json.dump(dict(resamples=RESAMPLES, surrogates=SURROGATES, candidate_states=K_RANGE,
               thresholds=dict(stable=STABLE, doubtful=DOUBTFUL, dissolved=DISSOLVED,
                               recognised=RECOGNISED),
               reference_states=ORDER, reference_sizes=sizes,
               stability=stab.to_dict('records'), persistence=prows,
               folds=frows, prediction_strength=rows, summary=summary),
          open('validation_tests.json', 'w'), indent=1)
print('\nwrote validation_stability.csv validation_prediction_strength.csv '
      'validation_folds.csv validation_persistence.csv validation_tests.json')
