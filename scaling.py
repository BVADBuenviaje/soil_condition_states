import json, time, resource, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import load, name_clusters, present, state_labels, FEAT, TEST_START, K, SEED
warnings.filterwarnings('ignore')

d, X = load(); cfg = json.load(open('best_configs.json'))
RF = {**cfg['random_forest']['fixed'], **cfg['random_forest']['tuned']}
LR = {**cfg['logistic_regression']['fixed'], **cfg['logistic_regression']['tuned']}
tr = (d.ts < TEST_START).values; te = ~tr
ytr_ref, yte_ref = state_labels(X[tr], X[te])
ORDER = present(ytr_ref)
rng = np.random.default_rng(SEED)

def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

def costs(Xs, ys, repeats=3):
    out = {}
    out['kmeans_seconds'] = median_time(lambda: KMeans(K, n_init=30, random_state=SEED).fit(Xs), repeats)
    out['forest_fit_seconds'] = median_time(lambda: RandomForestClassifier(**RF, n_jobs=-1).fit(Xs, ys), repeats)
    out['logistic_fit_seconds'] = median_time(lambda: LogisticRegression(**LR).fit(Xs, ys), repeats)
    rf = RandomForestClassifier(**RF, n_jobs=-1).fit(Xs, ys)
    lr = LogisticRegression(**LR).fit(Xs, ys)
    out['forest_assign_ms'] = median_time(lambda: rf.predict(X[te]), repeats) * 1000
    out['logistic_assign_ms'] = median_time(lambda: lr.predict(X[te]), repeats) * 1000
    return out, rf, lr

rows = []
Xtr, ytr = X[tr], ytr_ref
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    n = int(round(frac * len(Xtr)))
    idx = rng.choice(len(Xtr), n, replace=False) if n < len(Xtr) else np.arange(len(Xtr))
    c, rf, lr = costs(Xtr[idx], ytr[idx])
    rows.append(dict(readings=n, source='record', **{k: round(v, 4) for k, v in c.items()},
                     forest_macro_f1=round(f1_score(yte_ref, rf.predict(X[te]), average='macro',
                                                    labels=ORDER, zero_division=0), 4),
                     logistic_macro_f1=round(f1_score(yte_ref, lr.predict(X[te]), average='macro',
                                                      labels=ORDER, zero_division=0), 4)))
    print(f"{n:>7} readings from the record: k means {c['kmeans_seconds']:.3f} s, forest {c['forest_fit_seconds']:.3f} s")

for mult in [2, 5, 10, 20, 33]:
    Xs = np.repeat(Xtr, mult, axis=0) + rng.normal(0, 0.01, (len(Xtr) * mult, len(FEAT)))
    ys = np.repeat(ytr, mult)
    c, rf, lr = costs(Xs, ys, repeats=1 if mult >= 10 else 3)
    rows.append(dict(readings=len(Xs), source=f'record replicated {mult} times',
                     **{k: round(v, 4) for k, v in c.items()},
                     forest_macro_f1=None, logistic_macro_f1=None))
    print(f"{len(Xs):>7} readings, synthetic: k means {c['kmeans_seconds']:.3f} s, forest {c['forest_fit_seconds']:.3f} s")
scal = pd.DataFrame(rows); scal.to_csv('ch4_scaling_readings.csv', index=False)

FIT_FROM = 4000
big = scal[scal.readings >= FIT_FROM]
fit_exp, coefs = {}, {}
for col in ['kmeans_seconds', 'forest_fit_seconds', 'logistic_fit_seconds']:
    b, a = np.polyfit(np.log(big.readings), np.log(big[col]), 1)
    coefs[col] = (b, a)
    fit_exp[col] = dict(exponent=round(float(b), 3),
                        seconds_at_one_hundred_thousand=round(float(np.exp(a) * 100000 ** b), 2))
print(f'\nfitted power law exponents above {FIT_FROM} readings:', {k: v['exponent'] for k, v in fit_exp.items()})

rows = []
nodes = sorted(d.node.unique())
for m in range(2, len(nodes) + 1):
    keep = d.node.isin(nodes[:m]).values & tr
    lab, = state_labels(X[keep])
    c, rf, lr = costs(X[keep], lab)
    rows.append(dict(nodes=m, readings=int(keep.sum()), states=' | '.join(present(lab)),
                     **{k: round(v, 4) for k, v in c.items()}))
    print(f'{m} nodes ({keep.sum()} readings): {rows[-1]["states"]}')
pd.DataFrame(rows).to_csv('ch4_scaling_nodes.csv', index=False)

rows = []
for k in range(2, 9):
    t = median_time(lambda: KMeans(k, n_init=30, random_state=SEED).fit(Xtr), 3)
    km = KMeans(k, n_init=30, random_state=SEED).fit(Xtr)
    nm = name_clusters(km.cluster_centers_)
    rows.append(dict(states=k, kmeans_seconds=round(t, 4), inertia=round(float(km.inertia_), 1),
                     all_named=not any(v.startswith('Unnamed') for v in nm.values())))
pd.DataFrame(rows).to_csv('ch4_scaling_states.csv', index=False)
print('\ncost of the clustering from two to eight states:',
      {r['states']: r['kmeans_seconds'] for r in rows})

rows = []
tr_ts = d.ts[tr].sort_values()
for frac in [0.25, 0.5, 0.75, 1.0]:
    cut = tr_ts.iloc[int(frac * (len(tr_ts) - 1))]
    m = (d.ts <= cut).values
    lab, = state_labels(X[m])
    L = present(lab)
    rows.append(dict(share_of_training_period=frac, until=str(cut.date()), readings=int(m.sum()),
                     states=' | '.join(L), same_set_as_full=sorted(set(L)) == sorted(set(ORDER))))
    print(f'first {frac:.0%} of the training period ({m.sum()} readings): {" | ".join(L)}')
pd.DataFrame(rows).to_csv('ch4_scaling_length.csv', index=False)

per_reading = {c: fit_exp[c] for c in fit_exp}
rows = []
for n_nodes in [5, 20, 100]:
    for months, label in [(1, 'one month'), (6, 'six months'), (12, 'one year'), (60, 'five years')]:
        n = int(n_nodes * 24 * 30.4 * months)
        est = {c: float(np.exp(np.polyval(coefs[c], np.log(n))))
               for c in ['kmeans_seconds', 'forest_fit_seconds', 'logistic_fit_seconds']}
        total = est['kmeans_seconds'] + est['forest_fit_seconds']
        rows.append(dict(nodes=n_nodes, period=label, readings=n,
                         kmeans_seconds=round(est['kmeans_seconds'], 1),
                         forest_fit_seconds=round(est['forest_fit_seconds'], 1),
                         refit_seconds=round(total, 1),
                         inside_an_hourly_cycle=bool(total < 3600)))
viab = pd.DataFrame(rows); viab.to_csv('ch4_viability.csv', index=False)
print('\nestimated time for a complete refit:'); print(viab.to_string(index=False))

mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Xbig = np.repeat(Xtr, 33, axis=0) + rng.normal(0, 0.01, (len(Xtr) * 33, len(FEAT)))
RandomForestClassifier(**RF, n_jobs=-1).fit(Xbig, np.repeat(ytr, 33))
KMeans(K, n_init=30, random_state=SEED).fit(Xbig)
peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
print(f'\npeak memory of the process while fitting {len(Xbig)} readings: {peak_mb} MB')
del Xbig

json.dump(dict(power_law=fit_exp, fitted_above=FIT_FROM, peak_memory_mb=peak_mb,
               largest_measured=int(scal.readings.max()),
               largest_real=int(scal[scal.source == 'record'].readings.max()),
               refit_at_five_nodes_one_year=float(viab[(viab.nodes == 5) & (viab.period == 'one year')].refit_seconds.iloc[0]),
               all_inside_hourly_cycle=bool(viab.inside_an_hourly_cycle.all()),
               assign_ms_at_full_size=float(scal[scal.source == 'record'].forest_assign_ms.iloc[-1])),
          open('ch4_scaling.json', 'w'), indent=1)
print('\nwrote ch4_scaling_readings.csv, ch4_scaling_nodes.csv, ch4_scaling_states.csv, '
      'ch4_scaling_length.csv, ch4_viability.csv and ch4_scaling.json')
