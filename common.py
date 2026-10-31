import numpy as np, pandas as pd
from sklearn.cluster import KMeans

FEAT  = ['moisture_rel', 'ph_rel', 'temp_rel', 'dmoist_1h', 'dmoist_3h']
ORDER_ALL = ['Saturated', 'Rapid drainage', 'Moist / cooler', 'Drier',
             'Drier / higher pH', 'Drier / lower pH']
ORDER = ['Saturated', 'Moist / cooler', 'Drier / higher pH', 'Drier / lower pH']
K, SEED = 4, 0

TRAIN_START = pd.Timestamp('2025-10-13')
VAL_START   = pd.Timestamp('2025-10-30')
TEST_START  = pd.Timestamp('2025-11-06')

def load():
    d = pd.read_csv('soil_states_k4.csv', parse_dates=['ts']).sort_values(['node', 'ts']).reset_index(drop=True)
    X = d[FEAT].clip(-5, 5).values
    return d, X

def name_clusters(centres):
    prof = pd.DataFrame(centres, columns=FEAT)
    names = {int(prof.moisture_rel.idxmax()): 'Saturated'}
    for c in prof.index:
        if c in names: continue
        a = prof.loc[c].abs()
        if a.idxmax() in ('dmoist_1h', 'dmoist_3h') and prof.loc[c, a.idxmax()] < 0:
            names[int(c)] = 'Rapid drainage'
    rest = sorted([c for c in prof.index if c not in names], key=lambda c: prof.loc[c, 'moisture_rel'])
    if len(rest) == 2:
        names[rest[1]], names[rest[0]] = 'Moist / cooler', 'Drier'
    elif len(rest) == 3:
        lo, mid, hi = rest
        hiph, loph = (lo, mid) if prof.loc[lo, 'ph_rel'] > prof.loc[mid, 'ph_rel'] else (mid, lo)
        names[hi], names[hiph], names[loph] = 'Moist / cooler', 'Drier / higher pH', 'Drier / lower pH'
    else:
        for j, c in enumerate(rest): names[c] = f'Unnamed {j + 1}'
    if len(set(names.values())) < len(names):
        seen = {}
        for c in sorted(names):
            v = names[c]; seen[v] = seen.get(v, 0) + 1
            if seen[v] > 1: names[c] = f'{v} {seen[v]}'
    return {int(k): v for k, v in names.items()}

def present(*label_arrays):
    s = set(np.concatenate([np.asarray(a, dtype=object) for a in label_arrays]))
    return [n for n in ORDER_ALL if n in s] + sorted(x for x in s if x not in ORDER_ALL)

def state_labels(X_fit, *X_assign, k=K):
    km = KMeans(k, n_init=30, random_state=SEED).fit(X_fit)
    nm = name_clusters(km.cluster_centers_)
    out = [np.array([nm[c] for c in km.labels_], dtype=object)]
    out += [np.array([nm[c] for c in km.predict(Xa)], dtype=object) for Xa in X_assign]
    return out

def run_lengths(labels, node, ts):
    labels = np.asarray(labels, dtype=object); node = np.asarray(node); ts = np.asarray(ts, dtype='datetime64[ns]')
    runs = []
    for n in pd.unique(node):
        m = node == n; l = labels[m]; t = ts[m]; cur = 1
        for i in range(1, len(l)):
            gap = (t[i] - t[i - 1]) / np.timedelta64(1, 'h')
            if l[i] == l[i - 1] and 0.75 <= gap <= 1.25: cur += 1
            else: runs.append((l[i - 1], cur)); cur = 1
        runs.append((l[-1], cur))
    r = pd.DataFrame(runs, columns=['state', 'hours'])
    return r.groupby('state').hours.mean(), float(r.hours.mean())
