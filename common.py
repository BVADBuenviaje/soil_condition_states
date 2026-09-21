import numpy as np, pandas as pd
from sklearn.cluster import KMeans

FEAT  = ['moisture_rel', 'ph_rel', 'temp_rel', 'dmoist_1h', 'dmoist_3h']
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
    names = {prof.moisture_rel.idxmax(): 'Saturated'}
    rest = sorted([c for c in range(len(prof)) if c not in names], key=lambda c: prof.loc[c, 'moisture_rel'])
    lo, mid, hi = rest
    if prof.loc[lo, 'ph_rel'] > prof.loc[mid, 'ph_rel']:
        names[lo], names[mid] = 'Drier / higher pH', 'Drier / lower pH'
    else:
        names[mid], names[lo] = 'Drier / higher pH', 'Drier / lower pH'
    names[hi] = 'Moist / cooler'
    return names

def state_labels(X_fit, *X_assign):
    km = KMeans(K, n_init=30, random_state=SEED).fit(X_fit)
    nm = name_clusters(km.cluster_centers_)
    out = [np.array([nm[c] for c in km.labels_])]
    out += [np.array([nm[c] for c in km.predict(Xa)]) for Xa in X_assign]
    return out
