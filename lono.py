import pandas as pd, numpy as np, warnings, sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')
SEED = int(sys.argv[1]) if len(sys.argv)>1 else 0
d = pd.read_csv('soil_features.csv', parse_dates=['ts'])
FEAT=['moisture_rel','ph_rel','temp_rel','dmoist_1h','dmoist_3h']
X = d[FEAT].clip(-5,5).values
K = int(sys.argv[2]) if len(sys.argv)>2 else 5
ref = KMeans(K, n_init=30, random_state=SEED).fit(X)
prof = pd.DataFrame(ref.cluster_centers_, columns=FEAT)
names = {}
names[prof.moisture_rel.idxmax()] = 'Saturated'
if K==5: names[prof.dmoist_3h.idxmin()] = 'Draining'
rest = [c for c in range(K) if c not in names]
rest_sorted = sorted(rest, key=lambda c: prof.loc[c,'moisture_rel'])
lo, mid, hi = rest_sorted
if prof.loc[lo,'ph_rel'] > prof.loc[mid,'ph_rel']: names[lo]='Drier / higher pH'; names[mid]='Drier / lower pH'
else: names[mid]='Drier / higher pH'; names[lo]='Drier / lower pH'
names[hi]='Moist / cooler'
d['state_ref'] = [names[c] for c in ref.labels_]
print('Reference state profiles (all nodes):'); prof['state']=[names[c] for c in prof.index]; print(prof.round(2).to_string())
print('\nMedian date of each state:'); print(d.groupby('state_ref').ts.median().dt.strftime('%b %d').to_string())

def align(model):
    C = ((model.cluster_centers_[:,None,:]-ref.cluster_centers_[None,:,:])**2).sum(-1)
    r,c = linear_sum_assignment(C); return {ri: names[ci] for ri,ci in zip(r,c)}

def run_len(seq):
    runs=[]; cur=1
    for a,b in zip(seq[:-1],seq[1:]):
        if a==b: cur+=1
        else: runs.append(cur); cur=1
    runs.append(cur); return np.mean(runs)

order=[n for n in ['Saturated','Draining','Moist / cooler','Drier / higher pH','Drier / lower pH'] if n in names.values()]
rows=[]; heldout_labels={}
print('\n=== Leave-one-node-out ===')
for n in sorted(d.node.unique()):
    tr = d.node!=n; te = ~tr
    m = KMeans(K, n_init=30, random_state=SEED).fit(X[tr]); mp = align(m)
    lab_tr = np.array([mp[c] for c in m.labels_]); lab_te = np.array([mp[c] for c in m.predict(X[te])])
    heldout_labels[n]=lab_te
    share_tr = pd.Series(lab_tr).value_counts(normalize=True).reindex(order).fillna(0)
    share_te = pd.Series(lab_te).value_counts(normalize=True).reindex(order).fillna(0)
    sil_te = silhouette_score(X[te], lab_te) if len(set(lab_te))>1 else np.nan
    ari = adjusted_rand_score(d.state_ref[te], lab_te)
    hrs = d.hour[te].values
    sat_morning = np.mean((hrs>=7)&(hrs<=10)) if (lab_te=='Saturated').sum()==0 else np.mean(((hrs>=7)&(hrs<=10))[lab_te=='Saturated'])
    rows.append(dict(held_out=n, n_rows=te.sum(), silhouette_heldout=round(sil_te,3), agreement_with_full_model_ARI=round(ari,3),
                     saturated_rows=int((lab_te=='Saturated').sum()), share_of_saturated_in_7to10am=round(sat_morning,2),
                     mean_state_run_hours=round(run_len(lab_te),1)))
    print(f'\nHeld out {n}:  state share  (train 4 nodes  vs  held-out node)')
    print(pd.DataFrame({'train':share_tr.round(2),'held_out':share_te.round(2)}).T.to_string())
res = pd.DataFrame(rows); print('\nSummary:'); print(res.to_string(index=False))
res.to_csv(f'lono_summary_seed{SEED}_k{K}.csv', index=False)
pd.to_pickle(heldout_labels, f'lono_labels_seed{SEED}_k{K}.pkl'); d.to_csv(f'soil_states_k{K}.csv', index=False)
