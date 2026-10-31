import pandas as pd, numpy as np, warnings, sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from common import FEAT, name_clusters, present, run_lengths
warnings.filterwarnings('ignore')
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
d = pd.read_csv('soil_features.csv', parse_dates=['ts']).sort_values(['node', 'ts']).reset_index(drop=True)
X = d[FEAT].clip(-5, 5).values

ref = KMeans(K, n_init=30, random_state=SEED).fit(X)
names = name_clusters(ref.cluster_centers_)
d['state_ref'] = [names[c] for c in ref.labels_]
order = present(d.state_ref)
prof = pd.DataFrame(ref.cluster_centers_, columns=FEAT); prof['state'] = [names[c] for c in prof.index]
print('Reference state profiles (all nodes):'); print(prof.round(2).to_string())
print('\nMedian date of each state:'); print(d.groupby('state_ref').ts.median().dt.strftime('%b %d').to_string())

rows, heldout_labels, shares = [], {}, []
print('\n=== Leave one node out ===')
for n in sorted(d.node.unique()):
    tr = (d.node != n).values; te = ~tr
    m = KMeans(K, n_init=30, random_state=SEED).fit(X[tr])
    own = name_clusters(m.cluster_centers_)
    C = ((m.cluster_centers_[:, None, :] - ref.cluster_centers_[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C); mp = {ri: names[ci] for ri, ci in zip(r, c)}
    dist = {names[ci]: float(np.sqrt(C[ri, ci])) for ri, ci in zip(r, c)}
    own_of = {names[ci]: own[ri] for ri, ci in zip(r, c)}
    lab_tr = np.array([mp[x] for x in m.labels_]); lab_te = np.array([mp[x] for x in m.predict(X[te])])
    heldout_labels[n] = lab_te
    s_tr = pd.Series(lab_tr).value_counts(normalize=True).reindex(order).fillna(0)
    s_te = pd.Series(lab_te).value_counts(normalize=True).reindex(order).fillna(0)
    s_ref = d.state_ref[te].value_counts(normalize=True).reindex(order).fillna(0)
    for s in order:
        shares.append(dict(held_out=n, state=s, share_train_nodes=round(s_tr[s], 4), share_heldout=round(s_te[s], 4),
                           share_heldout_reference=round(s_ref[s], 4), matched_centre_distance=round(dist[s], 3),
                           fold_name_of_matched_cluster=own_of[s]))
    same_set = sorted(own.values()) == sorted(names.values())
    rl, _ = run_lengths(lab_te, d.node[te].values, d.ts[te].values)
    hrs = d.hour[te].values
    ref_te = d.state_ref[te].values
    tv = {}
    for s in order:
        a = hrs[lab_te == s]; b = hrs[ref_te == s]
        if len(a) and len(b):
            pa = np.bincount(a, minlength=24) / len(a); pb = np.bincount(b, minlength=24) / len(b)
            tv[s] = round(0.5 * float(np.abs(pa - pb).sum()), 3)
    worst_state = max(tv, key=tv.get) if tv else ''
    rows.append(dict(held_out=n, n_rows=int(te.sum()), same_state_set=same_set,
                     fold_states=' | '.join(present(list(own.values()))),
                     silhouette_heldout=round(silhouette_score(X[te], lab_te), 3) if len(set(lab_te)) > 1 else np.nan,
                     agreement_with_full_model_ARI=round(adjusted_rand_score(d.state_ref[te], lab_te), 3),
                     saturated_rows=int((lab_te == 'Saturated').sum()),
                     share_of_saturated_in_7to10am=round(np.mean(((hrs >= 7) & (hrs <= 10))[lab_te == 'Saturated']), 2)
                         if (lab_te == 'Saturated').any() else np.nan,
                     mean_per_state_run_hours=round(float(np.mean([rl[s] for s in rl.index])), 1),
                     hour_distribution_difference=(tv[worst_state] if tv else np.nan),
                     hour_distribution_worst_state=worst_state,
                     states_compared_by_hour=len(tv)))
    print(f'\nHeld out {n}: the fold on its own forms {present(list(own.values()))}  (same set as reference: {same_set})')
    print(pd.DataFrame({'train 4 nodes': s_tr.round(3), 'held out node': s_te.round(3),
                        'reference, same node': s_ref.round(3), 'centre distance': pd.Series(dist).reindex(order).round(2),
                        'fold calls it': pd.Series(own_of).reindex(order)}).to_string())
res = pd.DataFrame(rows); print('\nSummary:'); print(res.to_string(index=False))
res.to_csv(f'lono_summary_seed{SEED}_k{K}.csv', index=False)
pd.DataFrame(shares).to_csv(f'lono_shares_seed{SEED}_k{K}.csv', index=False)
pd.to_pickle(heldout_labels, f'lono_labels_seed{SEED}_k{K}.pkl'); d.to_csv(f'soil_states_k{K}.csv', index=False)
