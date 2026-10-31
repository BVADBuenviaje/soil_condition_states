import json, time, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score, davies_bouldin_score)
from scipy import stats
from common import load, name_clusters, present, FEAT, TEST_START, K
warnings.filterwarnings('ignore')
SEEDS = list(range(30))
d, X = load(); cfg = json.load(open('best_configs.json'))
RF = {**cfg['random_forest']['fixed'], **cfg['random_forest']['tuned']}
DT = {**cfg['decision_tree']['fixed'], **cfg['decision_tree']['tuned']}
LR = {**cfg['logistic_regression']['fixed'], **cfg['logistic_regression']['tuned']}
tr = (d.ts < TEST_START).values; te = ~tr
ref0 = KMeans(K, n_init=30, random_state=0).fit(X)

rows = []
for s in SEEDS:
    full = KMeans(K, n_init=30, random_state=s).fit(X)
    nm_full = name_clusters(full.cluster_centers_)
    lab_full = np.array([nm_full[c] for c in full.labels_], dtype=object)
    km = KMeans(K, n_init=30, random_state=s).fit(X[tr]); nm = name_clusters(km.cluster_centers_)
    ytr = np.array([nm[c] for c in km.labels_], dtype=object)
    yte = np.array([nm[c] for c in km.predict(X[te])], dtype=object)
    L = present(ytr)
    r = dict(seed=s,
             states_full=' | '.join(present(lab_full)),
             same_states_as_seed0=sorted(set(lab_full)) == sorted(set(np.array([name_clusters(ref0.cluster_centers_)[c] for c in ref0.labels_], dtype=object))),
             ari_with_seed0=round(adjusted_rand_score(ref0.labels_, full.labels_), 4),
             nmi_with_node=round(normalized_mutual_info_score(d.node, full.labels_), 4),
             inertia=round(float(full.inertia_), 1),
             silhouette=round(float(silhouette_score(X, full.labels_)), 4),
             davies_bouldin=round(float(davies_bouldin_score(X, full.labels_)), 4),
             states_training=' | '.join(L))
    for name, mdl in [('random_forest', RandomForestClassifier(**{**RF, 'random_state': s}, n_jobs=-1)),
                      ('logistic_regression', LogisticRegression(**LR)),
                      ('decision_tree', DecisionTreeClassifier(**{**DT, 'random_state': s}))]:
        p = mdl.fit(X[tr], ytr).predict(X[te])
        r[f'{name}_macro_f1'] = round(f1_score(yte, p, average='macro', labels=L, zero_division=0), 6)
        r[f'{name}_accuracy'] = round(accuracy_score(yte, p), 6)
        r[f'{name}_balanced_accuracy'] = round(balanced_accuracy_score(yte, p), 6)
    maj = np.full(te.sum(), pd.Series(ytr).mode()[0], dtype=object)
    r['majority_macro_f1'] = round(f1_score(yte, maj, average='macro', labels=L, zero_division=0), 6)
    r['majority_accuracy'] = round(accuracy_score(yte, maj), 6)
    r['majority_balanced_accuracy'] = round(balanced_accuracy_score(yte, maj), 6)
    nearest = np.array([nm[c] for c in km.predict(X[te])], dtype=object)
    r['nearest_centroid_accuracy'] = round(accuracy_score(yte, nearest), 4)
    rows.append(r); print(f"seed {s}: ARI with seed 0 {r['ari_with_seed0']:.3f}  RF {r['random_forest_macro_f1']:.4f}  "
                         f"LR {r['logistic_regression_macro_f1']:.4f}  tree {r['decision_tree_macro_f1']:.4f}")
seeds = pd.DataFrame(rows); seeds.to_csv('ch4_seeds.csv', index=False)
MODELS = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']
METRICS = ['macro_f1', 'balanced_accuracy', 'accuracy']
print(f'\nAcross {len(SEEDS)} independent runs:')
summary, mat = {}, []
for m in MODELS:
    v = seeds[f'{m}_macro_f1']
    summary[m] = dict(mean=round(float(v.mean()), 4), sd=round(float(v.std(ddof=1)), 4),
                      min=round(float(v.min()), 4), max=round(float(v.max()), 4))
    for k in METRICS:
        w = seeds[f'{m}_{k}']
        mat.append(dict(model=m, measure=k, mean=round(float(w.mean()), 4), sd=round(float(w.std(ddof=1)), 4),
                        minimum=round(float(w.min()), 4), median=round(float(w.median()), 4),
                        maximum=round(float(w.max()), 4)))
    print(f'  {m:<20} macro F1 {v.mean():.4f} (sd {v.std(ddof=1):.4f}, {v.min():.4f} to {v.max():.4f})')
for k, lab in [('inertia', 'within cluster sum of squares'), ('silhouette', 'silhouette'),
               ('davies_bouldin', 'Davies Bouldin'), ('ari_with_seed0', 'agreement with the first run')]:
    w = seeds[k]
    mat.append(dict(model='k means clustering', measure=lab, mean=round(float(w.mean()), 4),
                    sd=round(float(w.std(ddof=1)), 4), minimum=round(float(w.min()), 4),
                    median=round(float(w.median()), 4), maximum=round(float(w.max()), 4)))
matrix = pd.DataFrame(mat); matrix.to_csv('ch4_runs_matrix.csv', index=False)
print('\nComparative matrix over the runs:'); print(matrix.to_string(index=False))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj
wil = []
for other in ['logistic_regression', 'decision_tree', 'majority']:
    a = seeds['random_forest_macro_f1'].values; b = seeds[f'{other}_macro_f1'].values
    dif = a - b
    st, pv = stats.wilcoxon(a, b, alternative='two-sided')
    wil.append(dict(comparison=f'random_forest vs {other}', runs=len(SEEDS),
                    runs_proposed_better=int((dif > 0).sum()), runs_other_better=int((dif < 0).sum()),
                    runs_tied=int((dif == 0).sum()), median_difference=round(float(np.median(dif)), 4),
                    statistic=float(st), p=float(pv)))
for r, a_ in zip(wil, holm(np.array([r['p'] for r in wil]))):
    r['p'] = float(f"{r['p']:.3e}"); r['p_holm'] = float(f"{float(a_):.3e}")
wilcox = pd.DataFrame(wil); wilcox.to_csv('ch4_wilcoxon.csv', index=False)
print('\nPaired Wilcoxon signed rank over the runs, macro F1:'); print(wilcox.to_string(index=False))
same = int(seeds.same_states_as_seed0.sum()); ari = seeds.ari_with_seed0
print(f'  state set identical to seed 0 in {same} of {len(SEEDS)} seeds; ARI with seed 0 '
      f'{ari.mean():.3f} (min {ari.min():.3f}); inertia range {seeds.inertia.min():.1f} to {seeds.inertia.max():.1f}')
print(f'  nearest centroid accuracy: {seeds.nearest_centroid_accuracy.min():.4f} to {seeds.nearest_centroid_accuracy.max():.4f}')

km = KMeans(K, n_init=30, random_state=0).fit(X[tr]); nm = name_clusters(km.cluster_centers_)
ytr = np.array([nm[c] for c in km.labels_], dtype=object); L = present(ytr)
def timed(build, n=3):
    fit, pred = [], []
    for _ in range(n):
        m = build(); t0 = time.perf_counter(); m.fit(X[tr], ytr); t1 = time.perf_counter(); m.predict(X[te]); t2 = time.perf_counter()
        fit.append(t1 - t0); pred.append(t2 - t1)
    return round(float(np.median(fit)), 3), round(float(np.median(pred)) * 1000, 2)
times = []
for name, build in [('Random Forest', lambda: RandomForestClassifier(**RF, n_jobs=-1)),
                    ('Logistic regression', lambda: LogisticRegression(**LR)),
                    ('Classification tree', lambda: DecisionTreeClassifier(**DT))]:
    f, p = timed(build); times.append(dict(model=name, fit_seconds=f, predict_ms=p)); print(f'{name}: fit {f} s, predict {p} ms')
t0 = time.perf_counter(); KMeans(K, n_init=30, random_state=0).fit(X[tr]); t1 = time.perf_counter()
times.append(dict(model='k means, thirty restarts', fit_seconds=round(t1 - t0, 3), predict_ms=None))
tim = pd.DataFrame(times); tim.to_csv('ch4_timing.csv', index=False)
json.dump(dict(seeds=rows, summary=summary, matrix=mat, wilcoxon=wil, same_state_set=same, n_seeds=len(SEEDS),
               ari_mean=round(float(ari.mean()), 3), ari_min=round(float(ari.min()), 3),
               inertia_min=round(float(seeds.inertia.min()), 1), inertia_max=round(float(seeds.inertia.max()), 1),
               nearest_centroid_min=round(float(seeds.nearest_centroid_accuracy.min()), 4),
               timing=times), open('ch4_robustness.json', 'w'), indent=1)
print('\nwrote ch4_seeds.csv, ch4_runs_matrix.csv, ch4_wilcoxon.csv, ch4_timing.csv and ch4_robustness.json')
