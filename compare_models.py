import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             confusion_matrix, silhouette_score, davies_bouldin_score,
                             normalized_mutual_info_score, adjusted_rand_score)
from common import load, state_labels, name_clusters, ORDER, FEAT, TEST_START, K, SEED
warnings.filterwarnings('ignore')

d, X = load()
cfg = json.load(open('best_configs.json'))
def make(name):
    f, t = dict(cfg[name]['fixed']), dict(cfg[name]['tuned'])
    if name == 'random_forest': return RandomForestClassifier(**f, **t, n_jobs=-1)
    if name == 'decision_tree': return DecisionTreeClassifier(**f, **t)
    return LogisticRegression(**f, **t)
MODELS = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']

def fit_predict(name, Xtr, ytr, Xte):
    if name == 'majority':
        return np.full(len(Xte), pd.Series(ytr).mode()[0], dtype=object)
    return make(name).fit(Xtr, ytr).predict(Xte)

def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average='macro', labels=ORDER, zero_division=0),
                **{f'recall_{s}': recall_score(y, p, labels=[s], average='macro', zero_division=0) for s in ORDER})

def mcnemar(y, pa, pb):
    a_ok, b_ok = (pa == y), (pb == y)
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok))
    if b + c == 0: return b, c, 'none', float('nan'), 1.0
    if b + c < 25:
        return b, c, 'exact binomial', float('nan'), stats.binomtest(b, b + c, 0.5).pvalue
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, 'chi square, continuity corrected', chi2, stats.chi2.sf(chi2, 1)

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj

out = {}
tr = (d.ts < TEST_START).values; te = ~tr
ytr, yte = state_labels(X[tr], X[te])
print(f'chronological: train {tr.sum()}  test {te.sum()}')
print('test state counts:', pd.Series(yte).value_counts().reindex(ORDER).fillna(0).astype(int).to_dict())
preds, rows = {}, []
for m in MODELS:
    preds[m] = fit_predict(m, X[tr], ytr, X[te])
    rows.append(dict(model=m, **{k: round(v, 4) for k, v in metrics(yte, preds[m]).items()}))
rows.append(dict(model='nearest_centroid (reference)', accuracy=1.0, balanced_accuracy=1.0, macro_f1=1.0))
chrono = pd.DataFrame(rows); chrono.to_csv('comparison_chronological.csv', index=False)
print(chrono.to_string(index=False))
cm = pd.DataFrame(confusion_matrix(yte, preds['random_forest'], labels=ORDER), index=ORDER, columns=ORDER)
cm.to_csv('comparison_confusion_rf.csv'); print('\nRF confusion matrix (rows true, columns predicted):'); print(cm.to_string())

mc = []
for m in ['logistic_regression', 'decision_tree', 'majority']:
    b, c, kind, chi2, p = mcnemar(yte, preds['random_forest'], preds[m])
    mc.append(dict(comparison=f'random_forest vs {m}', b_rf_right_other_wrong=b, c_rf_wrong_other_right=c,
                   test=kind, chi2=None if np.isnan(chi2) else round(chi2, 3), p=p))
adj = holm(np.array([r['p'] for r in mc]))
for r, a in zip(mc, adj): r['p'] = round(r['p'], 6); r['p_holm'] = round(float(a), 6)
print('\nMcNemar, chronological test set:'); print(pd.DataFrame(mc).to_string(index=False))
out['mcnemar'] = mc

fold_rows = []
for n in sorted(d.node.unique()):
    trn = (d.node != n).values; ten = ~trn
    ytr_n, yte_n = state_labels(X[trn], X[ten])
    for m in MODELS:
        fold_rows.append(dict(held_out=n, model=m, **{k: round(v, 4) for k, v in
                          metrics(yte_n, fit_predict(m, X[trn], ytr_n, X[ten])).items()}))
lono = pd.DataFrame(fold_rows); lono.to_csv('comparison_lono_folds.csv', index=False)
wide = lono.pivot(index='held_out', columns='model', values='macro_f1')[MODELS]
print('\nLeave one node out, macro F1 per fold:'); print(wide.to_string())
print('\nmean (sd):', {m: f'{wide[m].mean():.3f} ({wide[m].std():.3f})' for m in MODELS})

N, k = wide.shape
ranks = wide.rank(axis=1, ascending=False)
R = ranks.mean()
chi2F, pF = stats.friedmanchisquare(*[wide[m].values for m in MODELS])
FF = (N - 1) * chi2F / (N * (k - 1) - chi2F) if N * (k - 1) != chi2F else float('inf')
pFF = stats.f.sf(FF, k - 1, (k - 1) * (N - 1))
q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}[k]
CD = q05 * np.sqrt(k * (k + 1) / (6 * N))
print(f'\nFriedman over {N} folds, {k} models: chi2_F = {chi2F:.3f}, p = {pF:.4f};  '
      f'Iman Davenport F = {FF:.3f}, p = {pFF:.4f};  Nemenyi CD (alpha 0.05) = {CD:.3f}')
print('mean ranks:', R.round(2).to_dict())
out['friedman'] = dict(N=int(N), k=int(k), chi2_F=round(float(chi2F), 4), p=round(float(pF), 6),
                       iman_davenport_F=round(float(FF), 4), p_F=round(float(pFF), 6),
                       nemenyi_CD=round(float(CD), 3), mean_ranks={m: round(float(v), 2) for m, v in R.items()})

iqr = d.groupby('node').moisture.transform(lambda s: max(s.quantile(.75) - s.quantile(.25), 1e-6))
std_feats = np.column_stack([d.moisture, d.ph, d.temp, d.dmoist_1h * iqr, d.dmoist_3h * iqr])
X_std = np.clip(StandardScaler().fit_transform(std_feats), -5, 5)
def run_len(lab, node):
    runs = []
    for _, g in pd.DataFrame({'l': lab, 'n': node}).groupby('n', sort=False):
        v = g.l.values; r = 1
        for a, b in zip(v[:-1], v[1:]):
            if a == b: r += 1
            else: runs.append(r); r = 1
        runs.append(r)
    return float(np.mean(runs))
def transfer_ari(Xm):
    full = KMeans(K, n_init=30, random_state=SEED).fit(Xm).labels_; aris = []
    for n in sorted(d.node.unique()):
        trn = (d.node != n).values
        pr = KMeans(K, n_init=30, random_state=SEED).fit(Xm[trn]).predict(Xm[~trn])
        aris.append(adjusted_rand_score(full[~trn], pr))
    return full, float(np.mean(aris)), float(np.min(aris))
cl = []
for label, Xm in [('proposed: per node normalisation', X), ('standard: pooled standardisation', X_std)]:
    lab, ari_mean, ari_min = transfer_ari(Xm)
    cl.append(dict(approach=label, silhouette=round(silhouette_score(Xm, lab), 3),
                   davies_bouldin=round(davies_bouldin_score(Xm, lab), 3),
                   nmi_with_node=round(normalized_mutual_info_score(d.node, lab), 3),
                   transfer_ari_mean=round(ari_mean, 3), transfer_ari_min=round(ari_min, 3),
                   mean_run_hours=round(run_len(lab, d.node.values), 2),
                   largest_cluster_share=round(pd.Series(lab).value_counts(normalize=True).max(), 3)))
clus = pd.DataFrame(cl); clus.to_csv('comparison_clustering.csv', index=False)
print('\nState identification, proposed vs standard approach (k = 4):'); print(clus.to_string(index=False))
json.dump(out, open('comparison_tests.json', 'w'), indent=2)
print('\nwrote comparison_*.csv and comparison_tests.json')
