import json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, adjusted_rand_score
from common import load, state_labels, name_clusters, present, FEAT, TEST_START, K, SEED
warnings.filterwarnings('ignore')

d, X = load()
cfg = json.load(open('best_configs.json'))
tr = (d.ts < TEST_START).values; te = ~tr
ytr, yte = state_labels(X[tr], X[te])
ORDER = present(ytr)
NAME = {'moisture_rel': 'moisture', 'ph_rel': 'pH', 'temp_rel': 'temperature',
        'dmoist_1h': 'moisture change, one hour', 'dmoist_3h': 'moisture change, three hours'}

def profile(Xfull, labels):
    return np.array([Xfull[labels == c].mean(0) for c in range(labels.max() + 1)])

def make(name, keep=None):
    f, t = dict(cfg[name]['fixed']), dict(cfg[name]['tuned'])
    if name == 'random_forest':
        if keep is not None and t.get('max_features', 1) > len(keep):
            t['max_features'] = len(keep)
        return RandomForestClassifier(**f, **t, n_jobs=-1)
    if name == 'decision_tree': return DecisionTreeClassifier(**f, **t)
    return LogisticRegression(**f, **t)

def scores(y, p, labels):
    return dict(accuracy=round(accuracy_score(y, p), 6),
                balanced_accuracy=round(balanced_accuracy_score(y, p), 6),
                macro_f1=round(f1_score(y, p, average='macro', labels=labels, zero_division=0), 6))

rf = make('random_forest').fit(X[tr], ytr)
lr = make('logistic_regression').fit(X[tr], ytr)
imp_trees = np.array([t.feature_importances_ for t in rf.estimators_])
def scorer(est, Xs, ys):
    return f1_score(ys, est.predict(Xs), average='macro', labels=ORDER, zero_division=0)
perm_rf = permutation_importance(rf, X[te], yte, scoring=scorer, n_repeats=30, random_state=SEED, n_jobs=1)
perm_lr = permutation_importance(lr, X[te], yte, scoring=scorer, n_repeats=30, random_state=SEED, n_jobs=1)
imp = pd.DataFrame(dict(
    feature=[NAME[f] for f in FEAT],
    forest_impurity=imp_trees.mean(0).round(4), forest_impurity_sd=imp_trees.std(0, ddof=1).round(4),
    forest_permutation=perm_rf.importances_mean.round(4), forest_permutation_sd=perm_rf.importances_std.round(4),
    logistic_permutation=perm_lr.importances_mean.round(4), logistic_permutation_sd=perm_lr.importances_std.round(4)))
imp.to_csv('ch4_importance.csv', index=False)
print('Importance of each input (impurity inside the forest, and permutation on the test week):')
print(imp.to_string(index=False))

coef = pd.DataFrame(lr.coef_, index=lr.classes_, columns=[NAME[f] for f in FEAT]).round(3)
coef.to_csv('ch4_lr_coefficients.csv')
print('\nLogistic regression coefficients (rows states, columns inputs, inputs on a common scale):')
print(coef.to_string())

SETS = [('none', [])] + [(NAME[f], [f]) for f in FEAT] + [
    ('both moisture changes', ['dmoist_1h', 'dmoist_3h']),
    ('pH and temperature', ['ph_rel', 'temp_rel'])]

def cols(drop):
    keep = [f for f in FEAT if f not in drop]
    return keep, [FEAT.index(f) for f in keep]

rows = []
for label, drop in SETS:
    keep, idx = cols(drop)
    for m in ['random_forest', 'logistic_regression', 'decision_tree']:
        p = make(m, keep).fit(X[tr][:, idx], ytr).predict(X[te][:, idx])
        r = scores(yte, p, ORDER)
        fold = []
        for n in sorted(d.node.unique()):
            trn = (d.node != n).values; ten = ~trn
            ytr_n, yte_n = state_labels(X[trn], X[ten])
            pn = make(m, keep).fit(X[trn][:, idx], ytr_n).predict(X[ten][:, idx])
            L = present(yte_n, pn)
            fold.append(f1_score(yte_n, pn, average='macro', labels=L, zero_division=0))
        rows.append(dict(removed=label, model=m, **r,
                         lono_macro_f1_mean=round(float(np.mean(fold)), 6),
                         lono_macro_f1_sd=round(float(np.std(fold, ddof=1)), 6)))
abl_c = pd.DataFrame(rows)
base = abl_c[abl_c.removed == 'none'].set_index('model')
abl_c['test_macro_f1_change'] = [round(r.macro_f1 - base.loc[r.model, 'macro_f1'], 6) for r in abl_c.itertuples()]
abl_c.to_csv('ch4_ablation_classifier.csv', index=False)
print('\nRemoval from the classifier only, states unchanged:')
print(abl_c.to_string(index=False))

rows = []
for label, drop in SETS:
    keep, idx = cols(drop)
    Xtr_, Xte_ = X[tr][:, idx], X[te][:, idx]
    km = KMeans(K, n_init=30, random_state=SEED).fit(Xtr_)
    nm = name_clusters(profile(X[tr], km.labels_))
    ytr2 = np.array([nm[c] for c in km.labels_], dtype=object)
    yte2 = np.array([nm[c] for c in km.predict(Xte_)], dtype=object)
    L2 = present(ytr2)
    same = sorted(set(L2)) == sorted(set(ORDER))
    moved = float(np.mean(ytr2 != ytr)) if same else float('nan')
    folds_same = 0
    for n in sorted(d.node.unique()):
        trn = ((d.node != n) & (d.ts < TEST_START)).values
        kf = KMeans(K, n_init=30, random_state=SEED).fit(X[trn][:, idx])
        folds_same += sorted(set(name_clusters(profile(X[trn], kf.labels_)).values())) == sorted(set(L2))
    row = dict(removed=label, states=' | '.join(L2), same_state_set=bool(same),
               folds_with_the_same_set=int(folds_same),
               ari_with_reference=round(adjusted_rand_score(ytr, ytr2), 4),
               share_of_readings_reassigned=None if np.isnan(moved) else round(moved, 4))
    for m in ['random_forest', 'logistic_regression']:
        p = make(m, keep).fit(Xtr_, ytr2).predict(Xte_)
        row[f'{m}_macro_f1'] = round(f1_score(yte2, p, average='macro', labels=L2, zero_division=0), 6)
    rows.append(row)
abl_k = pd.DataFrame(rows)
abl_k.to_csv('ch4_ablation_clustering.csv', index=False)
print('\nRemoval from the clustering as well, states re derived:')
print(abl_k.to_string(index=False))

S = dict(
    most_important_forest=NAME[FEAT[int(np.argmax(perm_rf.importances_mean))]],
    least_important_forest=NAME[FEAT[int(np.argmin(perm_rf.importances_mean))]],
    largest_classifier_loss=abl_c.loc[abl_c.test_macro_f1_change.idxmin(), ['removed', 'model', 'test_macro_f1_change']].to_dict(),
    removals_that_keep_the_states=[r.removed for r in abl_k.itertuples() if r.same_state_set],
    removals_that_change_the_states=[r.removed for r in abl_k.itertuples() if not r.same_state_set],
    n_settings=len(SETS))
json.dump(S, open('ch4_feature_summary.json', 'w'), indent=2)
print('\nwrote ch4_importance.csv, ch4_lr_coefficients.csv, ch4_ablation_classifier.csv, '
      'ch4_ablation_clustering.csv, ch4_feature_summary.json')
