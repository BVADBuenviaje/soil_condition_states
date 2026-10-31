import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, normalized_mutual_info_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from common import load, name_clusters, present, FEAT, TEST_START, K
warnings.filterwarnings('ignore')
d, X = load(); cfg = json.load(open('best_configs.json'))
tr = (d.ts < TEST_START).values; te = ~tr
km = KMeans(K, n_init=30, random_state=0).fit(X[tr]); nm = name_clusters(km.cluster_centers_)
ytr = np.array([nm[c] for c in km.labels_], dtype=object)
yte = np.array([nm[c] for c in km.predict(X[te])], dtype=object); L = present(ytr)
rf = RandomForestClassifier(**cfg['random_forest']['fixed'], **cfg['random_forest']['tuned'], n_jobs=-1).fit(X[tr], ytr)
lr = LogisticRegression(**cfg['logistic_regression']['fixed'], **cfg['logistic_regression']['tuned']).fit(X[tr], ytr)
p_rf, p_lr = rf.predict(X[te]), lr.predict(X[te])
rows = []
def chk(what, used, check, how, stochastic=False):
    rows.append(dict(quantity=what, value_used=round(float(used), 6), independent_check=round(float(check), 6),
                     difference=round(float(used) - float(check), 9), method_of_check=how,
                     stochastic=stochastic))

maj = pd.Series(ytr).mode()[0]
chk('Majority class baseline, accuracy on the test week', accuracy_score(yte, np.full(len(yte), maj, dtype=object)),
    (yte == maj).mean(), 'share of the test readings in that state, counted directly')
dist = np.sqrt(((X[te][:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1))
manual = np.array([nm[c] for c in dist.argmin(1)], dtype=object)
chk('Nearest centroid reference, agreement with the clustering', 1.0, (manual == yte).mean(),
    'distance to every centre computed by hand, smallest taken')
inertia_manual = ((X[tr] - km.cluster_centers_[km.labels_]) ** 2).sum()
chk('Within cluster sum of squares of the training partition', km.inertia_, inertia_manual,
    'squared distance of every reading to its own centre, summed by hand')
cm = confusion_matrix(yte, p_rf, labels=L)
rec = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(L)), where=cm.sum(1) > 0)
prec = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(len(L)), where=cm.sum(0) > 0)
f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(L)), where=(prec + rec) > 0)
chk('Random Forest, macro F1 on the test week', f1_score(yte, p_rf, average='macro', labels=L, zero_division=0), f1.mean(),
    'precision and recall per state taken from the confusion matrix')
chk('Random Forest, balanced accuracy on the test week', balanced_accuracy_score(yte, p_rf), rec.mean(),
    'mean of the per state recalls from the confusion matrix')
a_ok, b_ok = (p_rf == yte), (p_lr == yte)
b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok))
chi2 = (abs(b - c) - 1) ** 2 / (b + c); p_ours = stats.chi2.sf(chi2, 1)
sm = sm_mcnemar(np.array([[int(np.sum(a_ok & b_ok)), b], [c, int(np.sum(~a_ok & ~b_ok))]]), exact=False, correction=True)
chk('McNemar statistic, forest against logistic regression', chi2, sm.statistic, 'statsmodels 0.15.0, corrected chi square')
chk('McNemar p value, forest against logistic regression', p_ours, sm.pvalue, 'statsmodels 0.15.0, corrected chi square')
wide = pd.read_csv('comparison_lono_folds.csv').pivot(index='held_out', columns='model', values='macro_f1')
MODELS = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']
wide = wide[MODELS]; N, k = wide.shape
R = wide.rank(axis=1, ascending=False).mean()
chi2F_manual = 12 * N / (k * (k + 1)) * ((R ** 2).sum() - k * (k + 1) ** 2 / 4)
chi2F_scipy, _ = stats.friedmanchisquare(*[wide[m].values for m in MODELS])
chk('Friedman statistic over the five withheld nodes', chi2F_scipy, chi2F_manual,
    'the formula of Section 3.7.6 applied to the mean ranks by hand')
full = KMeans(K, n_init=30, random_state=0).fit(X)
lab = full.labels_; node = pd.factorize(d.node)[0]
ct = pd.crosstab(lab, node).values; n = ct.sum(); pij = ct / n; pi = pij.sum(1); pj = pij.sum(0)
mi = np.nansum(np.where(pij > 0, pij * np.log(pij / np.outer(pi, pj)), 0))
H = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
chk('Mutual information between state and node identity', normalized_mutual_info_score(d.node, lab), 2 * mi / (H(pi) + H(pj)),
    'entropies and mutual information computed from the contingency table')
fr = pd.read_csv('forecast_results.csv'); r6 = fr[(fr.target == 'moist') & (fr.horizon == 6)].iloc[0]
rowsf = pd.read_csv('forecast_rows_moist6.csv')
chk('Persistence baseline, moisture six hours ahead, mean absolute error', r6.naive,
    np.abs(rowsf.naive_moist_6 - rowsf.y_moist_6).mean(), 'mean of the absolute differences on the saved evaluation rows')
chk('Forest regressor, moisture six hours ahead, mean absolute error', r6.rf,
    np.abs(rowsf.rf - rowsf.y_moist_6).mean(), 'mean of the absolute differences on the saved evaluation rows')
chk('Forecast skill relative to persistence, moisture six hours ahead', r6.skill,
    1 - np.abs(rowsf.rf - rowsf.y_moist_6).mean() / np.abs(rowsf.naive_moist_6 - rowsf.y_moist_6).mean(),
    'one minus the ratio of the two errors above')
wl = pd.read_csv('ch4_wilcoxon.csv'); sd_ = pd.read_csv('ch4_seeds.csv')
dif = (sd_.random_forest_macro_f1 - sd_.logistic_regression_macro_f1).values
nz = dif[dif != 0]
rank = stats.rankdata(np.abs(nz))
w_hand = min(rank[nz > 0].sum(), rank[nz < 0].sum())
chk('Wilcoxon statistic, forest against logistic regression over the runs',
    wl.loc[wl.comparison == 'random_forest vs logistic_regression', 'statistic'].iloc[0], w_hand,
    'smaller of the two signed rank sums, ranked by hand')

Q = json.load(open('ch4_quality.json'))
c_ = X - X.mean(0)
lam = np.sort(np.linalg.eigvalsh(c_.T @ c_ / len(X)))[::-1]
chk('Relaxation lower bound on the clustering objective', Q['lower_bound'],
    float((c_ ** 2).sum() - len(X) * lam[:K - 1].sum()),
    'total scatter minus n times the three largest eigenvalues, computed by hand')

imp = pd.read_csv('ch4_importance.csv')
base = f1_score(yte, p_lr, average='macro', labels=L, zero_division=0)
rng_ = np.random.default_rng(0); drops = []
for _ in range(30):
    Xp = X[te].copy(); Xp[:, FEAT.index('temp_rel')] = rng_.permutation(Xp[:, FEAT.index('temp_rel')])
    drops.append(base - f1_score(yte, lr.predict(Xp), average='macro', labels=L, zero_division=0))
chk('Permutation importance of temperature for logistic regression',
    imp.loc[imp.feature == 'temperature', 'logistic_permutation'].iloc[0], float(np.mean(drops)),
    'thirty fresh permutations of that column by hand; a second random draw, so the two agree '
    'only to within the spread of a permutation', stochastic=True)

v = pd.DataFrame(rows); v.to_csv('ch4_verification.csv', index=False)
pd.set_option('display.width', 200); print(v.to_string(index=False))
worst = v[~v.stochastic].difference.abs().max()
print(f'\nlargest disagreement between the value used and its independent check, '
      f'over the checks that must agree exactly: {worst:.2e}')
sto = v[v.stochastic]
for _, r in sto.iterrows():
    print(f'stochastic check, {r.quantity.lower()}: {r.value_used:.4f} against {r.independent_check:.4f}')
