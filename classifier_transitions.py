import pandas as pd, numpy as np, warnings, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from common import load, state_labels, present, run_lengths, FEAT, ORDER, TEST_START
warnings.filterwarnings('ignore')
_c = json.load(open('best_configs.json'))['random_forest']
RF_PARAMS = {**_c['fixed'], **_c['tuned'], 'n_jobs': -1}
print('Random Forest configuration (from tune_models.py):', RF_PARAMS)
d, X = load()
print('Reference state counts (all nodes):'); print(d.state_ref.value_counts().reindex(ORDER).to_string())

tr = (d.ts < TEST_START).values; te = ~tr
y_tr, y_te = state_labels(X[tr], X[te]); L = present(y_tr)
print(f'\nstates of the training period partition: {L}')
rf = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y_tr); p = rf.predict(X[te])
print(f'\n=== RF state classifier, TIME split: train 13 Oct to 5 Nov ({tr.sum()} rows), test 6 to 13 Nov ({te.sum()} rows) ===')
print(f'accuracy {accuracy_score(y_te, p):.3f}   macro F1 {f1_score(y_te, p, average="macro", labels=L, zero_division=0):.3f}')
print(pd.DataFrame(confusion_matrix(y_te, p, labels=L), index=[f'true {o}' for o in L], columns=L).to_string())
print(classification_report(y_te, p, labels=L, digits=2, zero_division=0))
maj = pd.Series(y_tr).mode()[0]; print(f'majority class baseline accuracy: {np.mean(y_te == maj):.3f}  ({maj})')
print('feature importance:', {f: round(float(v), 3) for f, v in zip(FEAT, rf.feature_importances_)})

print('\n=== RF state classifier, LEAVE ONE NODE OUT ===')
rows = []
for n in sorted(d.node.unique()):
    trn = (d.node != n).values; ten = ~trn
    ytr, yte = state_labels(X[trn], X[ten])
    pp = RandomForestClassifier(**RF_PARAMS).fit(X[trn], ytr).predict(X[ten])
    Ln = present(yte, pp)
    rows.append(dict(held_out=n, n=int(ten.sum()), states=' | '.join(Ln), accuracy=round(accuracy_score(yte, pp), 3),
                     macro_F1=round(f1_score(yte, pp, average='macro', labels=Ln, zero_division=0), 3)))
print(pd.DataFrame(rows).to_string(index=False))

print('\n=== State transitions (all nodes, consecutive hours only) ===')
T = pd.DataFrame(0, index=ORDER, columns=ORDER, dtype=float)
for n, g in d.groupby('node'):
    s = g.state_ref.values; t = g.ts.values
    for i in range(1, len(g)):
        gap = (t[i] - t[i - 1]) / np.timedelta64(1, 'h')
        if 0.75 <= gap <= 1.25: T.loc[s[i - 1], s[i]] += 1
P = T.div(T.sum(axis=1), axis=0)
print('Transition counts:'); print(T.astype(int).to_string())
print('Transition probabilities (row = state now, column = state next hour):'); print(P.round(3).to_string())
P.round(4).to_csv('transition_probabilities.csv'); T.astype(int).to_csv('transition_counts.csv')
rl, _ = run_lengths(d.state_ref, d.node, d.ts)
print('\nMean run length (consecutive hours in the same state; a gap ends a run):')
for o in ORDER: print(f'  {o:20s} {rl[o]:5.2f} h')
sat_next = T.loc['Saturated'].drop('Saturated'); print('\nAfter a saturated hour ends, the next state is:', (sat_next / sat_next.sum()).round(2).to_dict())
rec = []
for n, g in d.groupby('node'):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        if g.state_ref[i] == 'Saturated' and (i == 0 or g.state_ref[i - 1] != 'Saturated'):
            for j in range(i + 1, len(g)):
                h = (g.ts[j] - g.ts[i]) / pd.Timedelta(hours=1)
                if h > 12: break
                if g.moisture_rel[j] < 0.5: rec.append(h); break
rec = np.array(rec)
print(f'Recovery after a saturated episode (hours until moisture is back within half an IQR of the node median): '
      f'median {np.median(rec):.1f} h, mean {rec.mean():.1f} h, max {rec.max():.1f} h, n = {len(rec)} episodes')
json.dump(dict(run_hours={o: round(float(rl[o]), 3) for o in ORDER},
               after_saturated={k: round(float(v), 3) for k, v in (sat_next / sat_next.sum()).items()},
               recovery=dict(n=int(len(rec)), median=round(float(np.median(rec)), 2), mean=round(float(rec.mean()), 2),
                             max=round(float(rec.max()), 2)),
               lono_rf=rows), open('transitions_summary.json', 'w'), indent=1)
print('wrote transition_counts.csv, transition_probabilities.csv, transitions_summary.json')
