import json, itertools, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import load, state_labels, ORDER, TRAIN_START, VAL_START, TEST_START
warnings.filterwarnings('ignore')

d, X = load()
train = (d.ts < TEST_START).values
sub   = (d.ts < VAL_START).values
val   = ((d.ts >= VAL_START) & (d.ts < TEST_START)).values
y_train_all, = state_labels(X[train])
y = np.empty(len(d), dtype=object); y[train] = y_train_all
ys, yv = y[sub], y[val]
print(f'sub training {sub.sum()} readings   validation {val.sum()} readings   (test week untouched)')
print('state counts, sub training :', pd.Series(ys).value_counts().reindex(ORDER).fillna(0).astype(int).to_dict())
print('state counts, validation   :', pd.Series(yv).value_counts().reindex(ORDER).fillna(0).astype(int).to_dict())

def score(model):
    model.fit(X[sub], ys)
    return (f1_score(ys, model.predict(X[sub]), average='macro', labels=ORDER, zero_division=0),
            f1_score(yv, model.predict(X[val]), average='macro', labels=ORDER, zero_division=0))

INF = 10**9
GRIDS = {
  'random_forest': dict(
      fixed=dict(n_estimators=400, class_weight='balanced', bootstrap=True, random_state=0, n_jobs=-1),
      grid=dict(min_samples_leaf=[1, 2, 3, 5, 10, 20], max_features=[1, 2, 3, 5], max_depth=[None, 4, 8, 16]),
      build=lambda p, f: RandomForestClassifier(**f, **p),
      simplicity=lambda p: (-p['min_samples_leaf'], p['max_depth'] or INF, p['max_features'])),
  'decision_tree': dict(
      fixed=dict(class_weight='balanced', random_state=0),
      grid=dict(min_samples_leaf=[1, 2, 3, 5, 10, 20], max_depth=[None, 2, 4, 6, 8, 12]),
      build=lambda p, f: DecisionTreeClassifier(**f, **p),
      simplicity=lambda p: (-p['min_samples_leaf'], p['max_depth'] or INF)),
  'logistic_regression': dict(
      fixed=dict(class_weight='balanced', solver='lbfgs', max_iter=5000),
      grid=dict(C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
      build=lambda p, f: LogisticRegression(**f, **p),
      simplicity=lambda p: (p['C'],)),
}

best = {}
for name, g in GRIDS.items():
    t0 = time.time(); rows = []
    keys = list(g['grid'])
    for combo in itertools.product(*[g['grid'][k] for k in keys]):
        p = dict(zip(keys, combo))
        tr_f1, va_f1 = score(g['build'](p, g['fixed']))
        rows.append({**p, 'train_macro_f1': round(tr_f1, 4), 'val_macro_f1': round(va_f1, 4)})
    res = pd.DataFrame(rows)
    res.to_csv(f'tuning_{name}.csv', index=False)
    top = res.val_macro_f1.max()
    near = [r for r in rows if r['val_macro_f1'] >= top - 0.005]
    chosen = min(near, key=lambda r: g['simplicity']({k: r[k] for k in keys}))
    params = {k: (None if (isinstance(chosen[k], float) and np.isnan(chosen[k])) else chosen[k]) for k in keys}
    best[name] = dict(fixed={k: v for k, v in g['fixed'].items() if k != 'n_jobs'}, tuned=params,
                      val_macro_f1=chosen['val_macro_f1'], train_macro_f1=chosen['train_macro_f1'],
                      best_val_macro_f1_any=float(top), n_configs=len(rows),
                      n_within_tie_band=len(near))
    print(f'\n{name}: {len(rows)} configurations (search completed in {time.time()-t0:.1f} s)')
    print(f'  best validation macro F1 {top:.4f}; {len(near)} within 0.005; chosen {params}')
    print(f'  chosen: train {chosen["train_macro_f1"]:.4f}  validation {chosen["val_macro_f1"]:.4f}')

json.dump(best, open('best_configs.json', 'w'), indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
print('\nwrote best_configs.json')
