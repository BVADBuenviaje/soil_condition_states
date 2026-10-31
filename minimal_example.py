import sys, openpyxl, numpy as np, pandas as pd, warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from common import name_clusters
warnings.filterwarnings('ignore')
XLSX = sys.argv[1] if len(sys.argv) > 1 else 'data/One_Month_Calibrated_Data.xlsx'

ws = openpyxl.load_workbook(XLSX, data_only=True)['Calibrated DATA']
rows = [[r[0], r[10], *r[12:20]] for r in ws.iter_rows(min_row=2, max_row=3188, values_only=True)
        if r[0] in ('Node-01', 'Node-02', 'Node-03', 'Node-04', 'Node-05') and hasattr(r[10], 'year')
        and all(isinstance(v, (int, float)) for v in r[12:20])]
d = pd.DataFrame(rows, columns=['node', 'ts', 'temp', 'moisture', 'ec', 'ph', 'n', 'p', 'k', 'salinity'])
d = d[d.temp < 45].sort_values(['node', 'ts']).reset_index(drop=True)
print(f'{len(d)} readings from {d.node.nunique()} nodes')

iqr = lambda s: max(s.quantile(.75) - s.quantile(.25), 1e-6)
for f in ['moisture', 'ph', 'temp']:
    d[f + '_rel'] = d.groupby('node')[f].transform(lambda s: (s - s.median()) / iqr(s))
for H in (1, 3):
    parts = []
    for _, g in d.groupby('node', sort=False):
        s = g.set_index('ts').moisture
        prev = s.reindex(s.index - pd.Timedelta(hours=H), method='nearest', tolerance=pd.Timedelta(minutes=20))
        parts.append(s.values - prev.values)
    d[f'dmoist_{H}h'] = np.nan_to_num(np.concatenate(parts), nan=0.0)
    d[f'dmoist_{H}h'] /= d.groupby('node').moisture.transform(iqr)
FEAT = ['moisture_rel', 'ph_rel', 'temp_rel', 'dmoist_1h', 'dmoist_3h']
X = d[FEAT].clip(-5, 5).values

train = (d.ts < pd.Timestamp('2025-11-06')).values; test = ~train
km = KMeans(4, n_init=30, random_state=0).fit(X[train])
name = name_clusters(km.cluster_centers_)
y_train = np.array([name[i] for i in km.labels_]); y_test = np.array([name[i] for i in km.predict(X[test])])

rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features=1, max_depth=8,
                            class_weight='balanced', bootstrap=True, random_state=0, n_jobs=-1)
p = rf.fit(X[train], y_train).predict(X[test])

print(f'train {train.sum()} readings, test {test.sum()} readings')
print(f'accuracy {accuracy_score(y_test, p):.4f}   balanced accuracy {balanced_accuracy_score(y_test, p):.4f}   '
      f'macro F1 {f1_score(y_test, p, average="macro"):.4f}')
