import pandas as pd, numpy as np, warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json
from common import name_clusters, present
_c = json.load(open('best_configs.json'))['random_forest']
RF_PARAMS = {**_c['fixed'], **_c['tuned'], 'n_jobs': -1}
warnings.filterwarnings('ignore')

FEAT = ['moisture_rel','ph_rel','temp_rel','dmoist_1h','dmoist_3h']
raw = pd.read_csv('soil_clean.csv', parse_dates=['ts'])

def build(d):
    d = d.sort_values(['node','ts']).reset_index(drop=True).copy()
    def robust(s):
        q1,q3 = s.quantile([.25,.75]); return (s - s.median())/max(q3-q1,1e-6)
    for f in ['moisture','ph','temp']:
        d[f+'_rel'] = d.groupby('node')[f].transform(robust)
    def delta(g,f,h):
        s=g.set_index('ts')[f]
        prev=s.reindex(s.index-pd.Timedelta(hours=h),method='nearest',tolerance=pd.Timedelta(minutes=20))
        return s.values-prev.values
    for h in (1,3):
        d[f'dmoist_{h}h']=np.nan_to_num(np.concatenate(
            [delta(g,'moisture',h) for _,g in d.groupby('node',sort=False)]),nan=0.0)
    iqr=d.groupby('node').moisture.transform(lambda s:max(s.quantile(.75)-s.quantile(.25),1e-6))
    d['dmoist_1h']/=iqr; d['dmoist_3h']/=iqr
    d['hour']=d.ts.dt.hour
    return d

def report(tag, d):
    d = build(d)
    X = d[FEAT].clip(-5,5).values
    km = KMeans(4, n_init=30, random_state=0).fit(X)
    prof = pd.DataFrame(km.cluster_centers_, columns=FEAT)
    names = name_clusters(km.cluster_centers_)
    d['state'] = [names[c] for c in km.labels_]
    sat = d[d.state=='Saturated']
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    print(f"rows {len(d):,}   silhouette {silhouette_score(X, km.labels_):.3f}   "
          f"NMI with node identity {normalized_mutual_info_score(d.node, d.state):.3f}")
    print("state sizes: " + ", ".join(f"{k} {v}" for k,v in d.state.value_counts().items()))
    if len(sat):
        in_window = ((sat.hour>=7)&(sat.hour<=10)).mean()
        print(f"Saturated: {len(sat)} rows on {sat.node.nunique()} nodes, "
              f"{100*in_window:.0f} percent inside the 07:00 to 11:00 window, "
              f"max moisture {sat.moisture.max():.1f}")
        print("  hours: " + ", ".join(f"{h}h:{n}" for h,n in sat.hour.value_counts().sort_index().items()))
    split = pd.Timestamp('2025-11-06')
    tr, te = d.ts < split, d.ts >= split
    km2 = KMeans(4, n_init=30, random_state=0).fit(X[tr.values])
    n2 = name_clusters(km2.cluster_centers_)
    ytr = [n2[c] for c in km2.labels_]
    yte = [n2[c] for c in km2.predict(X[te.values])]
    rf = RandomForestClassifier(**RF_PARAMS).fit(X[tr.values], ytr)
    pred = rf.predict(X[te.values])
    base = pd.Series(yte).value_counts(normalize=True).max()
    L = present(ytr)
    print(f"training period states: {L}")
    print(f"classifier, chronological split: accuracy {accuracy_score(yte,pred):.3f}  "
          f"macro F1 {f1_score(yte,pred,average='macro',labels=L,zero_division=0):.3f}  majority baseline {base:.3f}")

A = raw[raw.temp < 45]
ok_ph = raw.ph.between(3,10)
B = raw[(raw.temp<45) & ok_ph].copy(); B['moisture'] = B.moisture.clip(upper=100)
C = raw[(raw.temp<45) & ok_ph & raw.moisture.between(0,100) & raw.ec.between(0,2)]

print(f"raw readings {len(raw):,}")
print(f"  A as written       {len(A):,}")
print(f"  B ceiling censored {len(B):,}  ({len(raw)-len(B)} dropped, "
      f"{(raw.moisture>100).sum()} clipped to 100)")
print(f"  C strict           {len(C):,}  ({len(raw)-len(C)} dropped)")
report('A. AS WRITTEN (only temp < 45)', A)
report('B. CEILING CENSORED (moisture clipped at 100, bad pH dropped)', B)
report('C. STRICT (all out of range readings dropped)', C)
