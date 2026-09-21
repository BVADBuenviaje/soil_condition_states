import pandas as pd, numpy as np
d = pd.read_csv('soil_clean.csv', parse_dates=['ts']).sort_values(['node','ts']).reset_index(drop=True)
d = d[d.temp<45].copy()
gaps = d.groupby('node').ts.diff().dt.total_seconds()/3600
print('gap between consecutive readings (hours): median', gaps.median().round(2), '| share ~1h:', ((gaps>0.75)&(gaps<1.25)).mean().round(3), '| gaps >2h:', (gaps>2).sum())
def robust(s):
    q1,q3 = s.quantile([.25,.75]); iqr = max(q3-q1, 1e-6)
    return (s - s.median())/iqr
for f in ['moisture','ph','temp']:
    d[f+'_rel'] = d.groupby('node')[f].transform(robust)
def delta(g, f, h):
    s = g.set_index('ts')[f]
    prev = s.reindex(s.index - pd.Timedelta(hours=h), method='nearest', tolerance=pd.Timedelta(minutes=20))
    return (s.values - prev.values)
for h in (1,3):
    out=[]
    for n,g in d.groupby('node', sort=False):
        out.append(delta(g,'moisture',h))
    d[f'dmoist_{h}h'] = np.nan_to_num(np.concatenate(out), nan=0.0)
iqr = d.groupby('node').moisture.transform(lambda s: max(s.quantile(.75)-s.quantile(.25),1e-6))
d['dmoist_1h'] /= iqr; d['dmoist_3h'] /= iqr
d['hour']=d.ts.dt.hour
FEAT=['moisture_rel','ph_rel','temp_rel','dmoist_1h','dmoist_3h']
print(d[FEAT].describe().round(2).T)
d.to_csv('soil_features.csv', index=False)
print('rows:', len(d), '| per node:', d.node.value_counts().to_dict())
