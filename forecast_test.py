import pandas as pd, numpy as np, warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
d = pd.read_csv('soil_clean.csv', parse_dates=['ts'])
d = d[d.temp<45].sort_values(['node','ts'])
frames=[]
for n,g in d.groupby('node'):
    g = g.set_index('ts')[['moisture','ph','temp']].resample('h').mean().interpolate(limit=3)
    g['node']=n; frames.append(g)
h = pd.concat(frames).reset_index()
LAGS=[1,2,3,6,12,24]; HORIZONS=[1,3,6,12,24]
def make(g):
    g=g.copy()
    for v in ['moisture','ph','temp']:
        for L in LAGS: g[f'{v}_lag{L}'] = g[v]-g[v].shift(L)
        g[f'{v}_d3'] = g[v]-g[v].rolling(24, min_periods=6).mean()
    g['hour']=g.ts.dt.hour; g['sin_h']=np.sin(2*np.pi*g.hour/24); g['cos_h']=np.cos(2*np.pi*g.hour/24)
    for H in HORIZONS:
        g[f'y_moist_{H}']=g.moisture.shift(-H); g[f'y_ph_{H}']=g.ph.shift(-H)
        g[f'naive_moist_{H}']=g.moisture; g[f'naive_ph_{H}']=g.ph
        g[f'daily_moist_{H}']=g.moisture.shift(24-H); g[f'daily_ph_{H}']=g.ph.shift(24-H)
    return g
h = pd.concat([make(g) for _,g in h.groupby('node')])
feat=[c for c in h.columns if '_lag' in c or c.endswith('_d3') or c in ('sin_h','cos_h')]
split = pd.Timestamp('2025-11-06')
print('Forecasting test v2 (model predicts the CHANGE from now, inputs are recent changes) — train on first ~3.5 weeks of all nodes, test on the last week (Nov 6-13)')
print('MAE = average size of the error (lower is better).  Naive = "stays the same".  Daily = "same as this hour yesterday".')
VAL_START = pd.Timestamp('2025-10-30'); LEAVES = [5, 20, 50, 100, 200]
tune_rows, chosen = [], {}
for target in ('moist', 'ph'):
    for H in HORIZONS:
        y=f'y_{target}_{H}'; nv=f'naive_{target}_{H}'; cols=feat+[y,nv]
        a=h[h.ts<VAL_START].dropna(subset=cols); v=h[(h.ts>=VAL_START)&(h.ts<split)].dropna(subset=cols)
        maes={}
        for L in LEAVES:
            m=RandomForestRegressor(300, min_samples_leaf=L, random_state=0, n_jobs=-1).fit(a[feat], a[y]-a[nv])
            maes[L]=float(np.mean(np.abs(m.predict(v[feat])+v[nv]-v[y])))
            tune_rows.append(dict(target=target, horizon=H, min_samples_leaf=L, val_mae=round(maes[L],6),
                                  val_naive_mae=round(float(np.mean(np.abs(v[nv]-v[y]))),6)))
        best=min(maes.values()); chosen[(target,H)]=max(L for L in LEAVES if maes[L] <= best*1.005)
pd.DataFrame(tune_rows).to_csv('forecast_tuning.csv', index=False)
print('chosen minimum leaf size per target and horizon:', {f'{t}_{H}h': L for (t,H),L in chosen.items()})

rows=[]
for target,unit in (('moist','% moisture'),('ph','pH')):
    print(f'\nTarget: {target} ({unit})')
    print(' horizon  leaf   RF    naive   daily   RF-vs-naive')
    for H in HORIZONS:
        y=f'y_{target}_{H}'; cols=feat+[y,f'naive_{target}_{H}',f'daily_{target}_{H}']
        tr=h[(h.ts<split)].dropna(subset=cols); te=h[(h.ts>=split)].dropna(subset=cols)
        L=chosen[(target,H)]
        rf=RandomForestRegressor(300, min_samples_leaf=L, random_state=0, n_jobs=-1).fit(tr[feat], tr[y]-tr[f'naive_{target}_{H}'])
        mae=lambda p: np.mean(np.abs(p-te[y]))
        r=mae(rf.predict(te[feat])+te[f'naive_{target}_{H}']); nv=mae(te[f'naive_{target}_{H}']); dy=mae(te[f'daily_{target}_{H}'])
        print(f'  {H:3d}h   {L:4d}   {r:5.2f}   {nv:5.2f}   {dy:5.2f}     {100*(nv-r)/nv:+5.0f}%')
        rows.append(dict(target=target,horizon=H,min_samples_leaf=L,rf=r,naive=nv,daily=dy,skill=1-r/nv,n_test=len(te)))
print('\nSame test but hiding one node entirely (train on 4, forecast the 5th), moisture 6h ahead:')
H=6; y=f'y_moist_{H}'; cols=feat+[y,f'naive_moist_{H}']; L=chosen[('moist',H)]; lono_rows=[]
for n in sorted(h.node.unique()):
    tr=h[(h.node!=n)].dropna(subset=cols); te=h[(h.node==n)].dropna(subset=cols)
    rf=RandomForestRegressor(300, min_samples_leaf=L, random_state=0, n_jobs=-1).fit(tr[feat], tr[y]-tr[f'naive_moist_{H}'])
    r=np.mean(np.abs(rf.predict(te[feat])+te[f'naive_moist_{H}']-te[y])); nv=np.mean(np.abs(te[f'naive_moist_{H}']-te[y]))
    print(f'  hidden {n}:  RF {r:5.2f}   naive {nv:5.2f}   {100*(nv-r)/nv:+5.0f}%')
    lono_rows.append(dict(held_out=n, rf=r, naive=nv, skill=1-r/nv))
pd.DataFrame(lono_rows).round(6).to_csv('forecast_lono.csv', index=False)
pd.DataFrame(rows).round(6).to_csv('forecast_results.csv', index=False)
