import pandas as pd, numpy as np
raw = pd.read_csv('soil_clean.csv', parse_dates=['ts'])
d = raw[raw.temp < 45].copy()
d['hour'] = d.ts.dt.hour
print(f'raw {len(raw)}   analysis {len(d)}')

print('\n=== A. descriptive statistics, four measured channels ===')
for c, u in [('moisture','percent'),('ph','pH'),('temp','deg C'),('ec','dS/m')]:
    s = d[c]; q1, q3 = s.quantile([.25,.75])
    print(f'{c:<9} n={s.count()} mean={s.mean():.2f} median={s.median():.2f} sd={s.std():.2f} '
          f'min={s.min():.2f} q1={q1:.2f} q3={q3:.2f} max={s.max():.2f}')

print('\n=== B. per node ===')
span = (raw.ts.max()-raw.ts.min()).total_seconds()/3600
for n, g in d.groupby('node'):
    print(f'{n}: n={len(g)} cov={100*len(raw[raw.node==n])/span:.1f}%  '
          f'moist med={g.moisture.median():.1f} iqr={g.moisture.quantile(.75)-g.moisture.quantile(.25):.1f}  '
          f'pH med={g.ph.median():.2f} iqr={g.ph.quantile(.75)-g.ph.quantile(.25):.2f}  '
          f'temp med={g.temp.median():.1f} iqr={g.temp.quantile(.75)-g.temp.quantile(.25):.1f}  '
          f'ec med={g.ec.median():.3f}')

print('\n=== C. readings by hour of day ===')
h = d.groupby('hour').size()
print(h.to_string())
print(f'mean per hour {h.mean():.1f}   min {h.min()} at {h.idxmin()}h   max {h.max()} at {h.idxmax()}h')
low = h[h < 0.6*h.mean()]
print('hours below 60 percent of the mean:', dict(low))

print('\n=== D. minority conditions ===')
ceil = (raw.moisture > 100).sum()
print(f'moisture above 100: {ceil} of {len(raw)} = {100*ceil/len(raw):.1f}%')
print(f'pH outside 3 to 10: {(~raw.ph.between(3,10)).sum()}')
oor = int(((~raw.moisture.between(0, 100)) | (~raw.ph.between(3, 10)) | (~raw.temp.between(0, 45)) | (~raw.ec.between(0, 2))).sum())
print(f'out of range total, distinct readings (appendix): {oor} = {100*oor/len(raw):.1f}%')

print('\n=== E. random forest settings (selected by tune_models.py) ===')
import json
c = json.load(open('best_configs.json'))
for m in c: print(m, 'fixed', c[m]['fixed'], 'tuned', c[m]['tuned'], 'val macro F1', c[m]['val_macro_f1'], 'configs', c[m]['n_configs'])

print('\n=== F. chronological split sizes ===')
split = pd.Timestamp('2025-11-06')
print(f'train {int((d.ts<split).sum())}   test {int((d.ts>=split).sum())}   '
      f'test share {100*(d.ts>=split).mean():.1f}%')
