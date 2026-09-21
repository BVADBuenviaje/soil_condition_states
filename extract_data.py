import sys, openpyxl, pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else 'data/One_Month_Calibrated_Data.xlsx'
wb = openpyxl.load_workbook(src, data_only=True)
ws = wb['Calibrated DATA']

NODES = ('Node-01', 'Node-02', 'Node-03', 'Node-04', 'Node-05')
rows = []
for r in ws.iter_rows(min_row=2, max_row=3188, values_only=True):
    node, ts = r[0], r[10]
    calibrated = r[12:20]
    if node not in NODES:                             continue
    if not hasattr(ts, 'year'):                       continue
    if not all(isinstance(v, (int, float)) for v in calibrated): continue
    rows.append([node, ts, *calibrated])

df = pd.DataFrame(rows, columns=['node','ts','temp','moisture','ec','ph','n','p','k','salinity'])
df = df.sort_values('ts').reset_index(drop=True)
df.to_csv('soil_clean.csv', index=False)

print(f'wrote soil_clean.csv: {len(df)} rows, {df.node.nunique()} nodes, '
      f'{df.ts.min():%Y-%m-%d} to {df.ts.max():%Y-%m-%d}')
print(df.node.value_counts().sort_index().to_string())
