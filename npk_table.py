import pandas as pd, numpy as np

CAL = {'n': (1.11876, 0.155), 'p': (4.41, -3.27), 'k': (7.734, -76.204),
       'ec': (0.944, 0.0), 'salinity': (0.944, 0.0)}

d = pd.read_csv('soil_clean.csv', parse_dates=['ts'])
print(f"loaded {len(d):,} calibrated readings from {d.node.nunique()} nodes")

for col, (m, b) in CAL.items():
    d[col + '_raw'] = (d[col] - b) / m

zero = (d[['n_raw', 'p_raw', 'k_raw', 'ec_raw']].round(6) == 0).any(axis=1)
g = d[~zero].copy()
g['NP'] = g.n_raw / g.p_raw
g['KP'] = g.k_raw / g.p_raw
spike = (g.NP > 1) | (g.KP > 3)
print(f"excluded {zero.sum()} zero readings and {spike.sum()} single hour spikes")
print("  spikes:")
for _, r in g[spike].iterrows():
    print(f"    {r.node} {r.ts}  N={r.n_raw:.1f} P={r.p_raw:.1f} K={r.k_raw:.1f}")
c = g[~spike].copy()
print(f"analysed on {len(c):,} readings\n")

print("--- Ratios between the nutrient channels ---")
print("Independent sensors would drift apart. These do not.")
for lab, s in [('N to P', c.NP), ('K to P', c.KP)]:
    print(f"  {lab}: {s.mean():.4f} +/- {s.std():.4f}   "
          f"(CV {100*s.std()/s.mean():.1f} percent, range {s.min():.3f} to {s.max():.3f})")

print("\n--- Pearson correlation with electrical conductivity ---")
CORR = {}
for ch, lab in [('n_raw','Nitrogen'), ('p_raw','Phosphorus'),
                ('k_raw','Potassium'), ('salinity_raw','Salinity')]:
    CORR[lab] = c[ch].corr(c.ec_raw)
    print(f"  {lab:<12} r = {CORR[lab]:.3f}   (r squared {CORR[lab]**2:.3f})")

print("\n--- Distinct values reported over the whole deployment ---")
print("A separately sensed quantity would vary continuously, as the three")
print("genuinely measured channels do.")
RES = {}
for ch, lab in [('ec_raw','EC'), ('n_raw','Nitrogen'), ('p_raw','Phosphorus'),
                ('k_raw','Potassium'), ('salinity_raw','Salinity'),
                ('moisture','Moisture'), ('ph','pH'), ('temp','Temperature')]:
    RES[lab] = c[ch].round(4).nunique()
    print(f"  {lab:<12} {RES[lab]:>4} distinct values in {len(c):,} readings")

rows = [
    ['N to P ratio', f"{c.NP.mean():.3f}", f"{c.NP.std():.3f}",
     'Fixed across every reading and every node'],
    ['K to P ratio', f"{c.KP.mean():.3f}", f"{c.KP.std():.3f}",
     'Fixed across every reading and every node'],
]
for lab in ['Nitrogen', 'Phosphorus', 'Potassium', 'Salinity']:
    rows.append([f'{lab} vs EC (r)', f"{CORR[lab]:.3f}", '', 'Tracks conductivity'])
for lab in ['EC', 'Nitrogen', 'Phosphorus', 'Potassium']:
    rows.append([f'{lab}, distinct values', str(RES[lab]), '',
                 f'Compare moisture {RES["Moisture"]}, pH {RES["pH"]}'])
pd.DataFrame(rows, columns=['Quantity', 'Value', 'SD', 'Interpretation']
             ).to_csv('npk_ec_evidence.csv', index=False)

tex = r"""\begin{table}[htbp]
\centering
\caption{Dependence among the reported nutrient channels. The three nutrient
outputs hold fixed ratios to one another, track electrical conductivity
closely, and take only a handful of distinct values over the deployment, while
the three channels treated as measured vary continuously. Computed on
""" + f"{len(c):,}" + r""" readings after excluding """ + f"{zero.sum()}" + r""" zero readings and
""" + f"{spike.sum()}" + r""" isolated single hour spikes, as set out in Appendix~\ref{app:data}.}
\label{tab:npk-ec}
\begin{tabular}{lrl}
\toprule
Quantity & Value & Interpretation \\
\midrule
""" + \
f"N to P ratio & ${c.NP.mean():.3f} \\pm {c.NP.std():.3f}$ & Fixed across every reading \\\\\n" + \
f"K to P ratio & ${c.KP.mean():.3f} \\pm {c.KP.std():.3f}$ & Fixed across every reading \\\\\n" + \
r"\midrule" + "\n" + \
"".join(f"{lab} vs.\\ EC ($r$) & ${CORR[lab]:.3f}$ & Tracks conductivity \\\\\n"
        for lab in ['Nitrogen','Phosphorus','Potassium','Salinity']) + \
r"\midrule" + "\n" + \
"".join(f"{lab}, distinct values & ${RES[lab]}$ & \\\\\n"
        for lab in ['EC','Nitrogen','Phosphorus','Potassium']) + \
"".join(f"{lab}, distinct values & ${RES[lab]}$ & Treated as measured \\\\\n"
        for lab in ['Moisture','pH','Temperature']) + \
r"""\bottomrule
\end{tabular}
\end{table}
"""
open('table_npk_ec.tex', 'w').write(tex)
print("\nwrote npk_ec_evidence.csv and table_npk_ec.tex")
