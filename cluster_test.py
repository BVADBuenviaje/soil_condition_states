import pandas as pd, numpy as np, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings('ignore')
df = pd.read_csv('soil_clean.csv', parse_dates=['ts'])
F=['temp','moisture','ec','ph','n','p','k','salinity']
bad = (df.temp>=45)
dfc = df[~bad].reset_index(drop=True)
print(f'rows used after removing {bad.sum()} implausible readings: {len(dfc)}')

def run(X, labels_true, title, ks=range(2,9), seed=0):
    print(f'\n=== {title} ===')
    print(' k   silhouette   DB-index   ARI(node)  NMI(node)   purity(node)')
    out={}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        lab = km.labels_
        sil = silhouette_score(X, lab, sample_size=3000, random_state=0)
        db  = davies_bouldin_score(X, lab)
        ari = adjusted_rand_score(labels_true, lab)
        nmi = normalized_mutual_info_score(labels_true, lab)
        ct = pd.crosstab(lab, labels_true)
        purity = ct.max(axis=1).sum()/len(lab)
        print(f'{k:2d}   {sil:9.3f}   {db:8.3f}   {ari:8.3f}   {nmi:8.3f}   {purity:8.3f}')
        out[k]=lab
    return out

X_all = StandardScaler().fit_transform(dfc[F])
res_all = run(X_all, dfc.node, 'A) All 8 features, standardized (the naive plan)')
for k in (2,5):
    print(f'\nContingency k={k} (rows=cluster, cols=node):')
    print(pd.crosstab(res_all[k], dfc.node))

Fv=['temp','moisture','ph']
X_v = StandardScaler().fit_transform(dfc[Fv])
res_v = run(X_v, dfc.node, 'B) Only time-varying features (temp, moisture, pH)')
print('\nContingency k=3:'); print(pd.crosstab(res_v[3], dfc.node))

Xc = dfc[F].copy()
for f in F: Xc[f] = dfc.groupby('node')[f].transform(lambda s: s - s.mean())
X_c = StandardScaler().fit_transform(Xc)
res_c = run(X_c, dfc.node, 'C) Within-node centered (location removed), all 8 features')
print('\nContingency k=3:'); print(pd.crosstab(res_c[3], dfc.node))

X_o = StandardScaler().fit_transform(df[F])
run(X_o, df.node, 'D) Same as A but outliers kept', ks=(2,3,4,5))

dfc['cl_all_k5']=res_all[5]; dfc['cl_all_k2']=res_all[2]; dfc['cl_var_k3']=res_v[3]; dfc['cl_cen_k3']=res_c[3]
dfc.to_csv('soil_labeled.csv', index=False)
np.save('X_all.npy', X_all)
