import pandas as pd, numpy as np, warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
warnings.filterwarnings('ignore')
d = pd.read_csv('soil_features.csv', parse_dates=['ts'])
FEAT=['moisture_rel','ph_rel','temp_rel','dmoist_1h','dmoist_3h']
X = d[FEAT].clip(-5,5).values
print(' k  silhouette  NMI-vs-node  sizes')
labs={}
for k in (3,4,5,6):
    lab = KMeans(k, n_init=30, random_state=0).fit_predict(X); labs[k]=lab
    print(f'{k:2d}  {silhouette_score(X,lab,sample_size=3000,random_state=0):8.3f}   {normalized_mutual_info_score(d.node,lab):8.3f}   {np.bincount(lab).tolist()}')
for k in (3,4,5):
    print(f'\n--- k={k}: cluster profiles (mean of each feature) and peak hours ---')
    prof = pd.DataFrame(X, columns=FEAT).groupby(labs[k]).mean().round(2)
    prof['n']=np.bincount(labs[k])
    hrs = pd.crosstab(labs[k], d.hour, normalize='index')
    prof['peak_hours'] = [', '.join(str(h) for h in hrs.loc[c].nlargest(3).index) for c in prof.index]
    prof['share_7-10am'] = hrs.loc[:, 7:10].sum(axis=1).round(2).values
    print(prof.to_string())
    print('node mix (share of each node in each cluster):'); print(pd.crosstab(labs[k], d.node, normalize='columns').round(2).to_string())
