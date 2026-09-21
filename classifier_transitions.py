import pandas as pd, numpy as np, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import json
warnings.filterwarnings('ignore')
_c = json.load(open('best_configs.json'))['random_forest']
RF_PARAMS = {**_c['fixed'], **_c['tuned'], 'n_jobs': -1}
print('Random Forest configuration (from tune_models.py):', RF_PARAMS)
d = pd.read_csv('soil_states_k4.csv', parse_dates=['ts']).sort_values(['node','ts']).reset_index(drop=True)
FEAT=['moisture_rel','ph_rel','temp_rel','dmoist_1h','dmoist_3h']
X = d[FEAT].clip(-5,5).values; y = d['state_ref'].values
order=['Saturated','Moist / cooler','Drier / higher pH','Drier / lower pH']
print('State counts (all nodes):'); print(d.state_ref.value_counts().reindex(order).to_string())

split = pd.Timestamp('2025-11-06')
tr = d.ts < split; te = ~tr
km = KMeans(4, n_init=30, random_state=0).fit(X[tr])
prof = pd.DataFrame(km.cluster_centers_, columns=FEAT)
names={prof.moisture_rel.idxmax():'Saturated'}
rest=sorted([c for c in range(4) if c not in names], key=lambda c: prof.loc[c,'moisture_rel']); lo,mid,hi=rest
if prof.loc[lo,'ph_rel']>prof.loc[mid,'ph_rel']: names[lo]='Drier / higher pH'; names[mid]='Drier / lower pH'
else: names[mid]='Drier / higher pH'; names[lo]='Drier / lower pH'
names[hi]='Moist / cooler'
y_tr = np.array([names[c] for c in km.labels_]); y_te = np.array([names[c] for c in km.predict(X[te])])
rf = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y_tr)
p = rf.predict(X[te])
print(f'\n=== RF state classifier, TIME split: train 13 Oct to 5 Nov ({tr.sum()} rows), test 6 to 13 Nov ({te.sum()} rows) ===')
print(f'accuracy {accuracy_score(y_te,p):.3f}   macro-F1 {f1_score(y_te,p,average="macro"):.3f}')
print(pd.DataFrame(confusion_matrix(y_te,p,labels=order), index=[f'true {o}' for o in order], columns=[o.split(' /')[0] for o in order]).to_string())
print(classification_report(y_te,p,labels=order,digits=2))
maj = pd.Series(y_tr).mode()[0]; print(f'majority-class baseline accuracy: {np.mean(y_te==maj):.3f}  ({maj})')
print('feature importance:', dict(zip(FEAT, rf.feature_importances_.round(3))))

print('\n=== RF state classifier, LEAVE-ONE-NODE-OUT (cluster + RF fit on 4 nodes, tested on the 5th) ===')
rows=[]
for n in sorted(d.node.unique()):
    trn = d.node!=n; ten = ~trn
    km = KMeans(4, n_init=30, random_state=0).fit(X[trn]); prof = pd.DataFrame(km.cluster_centers_, columns=FEAT)
    names={prof.moisture_rel.idxmax():'Saturated'}
    rest=sorted([c for c in range(4) if c not in names], key=lambda c: prof.loc[c,'moisture_rel']); lo,mid,hi=rest
    if prof.loc[lo,'ph_rel']>prof.loc[mid,'ph_rel']: names[lo]='Drier / higher pH'; names[mid]='Drier / lower pH'
    else: names[mid]='Drier / higher pH'; names[lo]='Drier / lower pH'
    names[hi]='Moist / cooler'
    ytr = np.array([names[c] for c in km.labels_]); yte = np.array([names[c] for c in km.predict(X[ten])])
    rf = RandomForestClassifier(**RF_PARAMS).fit(X[trn], ytr)
    pp = rf.predict(X[ten]); rows.append(dict(held_out=n, n=int(ten.sum()), accuracy=round(accuracy_score(yte,pp),3), macro_F1=round(f1_score(yte,pp,average='macro'),3)))
print(pd.DataFrame(rows).to_string(index=False))

print('\n=== State transitions (all nodes, consecutive hours only) ===')
T = pd.DataFrame(0, index=order, columns=order, dtype=float); dwell={o:[] for o in order}
for n,g in d.groupby('node'):
    g=g.sort_values('ts'); s=g.state_ref.values; t=g.ts.values
    run=1
    for i in range(1,len(g)):
        gap=(t[i]-t[i-1])/np.timedelta64(1,'h')
        if 0.75<=gap<=1.25:
            T.loc[s[i-1], s[i]] += 1
            if s[i]==s[i-1]: run+=1
            else: dwell[s[i-1]].append(run); run=1
        else: dwell[s[i-1]].append(run); run=1
P = T.div(T.sum(axis=1), axis=0)
print('Transition probabilities (row = state now, column = state next hour):')
print(P.round(2).rename(columns=lambda c:c.split(' /')[0]).to_string())
print('\nMean dwell time (hours in a row before changing):')
for o in order: print(f'  {o:20s} {np.mean(dwell[o]):5.1f} h   (n runs {len(dwell[o])})')
sat_next = T.loc['Saturated'].drop('Saturated'); print('\nAfter a Saturated hour ends, the next state is:', (sat_next/sat_next.sum()).round(2).to_dict())
rec=[]
for n,g in d.groupby('node'):
    g=g.sort_values('ts').reset_index(drop=True)
    for i in range(len(g)):
        if g.state_ref[i]=='Saturated' and (i==0 or g.state_ref[i-1]!='Saturated'):
            for j in range(i+1, min(i+13,len(g))):
                if g.moisture_rel[j] < 0.5: rec.append(j-i); break
print(f'Recovery after a watering (hours until moisture is back within that node\'s normal range): median {np.median(rec):.0f} h, mean {np.mean(rec):.1f} h, n={len(rec)} events')
