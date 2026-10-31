import pandas as pd, numpy as np, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def flatten_png(name):
    im = Image.open(name)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1]); bg.save(name)

warnings.filterwarnings('ignore')
SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK2,
    'xtick.color':INK2,'ytick.color':INK2,'axes.titlecolor':INK,'figure.facecolor':SURF,'axes.facecolor':SURF,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6})
K=4
order=['Saturated','Moist / cooler','Drier / higher pH','Drier / lower pH']
COL={'Saturated':'#e34948','Draining':'#eda100','Moist / cooler':'#2a78d6','Drier / higher pH':'#4a3aa7','Drier / lower pH':'#1baf7a'}
d = pd.read_csv(f'soil_states_k{K}.csv', parse_dates=['ts'])
held = pd.read_pickle(f'lono_labels_seed0_k{K}.pkl')
FEAT=['moisture_rel','ph_rel','temp_rel','dmoist_1h','dmoist_3h']; X=d[FEAT].clip(-5,5).values
nodes=sorted(d.node.unique())

fig = plt.figure(figsize=(15,9.2))
gs = fig.add_gridspec(2,5, height_ratios=[1,1.35], hspace=0.55, wspace=0.35, left=0.10, right=0.985, top=0.88, bottom=0.08)

for i,n in enumerate(nodes):
    ax = fig.add_subplot(gs[0,i])
    te = d.node==n
    ref = KMeans(K,n_init=30,random_state=0).fit(X)
    prof = pd.DataFrame(ref.cluster_centers_, columns=FEAT)
    names={prof.moisture_rel.idxmax():'Saturated'}
    rest=sorted([c for c in range(K) if c not in names], key=lambda c: prof.loc[c,'moisture_rel']); lo,mid,hi=rest
    if prof.loc[lo,'ph_rel']>prof.loc[mid,'ph_rel']: names[lo]='Drier / higher pH'; names[mid]='Drier / lower pH'
    else: names[mid]='Drier / higher pH'; names[lo]='Drier / lower pH'
    names[hi]='Moist / cooler'
    m = KMeans(K,n_init=30,random_state=0).fit(X[~te])
    C=((m.cluster_centers_[:,None,:]-ref.cluster_centers_[None,:,:])**2).sum(-1); r,c=linear_sum_assignment(C); mp={ri:names[ci] for ri,ci in zip(r,c)}
    tr_share = pd.Series([mp[c] for c in m.labels_]).value_counts(normalize=True).reindex(order).fillna(0)
    te_share = pd.Series(held[n]).value_counts(normalize=True).reindex(order).fillna(0)
    y=np.arange(len(order)); h=0.38
    ax.barh(y-h/2, tr_share.values, h, color=[COL[s] for s in order], alpha=0.45, label='4 training nodes')
    ax.barh(y+h/2, te_share.values, h, color=[COL[s] for s in order], label=f"{n.replace('-', ' ')} (never seen)")
    for yy,(a,b) in enumerate(zip(tr_share.values, te_share.values)):
        ax.text(max(a,b)+0.01, yy, f'{a:.0%} → {b:.0%}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(order if i==0 else ['']*len(order), fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0,0.78); ax.set_xticks([0,.2,.4,.6]); ax.set_xticklabels(['0','20%','40%','60%'], fontsize=8)
    ax.set_title(f"Held out: {n.replace('-', ' ')}", loc='left', fontsize=10.5)
    ax.grid(axis='y', visible=False)
fig.text(0.10,0.925,'How the states transfer to a sensor the model never saw   (light bar: share on the 4 training nodes; solid bar: share on the held out node)', fontsize=11, color=INK)

for j,n in enumerate(['Node-01','Node-03']):
    ax = fig.add_subplot(gs[1, 0:2] if j==0 else gs[1, 2:5])
    sub = d[d.node==n].copy(); sub['state']=held[n]
    sub['day']=(sub.ts.dt.normalize()-sub.ts.dt.normalize().min()).dt.days; sub['hour']=sub.ts.dt.hour
    grid = np.full((24, sub.day.max()+1), -1)
    for _,r in sub.iterrows(): grid[r.hour, r.day]=order.index(r.state)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['#ffffff00']+[COL[s] for s in order]); norm=BoundaryNorm(np.arange(-1.5,len(order)+0.5,1), cmap.N)
    ax.imshow(grid, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest', origin='upper')
    ax.set_yticks([0,6,12,18,23]); ax.set_yticklabels(['00:00','06:00','12:00','18:00','23:00'], fontsize=8.5)
    days = pd.date_range(sub.ts.dt.normalize().min(), periods=sub.day.max()+1)
    tk = list(range(0, len(days), 4)); ax.set_xticks(tk); ax.set_xticklabels([days[t].strftime('%b %d') for t in tk], fontsize=8.5)
    ax.grid(False); ax.set_title(f"{n.replace('-', ' ')}: each hour of the month, labelled by a model that never saw it", loc='left', fontsize=10)
    ax.axhspan(6.5, 10.5, color='none', ec=INK, lw=1, ls='--', alpha=0.6)
    ax.text(0.3, 6.2, '07:00 to 11:00', fontsize=8, color=INK, va='bottom')
fig.legend(handles=[Patch(color=COL[s], label=s) for s in order], loc='lower center', ncol=len(order), frameon=False, fontsize=9, bbox_to_anchor=(0.5,0.0))
fig.text(0.10,0.47,'Where each state occurs in the day  (white: missing hour; dashed box: 07:00 to 11:00)', fontsize=11, color=INK)
fig.savefig('fig3_leave_one_node_out.png', dpi=160); print('saved'); flatten_png('fig3_leave_one_node_out.png')
