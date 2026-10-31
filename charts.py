import pandas as pd, numpy as np, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

def flatten_png(name):
    im = Image.open(name)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1]); bg.save(name)

SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e1'
NODE_COL = {'Node-01':'#2a78d6','Node-02':'#e34948','Node-03':'#1baf7a','Node-04':'#4a3aa7','Node-05':'#eda100'}
SHOW = lambda n: n.replace('-', ' ')
NODE_MK  = {'Node-01':'o','Node-02':'s','Node-03':'^','Node-04':'D','Node-05':'v'}
CL_COL   = ['#2a78d6','#e34948','#1baf7a','#4a3aa7','#eda100']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK2,
    'xtick.color':INK2,'ytick.color':INK2,'axes.titlecolor':INK,'figure.facecolor':SURF,'axes.facecolor':SURF,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6})

d = pd.read_csv('soil_labeled.csv', parse_dates=['ts'])
F=['temp','moisture','ec','ph','n','p','k','salinity']
X = StandardScaler().fit_transform(d[F])
pc = PCA(2, random_state=0).fit(X); P = pc.transform(X)
ev = pc.explained_variance_ratio_*100

fig, ax = plt.subplots(1,3, figsize=(15,4.8))
fig.subplots_adjust(wspace=0.32, left=0.05, right=0.98, top=0.86, bottom=0.14)

rng = np.random.default_rng(0); idx = rng.permutation(len(d))
for n,c in NODE_COL.items():
    m = (d.node==n).values
    ax[0].scatter(P[m,0],P[m,1], s=10, c=c, marker=NODE_MK[n], alpha=0.55, linewidths=0, label=SHOW(n))
ax[0].set_title('Coloured by sensor node (where the row came from)', loc='left', fontsize=11)
for k in range(5):
    m = (d.cl_all_k5==k).values
    ax[1].scatter(P[m,0],P[m,1], s=10, c=CL_COL[k], alpha=0.55, linewidths=0, label=f'Cluster {k}')
ax[1].set_title('Coloured by k means cluster (k = 5, all 8 features)', loc='left', fontsize=11)
xlo,xhi = np.percentile(P[:,0],[0.5,99.5]); ylo,yhi = np.percentile(P[:,1],[0.5,99.5])
off = ((P[:,0]<xlo-1)|(P[:,0]>xhi+1)|(P[:,1]<ylo-1)|(P[:,1]>yhi+1)).sum()
for a in ax[:2]:
    a.set_xlim(xlo-1, xhi+1); a.set_ylim(ylo-1, yhi+1)
    a.set_xlabel(f'PC1 ({ev[0]:.0f}% of variation)'); a.set_ylabel(f'PC2 ({ev[1]:.0f}%)')
    a.legend(frameon=False, fontsize=8.5, markerscale=2.2, loc='lower left', labelcolor=INK2)
    a.text(0.99,0.02,f'{off} extreme rows beyond the axes', transform=a.transAxes, ha='right', fontsize=8, color=INK2)
def sil_curve(Xs):
    Xs = np.clip(Xs, -5, 5)
    return [silhouette_score(Xs, KMeans(k,n_init=20,random_state=0).fit_predict(Xs), sample_size=3000, random_state=0) for k in range(2,9)]
Xc = d[F].copy()
for f in F: Xc[f] = d.groupby('node')[f].transform(lambda s: s-s.mean())
lab3 = KMeans(3,n_init=20,random_state=0).fit_predict(StandardScaler().fit_transform(Xc))
keep = np.bincount(lab3)[lab3]>20
curves = {
 'All 8 features (as planned)': sil_curve(X),
 'Only temp, moisture, pH':     sil_curve(StandardScaler().fit_transform(d[['temp','moisture','ph']])),
 'Location removed (within node)': sil_curve(StandardScaler().fit_transform(Xc[keep])),
}
ks=list(range(2,9))
for (name,ys),c in zip(curves.items(), ['#2a78d6','#e34948','#1baf7a']):
    ax[2].plot(ks, ys, color=c, lw=2, marker='o', ms=5, label=name)
ax[2].set_xlabel('number of clusters (k)'); ax[2].set_ylabel('silhouette score (higher = cleaner clusters)')
ax[2].set_title('Cluster quality vs k', loc='left', fontsize=11); ax[2].set_ylim(0,0.7)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='lower left')
fig.suptitle(f'Do the clusters reflect soil states, or just which sensor the row came from?  ({len(d):,} hourly rows, 5 nodes, 13 October to 13 November 2025)', x=0.05, ha='left', fontsize=12, color=INK)
fig.savefig('fig1_clusters_vs_nodes.png', dpi=160); flatten_png('fig1_clusters_vs_nodes.png')

dd = d.set_index('ts').groupby('node')[F].resample('D').mean().reset_index()
fig, ax = plt.subplots(2,2, figsize=(14,7), sharex=True)
fig.subplots_adjust(hspace=0.35, wspace=0.22, left=0.06, right=0.98, top=0.88, bottom=0.08)
panels = [('moisture','Moisture (calibrated %)'),('ph','pH'),('k','Potassium K (calibrated)'),('n','Nitrogen N (calibrated)')]
ends = []
for a,(f,t) in zip(ax.ravel(), panels):
    for n,c in NODE_COL.items():
        s = dd[dd.node==n]
        a.plot(s.ts, s[f], color=c, lw=1.8, label=SHOW(n))
        if f=='moisture': ends.append((SHOW(n), s.ts.iloc[-1], s[f].iloc[-1]))
    a.set_title(t, loc='left', fontsize=11)
    a.tick_params(axis='x', labelrotation=0, labelsize=8.5)
ends.sort(key=lambda e: e[2]); gap = 2.6; placed = []
for name, x, y in ends:
    yy = max(y, placed[-1] + gap) if placed else y
    placed.append(yy)
    ax[0,0].annotate(name, (x, y), xytext=(x, yy), textcoords='data', fontsize=8, color=INK2,
                     va='center', ha='left', annotation_clip=False,
                     xycoords='data').set_position((x + pd.Timedelta(hours=10), yy))
ax[0,0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=5, loc='upper left', bbox_to_anchor=(0,1.0))
import matplotlib.dates as mdates

for a in ax[1]: a.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.suptitle('Daily averages per node: moisture and pH move over the month, while N and K barely move except on one node', x=0.06, ha='left', fontsize=12, color=INK)
fig.savefig('fig2_what_changes_over_time.png', dpi=160); flatten_png('fig2_what_changes_over_time.png')
print('saved')
