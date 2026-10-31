import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from common import load, state_labels, present, ORDER, TEST_START
warnings.filterwarnings('ignore')
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.titlesize': 10.5, 'axes.titleweight': 'normal', 'legend.frameon': False})
COL = {'Saturated': '#e34948', 'Rapid drainage': '#eda100', 'Moist / cooler': '#2a78d6', 'Drier': '#4a3aa7',
       'Drier / higher pH': '#4a3aa7', 'Drier / lower pH': '#1baf7a'}
MCOL = {'random_forest': '#2a78d6', 'logistic_regression': '#e34948', 'decision_tree': '#eda100', 'majority': '#8a8985'}
MNAME = {'random_forest': 'Random Forest', 'logistic_regression': 'Logistic regression',
         'decision_tree': 'Classification tree', 'majority': 'Majority class'}
SHOW = lambda n: n.replace('-', ' ')
TRAIN_LS, VAL_LS = dict(ls='--', lw=1.6), dict(ls='-', lw=2.0)
d, X = load(); N = json.load(open('ch4_numbers.json')); cfg = json.load(open('best_configs.json'))
def save(fig, name):
    fig.savefig(name, dpi=200); plt.close(fig)
    im = Image.open(name)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1]); bg.save(name)
    print('saved', name)
def mark(ax, x, y): ax.plot([x], [y], 'o', ms=9, mfc='none', mec=INK, mew=1.4, zorder=5)

day = d.assign(day=d.ts.dt.normalize()).groupby(['day', 'state_ref']).size().unstack().reindex(columns=ORDER).fillna(0)
share = day.div(day.sum(1), axis=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.22)
bottom = np.zeros(len(share))
for s in ORDER:
    ax[0].bar(share.index, share[s], bottom=bottom, width=0.9, color=COL[s], label=s, lw=0); bottom += share[s].values
ax[0].axvspan(TEST_START - pd.Timedelta(hours=12), share.index.max() + pd.Timedelta(hours=12), color=INK, alpha=0.06, lw=0)
ax[0].text(TEST_START + pd.Timedelta(days=3.5), 1.03, 'test week', ha='center', fontsize=9, color=INK2)
ax[0].set_ylim(0, 1.08); ax[0].set_yticks([0, .25, .5, .75, 1]); ax[0].set_yticklabels(['0', '25%', '50%', '75%', '100%'])
ax[0].set_ylabel('share of the day\'s readings'); ax[0].grid(axis='x', visible=False)
ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b'))
ax[0].set_title('(a) State composition of each day, all five nodes', loc='left')
hr = d.groupby([d.ts.dt.hour, 'state_ref']).size().unstack().reindex(columns=ORDER).fillna(0)
hr = hr / hr.sum(0)
for s in ORDER: ax[1].plot(hr.index, hr[s] * 100, color=COL[s], lw=2 if s == 'Saturated' else 1.4, marker='o', ms=3)
ax[1].axvspan(6.5, 10.5, color=INK, alpha=0.06, lw=0); ax[1].text(11.0, 27, '07:00 to 11:00', ha='left', va='center', fontsize=8.5, color=INK2)
ax[1].set_xticks([0, 6, 12, 18, 23]); ax[1].set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'])
ax[1].set_ylabel('percent of the state\'s readings'); ax[1].set_title('(b) Hour of day at which each state occurs', loc='left')
fig.legend(handles=[Patch(color=COL[s], label=s) for s in ORDER], loc='lower center', ncol=4, fontsize=9)
save(fig, 'fig4_1_states_time.png')

sh = pd.read_csv('lono_shares_seed0_k4.csv')
fig, ax = plt.subplots(figsize=(6.4, 5.2)); fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.21)
ax.plot([0, .62], [0, .62], color=INK2, lw=0.8, ls=':')
MK = {'Node-01': 'o', 'Node-02': 's', 'Node-03': '^', 'Node-04': 'D', 'Node-05': 'v'}
for _, r in sh.iterrows():
    ax.scatter(r.share_heldout_reference, r.share_heldout, s=70, marker=MK[r.held_out], color=COL[r.state],
               edgecolor='white', lw=0.8, zorder=3)
for _, r in sh[(sh.held_out == 'Node-05') & sh.state.isin(['Drier / lower pH', 'Drier / higher pH'])].iterrows():
    ax.annotate(f'{SHOW(r.held_out)}, {r.state}', (r.share_heldout_reference, r.share_heldout),
                xytext=(10, -4 if r.state.endswith('lower pH') else 8), textcoords='offset points', fontsize=8.5, color=INK2)
ax.set_xlim(-0.02, 0.62); ax.set_ylim(-0.02, 0.62)
ax.set_xlabel('share on the withheld node, reference model (all five nodes)')
ax.set_ylabel('share on the withheld node, model fitted without it')
ax.set_title('Each state on each withheld node', loc='left')
h1 = [Patch(color=COL[s], label=s) for s in ORDER]
h2 = [plt.Line2D([], [], marker=MK[n], ls='', color=INK2, label=SHOW(n)) for n in MK]
fig.legend(handles=h1 + h2, loc='lower center', ncol=5, fontsize=8.2)
save(fig, 'fig4_2_transfer.png')

ch = cfg['random_forest']['tuned']
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2)); fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.12, hspace=0.45, wspace=0.28)
def vc(a, t, x, xl, chosen, title, xt=None, logx=False):
    t = t.reset_index(drop=True)
    if logx:
        xs = t[x].values; a.set_xscale('log'); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); cx = chosen
    else:
        xs = np.arange(len(t)); cx = int(np.where(t[x].values == chosen)[0][0])
        a.set_xticks(xs); a.set_xticklabels(xt if xt is not None else [str(v) for v in t[x]])
    a.plot(xs, t.train_macro_f1, color=INK2, **TRAIN_LS, marker='o', ms=4)
    a.plot(xs, t.val_macro_f1, color=MCOL['random_forest'], **VAL_LS, marker='o', ms=4)
    mark(a, cx, t.val_macro_f1[int(np.where(t[x].values == chosen)[0][0])])
    a.set_xlabel(xl); a.set_title(title, loc='left'); a.set_ylabel('macro F1')
t = pd.DataFrame(N['valcurve_min_samples_leaf']); vc(ax[0, 0], t, 'min_samples_leaf', 'minimum leaf size', ch['min_samples_leaf'], '(a) Forest: minimum leaf size')
t = pd.DataFrame(N['valcurve_max_features']); vc(ax[0, 1], t, 'max_features', 'features considered at each split', ch['max_features'], '(b) Forest: features per split')
t = pd.DataFrame(N['valcurve_max_depth']); t['x'] = t.max_depth.replace(0, 32)
t2 = t.rename(columns={'x': 'depthx'}).sort_values('depthx'); vc(ax[0, 2], t2, 'depthx', 'maximum depth', ch['max_depth'] or 32, '(c) Forest: maximum depth', xt=['4', '8', '16', 'none'])
dt = pd.DataFrame(N['dt_grid']); dch = cfg['decision_tree']['tuned']
t = dt[dt.min_samples_leaf == dch['min_samples_leaf']].copy(); t['x'] = t.max_depth.replace(0, 32); t = t.sort_values('x')
vc(ax[1, 0], t, 'x', 'maximum depth', dch['max_depth'] or 32, '(d) Single tree: maximum depth', xt=['2', '4', '6', '8', '12', 'none'])
for l in ax[1, 0].get_lines()[1:2]: l.set_color(MCOL['decision_tree'])
lr = pd.DataFrame(N['lr_grid']).sort_values('C')
vc(ax[1, 1], lr, 'C', 'inverse regularisation strength C', cfg['logistic_regression']['tuned']['C'], '(e) Logistic regression: C', logx=True)
for l in ax[1, 1].get_lines()[1:2]: l.set_color(MCOL['logistic_regression'])
lc = pd.DataFrame(N['learning_curve'])
a = ax[1, 2]; a.plot(lc.n, lc.train, color=INK2, **TRAIN_LS, marker='o', ms=4)
a.plot(lc.n, lc.val, color=MCOL['random_forest'], **VAL_LS, marker='o', ms=4)
a.fill_between(lc.n, lc.val - lc.val_sd.fillna(0), lc.val + lc.val_sd.fillna(0), color=MCOL['random_forest'], alpha=0.15, lw=0)
a.set_xlabel('training readings used'); a.set_ylabel('macro F1'); a.set_title('(f) Forest: learning curve', loc='left')
for a in ax.flat: a.set_ylim(0.65, 1.01)
ax[1, 1].set_ylim(0.3, 1.01); ax[1, 0].set_ylim(0.3, 1.01)
fig.legend(handles=[plt.Line2D([], [], color=INK2, **TRAIN_LS, label='training readings'),
                    plt.Line2D([], [], color=INK2, **VAL_LS, label='validation window (30 October to 5 November)'),
                    plt.Line2D([], [], marker='o', ms=9, mfc='none', mec=INK, ls='', label='selected value')],
           loc='lower center', ncol=3, fontsize=9)
save(fig, 'fig4_3_tuning_curves.png')

lt, lu = pd.DataFrame(N['loss_curve']), pd.DataFrame(N['loss_curve_unconstrained'])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3)); fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.25, wspace=0.25)
U = '#8a8985'
for a, col, yl in [(ax[0], 'log_loss', 'log loss'), (ax[1], 'macro_f1', 'macro F1')]:
    a.plot(lt.trees, lt['train_' + col], color=MCOL['random_forest'], **TRAIN_LS)
    a.plot(lt.trees, lt['val_' + col], color=MCOL['random_forest'], **VAL_LS)
    a.plot(lu.trees, lu['train_' + col], color=U, **TRAIN_LS)
    a.plot(lu.trees, lu['val_' + col], color=U, **VAL_LS)
    a.set_xscale('log'); a.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 400]); a.set_xticklabels(['1', '2', '5', '10', '20', '50', '100', '200', '400'])
    a.set_xlabel('number of trees'); a.set_ylabel(yl)
ax[0].set_ylim(0, 1.0); ax[0].set_title('(a) Log loss', loc='left')
ax[0].text(1.15, 0.93, f"validation loss with one tree: {lt.val_log_loss[0]:.2f} (selected), {lu.val_log_loss[0]:.2f} (unconstrained)", fontsize=8.5, color=INK2); ax[1].set_ylim(0.7, 1.01); ax[1].set_title('(b) Macro F1', loc='left')
fig.legend(handles=[plt.Line2D([], [], color=MCOL['random_forest'], lw=2, label='selected forest (leaf 2, depth 8, 1 feature per split)'),
                    plt.Line2D([], [], color=U, lw=2, label='unconstrained forest (leaf 1, no depth limit, all 5 features)'),
                    plt.Line2D([], [], color=INK2, **TRAIN_LS, label='training readings'),
                    plt.Line2D([], [], color=INK2, **VAL_LS, label='validation window')],
           loc='lower center', ncol=2, fontsize=9)
save(fig, 'fig4_4_loss_curves.png')

tr = (d.ts < TEST_START).values; te = ~tr
ytr, yte = state_labels(X[tr], X[te]); L = present(ytr)
cm_rf = pd.read_csv('comparison_confusion_rf.csv', index_col=0).reindex(index=L, columns=L).values
lrm = LogisticRegression(**cfg['logistic_regression']['fixed'], **cfg['logistic_regression']['tuned']).fit(X[tr], ytr)
cm_lr = confusion_matrix(yte, lrm.predict(X[te]), labels=L)
pd.DataFrame(cm_lr, index=L, columns=L).to_csv('ch4_confusion_lr.csv')
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6)); fig.subplots_adjust(left=0.12, right=0.99, top=0.9, bottom=0.2, wspace=0.42)
short = [s.replace(' / ', ' and ').replace('Rapid drainage', 'Rapid\ndrainage').replace('Moist and cooler', 'Moist and\ncooler') for s in L]
for a, cm, title in [(ax[0], cm_rf, '(a) Random Forest'), (ax[1], cm_lr, '(b) Logistic regression')]:
    rown = cm / cm.sum(1, keepdims=True)
    a.imshow(rown, cmap='Blues', vmin=0, vmax=1); a.grid(False)
    for i in range(len(L)):
        for j in range(len(L)):
            a.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=10, color='white' if rown[i, j] > 0.6 else INK)
    a.set_xticks(range(len(L))); a.set_xticklabels(short, fontsize=8.5); a.set_yticks(range(len(L))); a.set_yticklabels(short, fontsize=8.5)
    a.set_xlabel('predicted state'); a.set_ylabel('state assigned by the clustering'); a.set_title(title, loc='left')
save(fig, 'fig4_5_confusion.png')

lf = pd.read_csv('comparison_lono_folds.csv'); models = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']
fig, ax = plt.subplots(figsize=(7.2, 4.4)); fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.2)
data = [lf[lf.model == m].sort_values('held_out').macro_f1.values for m in models]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
for p, m in zip(bp['boxes'], models): p.set_facecolor(MCOL[m]); p.set_alpha(0.25); p.set_edgecolor(MCOL[m])
nodes = sorted(lf.held_out.unique()); MK = {'Node-01': 'o', 'Node-02': 's', 'Node-03': '^', 'Node-04': 'D', 'Node-05': 'v'}
for i, m in enumerate(models):
    sub = lf[lf.model == m].set_index('held_out')
    for j, n in enumerate(nodes):
        ax.scatter(i + 1 + (j - 2) * 0.07, sub.loc[n, 'macro_f1'], marker=MK[n], s=34, color=MCOL[m], edgecolor='white', lw=0.6, zorder=3)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([MNAME[m] for m in models]); ax.set_ylabel('macro F1 on the withheld node')
ax.set_ylim(0, 1.05); ax.set_title('Leave one node out, five folds', loc='left')
fig.legend(handles=[plt.Line2D([], [], marker=MK[n], ls='', color=INK2, label=SHOW(n)) for n in nodes], loc='lower center', ncol=5, fontsize=9)
save(fig, 'fig4_6_lono_boxplot.png')

sd = pd.read_csv('ch4_seeds.csv')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.16, wspace=0.28)
models = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']
data = [sd[f'{m}_macro_f1'].values for m in models]
bp = ax[0].boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
for p_, m in zip(bp['boxes'], models): p_.set_facecolor(MCOL[m]); p_.set_alpha(0.25); p_.set_edgecolor(MCOL[m])
rng = np.random.default_rng(0)
for i, m in enumerate(models):
    ax[0].scatter(i + 1 + rng.uniform(-0.12, 0.12, len(sd)), sd[f'{m}_macro_f1'], s=22, color=MCOL[m], edgecolor='white', lw=0.5, zorder=3)
ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels([MNAME[m] for m in models], fontsize=9)
ax[0].set_ylabel('macro F1 on the test week'); ax[0].set_ylim(0, 1.05)
ax[0].set_title('(a) Ten random runs, states and models refitted each time', loc='left')
ax[1].plot(sd.seed, sd.ari_with_seed0, 'o-', color=INK, lw=1.6, ms=5)
ax[1].set_ylim(0.9, 1.005); ax[1].set_xticks(sd.seed); ax[1].set_xlabel('random seed')
ax[1].set_ylabel('agreement with the seed 0 states')
ax[1].set_title("(b) Agreement of each run's states with the first run", loc='left')
save(fig, 'fig4_8_seeds.png')

fr = pd.read_csv('forecast_results.csv')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2)); fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.24, wspace=0.25)
for a, tgt, yl, title in [(ax[0], 'moist', 'mean absolute error (moisture percent)', '(a) Moisture'),
                          (ax[1], 'ph', 'mean absolute error (pH units)', '(b) pH')]:
    t = fr[fr.target == tgt].sort_values('horizon')
    a.plot(t.horizon, t.rf, color=MCOL['random_forest'], lw=2, marker='o', ms=5)
    a.plot(t.horizon, t.naive, color=INK, lw=2, marker='s', ms=5)
    a.plot(t.horizon, t.daily, color=U, lw=1.6, ls='--', marker='^', ms=5)
    a.set_xticks([1, 3, 6, 12, 24]); a.set_xlabel('forecast horizon (hours)'); a.set_ylabel(yl); a.set_title(title, loc='left')
    a.set_ylim(0, None)
fig.legend(handles=[plt.Line2D([], [], color=MCOL['random_forest'], lw=2, marker='o', label='Random Forest regressor'),
                    plt.Line2D([], [], color=INK, lw=2, marker='s', label='persistence (no change)'),
                    plt.Line2D([], [], color=U, lw=1.6, ls='--', marker='^', label='same hour on the previous day')],
           loc='lower center', ncol=3, fontsize=9)
save(fig, 'fig4_7_forecast.png')
