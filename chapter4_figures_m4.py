import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
warnings.filterwarnings('ignore')
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.titlesize': 10.5, 'axes.titleweight': 'normal', 'legend.frameon': False})
MCOL = {'random_forest': '#2a78d6', 'logistic_regression': '#e34948', 'decision_tree': '#eda100', 'majority': '#8a8985'}
MNAME = {'random_forest': 'Random Forest', 'logistic_regression': 'Logistic regression',
         'decision_tree': 'Classification tree', 'majority': 'Majority class'}
BLUE, RED, GOLD, GREEN, VIOLET = '#2a78d6', '#e34948', '#eda100', '#1baf7a', '#4a3aa7'

def save(fig, name):
    fig.savefig(name, dpi=200); plt.close(fig)
    im = Image.open(name)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1]); bg.save(name)
    print('saved', name)

imp = pd.read_csv('ch4_importance.csv')
coef = pd.read_csv('ch4_lr_coefficients.csv', index_col=0)
abl_c = pd.read_csv('ch4_ablation_classifier.csv')
abl_k = pd.read_csv('ch4_ablation_clustering.csv')
scal = pd.read_csv('ch4_scaling_readings.csv')
snod = pd.read_csv('ch4_scaling_nodes.csv')
ssta = pd.read_csv('ch4_scaling_states.csv')
bud = pd.read_csv('ch4_quality_budget.csv')
Q = json.load(open('ch4_quality.json'))
tests = json.load(open('comparison_tests.json'))
chrono = pd.read_csv('comparison_chronological.csv').set_index('model')
lsum = pd.read_csv('comparison_lono_summary.csv').set_index('model')
tim = pd.read_csv('ch4_timing.csv')

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.14, wspace=0.55)
y = np.arange(len(imp))[::-1]
h = 0.26
ax[0].barh(y + h, imp.forest_impurity, h, xerr=imp.forest_impurity_sd, color=BLUE, label='forest, impurity',
           error_kw=dict(ecolor=INK2, lw=0.8))
ax[0].barh(y, imp.forest_permutation, h, xerr=imp.forest_permutation_sd, color=GREEN, label='forest, permutation',
           error_kw=dict(ecolor=INK2, lw=0.8))
ax[0].barh(y - h, imp.logistic_permutation, h, xerr=imp.logistic_permutation_sd, color=RED,
           label='logistic regression, permutation', error_kw=dict(ecolor=INK2, lw=0.8))
ax[0].set_yticks(y); ax[0].set_yticklabels(imp.feature)
ax[0].set_xlabel('importance (impurity share, or loss of macro F1 when permuted)')
ax[0].set_title('(a) How much each input contributes', loc='left')
ax[0].set_xlim(right=0.58)
ax[0].legend(fontsize=8.5, loc='center right')
ax[0].axvline(0, color=INK2, lw=0.8)

M = coef.values
lim = np.abs(M).max()
im_ = ax[1].imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax[1].set_xticks(range(M.shape[1])); ax[1].set_xticklabels(['moisture', 'pH', 'temperature', 'change, 1 h', 'change, 3 h'],
                                                           fontsize=8.5, rotation=30, ha='right')
ax[1].set_yticks(range(M.shape[0])); ax[1].set_yticklabels(coef.index, fontsize=9)
ax[1].grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax[1].text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8,
                   color='white' if abs(M[i, j]) > 0.6 * lim else INK)
ax[1].set_title('(b) Logistic regression coefficients', loc='left')
fig.colorbar(im_, ax=ax[1], shrink=0.85, pad=0.02)
save(fig, 'fig4_9_importance.png')

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.16, wspace=0.5)
rem = [r for r in abl_c.removed.unique() if r != 'none']
y = np.arange(len(rem))[::-1]
for i, m in enumerate(['random_forest', 'logistic_regression', 'decision_tree']):
    v = [abl_c[(abl_c.removed == r) & (abl_c.model == m)].test_macro_f1_change.iloc[0] for r in rem]
    ax[0].barh(y + (1 - i) * 0.26, v, 0.26, color=MCOL[m], label=MNAME[m])
ax[0].axvline(0, color=INK2, lw=0.9)
ax[0].set_yticks(y); ax[0].set_yticklabels(rem)
ax[0].set_xlabel('change in macro F1 on the test week when the input is withheld')
ax[0].set_title('(a) Removed from the classifier, states unchanged', loc='left')
ax[0].set_xlim(right=0.33)
ax[0].legend(fontsize=8.5, loc='center right')

k = abl_k[abl_k.removed != 'none']
y = np.arange(len(k))[::-1]
cols = [GREEN if s else RED for s in k.same_state_set]
ax[1].barh(y, k.ari_with_reference, 0.55, color=cols)
ax[1].set_yticks(y); ax[1].set_yticklabels(k.removed)
ax[1].set_xlim(0, 1.05); ax[1].set_xlabel('agreement with the reference states')
ax[1].set_title('(b) Removed from the clustering as well', loc='left')
ax[1].legend(handles=[plt.Line2D([], [], color=GREEN, lw=6, label='same four states form'),
                      plt.Line2D([], [], color=RED, lw=6, label='a different state set forms')],
             fontsize=8.5, loc='lower right')
save(fig, 'fig4_10_ablation.png')

fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.21, wspace=0.3)
real = scal[scal.source == 'record']; syn = scal[scal.source != 'record']
for col, c, lab in [('kmeans_seconds', BLUE, 'identify the states'),
                    ('forest_fit_seconds', RED, 'fit the forest'),
                    ('logistic_fit_seconds', GOLD, 'fit logistic regression')]:
    ax[0].plot(scal.readings, scal[col], color=c, lw=1.6, zorder=2, label=lab)
    ax[0].plot(real.readings, real[col], 'o', color=c, ms=6, zorder=3)
    ax[0].plot(syn.readings, syn[col], 'o', mfc='none', mec=c, ms=6, zorder=3)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('readings'); ax[0].set_ylabel('seconds')
ax[0].set_title('(a) Cost against the size of the record', loc='left')
ax[0].legend(fontsize=8.5, loc='upper left')
ax[0].annotate('filled: the record\nopen: replicated readings', xy=(0.62, 0.08), xycoords='axes fraction',
               fontsize=8.5, color=INK2)

ax[1].plot(ssta.states, ssta.kmeans_seconds, color=BLUE, lw=1.8, marker='o', ms=5)
ax[1].set_xlabel('number of states'); ax[1].set_ylabel('seconds'); ax[1].set_ylim(0, None)
ax[1].set_title('(b) Cost against the number of states', loc='left')

ax[2].plot(snod.nodes, snod.kmeans_seconds, color=BLUE, lw=1.8, marker='o', ms=5, label='identify the states')
ax[2].plot(snod.nodes, snod.forest_fit_seconds, color=RED, lw=1.8, marker='s', ms=5, label='fit the forest')
ax[2].set_xticks(snod.nodes); ax[2].set_xlabel('number of nodes'); ax[2].set_ylabel('seconds'); ax[2].set_ylim(0, None)
ax[2].set_title('(c) Cost against the number of nodes', loc='left')
ax[2].legend(fontsize=8.5)
save(fig, 'fig4_11_scaling.png')

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1))
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.17, wspace=0.28)
ax[0].errorbar(bud.restarts, bud.objective_mean, yerr=bud.objective_sd, color=BLUE, lw=1.8, marker='o', ms=5,
               capsize=3, ecolor=INK2, elinewidth=0.9)
ax[0].axhline(Q['best_known'], color=INK, lw=1.2, ls='--')
ax[0].text(bud.restarts.max(), Q['best_known'], 'best known value, two thousand restarts ',
           fontsize=8.5, color=INK2, va='top', ha='right')
ax[0].set_xscale('log'); ax[0].set_xticks(bud.restarts); ax[0].set_xticklabels(bud.restarts)
ax[0].set_xlabel('restarts'); ax[0].set_ylabel('within cluster sum of squares')
ax[0].set_title('(a) What a restart budget buys', loc='left')

ax[1].plot(bud.seconds, bud.gap_to_best_known_mean_percent, color=BLUE, lw=1.8, marker='o', ms=5)
for _, r in bud.iterrows():
    ax[1].annotate(f"{int(r.restarts)}", (r.seconds, r.gap_to_best_known_mean_percent),
                   textcoords='offset points', xytext=(6, 6), fontsize=8.5, color=INK2)
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlim(bud.seconds.min() * 0.6, bud.seconds.max() * 2.2)
ax[1].set_xlabel('seconds'); ax[1].set_ylabel('gap to the best known value (percent)')
ax[1].set_title('(b) The same budget as cost against quality', loc='left')
save(fig, 'fig4_12_quality.png')

fr = tests['friedman']; R = fr['mean_ranks']; CD = fr['nemenyi_CD']
order = sorted(R, key=lambda m: R[m])
ranks = [R[m] for m in order]
lo, hi = 1, len(R)
fig, ax = plt.subplots(figsize=(9, 3.1)); fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
ax.set_xlim(lo - 1.9, hi + 1.9); ax.set_ylim(-1.35, 0.72); ax.axis('off')
ax.plot([lo, hi], [0, 0], color=INK, lw=1.4)
for t in range(lo, hi + 1):
    ax.plot([t, t], [0, 0.07], color=INK, lw=1.2)
    ax.text(t, 0.11, str(t), ha='center', fontsize=9, color=INK2)
ax.text((lo + hi) / 2, 0.62, 'mean rank over the five withheld nodes, 1 is best',
        ha='center', fontsize=9, color=INK2)
ax.plot([lo, lo + CD], [0.42, 0.42], color=INK2, lw=1.6)
for x in (lo, lo + CD):
    ax.plot([x, x], [0.39, 0.45], color=INK2, lw=1.2)
ax.text(lo + CD / 2, 0.48, f'critical difference {CD:.2f}', ha='center', fontsize=8.5, color=INK2)
half = (len(order) + 1) // 2
for i, m in enumerate(order):
    left = i < half
    y = -0.45 - 0.2 * (i if left else len(order) - 1 - i)
    x_end = lo - 0.55 if left else hi + 0.55
    ax.plot([R[m], R[m]], [0, y], color=MCOL[m], lw=1.4)
    ax.plot([R[m], x_end], [y, y], color=MCOL[m], lw=1.4)
    ax.text(x_end + (-0.08 if left else 0.08), y, f'{MNAME[m]} ({R[m]:.1f})',
            ha='right' if left else 'left', va='center', fontsize=9, color=INK)
cliques = []
for i in range(len(order)):
    j = max(j for j in range(len(order)) if ranks[j] - ranks[i] <= CD)
    if j > i and not any(a <= i and j <= b for a, b in cliques):
        cliques.append((i, j))
for n_, (i, j) in enumerate(cliques):
    yy = -0.1 - 0.1 * n_
    ax.plot([ranks[i] - 0.04, ranks[j] + 0.04], [yy, yy], color=INK, lw=3.6, solid_capstyle='butt')
ax.text((lo + hi) / 2, -1.27, 'a bar joins models the test cannot separate',
        ha='center', fontsize=8.5, color=INK2)
save(fig, 'fig4_13_critical_difference.png')

fig, ax = plt.subplots(figsize=(7.6, 4.3)); fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.16)
tmap = {'Random Forest': 'random_forest', 'Logistic regression': 'logistic_regression',
        'Classification tree': 'decision_tree'}
for _, r in tim.iterrows():
    if r.model not in tmap: continue
    m = tmap[r.model]
    x = r.predict_ms / 795 * 1000
    ax.plot([x], [chrono.loc[m, 'macro_f1']], 'o', color=MCOL[m], ms=11)
    ax.plot([x], [lsum.loc[m, 'macro_f1_mean']], 's', color=MCOL[m], ms=8, mfc='none', mew=1.6)
    ax.plot([x, x], [chrono.loc[m, 'macro_f1'], lsum.loc[m, 'macro_f1_mean']], color=MCOL[m], lw=1, alpha=0.5)
    right = x > 10
    ax.annotate(MNAME[m], (x, chrono.loc[m, 'macro_f1']), textcoords='offset points',
                xytext=(-12 if right else 12, 7), ha='right' if right else 'left', fontsize=9, color=INK)
ax.set_xscale('log')
xs = [r.predict_ms / 795 * 1000 for _, r in tim.iterrows() if r.model in tmap]
ax.set_xlim(min(xs) * 0.35, max(xs) * 3.2)
ax.set_xlabel('time to assign one reading (microseconds)')
ax.set_ylabel('macro F1')
ax.set_title('Cost of assigning a reading against the quality of the assignment', loc='left')
ax.legend(handles=[plt.Line2D([], [], color=INK2, marker='o', ls='', ms=9, label='test week'),
                   plt.Line2D([], [], color=INK2, marker='s', ls='', ms=8, mfc='none', label='withheld nodes, mean')],
          fontsize=8.5, loc='lower right')
save(fig, 'fig4_14_tradeoff.png')
VS = pd.read_csv('validation_stability.csv')
VP = pd.read_csv('validation_prediction_strength.csv')
VJ = json.load(open('validation_tests.json'))
VORDER = [x for x in VJ['reference_states']][::-1]
SHORT = {'Saturated': 'Saturated', 'Moist / cooler': 'Moist, cooler',
         'Drier / higher pH': 'Drier, higher pH', 'Drier / lower pH': 'Drier, lower pH'}

fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.9, 3.7))
fig.subplots_adjust(left=0.155, right=0.98, top=0.8, bottom=0.17, wspace=0.28)
axl.set_axisbelow(True); axr.set_axisbelow(True)
y = np.arange(len(VORDER))
day = [VS[(VS.state == s) & (VS.resampling == 'whole days')] for s in VORDER]
axl.barh(y, [float(r.mean_jaccard.iloc[0]) for r in day], height=0.52, color=BLUE, alpha=0.85,
         xerr=[float(r.standard_error.iloc[0]) for r in day], error_kw=dict(ecolor=INK2, elinewidth=1))
axl.plot([float(VS[(VS.state == s) & (VS.resampling == 'single readings')].mean_jaccard.iloc[0])
          for s in VORDER], y, 'o', color=INK, ms=6, mfc='none', mew=1.4)
axl.axvline(0.75, color=RED, ls='--', lw=1.2)
axl.text(0.752, len(VORDER) - 0.42, 'retained at 0.75', color=RED, fontsize=8.5, va='top')
axl.set_yticks(y); axl.set_yticklabels([SHORT[s] for s in VORDER])
axl.set_xlim(0.4, 1.02); axl.set_xlabel('mean Jaccard over resamples')
axl.set_title('Stability under resampling', loc='left', pad=22)
axl.legend(handles=[plt.Line2D([], [], color=BLUE, lw=7, alpha=0.85, label='whole days'),
                    plt.Line2D([], [], color=INK, marker='o', ls='', ms=6, mfc='none', label='single readings')],
           fontsize=8.5, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2,
           borderaxespad=0.2, handletextpad=0.5, columnspacing=1.4)

VF = pd.read_csv('validation_folds.csv')
band = (float(VF.surrogate_95th_minimum.min()), float(VF.surrogate_95th_minimum.max()))
axr.axvspan(band[0], band[1], color=GOLD, alpha=0.18)
axr.text(band[1] + 0.012, -0.44, 'reference level', color='#8a6b00', fontsize=8.5, ha='left')
for i, s in enumerate(VORDER):
    v = VP[VP.state == s]
    axr.plot(v.prediction_strength, np.full(len(v), i) + np.linspace(-0.17, 0.17, len(v)),
             'o', color=VIOLET, ms=5.5, alpha=0.85)
axr.axvline(0.8, color=RED, ls='--', lw=1.2)
axr.text(0.79, len(VORDER) - 0.42, 'proposed mark 0.8', color=RED, fontsize=8.5, va='top', ha='right')
axr.set_yticks(y); axr.set_yticklabels([])
axr.set_ylim(-0.6, len(VORDER) - 0.4)
axr.set_xlim(0.2, 1.04); axr.set_xlabel('prediction strength, one point per node')
axr.set_title('Recognisability on a withheld node', loc='left', pad=22)
axl.set_ylim(-0.6, len(VORDER) - 0.4)
save(fig, 'fig4_15_validation.png')
print('\nwrote fig4_9 to fig4_15')
