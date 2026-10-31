"""Figure for the comparison of clustering methods. Reads comparison_methods.csv and
writes ch4_methods.png."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = pd.read_csv('comparison_methods.csv')
LABEL = {'Ward agglomerative': 'Ward', 'Gaussian hidden Markov model': 'Hidden Markov model'}
names = [LABEL.get(m, m) for m in r.method]
BG, BLUE, GREY, RED = '#FBFAF8', '#2B79D6', '#BAB6B0', '#D93A3A'
colours = [BLUE if m == 'k means' else GREY for m in r.method]
y = list(range(len(r)))[::-1]

fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), dpi=150, sharey=True)
fig.patch.set_facecolor(BG)
panels = [
    ('mi_with_node', 'Dependence on sensor node\n(mutual information, lower is better)', (0, 0.068), '{:.3f}'),
    ('lowest_stability', 'Least stable group under resampling\n(mean Jaccard, retained at 0.75)', (0, 1.0), '{:.3f}'),
    ('transfer_mean', 'Agreement on a withheld node\n(adjusted Rand, mean and worst fold)', (0, 1.0), '{:.3f}'),
    ('silhouette', 'Separation of the groups\n(silhouette coefficient)', (0, 0.38), '{:.3f}'),
]
for ax, (col, title, lim, fmt) in zip(axes, panels):
    ax.set_facecolor(BG)
    vals = r[col].values
    ax.barh(y, [max(v, 0) for v in vals], color=colours, height=0.6, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(max(v, 0) + lim[1] * 0.01, yi, fmt.format(v), va='center', fontsize=10,
                ha='left' if v >= 0 else 'right' if False else 'left')
    if col == 'transfer_mean':
        ax.scatter([max(w, 0) for w in r.transfer_worst], y, s=60, facecolors='none',
                   edgecolors='black', linewidths=1.4, zorder=3)
    if col == 'lowest_stability':
        ax.axvline(0.75, color=RED, linestyle='--', linewidth=1.5, zorder=3)
    ax.set_xlim(*lim)
    ax.set_title(title, loc='left', fontsize=12)
    ax.grid(color='#E4E2DE', zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(names, fontsize=12)
hd = r.index[r.method == 'HDBSCAN']
if len(hd):
    i = int(hd[0])
    note = f"{int(r.groups[i])} groups, {round(100 * r.noise_share[i])} percent noise"
    axes[0].text(-0.005, y[i] - 0.28, note, color=RED, fontsize=9, ha='right', transform=axes[0].transData)
fig.suptitle('The k means states against five other clustering methods on the same five features',
             x=0.12, ha='left', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('ch4_methods.png', facecolor=BG)
print('wrote ch4_methods.png')
