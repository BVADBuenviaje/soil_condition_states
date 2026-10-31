import json, time, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from common import load, name_clusters, present, K, SEED
warnings.filterwarnings('ignore')

d, X = load()
n = len(X)

centred = X - X.mean(0)
total_scatter = float((centred ** 2).sum())
cov = centred.T @ centred / n
lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
lower_bound = float(total_scatter - n * lam[:K - 1].sum())
print(f'total scatter {total_scatter:.1f}, eigenvalues {np.round(lam, 3).tolist()}')
print(f'relaxation lower bound on the objective at k = {K}: {lower_bound:.1f}')

t0 = time.perf_counter()
best = KMeans(K, n_init=2000, random_state=SEED).fit(X)
search_seconds = time.perf_counter() - t0
best_known = float(best.inertia_)
print(f'best known objective after two thousand restarts: {best_known:.3f} '
      f'(search took {search_seconds:.1f} s)')
print('states of the best known solution:', present(np.array([name_clusters(best.cluster_centers_)[c]
                                                              for c in best.labels_], dtype=object)))

gap_best = lambda j: 100 * (j - best_known) / best_known
gap_bound = lambda j: 100 * (j - lower_bound) / lower_bound

rows = []
for budget in [1, 2, 5, 10, 30, 100]:
    vals, times = [], []
    for s in range(20):
        t0 = time.perf_counter()
        km = KMeans(K, n_init=budget, random_state=s).fit(X)
        times.append(time.perf_counter() - t0); vals.append(float(km.inertia_))
    v = np.array(vals)
    rows.append(dict(restarts=budget, objective_mean=round(float(v.mean()), 3),
                     objective_sd=round(float(v.std(ddof=1)), 3), objective_best=round(float(v.min()), 3),
                     objective_worst=round(float(v.max()), 3),
                     gap_to_best_known_mean_percent=round(float(np.mean([gap_best(x) for x in v])), 5),
                     gap_to_best_known_worst_percent=round(float(max(gap_best(x) for x in v)), 5),
                     gap_to_lower_bound_percent=round(float(np.mean([gap_bound(x) for x in v])), 2),
                     runs_within_a_hundredth_of_a_percent=int(sum(gap_best(x) < 0.01 for x in v)),
                     seconds=round(float(np.median(times)), 3)))
    print(f'{budget:>4} restarts: objective {v.mean():.3f} (sd {v.std(ddof=1):.3f}), '
          f'within a hundredth of a percent of the best known value in '
          f'{rows[-1]["runs_within_a_hundredth_of_a_percent"]} of 20 seeds, '
          f'{np.median(times):.3f} s')
budget = pd.DataFrame(rows); budget.to_csv('ch4_quality_budget.csv', index=False)

seeds = pd.read_csv('ch4_seeds.csv')
runs = pd.DataFrame(dict(seed=seeds.seed, objective=seeds.inertia,
                         gap_to_best_known_percent=[round(gap_best(j), 4) for j in seeds.inertia],
                         gap_to_lower_bound_percent=[round(gap_bound(j), 2) for j in seeds.inertia]))
runs.to_csv('ch4_quality_runs.csv', index=False)
print(f'\nover the {len(runs)} runs of the comparative matrix: objective '
      f'{runs.objective.min():.1f} to {runs.objective.max():.1f}, '
      f'gap to the best known value at most {runs.gap_to_best_known_percent.max():.4f} percent')

json.dump(dict(k=K, n=n, total_scatter=round(total_scatter, 1),
               eigenvalues=[round(float(x), 4) for x in lam],
               lower_bound=round(lower_bound, 6), best_known=round(best_known, 3),
               best_known_restarts=2000, best_known_search_seconds=round(search_seconds, 1),
               pipeline_objective=round(float(seeds.inertia.iloc[0]), 3),
               pipeline_gap_to_best_known_percent=round(gap_best(float(seeds.inertia.iloc[0])), 4),
               pipeline_gap_to_lower_bound_percent=round(gap_bound(float(seeds.inertia.iloc[0])), 2),
               worst_run_gap_percent=round(float(runs.gap_to_best_known_percent.max()), 4),
               single_restart_worst_gap_percent=round(float(budget.loc[budget.restarts == 1,
                                                                      'gap_to_best_known_worst_percent'].iloc[0]), 3),
               budget_rows=rows), open('ch4_quality.json', 'w'), indent=1)
print('\nwrote ch4_quality_budget.csv, ch4_quality_runs.csv and ch4_quality.json')
