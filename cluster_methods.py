"""Comparison of the k means states with five other clustering methods on the same
features: k medoids, Ward agglomerative clustering, a Gaussian mixture, HDBSCAN and a
Gaussian hidden Markov model. Writes comparison_methods.csv and comparison_methods.json."""
import json, sys, time, warnings
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             normalized_mutual_info_score, adjusted_rand_score)
from common import load, name_clusters, run_lengths, ORDER, K, SEED

warnings.filterwarnings('ignore')
RESAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
HMM_RESAMPLES = RESAMPLES // 2
STABLE = 0.75

d, X = load()
nodes = sorted(d.node.unique())
NODE = d.node.values
TS = d.ts.values
DAY_BLOCKS = [g.values for _, g in pd.Series(np.arange(len(d))).groupby(
    [d.node.values, d.ts.dt.normalize().values])]


def sequences(idx):
    """Split positions into unbroken hourly runs of one node, in the given order."""
    out, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        gap = (TS[b] - TS[a]) / np.timedelta64(1, 'h')
        if NODE[a] == NODE[b] and 0.75 <= gap <= 1.25:
            cur.append(b)
        else:
            out.append(cur); cur = [b]
    out.append(cur)
    return out


def nearest_mean(Xf, lab):
    ks = sorted(k for k in set(lab) if k >= 0)
    cen = np.array([Xf[lab == k].mean(0) for k in ks])
    return lambda Xn: np.array(ks)[cdist(Xn, cen).argmin(1)]


def fit_kmeans(Xf, seed=SEED, starts=30, idx=None):
    km = KMeans(K, n_init=starts, random_state=seed).fit(Xf)
    return km.labels_, km.predict


def kmedoids_once(D, rng):
    n = len(D)
    first = rng.integers(n)
    med = [first]
    dist = D[first].copy()
    for _ in range(K - 1):
        p = dist ** 2
        nxt = rng.choice(n, p=p / p.sum()) if p.sum() > 0 else rng.integers(n)
        med.append(nxt); dist = np.minimum(dist, D[nxt])
    med = np.array(med)
    for _ in range(100):
        lab = D[:, med].argmin(1)
        new = med.copy()
        for k in range(K):
            mem = np.flatnonzero(lab == k)
            if len(mem):
                new[k] = mem[D[np.ix_(mem, mem)].sum(1).argmin()]
        if np.array_equal(np.sort(new), np.sort(med)):
            med = new
            break
        med = new
    lab = D[:, med].argmin(1)
    return lab, med, D[np.arange(n), med[lab]].sum()


def fit_kmedoids(Xf, seed=SEED, starts=30, idx=None):
    D = cdist(Xf, Xf)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        lab, med, cost = kmedoids_once(D, rng)
        if best is None or cost < best[2]:
            best = (lab, med, cost)
    cen = Xf[best[1]]
    return best[0], (lambda Xn: cdist(Xn, cen).argmin(1))


def fit_ward(Xf, seed=SEED, starts=1, idx=None):
    lab = AgglomerativeClustering(K, linkage='ward').fit_predict(Xf)
    return lab, nearest_mean(Xf, lab)


def fit_gmm(Xf, seed=SEED, starts=10, idx=None):
    g = GaussianMixture(K, covariance_type='full', n_init=starts, random_state=seed).fit(Xf)
    return g.predict(Xf), g.predict


def fit_hdbscan(Xf, seed=SEED, starts=1, idx=None):
    lab = HDBSCAN(min_cluster_size=30).fit_predict(Xf)
    if (lab >= 0).sum() == 0:
        return lab, (lambda Xn: np.full(len(Xn), -1))
    return lab, nearest_mean(Xf, lab)


def fit_hmm(Xf, seed=SEED, starts=8, idx=None):
    from hmmlearn.hmm import GaussianHMM
    lengths = [len(s) for s in sequences(idx)]
    best = None
    for s in range(starts):
        m = GaussianHMM(K, covariance_type='full', n_iter=200, random_state=seed + s)
        try:
            m.fit(Xf, lengths)
            sc = m.score(Xf, lengths)
        except Exception:
            continue
        if best is None or sc > best[1]:
            best = (m, sc)
    m = best[0]

    def assign(Xn, idx_n):
        return m.predict(Xn, [len(s) for s in sequences(idx_n)])
    return m.predict(Xf, lengths), assign


METHODS = [
    ('k means', fit_kmeans, {}),
    ('k medoids', fit_kmedoids, {}),
    ('Ward agglomerative', fit_ward, {}),
    ('Gaussian mixture', fit_gmm, {}),
    ('HDBSCAN', fit_hdbscan, {}),
    ('Gaussian hidden Markov model', fit_hmm, {}),
]


def names_of(lab):
    ks = sorted(k for k in set(lab) if k >= 0)
    cen = np.array([X[lab == k].mean(0) for k in ks])
    nm = name_clusters(cen)
    return {ks[i]: nm[i] for i in range(len(ks))}


def stability(fit, lab_ref, is_hmm, n_res):
    rng = np.random.default_rng(SEED)
    groups = sorted(k for k in set(lab_ref) if k >= 0)
    per = {g: [] for g in groups}
    for r in range(n_res):
        blocks = rng.integers(0, len(DAY_BLOCKS), len(DAY_BLOCKS))
        idx = np.concatenate([DAY_BLOCKS[b] for b in blocks])
        if is_hmm:
            seen, keep = set(), []
            for b in blocks:
                if b not in seen:
                    seen.add(b); keep.append(DAY_BLOCKS[b])
            idx_u = np.concatenate(keep)
            idx_u = idx_u[np.lexsort((TS[idx_u], NODE[idx_u]))]
            lab_r, _ = fit(X[idx_u], seed=r, starts=1, idx=idx_u)
            idx = idx_u
        else:
            lab_r, _ = fit(X[idx], seed=r, idx=idx)
        drawn = np.unique(idx)
        clusters = {t: np.unique(idx[lab_r == t]) for t in set(lab_r) if t >= 0}
        for g in groups:
            a = drawn[lab_ref[drawn] == g]
            if not len(a):
                continue
            best = 0.0
            for b in clusters.values():
                inter = len(np.intersect1d(a, b, assume_unique=True))
                union = len(a) + len(b) - inter
                if union and inter / union > best:
                    best = inter / union
            per[g].append(best)
    return {g: float(np.mean(v)) for g, v in per.items() if len(v)}


def transfer(fit, lab_ref, is_hmm):
    scores = []
    for n in nodes:
        tr = np.flatnonzero(NODE != n)
        te = np.flatnonzero(NODE == n)
        kw = dict(starts=3) if is_hmm else {}
        lab_tr, assign = fit(X[tr], idx=tr, **kw)
        pred = assign(X[te], te) if is_hmm else assign(X[te])
        scores.append(adjusted_rand_score(lab_ref[te], pred))
    return float(np.mean(scores)), float(np.min(scores))


rows = []
for name, fit, _ in METHODS:
    t0 = time.time()
    is_hmm = fit is fit_hmm
    lab, _ = fit(X, idx=np.arange(len(X)))
    ok = lab >= 0
    nm = names_of(lab)
    same = sorted(nm.values()) == sorted(ORDER)
    groups = sorted(k for k in set(lab) if k >= 0)
    sil = silhouette_score(X[ok], lab[ok])
    db = davies_bouldin_score(X[ok], lab[ok])
    mi = normalized_mutual_info_score(NODE[ok], lab[ok])
    stab = stability(fit, lab, is_hmm, HMM_RESAMPLES if is_hmm else RESAMPLES)
    tmean, tmin = transfer(fit, lab, is_hmm)
    run = run_lengths(np.where(ok, lab, -1), NODE, TS)[1]
    named = {nm[g]: int((lab == g).sum()) for g in groups}
    wettest = max(groups, key=lambda g: X[lab == g, 0].mean())
    row = dict(method=name, groups=len(groups), same_states=bool(same),
               names=' | '.join(nm[g] for g in groups),
               silhouette=round(sil, 3), davies_bouldin=round(db, 3), mi_with_node=round(mi, 3),
               lowest_stability=round(min(stab.values()), 3),
               stable_groups=int(sum(v >= STABLE for v in stab.values())),
               transfer_mean=round(tmean, 3), transfer_worst=round(tmin, 3),
               mean_run_h=round(run, 2), noise_share=round(float((~ok).mean()), 3),
               group_sizes=json.dumps(named),
               wettest_size=int((lab == wettest).sum()),
               wettest_moisture=round(float(X[lab == wettest, 0].mean()), 2),
               wettest_above_100=int(((lab == wettest) & (d.moisture.values > 100)).sum()),
               resamples=HMM_RESAMPLES if is_hmm else RESAMPLES,
               seconds=round(time.time() - t0, 1))
    rows.append(row)
    print(json.dumps(row), flush=True)

out = pd.DataFrame(rows)
out.to_csv('comparison_methods.csv', index=False)
json.dump(rows, open('comparison_methods.json', 'w'), indent=1)
print(out[['method', 'groups', 'same_states', 'silhouette', 'davies_bouldin', 'mi_with_node',
           'lowest_stability', 'stable_groups', 'transfer_mean', 'transfer_worst',
           'mean_run_h']].to_string(index=False))
