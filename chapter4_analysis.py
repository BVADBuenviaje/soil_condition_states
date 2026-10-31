import json, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (silhouette_score, davies_bouldin_score, normalized_mutual_info_score,
                             adjusted_rand_score, f1_score, log_loss, accuracy_score, balanced_accuracy_score)
from scipy.optimize import linear_sum_assignment
from common import (load, FEAT, ORDER, name_clusters, present, state_labels, run_lengths,
                    TEST_START, VAL_START, SEED)
warnings.filterwarnings('ignore')
d, X = load()
nodes = sorted(d.node.unique())
cfg = json.load(open('best_configs.json'))
RF = {**cfg['random_forest']['fixed'], **cfg['random_forest']['tuned']}
LR = {**cfg['logistic_regression']['fixed'], **cfg['logistic_regression']['tuned']}
N = {}

rows = []
for k in [3, 4, 5, 6]:
    ref = KMeans(k, n_init=30, random_state=SEED).fit(X); nm = name_clusters(ref.cluster_centers_)
    L = np.array([nm[c] for c in ref.labels_], dtype=object)
    same, aris, shift = 0, [], 0.0
    worst = ('', '', 0.0, 0.0); missing = []
    for n in nodes:
        tr = (d.node != n).values
        m = KMeans(k, n_init=30, random_state=SEED).fit(X[tr]); own = name_clusters(m.cluster_centers_)
        same += sorted(own.values()) == sorted(nm.values())
        gone = sorted(set(nm.values()) - set(own.values()))
        if gone: missing.append(f"{n}: {', '.join(gone)}")
        C = ((m.cluster_centers_[:, None, :] - ref.cluster_centers_[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C); mp = dict(zip(r, c))
        lt = np.array([mp[x] for x in m.labels_]); lh = np.array([mp[x] for x in m.predict(X[~tr])])
        aris.append(adjusted_rand_score(ref.labels_[~tr], lh))
        for j in range(k):
            sf = float((lt == j).mean()); sr = float((ref.labels_[tr] == j).mean())
            if abs(sf - sr) > shift:
                shift = abs(sf - sr); worst = (n, nm[j], round(sf, 3), round(sr, 3))
    sizes = np.bincount(ref.labels_)
    rows.append(dict(k=k, silhouette=round(silhouette_score(X, ref.labels_), 3),
                     davies_bouldin=round(davies_bouldin_score(X, ref.labels_), 3),
                     nmi_node=round(normalized_mutual_info_score(d.node, ref.labels_), 3),
                     smallest=int(sizes.min()), all_named=not any(v.startswith('Unnamed') for v in nm.values()),
                     states=' | '.join(present(L)), folds_same_set=int(same),
                     ari_mean=round(float(np.mean(aris)), 3), ari_min=round(float(np.min(aris)), 3),
                     largest_share_shift=round(shift, 3), largest_shift_fold=worst[0],
                     largest_shift_state=worst[1], largest_shift_fold_share=worst[2],
                     largest_shift_reference_share=worst[3],
                     folds_missing_states='; '.join(missing)))
ksel = pd.DataFrame(rows); ksel.to_csv('ch4_k_selection.csv', index=False); print(ksel.to_string(index=False))
N['k_selection'] = ksel.to_dict('records')

rl, rl_all = run_lengths(d.state_ref, d.node, d.ts)
P = pd.read_csv('transition_probabilities.csv', index_col=0)
prof = []
for s in ORDER:
    g = d[d.state_ref == s]
    prof.append(dict(state=s, n=len(g), share=round(len(g) / len(d), 4), nodes=int(g.node.nunique()),
                     moisture_med=round(g.moisture.median(), 1), ph_med=round(g.ph.median(), 2), temp_med=round(g.temp.median(), 1),
                     moisture_rel=round(g.moisture_rel.mean(), 2), ph_rel=round(g.ph_rel.mean(), 2), temp_rel=round(g.temp_rel.mean(), 2),
                     dmoist_3h=round(g.dmoist_3h.clip(-5, 5).mean(), 2),
                     run_h=round(rl[s], 2), self_p=round(P.loc[s, s], 3),
                     share_0710=round(((g.ts.dt.hour >= 7) & (g.ts.dt.hour <= 10)).mean(), 2),
                     share_test_week=round((g.ts >= TEST_START).mean(), 2),
                     share_from_3nov=round((g.ts >= pd.Timestamp('2025-11-03')).mean(), 2),
                     median_date=g.ts.median().strftime('%d %B')))
prof = pd.DataFrame(prof); prof.to_csv('ch4_state_profiles.csv', index=False); print(prof.to_string(index=False))
N['profiles'] = prof.to_dict('records'); N['run_all_mean'] = round(rl_all, 2)
N['share_test_week_all'] = round(float((d.ts >= TEST_START).mean()), 3)
wk = d.groupby([d.node, (d.ts - pd.Timestamp('2025-10-13')).dt.days // 7]).ph.median().unstack().round(2)
wk.to_csv('ch4_weekly_ph.csv'); N['weekly_ph'] = {n: wk.loc[n].tolist() for n in wk.index}
print(wk)

tr = (d.ts < TEST_START).values; te = ~tr
km = KMeans(4, n_init=30, random_state=SEED).fit(X[tr]); nm = name_clusters(km.cluster_centers_)
cen = pd.DataFrame(km.cluster_centers_, columns=FEAT); cen['state'] = [nm[i] for i in cen.index]
cen['n_train'] = np.bincount(km.labels_); cen['n_test'] = np.bincount(km.predict(X[te]), minlength=4)
cen.round(3).to_csv('ch4_training_partition.csv', index=False); print(cen.round(2).to_string())
N['train_partition'] = cen.round(3).to_dict('records')
D = pd.DataFrame(np.sqrt(((km.cluster_centers_[:, None] - km.cluster_centers_[None]) ** 2).sum(-1)),
                 index=cen.state, columns=cen.state).round(3)
D.to_csv('ch4_centre_distances.csv'); print(D)
N['centre_distances'] = D.to_dict()
yt, yv = state_labels(X[tr], X[te]); lab = np.empty(len(d), dtype=object); lab[tr] = yt; lab[te] = yv
dd = d.assign(lab=lab)
after_sat = []
for n, g in dd.groupby('node'):
    g = g.reset_index(drop=True)
    for i in np.where(g.lab.values == 'Rapid drainage')[0]:
        w = g[(g.ts < g.ts[i]) & (g.ts >= g.ts[i] - pd.Timedelta(hours=3))]
        after_sat.append(bool((w.state_ref == 'Saturated').any()))
N['drainage_n'] = int((lab == 'Rapid drainage').sum()); N['drainage_after_sat'] = int(np.sum(after_sat))
N['drainage_nodes'] = dd[dd.lab == 'Rapid drainage'].node.value_counts().to_dict()
print('rapid drainage readings', N['drainage_n'], 'preceded by saturation within 3 h:', N['drainage_after_sat'], N['drainage_nodes'])
km_ref_init = KMeans(4, init=KMeans(4, n_init=30, random_state=SEED).fit(X).cluster_centers_, n_init=1).fit(X[tr])
N['inertia_train_best'] = round(float(km.inertia_), 1); N['inertia_train_phsplit'] = round(float(km_ref_init.inertia_), 1)
tr5 = (d.node != 'Node-05').values
k5a = KMeans(4, n_init=30, random_state=SEED).fit(X[tr5])
k5b = KMeans(4, init=KMeans(4, n_init=30, random_state=SEED).fit(X).cluster_centers_, n_init=1).fit(X[tr5])
N['inertia_no5_best'] = round(float(k5a.inertia_), 1); N['inertia_no5_phsplit'] = round(float(k5b.inertia_), 1)
print('inertia: training period best', N['inertia_train_best'], 'pH split', N['inertia_train_phsplit'],
      '| without Node 05 best', N['inertia_no5_best'], 'pH split', N['inertia_no5_phsplit'])

sub = (d.ts < VAL_START).values; val = ((d.ts >= VAL_START) & (d.ts < TEST_START)).values
ys, yvv = lab[sub], lab[val]; Ltr = present(yt)
def mf1(y, p): return f1_score(y, p, average='macro', labels=Ltr, zero_division=0)
rfg = pd.read_csv('tuning_random_forest.csv'); rfg['max_depth'] = rfg.max_depth.fillna(0).astype(int)
ch = cfg['random_forest']['tuned']; chd = ch['max_depth'] or 0
N['rf_grid'] = dict(val_min=float(rfg.val_macro_f1.min()), val_median=float(rfg.val_macro_f1.median()),
                    val_max=float(rfg.val_macro_f1.max()), n=int(len(rfg)),
                    n_within_001=int((rfg.val_macro_f1 >= rfg.val_macro_f1.max() - 0.01).sum()),
                    train_min=float(rfg.train_macro_f1.min()), train_max=float(rfg.train_macro_f1.max()))
marg = {p: rfg.groupby(p).val_macro_f1.agg(['mean', 'min', 'max']).round(4).reset_index().to_dict('records')
        for p in ['min_samples_leaf', 'max_features', 'max_depth']}
N['rf_marginal'] = marg
top = rfg.sort_values('val_macro_f1', ascending=False).head(8); top.to_csv('ch4_rf_top.csv', index=False)
N['rf_top'] = top.to_dict('records'); print(top.to_string(index=False))
vc = {}
vc['min_samples_leaf'] = rfg[(rfg.max_features == ch['max_features']) & (rfg.max_depth == chd)].sort_values('min_samples_leaf')
vc['max_features'] = rfg[(rfg.min_samples_leaf == ch['min_samples_leaf']) & (rfg.max_depth == chd)].sort_values('max_features')
vc['max_depth'] = rfg[(rfg.min_samples_leaf == ch['min_samples_leaf']) & (rfg.max_features == ch['max_features'])].copy()
vc['max_depth']['o'] = vc['max_depth'].max_depth.replace(0, 99); vc['max_depth'] = vc['max_depth'].sort_values('o').drop(columns='o')
for p, t in vc.items():
    t.to_csv(f'ch4_valcurve_{p}.csv', index=False); N[f'valcurve_{p}'] = t.to_dict('records')
dtg = pd.read_csv('tuning_decision_tree.csv'); dtg['max_depth'] = dtg.max_depth.fillna(0).astype(int)
N['dt_grid'] = dtg.to_dict('records')
lrg = pd.read_csv('tuning_logistic_regression.csv'); N['lr_grid'] = lrg.to_dict('records')

sizes = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400]
rf = RandomForestClassifier(**{**RF, 'n_estimators': 1}, warm_start=True, n_jobs=-1)
loss = []
for t in sizes:
    rf.set_params(n_estimators=t); rf.fit(X[sub], ys)
    ptr, pva = rf.predict_proba(X[sub]), rf.predict_proba(X[val])
    loss.append(dict(trees=t, train_log_loss=round(log_loss(ys, ptr, labels=rf.classes_), 4),
                     val_log_loss=round(log_loss(yvv, pva, labels=rf.classes_), 4),
                     train_macro_f1=round(mf1(ys, rf.classes_[ptr.argmax(1)]), 4),
                     val_macro_f1=round(mf1(yvv, rf.classes_[pva.argmax(1)]), 4)))
loss = pd.DataFrame(loss); loss.to_csv('ch4_loss_curve.csv', index=False); print(loss.to_string(index=False))
N['loss_curve'] = loss.to_dict('records')
rfu = RandomForestClassifier(**{**RF, 'n_estimators': 1, 'min_samples_leaf': 1, 'max_depth': None, 'max_features': 5},
                             warm_start=True, n_jobs=-1)
lossu = []
for t in sizes:
    rfu.set_params(n_estimators=t); rfu.fit(X[sub], ys)
    ptr, pva = rfu.predict_proba(X[sub]), rfu.predict_proba(X[val])
    lossu.append(dict(trees=t, train_log_loss=round(log_loss(ys, np.clip(ptr, 1e-15, 1), labels=rfu.classes_), 4),
                      val_log_loss=round(log_loss(yvv, np.clip(pva, 1e-15, 1), labels=rfu.classes_), 4),
                      train_macro_f1=round(mf1(ys, rfu.classes_[ptr.argmax(1)]), 4),
                      val_macro_f1=round(mf1(yvv, rfu.classes_[pva.argmax(1)]), 4)))
lossu = pd.DataFrame(lossu); lossu.to_csv('ch4_loss_curve_unconstrained.csv', index=False); print(lossu.to_string(index=False))
N['loss_curve_unconstrained'] = lossu.to_dict('records')

rng = np.random.default_rng(SEED); fr = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]; lc = []
idx_sub = np.where(sub)[0]
for f in fr:
    for rep in range(5 if f < 1 else 1):
        pick = []
        for s in Ltr:
            ids = idx_sub[lab[idx_sub] == s]; m = max(2, int(round(f * len(ids))))
            pick.append(rng.choice(ids, size=min(m, len(ids)), replace=False))
        pick = np.concatenate(pick)
        mdl = RandomForestClassifier(**RF, n_jobs=-1).fit(X[pick], lab[pick])
        lc.append(dict(fraction=f, repeat=rep, n=len(pick), train_macro_f1=mf1(lab[pick], mdl.predict(X[pick])),
                       val_macro_f1=mf1(yvv, mdl.predict(X[val]))))
lc = pd.DataFrame(lc); lcs = lc.groupby('fraction').agg(n=('n', 'mean'), train=('train_macro_f1', 'mean'), val=('val_macro_f1', 'mean'),
                                                     val_sd=('val_macro_f1', 'std')).round(4).reset_index()
lcs.to_csv('ch4_learning_curve.csv', index=False); print(lcs.to_string(index=False)); N['learning_curve'] = lcs.to_dict('records')

fin = {}
for name, mdl in [('random_forest', RandomForestClassifier(**RF, n_jobs=-1)), ('logistic_regression', LogisticRegression(**LR))]:
    mdl.fit(X[tr], yt)
    fin[name] = dict(train_macro_f1=round(mf1(yt, mdl.predict(X[tr])), 4), test_macro_f1=round(mf1(yv, mdl.predict(X[te])), 4),
                     test_log_loss=round(log_loss(yv, mdl.predict_proba(X[te]), labels=mdl.classes_), 4))
N['final_fit'] = fin; print(fin)

cm = pd.read_csv('comparison_confusion_rf.csv', index_col=0)
err = cm.values.sum() - np.trace(cm.values)
mc = cm.loc['Moist / cooler', 'Drier'] + cm.loc['Drier', 'Moist / cooler']
N['rf_errors'] = int(err); N['rf_errors_moist_drier'] = int(mc)
def margin(Xs):
    dist = np.sqrt(((Xs[:, None] - km.cluster_centers_[None]) ** 2).sum(-1)); dist.sort(1); return dist[:, 1] - dist[:, 0]
rf_final = RandomForestClassifier(**RF, n_jobs=-1).fit(X[tr], yt); p = rf_final.predict(X[te])
mg = margin(X[te]); N['margin_median_correct'] = round(float(np.median(mg[p == yv])), 3)
N['margin_median_wrong'] = round(float(np.median(mg[p != yv])), 3)
N['margin_q90_wrong'] = round(float(np.quantile(mg[p != yv], 0.9)), 3)
print('boundary margin, median correct', N['margin_median_correct'], 'wrong', N['margin_median_wrong'])

def second_nearest_share(km_, Xte, y_true, y_pred, names_):
    dist = np.sqrt(((Xte[:, None] - km_.cluster_centers_[None]) ** 2).sum(-1)); order_ = dist.argsort(1)
    second = np.array([names_[i] for i in order_[:, 1]], dtype=object); wrong = y_true != y_pred
    return int(wrong.sum()), int((wrong & (y_pred == second)).sum())
adj = {}
for name, mdl in [('random_forest', RandomForestClassifier(**RF, n_jobs=-1)), ('logistic_regression', LogisticRegression(**LR))]:
    pp = mdl.fit(X[tr], yt).predict(X[te]); adj[f'chrono_{name}'] = second_nearest_share(km, X[te], yv, pp, nm)
    tot = [0, 0]
    for n in nodes:
        trn = (d.node != n).values; kf = KMeans(4, n_init=30, random_state=SEED).fit(X[trn]); nf = name_clusters(kf.cluster_centers_)
        ytr_n = np.array([nf[c] for c in kf.labels_], dtype=object); yte_n = np.array([nf[c] for c in kf.predict(X[~trn])], dtype=object)
        m2 = (RandomForestClassifier(**RF, n_jobs=-1) if name == 'random_forest' else LogisticRegression(**LR)).fit(X[trn], ytr_n)
        w, a2 = second_nearest_share(kf, X[~trn], yte_n, m2.predict(X[~trn]), nf); tot[0] += w; tot[1] += a2
    adj[f'lono_{name}'] = tuple(tot)
N['adjacent_errors'] = adj; print('errors (total, to the second nearest centre):', adj)
held = pd.read_pickle('lono_labels_seed0_k4.pkl'); rlh = {}
for n in nodes:
    m = (d.node == n).values
    rlh[n] = dict(fold=round(run_lengths(held[n], d.node[m].values, d.ts[m].values)[1], 2),
                  reference=round(run_lengths(d.state_ref[m].values, d.node[m].values, d.ts[m].values)[1], 2))
N['lono_run_lengths'] = rlh; print('mean run length on each withheld node, fold vs reference:', rlh)
sat = d[d.state_ref == 'Saturated']
N['sat_at_ceiling'] = int((sat.moisture >= 100).sum()); N['sat_ph_below_3'] = int((sat.ph < 3).sum())
N['sat_ph_3_to_5_5'] = int(((sat.ph >= 3) & (sat.ph < 5.5)).sum()); N['sat_node02'] = int((sat.node == 'Node-02').sum())
N['sat_node02_max_moisture'] = round(float(d[d.node == 'Node-02'].moisture.max()), 1)
N['recovery_episodes_total'] = int(sum(1 for n, g in d.groupby('node') for i, v in enumerate(g.state_ref.values)
                                   if v == 'Saturated' and (i == 0 or g.state_ref.values[i - 1] != 'Saturated')))
lfo = pd.read_csv('comparison_lono_folds.csv')
N['lono_means'] = {m: dict(accuracy=round(float(g.accuracy.mean()), 3), balanced_accuracy=round(float(g.balanced_accuracy.mean()), 3))
                   for m, g in lfo.groupby('model')}

tr_d = d[tr]
def feats(stats_from):
    f = d.copy()
    for c in ['moisture', 'ph', 'temp']:
        med = stats_from.groupby('node')[c].median(); q = stats_from.groupby('node')[c].quantile([.25, .75]).unstack()
        iq = (q[0.75] - q[0.25]).clip(lower=1e-6)
        f[c + '_rel'] = (f[c] - f.node.map(med)) / f.node.map(iq)
    iqm = (stats_from.groupby('node').moisture.quantile(.75) - stats_from.groupby('node').moisture.quantile(.25)).clip(lower=1e-6)
    iq_all = d.groupby('node').moisture.transform(lambda s: max(s.quantile(.75) - s.quantile(.25), 1e-6))
    for h in ('dmoist_1h', 'dmoist_3h'):
        f[h] = d[h] * iq_all / f.node.map(iqm)
    return f[FEAT].clip(-5, 5).values
Xt = feats(tr_d)
lo_, hi_ = Xt[tr][:, :3].min(0), Xt[tr][:, :3].max(0)
N['train_only_outside_share'] = round(float(((Xt[te][:, :3] < lo_) | (Xt[te][:, :3] > hi_)).any(1).mean()), 3)
yt2, yv2 = state_labels(Xt[tr], Xt[te]); L2 = present(yt2)
res2 = {}
for name, mdl in [('random_forest', RandomForestClassifier(**RF, n_jobs=-1)), ('logistic_regression', LogisticRegression(**LR))]:
    pp = mdl.fit(Xt[tr], yt2).predict(Xt[te])
    res2[name] = dict(accuracy=round(accuracy_score(yv2, pp), 4), balanced_accuracy=round(balanced_accuracy_score(yv2, pp), 4),
                      macro_f1=round(f1_score(yv2, pp, average='macro', labels=L2, zero_division=0), 4))
N['train_only_baseline'] = dict(states=L2, counts_test=pd.Series(yv2).value_counts().to_dict(), results=res2)
print('baseline from the training period only:', N['train_only_baseline'])

rules = {}
sat_runs = []
for n, g in d.groupby('node'):
    g = g.reset_index(drop=True); i = 0
    while i < len(g):
        if g.state_ref[i] == 'Saturated':
            j = i
            while j + 1 < len(g) and g.state_ref[j + 1] == 'Saturated' and (g.ts[j + 1] - g.ts[j]) <= pd.Timedelta(minutes=75): j += 1
            sat_runs.append(dict(node=n, start=g.ts[i], hours=j - i + 1)); i = j + 1
        else: i += 1
sr = pd.DataFrame(sat_runs)
rules['saturated_episodes'] = len(sr); rules['saturated_episodes_by_node'] = sr.node.value_counts().to_dict()
rules['saturated_longest_h'] = int(sr.hours.max()) if len(sr) else 0
rules['saturated_episodes_over_1h'] = int((sr.hours > 1).sum())
sd = d[d.state_ref == 'Saturated'].assign(day=lambda x: x.ts.dt.normalize(), wk=lambda x: x.ts.dt.to_period('W'))
per = sd.groupby(['node', 'wk']).day.nunique()
rules['r2_max_days_in_a_week'] = int(per.max()); rules['r2_node_weeks_with_4plus_days'] = int((per >= 4).sum())
dr = np.where(d.state_ref.str.startswith('Drier'), 'Drier', d.state_ref)
runs = []
for n in nodes:
    m = (d.node == n).values; l = dr[m]; t = d.ts.values[m]; cur = 1
    for i in range(1, len(l)):
        gap = (t[i] - t[i - 1]) / np.timedelta64(1, 'h')
        if l[i] == l[i - 1] and 0.75 <= gap <= 1.25: cur += 1
        else: runs.append((n, l[i - 1], cur)); cur = 1
    runs.append((n, l[-1], cur))
runs = pd.DataFrame(runs, columns=['node', 'state', 'h'])
rules['r3_drier_runs_over_24h'] = int(((runs.state == 'Drier') & (runs.h > 24)).sum())
rules['r3_longest_drier_run_h'] = int(runs[runs.state == 'Drier'].h.max())
rules['r5_valid_readings_ph_below_5_5'] = int(((d.ph < 5.5) & (d.ph >= 3)).sum())
ec = d.groupby('node').ec.median().round(3).to_dict(); rules['r6_ec_median_by_node'] = ec
N['rules'] = rules; print(json.dumps(rules, indent=1, default=str))

lab_raw = pd.read_csv('soil_labeled.csv')
N['raw_nmi_k5'] = round(float(normalized_mutual_info_score(lab_raw.node, lab_raw.cl_all_k5)), 3)
raw = pd.read_csv('soil_clean.csv')
oor = (~raw.moisture.between(0, 100)) | (~raw.ph.between(3, 10)) | (~raw.temp.between(0, 45)) | (~raw.ec.between(0, 2))
N['out_of_range'] = int(oor.sum()); N['ceiling_readings'] = int((raw.moisture > 100).sum())
n4 = d[d.node == 'Node-04'].moisture
N['node04'] = dict(median=round(float(n4.median()), 1), iqr=round(float(n4.quantile(.75) - n4.quantile(.25)), 1),
                   max=round(float(n4.max()), 1), max_rel=round(float(d[d.node == 'Node-04'].moisture_rel.max()), 1))
N['sub_counts'] = pd.Series(ys).value_counts().to_dict(); N['val_counts'] = pd.Series(yvv).value_counts().to_dict()
N['n_sub'] = int(sub.sum()); N['n_val'] = int(val.sum())
lowruns = runs = None
rr = []
for n in nodes:
    m = (d.node == n).values; l = d.state_ref.values[m]; t = d.ts.values[m]; cur = 1
    for i in range(1, len(l)):
        gap = (t[i] - t[i - 1]) / np.timedelta64(1, 'h')
        if l[i] == l[i - 1] and 0.75 <= gap <= 1.25: cur += 1
        else: rr.append((l[i - 1], cur)); cur = 1
    rr.append((l[-1], cur))
rr = pd.DataFrame(rr, columns=['s', 'h']); N['longest_run'] = {k: int(v) for k, v in rr.groupby('s').h.max().items()}
merged = np.where(d.state_ref.str.startswith('Drier'), 'Drier', d.state_ref)
rl_merged, _ = run_lengths(merged, d.node.values, d.ts.values)
N['run_merged_drier'] = round(float(rl_merged['Drier']), 2)
an = raw[raw.temp < 45]
N['out_of_range_analysis'] = int((((~an.moisture.between(0, 100)) | (~an.ph.between(3, 10)) | (~an.temp.between(0, 45)) | (~an.ec.between(0, 2)))).sum())
n1 = d[(d.node == 'Node-01') & (d.moisture.between(0, 100))]
hm = n1.groupby(n1.ts.dt.hour).moisture.median()
N['node01_hourly_median_min'] = round(float(hm.min()), 1); N['node01_hourly_median_max'] = round(float(hm.max()), 1)
N['node01_iqr'] = round(float(n1.moisture.quantile(.75) - n1.moisture.quantile(.25)), 2)
scan = pd.read_csv('raw_cluster_scan.csv').set_index('k')
N['raw_nmi_k4'] = float(scan.loc[4, 'nmi_node']); N['raw_purity_k4'] = float(scan.loc[4, 'purity_node'])
ceil_by_state = d[d.moisture >= 100].state_ref.value_counts().to_dict(); N['ceiling_by_state'] = ceil_by_state
N['ceiling_node03_moist'] = int(((d.moisture >= 100) & (d.node == 'Node-03') & (d.state_ref == 'Moist / cooler')).sum())
wk_raw = d.groupby([d.node, (d.ts - pd.Timestamp('2025-10-13')).dt.days // 7]).ph.median().unstack()
wk_dev = {n: round(float(max(abs(v - wk_raw.loc[n].iloc[0]) for v in wk_raw.loc[n])), 2) for n in wk_raw.index}
N['weekly_ph_max_departure'] = wk_dev
N['ph_below_5_5_all'] = int((d.ph < 5.5).sum()); N['ph_below_3_outside_sat'] = int(((d.ph < 3) & (d.state_ref != 'Saturated')).sum())
N['lc_full_vs_selected'] = round(float(lcs.set_index('fraction').loc[1.0, 'val']), 4)
print('extra:', {k: N[k] for k in ['raw_nmi_k5', 'out_of_range', 'ceiling_readings', 'node04', 'sub_counts', 'val_counts',
                                   'longest_run', 'ceiling_by_state', 'weekly_ph_max_departure', 'ph_below_5_5_all', 'ph_below_3_outside_sat']})

json.dump(N, open('ch4_numbers.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('\nwrote ch4_numbers.json and ch4_*.csv')
