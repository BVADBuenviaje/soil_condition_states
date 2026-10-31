import json, sys, shutil, os, re
import numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results_and_discussion.tex'
N = json.load(open('ch4_numbers.json')); cfg = json.load(open('best_configs.json'))
TS = json.load(open('transitions_summary.json')); CT = json.load(open('comparison_tests.json'))
RB = json.load(open('ch4_robustness.json')); VER = pd.read_csv('ch4_verification.csv')
TIM = pd.read_csv('ch4_timing.csv'); STEPS = pd.read_csv('pipeline_step_times.csv') if os.path.exists('pipeline_step_times.csv') else None
TOP3 = ''
if STEPS is not None:
    _t = STEPS.sort_values('seconds', ascending=False).head(3)
    _nm = [re.sub(r'^[0-9]+\.\s*', '', s) for s in _t.step]
    _nm = [(s if s.startswith('Chapter') else s[0].lower() + s[1:]).replace('Chapter 4', 'Chapter Four') for s in _nm]
    _sc = list(_t.seconds)
    _parts = [f'{n} ({s:.0f} seconds)' for n, s in zip(_nm, _sc)]
    TOP3 = ', '.join(_parts[:-1]) + ' and ' + _parts[-1]
SEEDS = pd.read_csv('ch4_seeds.csv')
MATRIX = pd.read_csv('ch4_runs_matrix.csv'); WIL = pd.read_csv('ch4_wilcoxon.csv')
LSUM = pd.read_csv('comparison_lono_summary.csv').set_index('model')
KFOLD = pd.read_csv('comparison_random_kfold.csv').set_index('model')
IMP = pd.read_csv('ch4_importance.csv'); LRC = pd.read_csv('ch4_lr_coefficients.csv', index_col=0)
ABLC = pd.read_csv('ch4_ablation_classifier.csv'); ABLK = pd.read_csv('ch4_ablation_clustering.csv')
FS = json.load(open('ch4_feature_summary.json'))
VSTAB = pd.read_csv('validation_stability.csv'); VPS = pd.read_csv('validation_prediction_strength.csv')
VPERS = pd.read_csv('validation_persistence.csv'); VJ = json.load(open('validation_tests.json'))
VFOLD = pd.read_csv('validation_folds.csv').set_index('held_out')
SCAL = pd.read_csv('ch4_scaling_readings.csv'); SNOD = pd.read_csv('ch4_scaling_nodes.csv')
SSTA = pd.read_csv('ch4_scaling_states.csv'); SLEN = pd.read_csv('ch4_scaling_length.csv')
VIAB = pd.read_csv('ch4_viability.csv'); SC = json.load(open('ch4_scaling.json'))
BUD = pd.read_csv('ch4_quality_budget.csv'); Q = json.load(open('ch4_quality.json'))
ks = pd.read_csv('ch4_k_selection.csv'); pr = pd.read_csv('ch4_state_profiles.csv').set_index('state')
P = pd.read_csv('transition_probabilities.csv', index_col=0); T = pd.read_csv('transition_counts.csv', index_col=0)
lono = pd.read_csv('lono_summary_seed0_k4.csv').set_index('held_out'); sh = pd.read_csv('lono_shares_seed0_k4.csv')
ch = pd.read_csv('comparison_chronological.csv').set_index('model'); lf = pd.read_csv('comparison_lono_folds.csv')
cl = pd.read_csv('comparison_clustering.csv'); fr = pd.read_csv('forecast_results.csv'); fl = pd.read_csv('forecast_lono.csv')
tp = pd.DataFrame(N['train_partition']).set_index('state'); D = pd.DataFrame(N['centre_distances'])
cm = pd.read_csv('comparison_confusion_rf.csv', index_col=0); cml = pd.read_csv('ch4_confusion_lr.csv', index_col=0)
lt = pd.DataFrame(N['loss_curve']).set_index('trees'); lu = pd.DataFrame(N['loss_curve_unconstrained']).set_index('trees')
lc = pd.DataFrame(N['learning_curve']).set_index('fraction'); rules = N['rules']

def f(x, n=3): return f'{x:.{n}f}'
def pc(x, n=0): return f'{100 * x:.{n}f}'
def th(n): return f'{int(n):,}'.replace(',', '{,}')
def sg(x, n=2):
    v = round(float(x), n)
    return (f'$-${abs(v):.{n}f}' if v < 0 else f'{abs(v):.{n}f}')
def numw(n):
    return WORD.get(int(n), str(int(n))).lower() if n else 'none'
WORD = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven',
        8: 'Eight', 9: 'Nine', 10: 'Ten', 11: 'Eleven', 12: 'Twelve', 13: 'Thirteen',
        14: 'Fourteen', 15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen',
        19: 'Nineteen', 20: 'Twenty', 30: 'Thirty'}
def sci(x, n=1):
    if abs(x) < 5e-10: return '0'
    a, b = f'{x:.{n}e}'.split('e')
    return f'${a} \\times 10^{{{int(b)}}}$'
SN = lambda n: n.replace('-', ' ')
NODES = ['Node-01', 'Node-02', 'Node-03', 'Node-04', 'Node-05']
ORDER = ['Saturated', 'Moist / cooler', 'Drier / higher pH', 'Drier / lower pH']
TP = ['Saturated', 'Rapid drainage', 'Moist / cooler', 'Drier']
NAME = {'Saturated': 'saturated', 'Moist / cooler': 'moist and cooler', 'Drier / higher pH': 'drier and higher pH',
        'Drier / lower pH': 'drier and lower pH', 'Rapid drainage': 'rapid drainage', 'Drier': 'drier'}
TNAME = {'Saturated': 'Saturated', 'Moist / cooler': 'Moist and cooler', 'Drier / higher pH': 'Drier, higher pH',
         'Drier / lower pH': 'Drier, lower pH', 'Rapid drainage': 'Rapid drainage', 'Drier': 'Drier'}
MN = {'random_forest': 'Random Forest', 'logistic_regression': 'Logistic regression', 'decision_tree': 'Classification tree',
      'majority': 'Majority class', 'nearest_centroid (reference)': 'Nearest centroid (reference)'}
k4 = ks.set_index('k').loc[4]; k3 = ks.set_index('k').loc[3]; k5 = ks.set_index('k').loc[5]; k6 = ks.set_index('k').loc[6]
n_all = int(pr.n.sum())
rf_cfg, dt_cfg, lr_cfg = cfg['random_forest'], cfg['decision_tree'], cfg['logistic_regression']
shr = lambda node, s, col: float(sh[(sh.held_out == node) & (sh.state == s)][col].iloc[0])
wide = lf.pivot(index='held_out', columns='model', values='macro_f1')
ranks = CT['friedman']['mean_ranks']; CD = CT['friedman']['nemenyi_CD']
mc = {r['comparison'].split(' vs ')[1]: r for r in CT['mcnemar']}
_n5 = int(sh[(sh.held_out == 'Node-05')].n_heldout.iloc[0]) if 'n_heldout' in sh.columns else int(lono.loc['Node-05', 'n_rows'])
_lo5 = int((pd.read_csv('soil_states_k4.csv').query("node == 'Node-05' and state_ref == 'Drier / lower pH'")).shape[0])
ref_share_on_train5 = (pr.loc['Drier / lower pH', 'n'] - _lo5) / (n_all - _n5)
fchk = lambda t, h: fr[(fr.target == t) & (fr.horizon == h)].iloc[0]
_nd = N['val_counts']['Rapid drainage']; _r = (_nd - 1) / _nd
one_error = (1 - (2 * _r / (1 + _r))) / 4
wkph = N['weekly_ph']
L = []; A = L.append

A(r"""\chapter{Results and Discussion}
\label{ch:results}

This chapter reports the outcome of each procedure set out in
Chapter Three, in the order of the evaluation plan of
Section~\ref{sec:evaluation}. Every value is produced by the pipeline of
Section~\ref{sec:reproducibility} and can be regenerated from the source
workbook, and every number in the chapter is inserted by a script from the
pipeline's outputs and not transcribed by hand. Section~\ref{sec:discussion} draws
the findings together. Five procedural matters that arose during the
analysis, and that affect how several results should be read, are declared where
they arise rather than deferred: the choice of the number of states, which the
rule of precedence could not decide and which was therefore made after the
results were seen, in Section~\ref{sec:res_states}; the retention of a state that
the transfer rule of Section~\ref{sec:lono} would have removed, in
Section~\ref{sec:res_lono}; an extension of the state naming rule in
Section~\ref{sec:res_training}; an additional check on the normalisation
baseline in Section~\ref{sec:res_baseline}; and the revision of the recurrence
criteria themselves, in Section~\ref{sec:res_stability}. None of the five was
specified in advance, and all are reported as departures from the plan. The last
of them concerns the criteria by which the others are judged, and is therefore
stated in full where it arises.
""")

c_p, c_s = cl.iloc[0], cl.iloc[1]
A(rf"""
\section{{Separation of the States from Sensor Location}}
\label{{sec:res_location}}

Section~\ref{{sec:nodedominance}} showed that clustering the raw readings recovers
the sensor layout: at four clusters the normalised mutual information between
cluster and node is {N['raw_nmi_k4']:.2f}, and {pc(N['raw_purity_k4'])} percent of the readings fall in a
cluster whose largest contributor is their own node. After per node normalisation
the same measure for the four states is {f(c_p.nmi_with_node)}, so that knowing a reading's state
conveys almost nothing about which sensor produced it. Table~\ref{{tab:res_clustering}}
compares the proposed normalisation against the standard approach of
Section~\ref{{sec:baselines}}, in which the readings of all nodes are pooled and
standardised together before clustering with the same number of states, seed,
and clipping.

\begin{{table}}[htbp]
\centering
\caption[Proposed normalisation against the standard approach]{{\textbf{{Proposed
per node normalisation against pooled standardisation}}, four states in each
case. Run length breaks at gaps in the record.}}
\label{{tab:res_clustering}}
\small
\begin{{tabular}}{{lrr}}
\toprule
 & Per node normalisation & Pooled standardisation \\
\midrule
Mutual information with node identity & {f(c_p.nmi_with_node)} & {f(c_s.nmi_with_node)} \\
Silhouette coefficient & {f(c_p.silhouette)} & {f(c_s.silhouette)} \\
Davies and Bouldin index & {f(c_p.davies_bouldin)} & {f(c_s.davies_bouldin)} \\
Transfer agreement, mean over folds (adjusted Rand) & {f(c_p.transfer_ari_mean)} & {f(c_s.transfer_ari_mean)} \\
Transfer agreement, worst fold (adjusted Rand) & {f(c_p.transfer_ari_min)} & {f(c_s.transfer_ari_min)} \\
Mean run length (hours) & {f(c_p.mean_run_hours, 2)} & {f(c_s.mean_run_hours, 2)} \\
Share of readings in the largest state & {f(c_p.largest_cluster_share)} & {f(c_s.largest_cluster_share)} \\
\bottomrule
\end{{tabular}}
\end{{table}}

The standard approach scores better on both internal indices, and the reason is
the one Section~\ref{{sec:nodedominance}} anticipated: its clusters are largely
the nodes themselves, which differ from one another more cleanly than conditions
within a node do. Its mutual information with node identity is {f(c_s.nmi_with_node)},
more than thirty times that of the proposed states, and its largest cluster holds
{pc(c_s.largest_cluster_share)} percent of all readings. The proposed states are
also the more transferable of the two: the partition obtained without a node
agrees more closely with the partition obtained with it, on average and in the
worst fold. The standard approach's states do last longer, {f(c_s.mean_run_hours, 2)} hours against
{f(c_p.mean_run_hours, 2)}, but its most node dominated cluster draws
{pc(c_s.most_node_dominated_cluster)} percent of its readings from a single node, against
{pc(c_p.most_node_dominated_cluster)} percent for the proposed states, and a cluster of that kind
changes only when that node's readings move into the range of another, so persistence
of this kind is inherited from the sensor and is not evidence about the soil. A better
silhouette and a lower Davies and Bouldin index are therefore not
evidence in favour of the standard approach here; they are the signature of the
location effect that the normalisation exists to remove. This is the central
methodological result of the study, and the remainder of the chapter concerns
what the location free states are and how reliably they can be assigned.
""")

def krow(r):
    nm = 'yes' if r.all_named else 'no'
    return (f"{int(r.k)} & {f(r.silhouette)} & {f(r.davies_bouldin)} & {f(r.nmi_node)} & {int(r.smallest)} & {nm} & "
            f"{int(r.folds_same_set)} of 5 & {f(r.ari_mean)} & {f(r.largest_share_shift, 2)} \\\\")
A(rf"""
\section{{Number of States}}
\label{{sec:res_k}}

Table~\ref{{tab:res_k}} reports every quantity that Sections~\ref{{sec:clustering}}
and~\ref{{sec:lono}} use to select the number of states, for each candidate from
three to six.

\begin{{table}}[htbp]
\centering
\caption[Selection of the number of states]{{\textbf{{Selection of the number of
states.}} Same set: folds in which the model fitted without the withheld node
forms the same set of named states as the reference model. Share shift: the
largest change, over all folds and states, in the share of the four training
nodes' readings assigned to a state, relative to the reference model.}}
\label{{tab:res_k}}
\small
\begin{{tabular}}{{rrrrrcccr}}
\toprule
$k$ & Silhouette & Davies Bouldin & MI with node & Smallest & Named & Same set & Mean ARI & Share shift \\
\midrule
""" + "\n".join(krow(r) for _, r in ks.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

The two internal indices agree: both favour five states, the silhouette
coefficient ties five and six, and both prefer four to three. Under the order of
precedence fixed in Section~\ref{{sec:clustering}}, however, the definition of a
state is applied before the indices, and on that test the candidates fare as
follows. Six states cannot be named by the rule: it names the saturated and the
rapid drainage clusters, but the remaining four divide the ordinary readings more
finely than the definition of a state provides for, and the solution fails the
interpretability requirement. Three, four, and five states can each be named,
and each fails the transfer criterion of Section~\ref{{sec:lono}} in one of the
five folds, in addition to the absence of the saturated state from Node 04, which
Section~\ref{{sec:res_lono}} discusses and which affects every candidate equally. With three states, when {SN(k3.largest_shift_fold)} is
withheld the cluster that the rule still names saturated no longer isolates the
saturated readings: it takes {pc(k3.largest_shift_fold_share)} percent of the four training
nodes' readings, against {pc(k3.largest_shift_reference_share)} percent under the reference model.
With four states, when Node 05 is withheld the division of the drier readings by
pH does not form, as Section~\ref{{sec:res_lono}} describes. With five states, the
named set is not reproduced when {SN(k5.largest_shift_fold)} is withheld, and the cluster
matched to the rapid drainage state takes {pc(k5.largest_shift_fold_share)} percent of the
training nodes' readings against {pc(k5.largest_shift_reference_share)} percent under the
reference model, so that state does not form either.

No candidate therefore satisfies the definition in full, and the rule of
precedence, which separates candidates only when both satisfy it, cannot decide
between them. The choice was accordingly made on grounds stated here, after the
results were seen, and it should be read as such. Four states are retained. The
single failure of the three state solution concerns the saturated state, on
which rules R1 and R2 act, whereas with four states the saturated state forms in
every fold. The state that five states add to four is small, holding
{int(k5.smallest)} readings, and {N['drainage_after_sat']} of the {N['drainage_n']}
readings that the training period partition of Section~\ref{{sec:res_training}}
places in the same state occur within three hours after a saturated reading, so
it largely describes the falling phase of the saturation event and not a
distinct condition. Among the three candidates that can be named, four states have the
second best value of each index. The
cost of this choice is that its fourth state, the division of the drier readings
by pH, is the least secure of the four, and it is reported as provisional
throughout this chapter.
""")

def srow(s):
    r = pr.loc[s]
    return (f"{TNAME[s]} & {th(r.n)} & {pc(r.share, 1)} & {int(r.nodes)} & {sg(r.moisture_rel)} & {sg(r.ph_rel)} & "
            f"{sg(r.temp_rel)} & {f(r.moisture_med, 1)} & {f(r.ph_med, 2)} & {pc(r.share_0710)} & {pc(r.share_test_week)} \\\\")
S_, M_, H_, W_ = pr.loc['Saturated'], pr.loc['Moist / cooler'], pr.loc['Drier / higher pH'], pr.loc['Drier / lower pH']
A(rf"""
\section{{Identified States}}
\label{{sec:res_states}}

Table~\ref{{tab:res_states}} profiles the four states of the reference model,
fitted on all {th(n_all)} readings, and Figure~\ref{{fig:res_time}} shows when
they occur.

\begin{{table}}[htbp]
\centering
\caption[Profiles of the four states]{{\textbf{{Profiles of the four states.}}
Normalised values are means, before clipping, in units of the node's
interquartile range from the node median. Raw medians pool all nodes. The last two columns give the share of
the state's readings falling between 07:00 and 11:00, four hours of the
twenty four, where one sixth would be expected by chance, and in the test week of 6 to 13 November, which holds
{pc(N['share_test_week_all'])} percent of all readings.}}
\label{{tab:res_states}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrrrrrrr}}
\toprule
 & & & & \multicolumn{{3}}{{c}}{{Normalised mean}} & \multicolumn{{2}}{{c}}{{Raw median}} & \multicolumn{{2}}{{c}}{{Percent in}} \\
\cmidrule(lr){{5-7}} \cmidrule(lr){{8-9}} \cmidrule(lr){{10-11}}
State & Readings & Percent & Nodes & Moisture & pH & Temp. & Moisture & pH & 07:00 to 11:00 & Test week \\
\midrule
""" + "\n".join(srow(s) for s in ORDER) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_states_time.png}}
\caption[When each state occurs]{{\textbf{{When each state occurs.}} (a) Share of
each day's readings, all nodes together; the shaded band is the test week. (b)
Distribution of each state's readings over the hours of the day.}}
\label{{fig:res_time}}
\end{{figure}}

\textbf{{Saturated.}} The smallest state holds {th(S_.n)} readings, {pc(S_.share, 1)} percent of
the record, on four of the five nodes; Node 04 contributes none. Its median raw
moisture of {f(S_.moisture_med, 1)} percent lies beyond the 100 percent ceiling of
the probe, and {N['sat_at_ceiling']} of its {th(S_.n)} readings are at or beyond the
ceiling. The other {int(S_.n) - N['sat_at_ceiling']} do not reach it, {N['sat_node02']} of them on Node 02,
whose moisture never exceeds {N['sat_node02_max_moisture']:.1f} percent; all but one lie at least
two and a half interquartile ranges above their node's median moisture, and the
exception is a reading whose pH is recorded as zero. The state is also marked by the pH channel: its
median pH is {f(S_.ph_med, 2)}, {N['sat_ph_below_3']} of its readings lie below the valid
minimum of 3, and all {rules['r5_valid_readings_ph_below_5_5']} readings in the record with a pH
between 3 and 5.5 fall within it, values that accompany the moisture excursion and
are not credible as soil pH. It is concentrated in the morning,
with {pc(S_.share_0710)} percent of its readings between 07:00 and 11:00, and it is
brief: its mean run lasts {f(TS['run_hours']['Saturated'], 2)} hours, and no episode
lasts longer than {rules['saturated_longest_h']} hours. As Appendix~\ref{{app:sensitivity}}
establishes, this character depends on the treatment of out of range readings,
and the state is accordingly interpreted as an excursion of the moisture reading
far above the node's own normal, in most of its readings to or beyond the ceiling
of the probe and accompanied by an implausible pH reading, which is operationally
worth flagging, and not as a measured degree of wetness or a detected irrigation
schedule. Not every ceiling reading falls in the state: of the
{N['ceiling_readings']} readings above 100 percent, {N['ceiling_by_state'].get('Saturated', 0)} do, and the other
{N['ceiling_by_state'].get('Moist / cooler', 0)} fall in the moist and cooler state, {N['ceiling_node03_moist']} of them on Node 03,
whose normal range is the widest of the five nodes.

\textbf{{Moist and cooler.}} The largest state, {pc(M_.share, 1)} percent of the
record, lies {f(M_.moisture_rel, 2)} interquartile ranges above the node's median
moisture and {f(-M_.temp_rel, 2)} below its median temperature, with pH near the
median. It occurs on every node and at every hour.

\textbf{{Drier and higher pH.}} This state lies below the median moisture and
above the median temperature and pH. It dominates the first two weeks of the
record, and its readings have a median date of {H_.median_date}.

\textbf{{Drier and lower pH.}} Like the previous state this one lies below the
median moisture. It differs from it chiefly in pH, which lies
{f(-W_.ph_rel, 2)} interquartile ranges below the node median, and it is somewhat
cooler. It is strongly
concentrated in time: {pc(W_.share_from_3nov)} percent of its readings occur from
3 November onwards and {pc(W_.share_test_week)} percent fall in the test week,
against {pc(N['share_test_week_all'])} percent of the record as a whole. The
concentration reflects a decline in pH over the month at three of the five
nodes. The weekly median pH fell from {wkph['Node-01'][0]:.2f} to {wkph['Node-01'][4]:.2f}
at Node 01, from {wkph['Node-03'][0]:.2f} to {wkph['Node-03'][4]:.2f} at Node 03, and from
{wkph['Node-05'][0]:.2f} to {wkph['Node-05'][4]:.2f} at Node 05, while at Nodes 02 and 04
it stayed within {max(N['weekly_ph_max_departure']['Node-02'], N['weekly_ph_max_departure']['Node-04']):.2f} of its first week's value
in every week. Because each node is normalised against the median of its whole
record, a node whose pH falls over the month reads as higher than its median early
in the month and lower late in it. The two drier states therefore
separate, in large part, the earlier and the later halves of the record.
Whether the decline is a change in the soil or a drift of the pH electrodes
cannot be determined from these data, since no reference pH measurement was
taken, and a laboratory measurement at the start and end of a deployment is the
evidence that would settle it.
""")

def prow(s):
    return (f"{TNAME[s]} & " + " & ".join(f"{P.loc[s, t]:.3f}" for t in ORDER)
            + f" & {th(T.loc[s].sum())} & {1 / max(1 - P.loc[s, s], 1e-9):.1f} & {TS['run_hours'][s]:.2f} \\\\")
A(rf"""
\section{{Persistence and Transitions}}
\label{{sec:res_transitions}}

Table~\ref{{tab:res_transitions}} gives the transition probabilities between
consecutive hourly readings, counting only pairs of readings from the same node
about one hour apart, as Section~\ref{{sec:transitions}} specifies, together with
the mean run length of each state.

\begin{{table}}[htbp]
\centering
\caption[Transition probabilities and run lengths]{{\textbf{{Hourly transition
probabilities}} (row: state in the current hour; column: state in the next) and
mean run length. Expected: the duration a first order chain implies, one over one
minus the self transition probability. Pairs spanning a gap are excluded, and a gap
ends a run, so observed runs are shorter than the expectation.}}
\label{{tab:res_transitions}}
\small
\begin{{tabular}}{{lrrrrrrr}}
\toprule
From \textbackslash{{}} to & Saturated & Moist, cooler & Drier, higher pH & Drier, lower pH & Pairs & Expected (h) & Run (h) \\
\midrule
""" + "\n".join(prow(s) for s in ORDER) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

The three large states satisfy the persistence requirement of
Section~\ref{{sec:background}}. Each is followed by itself in between
{pc(min(P.loc[s, s] for s in ORDER[1:]))} and {pc(max(P.loc[s, s] for s in ORDER[1:]))}
percent of hourly transitions, and each is held for about six hours on average.
Their mean runs are shorter than the self transition probabilities alone would
imply, because a gap in the record also ends a run, and
Appendix~\ref{{app:data}} reports that gaps are frequent.

The saturated state is not persistent in this sense, and is not expected to be:
it is followed by itself in only {pc(P.loc['Saturated', 'Saturated'])} percent of
transitions. When it ends, the node passes to the moist and cooler state in
{pc(TS['after_saturated']['Moist / cooler'])} percent of cases and to the drier and
higher pH state in {pc(TS['after_saturated']['Drier / higher pH'])} percent. In
{TS['recovery']['n']} of the {N['recovery_episodes_total']} saturated episodes, normalised moisture returned
to within half an interquartile range of the node median within twelve hours,
after a median of {TS['recovery']['median']:.1f} hours, a mean of {TS['recovery']['mean']:.1f} hours,
and at most {TS['recovery']['max']:.1f} hours; in the other
{N['recovery_episodes_total'] - TS['recovery']['n']} it did not do so within twelve hours. This is the recovery interval on which the
duration condition of rule R1 depends, and, as Section~\ref{{sec:rules}} cautions, it
measures how long the probe remains at its ceiling rather than how long the soil
remains waterlogged.
""")

def vpersrow(st):
    r_ = VPERS[VPERS.state == st].iloc[0]
    return (f"{TNAME[st]} & {r_.observed_mean_run_hours:.2f} & {r_.surrogate_median_hours:.2f}"
            f" & {r_.surrogate_95th_hours:.2f} & {'yes' if r_.exceeds_surrogate else 'no'} \\\\")

EXC = [st for st in ORDER if bool(VPERS[VPERS.state == st].iloc[0].exceeds_surrogate)]
NOEXC = [st for st in ORDER if st not in EXC]
def nlist(items, join='and'):
    items = [NAME.get(x, x) for x in items]
    if len(items) == 1: return items[0]
    return ', '.join(items[:-1]) + f' {join} ' + items[-1]
OBS = {st: float(VPERS[VPERS.state == st].iloc[0].observed_mean_run_hours) for st in ORDER}
B95 = {st: float(VPERS[VPERS.state == st].iloc[0].surrogate_95th_hours) for st in ORDER}

A(rf"""
Table~\ref{{tab:res_rednoise}} compares the observed mean run length of each state
with the run lengths of the {th(VJ['surrogates'])} surrogate records of
Section~\ref{{sec:transitions}}, which vary as smoothly as the readings do but
contain no states.

\begin{{table}}[htbp]
\centering
\caption[Run lengths against the surrogate records]{{\textbf{{Mean run length
against smooth variation.}} The observed mean run length of each state beside the
run lengths obtained on {th(VJ['surrogates'])} surrogate records in which each of
the three level channels is replaced by an autoregressive series matching the
mean, variance, and lag one autocorrelation of the channel it replaces, and the
two change features are recomputed from the surrogate moisture series. The
comparison is reported and is not a condition for retaining a state.}}
\label{{tab:res_rednoise}}
\small
\begin{{tabular}}{{lrrrc}}
\toprule
State & Observed (h) & Surrogate median (h) & Surrogate 95th (h) & Exceeds \\
\midrule
""" + "\n".join(vpersrow(st) for st in ORDER) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

{'The three large states' if set(EXC) == {st for st in ORDER if st != 'Saturated'} else 'The ' + nlist(EXC)} {'last' if len(EXC) > 1 else 'lasts'} longer than
smooth variation of the readings alone would produce, at {nlist([f'{OBS[st]:.2f}' for st in EXC])}
hours against reference levels of {nlist([f'{B95[st]:.2f}' for st in EXC])} hours
respectively. {'The ' + nlist(NOEXC) + (' states do not' if len(NOEXC) > 1 else ' state does not') + ', at ' + nlist([f'{OBS[st]:.2f}' for st in NOEXC]) + ' hours against ' + nlist([f'{B95[st]:.2f}' for st in NOEXC]) + '. ' if NOEXC else ''}The
margin in the saturated case is {abs(OBS['Saturated'] - B95['Saturated']):.2f} hours against a spread of
{float(VPERS[VPERS.state == 'Saturated'].iloc[0].surrogate_spread_hours):.2f} hours across the surrogate records, so the
comparison places that state below its reference level without placing it far
below. The result is in any case consistent with the account of it given above, as
a brief episode during which the probe stands at its ceiling rather than a
condition the soil holds.
"""
)

def vsrow(st):
    d_ = VSTAB[(VSTAB.state == st) & (VSTAB.resampling == 'whole days')].iloc[0]
    r_ = VSTAB[(VSTAB.state == st) & (VSTAB.resampling == 'single readings')].iloc[0]
    return (f"{TNAME[st]} & {th(d_.readings)} & {d_.mean_jaccard:.3f} ({d_.standard_error:.3f})"
            f" & {r_.mean_jaccard:.3f} ({r_.standard_error:.3f}) & {d_.verdict} \\\\")

VD = {st: VSTAB[(VSTAB.state == st) & (VSTAB.resampling == 'whole days')].iloc[0] for st in ORDER}
VR = {st: VSTAB[(VSTAB.state == st) & (VSTAB.resampling == 'single readings')].iloc[0] for st in ORDER}
V_STABLE = [st for st in ORDER if VD[st].mean_jaccard >= 0.75]
V_LOWEST = min(ORDER, key=lambda st: VD[st].mean_jaccard)
V_GAP = abs(VD[V_LOWEST].mean_jaccard - 0.75) / max(VD[V_LOWEST].standard_error, 1e-9)
V_SIDE = 'above' if VD[V_LOWEST].mean_jaccard >= 0.75 else 'below'
V_MARGIN = (f'Its margin {V_SIDE} the threshold is {V_GAP:.1f} standard errors, so the test places '
            f'it {V_SIDE} the threshold securely, though by the narrowest margin of the four.'
            if V_GAP >= 2 else
            f'Its margin {V_SIDE} the threshold is {V_GAP:.1f} standard errors, so the test places it '
            f'{V_SIDE} the threshold but not securely, which is consistent with the provisional standing '
            f'that Section~\\ref{{sec:res_lono}} gives that state on other grounds.')
V_OPT = float(np.mean([VR[st].mean_jaccard - VD[st].mean_jaccard for st in ORDER]))

A(rf"""
\section{{Stability of the States under Resampling}}
\label{{sec:res_stability}}

Table~\ref{{tab:res_stability}} reports the mean Jaccard coefficient of each state
over the {th(VJ['resamples'])} resamples specified in Section~\ref{{sec:stability}},
under resampling of whole days and of single readings, and the left panel of
Figure~\ref{{fig:res_validation}} shows the same values against the retention
threshold.

\begin{{table}}[htbp]
\centering
\caption[Stability of the states under resampling]{{\textbf{{Stability under
resampling.}} Mean Jaccard coefficient of each state over {th(VJ['resamples'])}
resamples, with its standard error in brackets. The criterion of
Section~\ref{{sec:stability}} is applied to the values obtained by resampling whole
days: 0.75 or more retains a state and 0.5 or less marks it dissolved.}}
\label{{tab:res_stability}}
\small
\begin{{tabular}}{{lrrrl}}
\toprule
State & Readings & Whole days & Single readings & Verdict \\
\midrule
""" + "\n".join(vsrow(st) for st in ORDER) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_validation.png}}
\caption[Stability and recognisability of the states]{{\textbf{{Stability and
recognisability.}} Left: mean Jaccard coefficient of each state under resampling of
whole days, with its standard error, and the value obtained by resampling single
readings shown as a hollow marker. Right: prediction strength of each state on each
withheld node, one point per node, with the shaded band giving the range of the
reference levels obtained from the fold minima of the surrogate records.}}
\label{{fig:res_validation}}
\end{{figure}}

{'All four states' if len(V_STABLE) == len(ORDER) else (WORD[len(V_STABLE)] + ' of the four states') if V_STABLE else 'None of the four states'} reach the retention
threshold of 0.75 under resampling of whole days.
The lowest value belongs to the {NAME[V_LOWEST]} state, {VD[V_LOWEST].mean_jaccard:.3f}
with a standard error of {VD[V_LOWEST].standard_error:.3f}. {V_MARGIN}
Resampling single readings raises every value, by {V_OPT:.3f} on average, which is
the optimism that Section~\ref{{sec:stability}} anticipates when neighbouring
readings are near duplicates, and it is the reason the criterion is applied to the
day blocks and not to these.

The saturated state records the highest stability of the four,
{VD['Saturated'].mean_jaccard:.3f}, on {th(VD['Saturated'].readings)} readings. A small
group of extreme and well separated readings is recovered by the algorithm
whenever the resample contains any of it, so this value measures how distinct the
group is and not whether it describes the soil; the reading of that state given in
Section~\ref{{sec:res_states}}, as the moisture channel at its ceiling, is unaffected
by it. Stability is in any case a necessary and not a sufficient condition, as
Section~\ref{{sec:stability}} records from \cite{{hennig2007}}: a method that splits
one true group in the same way on every resample is stable and wrong. What bears on
that possibility here is not this test but the comparison of candidate numbers of
states in Section~\ref{{sec:res_k}}, where the drier readings were examined both as
one state and as two, and the requirement that every state be interpretable against
published soil processes.

\textbf{{Departure from the plan.}} The criteria applied in this section and in
Section~\ref{{sec:res_lono}} are not the criteria with which this study began. The
original plan required a state to recur on a withheld node in comparable
proportions and with comparable run lengths, without stating what counted as
comparable, so that no value could have decided the question either way. The
criteria were replaced, after a first analysis of the record had been carried out
under the original wording, by the published criteria that Sections
\ref{{sec:stability}} and~\ref{{sec:lono}} now state. The thresholds were taken from
the sources cited there and not chosen to suit the values obtained here. The
revision also considered a permutation test of persistence, comparing the self
transition probability of a state with the share of readings in it, and did not
adopt it, for the reason Section~\ref{{sec:transitions}} gives: on a record of
hourly readings that comparison is passed by any division, so it would have
distinguished nothing. The thresholds that were adopted are nevertheless not
thresholds fixed before the data were seen, and nothing in this chapter should be
read as though they were.
"""
)

def lrow(n):
    r = lono.loc[n]
    cells = " & ".join(f"{shr(n, s, 'share_train_nodes'):.2f} / {shr(n, s, 'share_heldout'):.2f}" for s in ORDER)
    rl_ = N['lono_run_lengths'][n]
    return (f"{SN(n)} & {'yes' if r.same_state_set else 'no'} & {cells} & {rl_['fold']:.1f} / {rl_['reference']:.1f}"
            f" & {r.hour_distribution_difference:.2f} & {r.agreement_with_full_model_ARI:.3f} \\\\")
A(rf"""
\section{{Transfer to a Withheld Node}}
\label{{sec:res_lono}}

Table~\ref{{tab:res_lono}} reports, fold by fold, the presence criterion of
Section~\ref{{sec:lono}} together with the comparisons that section records as
descriptive, and Figure~\ref{{fig:res_transfer}} compares, for every state and every
withheld node, the share of that node's readings assigned to the state by the
model fitted without it against the share assigned by the reference model.

\begin{{table}}[htbp]
\centering
\caption[Leave one node out transfer of the states]{{\textbf{{Leave one node out
transfer.}} Same set: whether the model fitted on the other four nodes forms the
same named states. Each state cell gives the share of the four training nodes'
readings and the share of the withheld node's readings assigned to the state.
Run: mean run length in hours on the withheld node, under the fold's model and
under the reference model. Hour: the largest difference, over the states compared,
between the fold's and the reference model's distribution of a state's readings over
the hours of the day, measured as the share of the readings that would have to move
between hours. ARI: agreement with the reference model on the withheld node.}}
\label{{tab:res_lono}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccccccr}}
\toprule
Withheld & Same set & Saturated & Moist, cooler & Drier, higher pH & Drier, lower pH & Run (h) & Hour & ARI \\
\midrule
""" + "\n".join(lrow(n) for n in NODES) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.72\textwidth]{{figs/ch4_transfer.png}}
\caption[Transfer of each state to each withheld node]{{\textbf{{Transfer of each
state to each withheld node.}} Points on the diagonal are states whose share of
the withheld node is the same whether or not that node was used to derive them.}}
\label{{fig:res_transfer}}
\end{{figure}}

When Nodes 01, 02, and 03 are withheld, the presence criterion is met and the
descriptive comparisons agree with it. The same four states form on the other four
nodes, all four appear on the withheld node, the withheld node's shares are comparable with those of the
training nodes, its mean run length under the fold's model differs from that under
the reference model by {max(abs(N['lono_run_lengths'][n]['fold'] - N['lono_run_lengths'][n]['reference']) for n in ['Node-01', 'Node-02', 'Node-03']):.2f} hours at most, and no state's
distribution over the hours of the day moves by more than
{max(lono.loc[n, 'hour_distribution_difference'] for n in ['Node-01', 'Node-02', 'Node-03']):.2f} of its readings. The saturated state is
reproduced in number: when Node 01, which
contributes {int(lono.loc['Node-01', 'saturated_rows'])} of its {int(S_.n)} readings, is withheld,
the state still captures {shr('Node-01', 'Saturated', 'share_heldout'):.3f} of that node's
readings against {shr('Node-01', 'Saturated', 'share_heldout_reference'):.3f} under the
reference model, although its centre moves by
{shr('Node-01', 'Saturated', 'matched_centre_distance'):.2f} units, the largest displacement
of any state in the three folds that meet the criterion, because so few readings
define it.

When Node 04 is withheld, the saturated state is absent from it, which the
presence criterion, taken literally, does not allow. The reference model, fitted with Node
04 included, assigns none of that node's readings to the state either. Node 04
has a high median moisture, {N['node04']['median']:.1f} percent, and an interquartile range of
{N['node04']['iqr']:.1f} percentage points, so that even its highest reading,
{N['node04']['max']:.1f} percent, lies only {N['node04']['max_rel']:.1f} interquartile ranges above its median,
against a state centre above four. The
absence therefore describes the record of that node and not a failure of the
state to transfer. It is reported as a limitation of a criterion that was written
for conditions every node experiences, and it applies equally to every candidate
number of states in Table~\ref{{tab:res_k}}.

When Node 05 is withheld, the presence criterion fails. The four remaining nodes do not form
the drier and lower pH state: only {pc(shr('Node-05', 'Drier / lower pH', 'share_train_nodes'), 1)}
percent of their readings are assigned to the cluster matched to it, against
{pc(ref_share_on_train5, 1)} percent of the same readings under the reference model, and
that cluster's centre lies {shr('Node-05', 'Drier / lower pH', 'matched_centre_distance'):.2f}
units from the reference centre, against at most
{max(shr('Node-05', s, 'matched_centre_distance') for s in ORDER[:3]):.2f} for the other
three states of the fold. The model fitted without Node 05 has instead isolated a
small group of readings in which moisture falls rapidly, the rapid drainage state
of Section~\ref{{sec:clustering}}, and merged the two drier states into one. On
the withheld node, the readings that the reference model places in the lower pH
state are consequently assigned to the drier and higher pH state, which is the
point far above the diagonal in Figure~\ref{{fig:res_transfer}}, agreement with the
reference model falls to {lono.loc['Node-05', 'agreement_with_full_model_ARI']:.3f}, and the hour of day
distribution of that state moves by {lono.loc['Node-05', 'hour_distribution_difference']:.2f} of its readings, an order
of magnitude more than in any other fold.

The two four state solutions are close competitors. On the readings of the four
nodes other than Node 05, the solution with a rapid drainage state has a within
cluster sum of squares of {N['inertia_no5_best']:.1f}, and the solution that divides the
drier readings by pH, obtained by starting the algorithm from the reference
centres, has {N['inertia_no5_phsplit']:.1f}, a difference of
{pc((N['inertia_no5_phsplit'] - N['inertia_no5_best']) / N['inertia_no5_best'], 1)}
percent. Which of them the algorithm returns depends on which readings are
present. Taken with the temporal concentration reported in
Section~\ref{{sec:res_states}}, the result means that the moist and cooler state
recurs on every withheld node, and so does a drier condition, divided by pH in four
folds and undivided in the fifth; the saturated event recurs wherever a node
produces readings far enough from its normal; and the division of the drier
readings by pH is sustained by the late part of the record and by Node 05 in
particular.

Section~\ref{{sec:lono}} states that a state confined to the nodes used to derive it
is not reported as a state, and in the Node 05 fold the drier and lower pH state is
close to that condition. It is nevertheless retained here, as a provisional state
that is not treated as established, for two reasons: it forms in the reference model
and in four of the five folds, and removing it would leave the three state solution,
whose own failure falls on the saturated state that rules R1 and R2 act upon. The
retention is a departure from the plan of Chapter Three, it is listed among the
departures at the start of this chapter, and every result that depends on the state
is reported with the state marked provisional.
""")

def vprow(st):
    v = VPS[VPS.state == st].set_index('held_out')
    return (f"{TNAME[st]} & " + " & ".join(
        f"{v.loc[n, 'prediction_strength']:.3f} ({int(v.loc[n, 'readings_on_withheld_node'])})"
        for n in NODES) + f" & {v.prediction_strength.min():.3f} \\\\")

PS_MIN = {st: float(VPS[VPS.state == st].prediction_strength.min()) for st in ORDER}
PS_MAX = {st: float(VPS[VPS.state == st].prediction_strength.max()) for st in ORDER}
PS_REACH = int(VPS.reaches_threshold.sum())
PS_TOTAL = int(len(VPS))
PS_BY_STATE = {st: int(VPS[VPS.state == st].reaches_threshold.sum()) for st in ORDER}
PS_NONE = [st for st in ORDER if PS_BY_STATE[st] == 0]
BEAT = [n for n in NODES if bool(VFOLD.loc[n, 'minimum_beats_surrogate'])]
FAILN = [n for n in NODES if n not in BEAT]
NARROW = [n for n in BEAT if VFOLD.loc[n, 'observed_minimum'] - VFOLD.loc[n, 'surrogate_95th_minimum'] < 0.02]
POOL_LO, POOL_HI = VFOLD.surrogate_95th_pooled.min(), VFOLD.surrogate_95th_pooled.max()
PS_ABOVE_POOL = int(VPS.beats_pooled_surrogate.sum())
N1 = VPS[VPS.held_out == 'Node-01'].sort_values('readings_on_withheld_node', ascending=False)
N1_TOTAL = int(N1.readings_on_withheld_node.sum())
N1_BIG = int(N1.readings_on_withheld_node.iloc[0])
_rest = [str(int(x)) for x in N1.readings_on_withheld_node.iloc[1:]]
N1_REST = ', '.join(_rest[:-1]) + ' and ' + _rest[-1]
N1_SMALL = int((N1.readings_on_withheld_node < 20).sum())
LARGE = [st for st in ORDER if st != 'Saturated']
BESTFOLD = max(NODES, key=lambda n: VPS[(VPS.held_out == n)
                                        & (VPS.state.isin(LARGE))].prediction_strength.min())
BF = sorted(VPS[(VPS.held_out == BESTFOLD) & (VPS.state.isin(LARGE))].prediction_strength)
KDIFF = [n for n in NODES if not bool(VFOLD.loc[n, 'agrees_with_reference_k'])]
KVALS = sorted({int(VFOLD.loc[n, 'reselected_k']) for n in KDIFF})
KCOUNT = {k: [n for n in KDIFF if int(VFOLD.loc[n, 'reselected_k']) == k] for k in KVALS}

A(rf"""
\subsection{{Recognisability of the States on the Withheld Node}}
\label{{sec:res_recognisability}}

Table~\ref{{tab:res_recognisability}} reports the prediction strength of each state
on each withheld node, defined in Section~\ref{{sec:lono}}, with the number of the
withheld node's readings on which each proportion is computed, the minimum over
the states of each fold, which is the statistic the original procedure defines,
and the two reference levels obtained from the surrogate records. The right panel
of Figure~\ref{{fig:res_validation}} shows the same values.

\begin{{table}}[htbp]
\centering
\caption[Recognisability of each state on each withheld node]{{\textbf{{Prediction
strength of each state on each withheld node.}} Each cell gives the share of pairs
of the withheld node's readings that its own clustering and the training centroids
both place together, with the number of readings in brackets. Fold minimum: the
smallest of the four, the statistic of \cite{{tibshirani2005}}. Reference levels:
the 95th percentile of the fold minima of {th(VJ['surrogates'])} surrogate records,
and the 95th percentile of their proportions pooled over states. Selected states:
the number of states the four training nodes of the fold would select on their
own.}}
\label{{tab:res_recognisability}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccccr}}
\toprule
State & Node 01 & Node 02 & Node 03 & Node 04 & Node 05 & Lowest \\
\midrule
""" + "\n".join(vprow(st) for st in ORDER) + r"""
\addlinespace
Fold minimum & """ + " & ".join(f"{VFOLD.loc[n, 'observed_minimum']:.3f}" for n in NODES) + r""" & \\
Reference, minimum & """ + " & ".join(f"{VFOLD.loc[n, 'surrogate_95th_minimum']:.3f}" for n in NODES) + r""" & \\
Reference, pooled & """ + " & ".join(f"{VFOLD.loc[n, 'surrogate_95th_pooled']:.3f}" for n in NODES) + r""" & \\
Selected states & """ + " & ".join(str(int(VFOLD.loc[n, 'reselected_k'])) for n in NODES) + r""" & \\""" + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

The threshold of 0.8 is not met by any state in every fold. It is reached in
{numw(PS_REACH)} of the {numw(PS_TOTAL)} combinations of a state and a withheld node:
{nlist([f'{numw(PS_BY_STATE[st])} by the {NAME[st]} state' for st in ORDER if PS_BY_STATE[st]])},
and in no fold at all by the {nlist([NAME[st] for st in PS_NONE])}
{'states' if len(PS_NONE) > 1 else 'state'}. Of the {numw(PS_REACH)} values that do reach
it, {numw(int(((VPS.reaches_threshold == True) & (VPS.state == 'Saturated')).sum()))} are the
saturated state on {th(int(VPS[(VPS.state == 'Saturated') & (VPS.reaches_threshold == True)].readings_on_withheld_node.max()))}
readings or fewer, where the proportion is unstable. Applied as a condition for
retention, the measure would remove every state and leave the chapter with none.

The comparison with the surrogate records is what the measure itself can be judged
on, and the matched comparison, made on the fold minimum, is the one that bears on
it. The observed minimum exceeds the 95th percentile of the surrogate minima in
{numw(len(BEAT))} of the five folds and falls below it in the
{nlist([SN(n) for n in FAILN])} {'folds' if len(FAILN) > 1 else 'fold'}, where
{VFOLD.loc[FAILN[0], 'observed_minimum']:.3f} stands against
{VFOLD.loc[FAILN[0], 'surrogate_95th_minimum']:.3f}. The folds that exceed it do so by margins of
{nlist([f"{VFOLD.loc[n, 'observed_minimum'] - VFOLD.loc[n, 'surrogate_95th_minimum']:.3f}" for n in BEAT])},
{numw(len(NARROW))} of them by less than 0.02, so the result is positive but slight.
On this evidence the grouping of {numw(len(BEAT))} folds is distinguishable from records
that contain no states and the grouping of the
{nlist([SN(n) for n in FAILN])} {'folds is' if len(FAILN) > 1 else 'fold is'} not.

The state by state comparison against the pooled reference level says less, and
Section~\ref{{sec:lono}} records why. That level lies between {POOL_LO:.3f} and
{POOL_HI:.3f}, above {'every observed value' if PS_ABOVE_POOL == 0 else 'all but ' + WORD[PS_ABOVE_POOL].lower() + ' of the twenty observed values'},
because a cluster holding few readings has nearly all of its pairs held together
whatever record it comes from, which lifts the upper tail of the pooled surrogate
distribution towards one. The same effect produces the values of 1.000 in the
table, each computed on seventeen readings. The pooled comparison is reported for
completeness and carries no weight for or against any single state.

The composition of the folds shows where the measure loses its grip. The
computation requires the withheld node to be clustered on its own, and a single
node's readings do not divide into four comparable groups. When Node 01 is
withheld, its own clustering places {th(N1_BIG)} of its {th(N1_TOTAL)} readings in
one cluster and {N1_REST} in the other three, so {numw(N1_SMALL)} of its
four proportions rest on fewer than twenty readings, and that fold is the one
whose minimum falls below its reference level. This follows from the
normalisation of Section~\ref{{sec:normalisation}}, which expresses every reading
relative to its own node's median and interquartile range, so that the readings of
one node are centred on the origin whether or not the states are meaningful.

Read as a description, the measure ranks the nodes. {SN(BESTFOLD)} is the node whose
own readings come closest to reproducing the grouping: its three states other than
the saturated one reach {nlist([f'{v:.3f}' for v in BF])}, the highest such set in
any fold. That is also the node whose absence prevents the remaining four from
forming the drier and lower pH state at all, which is consistent with the account
given above: the division of the drier readings rests on Node 05 and on the late
part of the record.

Section~\ref{{sec:lono}} also requires the number of states to be selected again on
the four training nodes of each fold, under the mechanical form of the selection
rule stated there. {WORD[len(KDIFF)] if KDIFF else 'None'} of the five folds
{'select' if len(KDIFF) != 1 else 'selects'} a number other than four:
{nlist([numw(k) + ' when ' + nlist([SN(n) for n in KCOUNT[k]], 'or') + (' is withheld' if len(KCOUNT[k]) == 1 else ' is withheld') for k in KVALS])}.
This repeats within the folds what Section~\ref{{sec:res_k}} reports for the whole
record, where the internal indices favoured more than four states and the naming
rule and the transfer test decided four, and it is reported for the same reason:
the choice of four is a judgement made under a stated rule of precedence and not a
value the indices force.

Taken with Section~\ref{{sec:res_stability}}, the two procedures place the states as
follows. The grouping is a property of the record of the five nodes considered
together. It survives resampling of that record, the same states form when any one
node is removed from the derivation, the model fitted on four nodes assigns the
fifth node's readings to those states in comparable proportions, with the
qualifications already reported for Nodes 04 and 05, and four of the five folds
are distinguishable from records containing no states at all. It is not a property
that a single node's record reproduces in isolation, and this study does not claim
that one node, analysed on its own, would recover these states. The claim the
evidence supports is the one the research question asks: that the states recur on a
node that was not used to derive them, in the sense that the readings of such a
node are assigned to them as the readings of the other nodes are.
"""
)

drain = tp.loc['Rapid drainage']
A(rf"""
\section{{Hyperparameter Tuning of the State Classifier}}
\label{{sec:res_tuning}}

\subsection{{States of the Training Period}}
\label{{sec:res_training}}

As Section~\ref{{sec:tuning}} specifies, the classifier is trained to reproduce the
states of the k means solution fitted on the training period, 13 October to 5
November, which never sees the test week. That solution differs from the
reference model in its fourth state, for the reason Section~\ref{{sec:res_lono}}
has just given: most of the readings that form the lower pH state lie in the
test week, so the training period alone does not support that division, and the
algorithm isolates a rapid drainage state instead. Its four states are saturated
({int(tp.loc['Saturated', 'n_train'])} training readings, {int(tp.loc['Saturated', 'n_test'])} test readings), rapid drainage
({int(drain.n_train)} and {int(drain.n_test)}), moist and cooler ({th(tp.loc['Moist / cooler', 'n_train'])} and {int(tp.loc['Moist / cooler', 'n_test'])}), and a single drier
state ({th(tp.loc['Drier', 'n_train'])} and {int(tp.loc['Drier', 'n_test'])}). The rapid drainage centre lies close to the
node medians in moisture, pH, and temperature, and {f(-drain.dmoist_3h, 2)} interquartile
ranges below them in its three hour moisture change. Of the {N['drainage_n']}
readings assigned to it, {N['drainage_after_sat']} follow a saturated reading within three
hours, and all but one lie on Nodes 01, 03, and 05.

The naming rule as first written in Section~\ref{{sec:clustering}} did not provide
for such a cluster, and applied literally it would have named this one the drier
and lower pH state, although its moisture and pH both lie above their node
medians. The rule was therefore extended to name a cluster defined by a rapid
fall in moisture as rapid drainage, and Section~\ref{{sec:clustering}} states the
extended rule. The extension changes names only. No assignment, and therefore no
score in this chapter, depends on it.

\subsection{{Search and Selected Configurations}}
\label{{sec:res_search}}

Every configuration was fitted on the sub training window, 13 to 29 October, and
scored on the validation window, 30 October to 5 November, by exhaustive grid search.
The two windows differ in composition: the validation week holds
{N['val_counts']['Moist / cooler']} moist and cooler readings out of {N['n_val']}, against {N['sub_counts']['Moist / cooler']} out of
{th(N['n_sub'])} in the sub training window, and only {N['val_counts']['Saturated']} saturated and
{N['val_counts']['Rapid drainage']} rapid drainage readings. Table~\ref{{tab:res_config}}
gives the search range, the selected value, and the resulting scores for each model.

\begin{{table}}[htbp]
\centering
\caption[Hyperparameter search and selected configurations]{{\textbf{{Hyperparameter
search and selected configurations.}} Selection by macro F1 on the validation
window; among configurations within 0.005 of the best, the simplest is chosen.}}
\label{{tab:res_config}}
\small
\begin{{tabular}}{{llll}}
\toprule
Model & Hyperparameter & Search range & Selected \\
\midrule
Random Forest & number of trees & fixed & 400 \\
 & class weights & fixed & balanced \\
 & minimum leaf size & 1, 2, 3, 5, 10, 20 & {rf_cfg['tuned']['min_samples_leaf']} \\
 & features per split & 1, 2, 3, 5 & {rf_cfg['tuned']['max_features']} \\
 & maximum depth & 4, 8, 16, none & {rf_cfg['tuned']['max_depth']} \\
 & \multicolumn{{3}}{{l}}{{{rf_cfg['n_configs']} configurations; training macro F1 {rf_cfg['train_macro_f1']:.4f}, validation macro F1 {rf_cfg['val_macro_f1']:.4f}}} \\
\addlinespace
Classification tree & class weights & fixed & balanced \\
 & minimum leaf size & 1, 2, 3, 5, 10, 20 & {dt_cfg['tuned']['min_samples_leaf']} \\
 & maximum depth & 2, 4, 6, 8, 12, none & {dt_cfg['tuned']['max_depth']} \\
 & \multicolumn{{3}}{{l}}{{{dt_cfg['n_configs']} configurations; training macro F1 {dt_cfg['train_macro_f1']:.4f}, validation macro F1 {dt_cfg['val_macro_f1']:.4f}}} \\
\addlinespace
Logistic regression & class weights, solver & fixed & balanced, lbfgs \\
 & inverse regularisation $C$ & $10^{{-3}}$ to $10^{{3}}$, seven values & {lr_cfg['tuned']['C']:g} \\
 & \multicolumn{{3}}{{l}}{{{lr_cfg['n_configs']} configurations; training macro F1 {lr_cfg['train_macro_f1']:.4f}, validation macro F1 {lr_cfg['val_macro_f1']:.4f}}} \\
\addlinespace
Forest regressor & number of trees, minimum leaf size & fixed 300; 5, 20, 50, 100, 200 & per horizon, Table~\ref{{tab:res_forecast}} \\
\addlinespace
\multicolumn{{4}}{{l}}{{\textit{{Solver settings, fixed for every configuration}}}} \\
k means & restarts, algorithm, tolerance & fixed & 30, Lloyd, $10^{{-4}}$ \\
Logistic regression & solver, tolerance, iteration limit & fixed & lbfgs, $10^{{-4}}$, 5{{,}}000 \\
Forest and tree & split criterion, splitter & fixed & Gini impurity, best split \\
All models & random seed, library & fixed & 0, scikit learn 1.8.0 \\
\bottomrule
\end{{tabular}}
\end{{table}}

For the forest, the best validation score, {rf_cfg['best_val_macro_f1_any']:.4f}, was reached with
a minimum leaf size of one; {rf_cfg['n_within_tie_band']} configurations lay within the tie band, and
the selection rule took the one with the larger leaf, which scored {rf_cfg['val_macro_f1']:.4f}.
For the single tree, three configurations tied and the rule took the shallowest;
for logistic regression no other value lay within the band. Across all
{N['rf_grid']['n']} forest configurations, validation macro F1 ranged from
{N['rf_grid']['val_min']:.3f} to {N['rf_grid']['val_max']:.3f}, with a median of
{N['rf_grid']['val_median']:.3f}.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_tuning_curves.png}}
\caption[Validation curves and learning curve]{{\textbf{{Validation curves and
learning curve.}} (a) to (c) each vary one forest hyperparameter with the other
two at their selected values; (d) varies the depth of the single tree at its
selected leaf size; (e) varies $C$ for logistic regression; (f) trains the
selected forest on stratified random subsets of the sub training window, five
repeats for each size, with the band showing one standard deviation.}}
\label{{fig:res_tuning}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_loss_curves.png}}
\caption[Training and validation loss curves]{{\textbf{{Training and validation
loss as trees are added}}, for the selected forest and for an unconstrained
forest with a minimum leaf size of one, no depth limit, and all five features at
every split. Log loss is defined in Section~\ref{{sec:metrics}}.}}
\label{{fig:res_loss}}
\end{{figure}}

Figure~\ref{{fig:res_loss}} gives the training and validation loss curves that
Section~\ref{{sec:tuning}} describes, with the number of trees in place of training
epochs. For the selected forest, validation log loss falls steeply over the first
few trees and is essentially flat from fifty trees onwards, at
{lt.loc[50, 'val_log_loss']:.3f} with fifty trees and {lt.loc[400, 'val_log_loss']:.3f} with four
hundred, while training log loss settles near {lt.loc[400, 'train_log_loss']:.2f}. From fifty trees onwards the
validation loss varies only between {lt.loc[50:, 'val_log_loss'].min():.3f} and {lt.loc[50:, 'val_log_loss'].max():.3f}, so adding trees
does not raise it materially, as Section~\ref{{sec:overfitting}} expects, and the fixed choice
of four hundred trees costs computation but not accuracy.
""")

fin = N['final_fit']
vcl = pd.DataFrame(N['valcurve_min_samples_leaf']).set_index('min_samples_leaf')
vcd = pd.DataFrame(N['valcurve_max_depth']).set_index('max_depth')
vcf = pd.DataFrame(N['valcurve_max_features']).set_index('max_features')
dtg = pd.DataFrame(N['dt_grid']); dt1 = dtg[dtg.min_samples_leaf == 1].set_index('max_depth')
lrg = pd.DataFrame(N['lr_grid']).set_index('C')
A(rf"""
\subsection{{Underfitting, Overfitting, and Sensitivity}}
\label{{sec:res_fitting}}

\textbf{{Overfitting.}} The selected forest scores {rf_cfg['train_macro_f1']:.3f} on the readings
it was trained on and {rf_cfg['val_macro_f1']:.3f} on the validation window, a gap of
{rf_cfg['train_macro_f1'] - rf_cfg['val_macro_f1']:.3f}. Refitted on the whole training period and
scored on the test week, the corresponding values are {fin['random_forest']['train_macro_f1']:.3f} and
{fin['random_forest']['test_macro_f1']:.3f}. The unconstrained forest of
Figure~\ref{{fig:res_loss}} shows what the constraints prevent: it reproduces its
training readings perfectly, with a training log loss of {lu.loc[400, 'train_log_loss']:.3f} and a
training macro F1 of {lu.loc[400, 'train_macro_f1']:.3f}, but reaches a validation macro F1 of only
{lu.loc[400, 'val_macro_f1']:.3f}. The single tree shows the same pattern with depth: from a
depth of twelve its training macro F1 is {dt1.loc[12, 'train_macro_f1']:.3f} while its
validation score stays near {dt1.loc[12, 'val_macro_f1']:.2f}, and the rule selected a depth of six,
where validation performance peaks at {dt1.loc[6, 'val_macro_f1']:.3f}. For the forest, allowing
every feature at every split lowers validation macro F1 from {vcf.loc[1, 'val_macro_f1']:.3f} to
{vcf.loc[5, 'val_macro_f1']:.3f} while leaving training macro F1 near
{vcf.loc[5, 'train_macro_f1']:.2f}, which is the loss of decorrelation between trees that
Section~\ref{{sec:overfitting}} describes.

\textbf{{Underfitting.}} Underfitting appears at the other end of each range. A
forest limited to a depth of four scores {vcd.loc[4, 'train_macro_f1']:.3f} on training and
{vcd.loc[4, 'val_macro_f1']:.3f} on validation; one with a minimum leaf size of twenty scores
{vcl.loc[20, 'train_macro_f1']:.3f} and {vcl.loc[20, 'val_macro_f1']:.3f}, because the two event
states have only {N['sub_counts']['Rapid drainage']} and {N['sub_counts']['Saturated']} readings in the sub training window, and leaves of at
least twenty readings can barely isolate them. A single tree of depth two
reaches only {dt1.loc[2, 'val_macro_f1']:.3f} on validation. Logistic regression underfits under
the strongest regularisation, scoring {lrg.loc[0.001, 'train_macro_f1']:.3f} on training and
{lrg.loc[0.001, 'val_macro_f1']:.3f} on validation at $C = 0.001$, improves steadily to
$C = 10$, and declines slightly beyond it.

\textbf{{Amount of data.}} The learning curve in Figure~\ref{{fig:res_tuning}}(f) rises
from {lc.loc[0.1, 'val']:.3f} with {int(lc.loc[0.1, 'n'])} training readings to {lc.loc[1.0, 'val']:.3f}
with all {th(lc.loc[1.0, 'n'])}, while training performance stays between
{lc.train.min():.3f} and {lc.train.max():.3f} throughout. The full size point, {lc.loc[1.0, 'val']:.3f}, uses the same
readings and configuration as the selected forest, which scores {rf_cfg['val_macro_f1']:.3f}; the only
difference is the order in which the readings are presented, which changes the
bootstrap samples drawn under the fixed seed. The difference, {abs(lc.loc[1.0, 'val'] - rf_cfg['val_macro_f1']):.3f}, indicates how much the
forest moves from one random draw to another, and it is about the width of the tie
band. Section~\ref{{sec:res_seeds}} measures that spread properly. The curve is still
rising at the full size point, so a longer record may improve the classifier further,
although by how much cannot be read from a curve that ends where the data end.

\textbf{{Sensitivity.}} Validation performance is most sensitive to the number of
features considered at each split, which is best at one, and to the minimum leaf
size beyond three, and is least sensitive to depth once depth reaches eight. The
differences among the best configurations are, however, smaller than the
resolution of the validation window. The rapid drainage state has {N['val_counts']['Rapid drainage']} validation
readings and the saturated state {N['val_counts']['Saturated']}, and a single missed rapid drainage
reading lowers macro F1 on the validation window by about {one_error:.3f}, which is
several times the tie band and more than the difference of
{rf_cfg['best_val_macro_f1_any'] - rf_cfg['val_macro_f1']:.3f} between the best and the selected
configurations. The selected configuration should therefore be read as one of a
small group of practically equivalent settings, not as a sharply determined
optimum, and it is for such cases that the tie is broken in favour of the simpler
configuration.

\textbf{{Probability estimates.}} The constraints that improve the forest's macro
F1 make its probabilities less confident. The unconstrained forest reaches a lower
validation log loss, {lu.loc[400, 'val_log_loss']:.3f} against {lt.loc[400, 'val_log_loss']:.3f}, because its
probabilities are sharper, even though it assigns more readings to the wrong
state. On the test week the selected forest has a log loss of
{fin['random_forest']['test_log_loss']:.3f} and logistic regression one of
{fin['logistic_regression']['test_log_loss']:.3f}. Selection by macro F1 therefore did not
select for well calibrated probabilities, and Section~\ref{{sec:discussion}} returns to the
consequence of this for the first of the two reasons Section~\ref{{sec:classifier}} gives for
using a classifier at all.
""")

def crow(m):
    r = ch.loc[m]
    rec = " & ".join('' if pd.isna(r.get(f'recall_{s}', np.nan)) else f"{r[f'recall_{s}']:.3f}" for s in TP)
    return f"{MN[m]} & {r.accuracy:.3f} & {r.balanced_accuracy:.3f} & {r.macro_f1:.3f} & {rec} \\\\"
err = N['rf_errors']; err_lr = int(cml.values.sum() - np.trace(cml.values))
A(rf"""
\section{{Classifier Performance}}
\label{{sec:res_classifier}}

\subsection{{Chronological Partition}}

Table~\ref{{tab:res_chrono}} reports every model on the test week, 6 to 13
November, after refitting on the whole training period with its selected
configuration.

\begin{{table}}[htbp]
\centering
\caption[Classifier performance on the test week]{{\textbf{{Classifier performance
on the test week}} ({th(795)} readings: {int(tp.loc['Saturated', 'n_test'])} saturated, {int(drain.n_test)} rapid
drainage, {int(tp.loc['Moist / cooler', 'n_test'])} moist and cooler, {int(tp.loc['Drier', 'n_test'])} drier). Recall is
given for each state.}}
\label{{tab:res_chrono}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
 & & Balanced & Macro & \multicolumn{{4}}{{c}}{{Recall}} \\
\cmidrule(lr){{5-8}}
Model & Accuracy & accuracy & F1 & Saturated & Drainage & Moist, cooler & Drier \\
\midrule
""" + "\n".join(crow(m) for m in ['random_forest', 'logistic_regression', 'decision_tree', 'majority']) + rf"""
Nearest centroid (reference) & 1.000 & 1.000 & 1.000 & & & & \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{figs/ch4_confusion.png}}
\caption[Confusion matrices on the test week]{{\textbf{{Confusion matrices on the
test week}} for the selected forest and for logistic regression. Rows are the states
assigned by the clustering; shading is the share of each row.}}
\label{{fig:res_confusion}}
\end{{figure}}

The forest assigns {pc(ch.loc['random_forest', 'accuracy'], 1)} percent of test readings to the
correct state, with a macro F1 of {ch.loc['random_forest', 'macro_f1']:.3f}, against
{ch.loc['majority', 'macro_f1']:.3f} for the majority class baseline, whose balanced accuracy
of {ch.loc['majority', 'balanced_accuracy']:.2f} is exactly the chance level of one quarter. It
recovers every saturated and every rapid drainage reading. Logistic regression,
however, performs better on every metric except the two event state recalls, on
which both models are perfect, with a macro F1 of
{ch.loc['logistic_regression', 'macro_f1']:.3f}, and the single tree performs worst of the three
trained models. The nearest centroid rule reproduces the partition exactly, as
Section~\ref{{sec:classifier}} explains it must.

The errors of both models, shown in Figure~\ref{{fig:res_confusion}}, fall between
adjacent states, which is the question Objective 2 poses. All {err} of the forest's errors and all {err_lr} of logistic
regression's lie between the moist and cooler state and the drier state, whose
centres are {D.loc['Drier', 'Moist / cooler']:.2f} units apart, against at least
{min(D.loc['Drier', 'Rapid drainage'], D.loc['Moist / cooler', 'Rapid drainage'], D.loc['Moist / cooler', 'Saturated']):.2f} for every
other pair, and every one of these errors assigns the reading to the state whose
centre is second nearest to it. No reading of either event state is misassigned, and no ordinary
reading is mistaken for an event. The misassigned readings lie on the boundary
itself: the difference between a reading's distance to its nearest and to its
second nearest centre has a median of {N['margin_median_wrong']:.3f} for the forest's errors,
against {N['margin_median_correct']:.3f} for its correct assignments. Of the forest's errors,
{int(cm.loc['Drier', 'Moist / cooler'])} are drier readings assigned to the moist and cooler state,
which in operation would understate dryness and delay a scheduling
recommendation under rule R3 instead of triggering a false one.

\begin{{table}}[htbp]
\centering
\caption[McNemar tests on the test week]{{\textbf{{McNemar tests on the test week}},
forest against each baseline. $n_{{01}}$: readings the forest assigns correctly and
the baseline incorrectly; $n_{{10}}$: the reverse. Holm adjusted $p$ values.}}
\label{{tab:res_mcnemar}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Forest against & $n_{{01}}$ & $n_{{10}}$ & $\chi^2$ & Adjusted $p$ \\
\midrule
Logistic regression & {mc['logistic_regression']['b_rf_right_other_wrong']} & {mc['logistic_regression']['c_rf_wrong_other_right']} & {mc['logistic_regression']['chi2']:.2f} & $< 0.001$ \\
Classification tree & {mc['decision_tree']['b_rf_right_other_wrong']} & {mc['decision_tree']['c_rf_wrong_other_right']} & {mc['decision_tree']['chi2']:.2f} & $< 0.001$ \\
Majority class & {mc['majority']['b_rf_right_other_wrong']} & {mc['majority']['c_rf_wrong_other_right']} & {mc['majority']['chi2']:.2f} & $< 0.001$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

Table~\ref{{tab:res_mcnemar}} tests these differences. The forest is significantly
better than the single tree and the majority class baseline, and significantly
worse than logistic regression, which is correct on
{mc['logistic_regression']['c_rf_wrong_other_right']} readings that the forest misassigns while the
converse holds for {mc['logistic_regression']['b_rf_right_other_wrong']}. As
Section~\ref{{sec:statcompare}} cautions, consecutive readings are correlated and these $p$
values are optimistic; the direction and size of each difference, which are large,
carry more weight than the $p$ values themselves.
""")

MODS = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']
def lcrow(n): return f"{SN(n)} & " + " & ".join(f"{wide.loc[n, m]:.3f}" for m in MODS) + " \\\\"
fold_acc = lf[lf.model == 'random_forest'].set_index('held_out').accuracy
A(rf"""
\subsection{{Leave One Node Out Partition}}

Table~\ref{{tab:res_lonoclf}} and Figure~\ref{{fig:res_lonobox}} report macro F1 on
each withheld node, with the clustering and the classifier both fitted on the other
four nodes.

\begin{{table}}[htbp]
\centering
\caption[Classifier performance on each withheld node]{{\textbf{{Macro F1 on each
withheld node.}} A state with no readings on the withheld node and never predicted
there has no defined F1 and is omitted from that node's average.}}
\label{{tab:res_lonoclf}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Withheld & Random Forest & Logistic regression & Classification tree & Majority class \\
\midrule
""" + "\n".join(lcrow(n) for n in NODES) + rf"""
\midrule
Mean (standard deviation) & """ + " & ".join(f"{wide[m].mean():.3f} ({wide[m].std():.3f})" for m in MODS) + r""" \\
Mean rank & """ + " & ".join(f"{ranks[m]:.1f}" for m in MODS) + rf""" \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{{figs/ch4_lono_boxplot.png}}
\caption[Macro F1 on each withheld node]{{\textbf{{Macro F1 on each withheld node}},
five folds per model.}}
\label{{fig:res_lonobox}}
\end{{figure}}

The Friedman test rejects the hypothesis that the four models perform equally,
with $\chi^2_F = {CT['friedman']['chi2_F']:.2f}$ ($p = {CT['friedman']['p']:.4f}$) and an Iman and
Davenport statistic of $F_F = {CT['friedman']['iman_davenport_F']:.1f}$ on 3 and 12 degrees of freedom
($p < 0.001$). Logistic regression ranks first on every node. With five folds the
Nemenyi critical difference is {CD:.2f} ranks, and only the difference between
logistic regression and the majority class baseline, {ranks['majority'] - ranks['logistic_regression']:.1f} ranks,
exceeds it. The forest's advantage over the baseline, {ranks['majority'] - ranks['random_forest']:.1f} ranks,
and logistic regression's advantage over the forest, {ranks['random_forest'] - ranks['logistic_regression']:.1f} ranks, are
consistent across folds but cannot be declared significant at this number of folds,
which is the limitation of power Section~\ref{{sec:statcompare}} anticipates.

Across the five folds the forest's mean accuracy is {N['lono_means']['random_forest']['accuracy']:.3f} and its
mean balanced accuracy {N['lono_means']['random_forest']['balanced_accuracy']:.3f}, against
{N['lono_means']['logistic_regression']['accuracy']:.3f} and {N['lono_means']['logistic_regression']['balanced_accuracy']:.3f} for
logistic regression. The errors again fall predominantly between adjacent states: of the
forest's {N['adjacent_errors']['lono_random_forest'][0]} errors on the withheld nodes,
{N['adjacent_errors']['lono_random_forest'][1]} assign a reading to the state whose centre is the second
nearest to it, and so do {N['adjacent_errors']['lono_logistic_regression'][1]} of logistic regression's
{N['adjacent_errors']['lono_logistic_regression'][0]}.

The weakest fold for the forest is Node 04, at {wide.loc['Node-04', 'random_forest']:.3f}, although its
accuracy there, {fold_acc['Node-04']:.3f}, is the second highest of the five. The low macro F1 is
caused by a single reading. Node 04 has no saturated readings, and the forest assigns
one of its readings to the saturated state: a moisture value of {N['node04']['max']:.1f} percent, at the
probe's ceiling, which the clustering had placed in the moist and cooler state. That
one assignment gives the saturated state an F1 of zero on this node and removes a
quarter of the macro average. The reading is at the probe's ceiling, which is the
condition the saturated state usually marks, so the assignment is understandable,
but it is not the state the clustering assigned. Logistic regression neither holds
nor predicts the saturated state on this node, so its average on Node 04 is taken
over three states and the forest's over four, and the two are not like for like.
The case shows how strongly macro F1 on a single node depends on the handling of
states with few or no readings, and it is the reason the mean ranks of
Table~\ref{{tab:res_lonoclf}} should be read together with the state sets its folds
actually contain.
""")

MODS_ = ['random_forest', 'logistic_regression', 'decision_tree', 'majority']
def fsrow(m):
    r = LSUM.loc[m]
    return (f"{MN[m]} & {r.accuracy_mean:.3f} ({r.accuracy_std:.3f}) & "
            f"{r.balanced_accuracy_mean:.3f} ({r.balanced_accuracy_std:.3f}) & "
            f"{r.macro_f1_mean:.3f} ({r.macro_f1_std:.3f}) \\\\")
def kfrow(m):
    r = KFOLD.loc[m]
    return (f"{MN[m]} & {r.accuracy_mean:.3f} ({r.accuracy_std:.3f}) & "
            f"{r.balanced_accuracy_mean:.3f} ({r.balanced_accuracy_std:.3f}) & "
            f"{r.macro_f1_mean:.3f} ({r.macro_f1_std:.3f}) \\\\")
A(rf"""
Table~\ref{{tab:res_foldspread}} gives the mean and the standard deviation of each measure
over the five folds, which is how the spread of a cross validated result is normally
reported, and Figure~\ref{{fig:res_cd}} places the four models on the rank scale of the
Friedman test with the critical difference marked.

\begin{{table}}[htbp]
\centering
\caption[Spread over the withheld node folds]{{\textbf{{Mean and, in brackets, standard
deviation over the five withheld node folds.}} Each fold refits the clustering and the
classifier on four nodes and scores the fifth. The spread is wide because the folds differ
in which states occur on the withheld node, not only in difficulty.}}
\label{{tab:res_foldspread}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Model & Accuracy & Balanced accuracy & Macro F1 \\
\midrule
""" + "\n".join(fsrow(m) for m in MODS_) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{{figs/ch4_critical_difference.png}}
\caption[Critical difference diagram]{{\textbf{{Mean rank of each model over the five
withheld nodes.}} A bar joins models whose mean ranks differ by less than the Nemenyi
critical difference, which with four models and five folds is {CT['friedman']['nemenyi_CD']:.2f} ranks.}}
\label{{fig:res_cd}}
\end{{figure}}

The diagram makes the limitation of Section~\ref{{sec:statcompare}} visible: with five
folds the critical difference is {CT['friedman']['nemenyi_CD']:.2f} ranks, wider than the
distance between logistic regression and the single tree, so the only pair the rank test
separates is logistic regression and the majority class baseline. The ranks themselves are
consistent across every fold, and Section~\ref{{sec:res_seeds}} applies the paired test over
repeated runs, where the number of observations is sufficient.

\subsection{{What a Random Partition Would Have Shown}}
\label{{sec:res_randomkfold}}

Section~\ref{{sec:classifiereval}} rejects random partitioning because consecutive hourly
readings are near duplicates, so a random split trains the model on a reading's own
neighbours. The marking of such designs in the literature is unambiguous: random
partitioning underestimates prediction error on data with temporal or spatial structure
\cite{{roberts2017}}, and in digital soil mapping the reported accuracy has been found
between twenty nine and sixty two percent higher under random splitting than under a
blocked design \cite{{john2025}}. Table~\ref{{tab:res_kfold}} computes the random five
fold result anyway, so that the size of the optimism in this record can be seen.

\begin{{table}}[htbp]
\centering
\caption[Random five fold, for comparison only]{{\textbf{{Random five fold on all readings,
mean and standard deviation.}} This design is reported for comparison and supports no
conclusion in this thesis: it places a reading's own neighbours in the training part, so its
scores are optimistic.}}
\label{{tab:res_kfold}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Model & Accuracy & Balanced accuracy & Macro F1 \\
\midrule
""" + "\n".join(kfrow(m) for m in MODS_) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

The optimism is large and it is not uniform. The forest's macro F1 is
{KFOLD.loc['random_forest', 'macro_f1_mean']:.3f} under random partitioning against
{LSUM.loc['random_forest', 'macro_f1_mean']:.3f} on a withheld node, a difference of
{KFOLD.loc['random_forest', 'macro_f1_mean'] - LSUM.loc['random_forest', 'macro_f1_mean']:.3f}, and the single
tree gains {KFOLD.loc['decision_tree', 'macro_f1_mean'] - LSUM.loc['decision_tree', 'macro_f1_mean']:.3f},
which would have made it look competitive with the forest. A study that reported only the
random split would have published a classifier that appears to work on an unseen node and
does not.
""")

tob = N['train_only_baseline']; tr_ = tob['results']
A(rf"""
\subsection{{Normalisation Baseline from the Training Period Only}}
\label{{sec:res_baseline}}

Section~\ref{{sec:normalisation}} normalises each node against the median and
interquartile range of its whole record, which includes the test week. The labels
and the classifier of the chronological evaluation never see the test week, but the
normalisation statistics do. This check was not part of the plan in
Chapter Three and is added because a deployed system can only
compute its baseline from readings already taken. The chronological evaluation
was repeated with each node's median and interquartile range computed from the
training period alone, everything else unchanged.

Under this baseline the training period partition does divide the drier readings
by pH, and the test week, measured against a baseline that no longer contains it,
contains {tob['counts_test'].get('Drier / lower pH', 0)} lower pH readings, {tob['counts_test'].get('Drier / higher pH', 0)} higher pH readings,
{tob['counts_test'].get('Moist / cooler', 0)} moist and cooler readings, and {tob['counts_test'].get('Saturated', 0)} saturated readings. The
forest's macro F1 falls from {ch.loc['random_forest', 'macro_f1']:.3f} to {tr_['random_forest']['macro_f1']:.3f} and its
accuracy from {ch.loc['random_forest', 'accuracy']:.3f} to {tr_['random_forest']['accuracy']:.3f}, while logistic regression falls
only from {ch.loc['logistic_regression', 'macro_f1']:.3f} to {tr_['logistic_regression']['macro_f1']:.3f}. The comparison is not like for like, since
the state set itself changes with the baseline, and logistic regression's accuracy
in fact rises, to {tr_['logistic_regression']['accuracy']:.3f}, while its macro F1 falls. Two conclusions nonetheless
follow. The chronological results of Table~\ref{{tab:res_chrono}} depend on the choice of
baseline, and should not be read as an estimate of deployed performance, which
depends on how each node's baseline is maintained. And under this one change of
baseline the forest loses more than logistic regression does, which is what the two
models' boundaries would lead one to expect, although a single comparison cannot
establish it as a property of either model. Measured against the training period baseline,
{pc(N['train_only_outside_share'])} percent of test readings lie outside the training range of at least
one of the normalised moisture, pH, and temperature values. Beyond that range the
forest's piecewise constant boundaries hold their last value, while the linear
boundaries of logistic regression, like those of the k means partition itself,
extend.
""")

def frow(t, h, unit):
    r = fchk(t, h)
    return f"{'Moisture' if t == 'moist' else 'pH'} & {h} & {int(r.min_samples_leaf)} & {r.rf:.2f} & {r.naive:.2f} & {r.daily:.2f} & {100 * r.skill:+.0f} \\\\".replace('+', '$+$').replace('-', '$-$')
SU = RB['summary']
def srow_(m, label):
    v = SU[m]; return f"{label} & {v['mean']:.4f} & {v['sd']:.4f} & {v['min']:.4f} & {v['max']:.4f} \\\\"
A(rf"""
\subsection{{Thirty Independent Runs}}
\label{{sec:res_seeds}}

Every result above comes from one random seed. The whole procedure was repeated with
{RB['n_seeds']} seeds, and in each run the states were identified again from the readings and every
model was refitted, so that the spread in Tables~\ref{{tab:res_seeds}} and
\ref{{tab:res_matrix}} and in Figure~\ref{{fig:res_seeds}} covers the clustering as well as
the classifiers. Thirty runs are enough for the paired test that five folds could not
support.

\begin{{table}}[htbp]
\centering
\caption[Repeated independent runs]{{\textbf{{Macro F1 on the test week over {RB['n_seeds']} independent runs.}} In each run the
k means solution, the forest and the single tree were refitted with a new seed. Logistic
regression and the majority class baseline have no random element of their own and vary only
with the states they are given.}}
\label{{tab:res_seeds}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Mean & Standard deviation & Lowest & Highest \\
\midrule
""" + "\n".join(srow_(m, MN[m]) for m in ['random_forest', 'logistic_regression', 'decision_tree', 'majority']) + rf"""
Nearest centroid (reference) & 1.0000 & 0.0000 & 1.0000 & 1.0000 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_seeds.png}}
\caption[Repeated independent runs]{{\textbf{{Thirty independent runs.}} (a) Macro F1 on the test week for each
model, one point per run. (b) Agreement, by the adjusted Rand index, between the states found in
each run and those found in the first.}}
\label{{fig:res_seeds}}
\end{{figure}}

The four state solution is stable under reseeding. The same four named states form in all
{RB['same_state_set']} runs, their agreement with the first run has a mean adjusted Rand index of
{RB['ari_mean']:.3f} and never falls below {RB['ari_min']:.3f}, and the within cluster sum of squares varies only
between {RB['inertia_min']:.1f} and {RB['inertia_max']:.1f}. The nearest centroid reference reproduces the partition
exactly in every run, as it must.

The ordering of the models does not depend on the seed either. Logistic regression scores
between {SU['logistic_regression']['min']:.4f} and {SU['logistic_regression']['max']:.4f}, the forest between {SU['random_forest']['min']:.4f} and
{SU['random_forest']['max']:.4f}, and the two ranges do not overlap, so the conclusion of
Section~\ref{{sec:res_classifier}} survives repetition. The forest's own spread, a standard deviation of
{SU['random_forest']['sd']:.4f}, is of the same size as the differences between the best few configurations in
Section~\ref{{sec:res_search}}, which is the practical reason for preferring the simpler
configuration when the validation window cannot separate them.
""")

MEAS = {'macro_f1': 'macro F1', 'balanced_accuracy': 'balanced accuracy', 'accuracy': 'accuracy'}
def mrow(m):
    out = []
    for k, lab in MEAS.items():
        r = MATRIX[(MATRIX.model == m) & (MATRIX.measure == k)].iloc[0]
        out.append(f"{MN[m] if k == 'macro_f1' else ''} & {lab} & {r['mean']:.4f} & {r['sd']:.4f} & "
                   f"{r['minimum']:.4f} & {r['median']:.4f} & {r['maximum']:.4f} \\\\")
    return "\n".join(out)
def crow(lab):
    r = MATRIX[(MATRIX.model == 'k means clustering') & (MATRIX.measure == lab)].iloc[0]
    return (f" & {lab} & {r['mean']:.4f} & {r['sd']:.4f} & {r['minimum']:.4f} & {r['median']:.4f} & "
            f"{r['maximum']:.4f} \\\\")
def wrow(r):
    other = r.comparison.split(' vs ')[1]
    return (f"{MN['random_forest']} against {MN[other].lower()} & {int(r.runs_proposed_better)} & "
            f"{int(r.runs_other_better)} & {sg(r.median_difference, 4)} & {r.statistic:.0f} & "
            f"{sci(r.p)} & {sci(r.p_holm)} \\\\")
WL = WIL.set_index('comparison')
A(rf"""
Table~\ref{{tab:res_matrix}} aggregates the runs into the comparative matrix: for every model
and every measure of Section~\ref{{sec:metrics}}, the mean, the standard deviation, the
extremes and the median over the {RB['n_seeds']} runs, together with the quantities that
describe the clustering itself.

\begin{{table}}[htbp]
\centering
\caption[Comparative matrix over thirty runs]{{\textbf{{Every measure over
{RB['n_seeds']} independent runs.}} Classifier measures are on the test week. The clustering
rows describe the partition of the whole record found in each run: its objective, the two
internal indices, and its agreement with the partition found in the first run.}}
\label{{tab:res_matrix}}
\small
\begin{{tabular}}{{llrrrrr}}
\toprule
Model & Measure & Mean & Standard deviation & Lowest & Median & Highest \\
\midrule
""" + ("\n" + chr(92) + "addlinespace\n").join(mrow(m) for m in MODS_) + rf"""
\addlinespace
k means clustering""" + "\n".join(
    crow(l) for l in ['within cluster sum of squares', 'silhouette', 'Davies Bouldin',
                      'agreement with the first run']) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[htbp]
\centering
\caption[Paired signed rank test over the runs]{{\textbf{{Paired Wilcoxon signed rank test on
macro F1 over the {RB['n_seeds']} runs.}} Each run pairs the models on the same states and the
same test week. Wins count the runs in which each model scores higher. The $p$ values are two
sided, and the last column applies Holm's correction across the three comparisons.}}
\label{{tab:res_wilcoxon}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Comparison & Forest wins & Other wins & Median difference & $W$ & $p$ & $p$ (Holm) \\
\midrule
""" + "\n".join(wrow(r) for _, r in WIL.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

Table~\ref{{tab:res_wilcoxon}} reports that test. It is decisive where the fold level test
was not. Logistic regression scores
higher than the forest in {int(WL.loc['random_forest vs logistic_regression', 'runs_other_better'])} of the
{RB['n_seeds']} runs, by a median of
{abs(WL.loc['random_forest vs logistic_regression', 'median_difference']):.4f} macro F1, and the test
rejects equality at {sci(WL.loc['random_forest vs logistic_regression', 'p_holm'])} after correction. The
forest in turn beats the single tree and the majority class baseline in every run. Because each
run repeats the whole procedure, the comparison is paired on the states as well as on the test
week, which is the design Section~\ref{{sec:statcompare}} prescribes for repeated measurements
on one dataset \cite{{demsar2006}}. It should still be read for what it is: thirty runs on one
record, not thirty datasets, so it establishes that the ordering is not an accident of the seed,
and nothing about other sites.
""")


def improw(r):
    return (f"{r.feature} & {r.forest_impurity:.3f} ({r.forest_impurity_sd:.3f}) & "
            f"{sg(r.forest_permutation, 3)} ({r.forest_permutation_sd:.3f}) & "
            f"{sg(r.logistic_permutation, 3)} ({r.logistic_permutation_sd:.3f}) \\\\")
def ablrow(lbl):
    g = ABLC[ABLC.removed == lbl].set_index('model')
    return (f"{lbl} & {g.loc['random_forest', 'macro_f1']:.3f} & "
            f"{sg(g.loc['random_forest', 'test_macro_f1_change'], 3)} & "
            f"{g.loc['logistic_regression', 'macro_f1']:.3f} & "
            f"{sg(g.loc['logistic_regression', 'test_macro_f1_change'], 3)} & "
            f"{g.loc['decision_tree', 'macro_f1']:.3f} \\\\")
def ablkrow(r):
    sh = '' if pd.isna(r.share_of_readings_reassigned) else f"{pc(r.share_of_readings_reassigned, 2)}"
    return (f"{r.removed} & {'yes' if r.same_state_set else 'no'} & {r.ari_with_reference:.3f} & "
            f"{sh} & {int(r.folds_with_the_same_set)} of 5 & "
            f"{r.random_forest_macro_f1:.3f} & {r.logistic_regression_macro_f1:.3f} \\\\")
ONE_H = IMP[IMP.feature.str.contains('one hour')].iloc[0]
THREE_H = IMP[IMP.feature.str.contains('three hours')].iloc[0]
A1 = ABLC[(ABLC.removed == 'moisture change, one hour') & (ABLC.model == 'random_forest')].iloc[0]
A1L = ABLC[(ABLC.removed == 'moisture change, one hour') & (ABLC.model == 'logistic_regression')].iloc[0]
A3 = ABLC[(ABLC.removed == 'moisture change, three hours') & (ABLC.model == 'random_forest')].iloc[0]
A3L = ABLC[(ABLC.removed == 'moisture change, three hours') & (ABLC.model == 'logistic_regression')].iloc[0]
K1 = ABLK[ABLK.removed == 'moisture change, one hour'].iloc[0]
K3 = ABLK[ABLK.removed == 'moisture change, three hours'].iloc[0]
KPT = ABLK[ABLK.removed == 'pH and temperature'].iloc[0]
A(rf"""
\section{{The Contribution of Each Input}}
\label{{sec:res_inputs}}

Section~\ref{{sec:features}} gives the classifier five inputs. This section measures what
each of them contributes, in three ways that answer different questions: how much the
fitted models rely on an input, how much is lost when the classifier is denied it, and
whether the states themselves survive its removal. The third is the strongest test. A state
set that dissolves when one input is withheld was a property of that input and not of the
record, which is the distinction between validating a method and validating a particular
partition \cite{{ullmann2022}}.

\subsection{{Importance}}
\label{{sec:res_importance}}

\begin{{table}}[htbp]
\centering
\caption[Importance of each input]{{\textbf{{How much each input contributes, with standard
deviations in brackets.}} Impurity is the share of the decrease in impurity attributed to the
input across the four hundred trees. Permutation is the loss of macro F1 on the test week when
that column alone is shuffled, over thirty permutations. Inputs are on a common normalised
scale, so the measures are comparable between inputs.}}
\label{{tab:res_importance}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Input & Forest, impurity & Forest, permutation & Logistic regression, permutation \\
\midrule
""" + "\n".join(improw(r) for _, r in IMP.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_importance.png}}
\caption[Importance of each input]{{\textbf{{Contribution of each input.}} (a) Impurity and
permutation importance, with standard deviations. (b) The coefficients of logistic regression,
one row per state; a positive value raises the odds of that state as the input rises.}}
\label{{fig:res_importance}}
\end{{figure}}

The two measures agree on the extremes and differ in the middle. The change in moisture over
three hours is the largest contributor by permutation for both models,
{THREE_H.forest_permutation:.3f} for the forest and {THREE_H.logistic_permutation:.3f} for logistic
regression, and the change over one hour is the smallest, {ONE_H.forest_permutation:.3f} and
{sg(ONE_H.logistic_permutation, 3)}, which is within its own standard deviation of zero. The
disagreement in the middle is expected: impurity importance divides credit between inputs that
carry the same information, and moisture and its three hour change are related by construction,
whereas permutation asks what is lost when one column is destroyed and the others remain
\cite{{breiman2001}}. The coefficients in Figure~\ref{{fig:res_importance}}(b) show how the
linear model separates the states: the saturated state is high moisture and low pH, the drier
state is low moisture with high pH and high temperature, and the rapid drainage state is
distinguished almost entirely by a falling three hour moisture change, with a coefficient of
{sg(LRC.loc['Rapid drainage'].iloc[4], 1)} against {sg(abs(LRC.loc['Rapid drainage'].iloc[3]), 1)} for the one hour change.

\subsection{{Removing an Input from the Classifier}}
\label{{sec:res_ablation}}

Table~\ref{{tab:res_ablation}} and Figure~\ref{{fig:res_ablation}}(a) withhold one input at a
time, and two pairs of inputs, from the classifier alone.

\begin{{table}}[htbp]
\centering
\caption[Removing an input from the classifier]{{\textbf{{Macro F1 on the test week when the
classifier is denied an input, with the change from the full set.}} The states are unchanged
throughout: only the classifier's view of each reading is reduced.}}
\label{{tab:res_ablation}}
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Input withheld & Forest & Change & Logistic regression & Change & Classification tree \\
\midrule
""" + "\n".join(ablrow(l) for l in ABLC.removed.unique()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_ablation.png}}
\caption[Effect of removing an input]{{\textbf{{What removing an input costs.}} (a) Change in
macro F1 on the test week when the classifier is denied the input, the states being unchanged.
(b) Agreement with the reference states when the clustering is denied the input as well; a red
bar marks a removal after which a different set of states forms.}}
\label{{fig:res_ablation}}
\end{{figure}}

Three results follow. The change in moisture over one hour can be removed at no cost: macro F1
moves by {abs(A1.test_macro_f1_change):.3f} for the forest and {abs(A1L.test_macro_f1_change):.3f} for
logistic regression, in both cases less than the spread between repeated runs reported in
Section~\ref{{sec:res_seeds}}. The change over three hours is the input the classifier can least
afford to lose, costing the forest {abs(A3.test_macro_f1_change):.3f} and logistic regression
{abs(A3L.test_macro_f1_change):.3f}. And the single tree behaves differently from the other two:
withholding moisture or pH raises its macro F1 slightly, because a tree limited to depth six
spends its first splits on whichever input separates the largest group and gains from having
fewer candidates, which is the behaviour the depth limit of Section~\ref{{sec:tuning}} was
chosen to impose.

\subsection{{Removing an Input from the States Themselves}}
\label{{sec:res_ablation_states}}

Table~\ref{{tab:res_ablation_states}} and Figure~\ref{{fig:res_ablation}}(b) repeat the removal
with the clustering refitted, so that the states themselves are derived again without the input.

\begin{{table}}[htbp]
\centering
\caption[Removing an input from the clustering]{{\textbf{{What the states become when the
clustering is denied an input.}} Same set: whether the naming rule produces the same four
states. Agreement: adjusted Rand index against the reference states. Moved: the share of
readings that change state, defined only where the state set is unchanged. Folds: the number of
the five withheld node folds in which that same set forms.}}
\label{{tab:res_ablation_states}}
\small
\begin{{tabular}}{{lccrcrr}}
\toprule
Input withheld & Same set & Agreement & Moved (percent) & Folds & Forest macro F1 & Logistic macro F1 \\
\midrule
""" + "\n".join(ablkrow(r) for _, r in ABLK.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

The one hour change is redundant here too: without it the same four states form, they agree with
the reference states at an adjusted Rand index of {K1.ari_with_reference:.3f}, and
{pc(K1.share_of_readings_reassigned, 2)} percent of readings change state. On the evidence of
this record the input set could be reduced to four without consequence. It is reported rather
than applied, because Chapter Three fixed the inputs in advance and every result above was
produced with the five it names.

The three hour change is the opposite case. Without it the training period no longer produces a
rapid drainage state at all, and the clustering instead divides the drier readings by pH, which
is the division the whole record produces and Section~\ref{{sec:res_lono}} reports as
provisional. This is a direct answer to a question Section~\ref{{sec:res_training}} left open:
the rapid drainage state of the training period exists because the three hour change is an
input, and it is the only state of the four that depends on a single input in this way.

The last row is a caution about names. Withholding pH and temperature together leaves a
clustering the rule still names with the same four names, but its agreement with the reference
states falls to {KPT.ari_with_reference:.3f} and {pc(KPT.share_of_readings_reassigned, 1)} percent of
readings change state: the same words describe a materially different partition. A naming rule
fixes the vocabulary, not the content, which is why the transfer evidence of
Section~\ref{{sec:res_lono}} and the stability evidence of Section~\ref{{sec:res_seeds}} are
reported alongside it \cite{{hennig2007,ullmann2022}}.
""")

A(rf"""
\section{{Verification and Cost of the Implementations}}
\label{{sec:res_verification}}

\subsection{{Verification}}

Every baseline, metric, and test used in this chapter was computed a second time, either by
hand from its definition or with a different library, and the two values were compared.
Table~\ref{{tab:res_verify}} reports the comparison.

\begin{{table}}[htbp]
\centering
\caption[Verification of the implementations]{{\textbf{{Each quantity, recomputed
independently.}} The difference column is the value used in this chapter minus the value
obtained by the check. The last row is a random quantity, so its two values agree only to
within the spread of a permutation.}}
\label{{tab:res_verify}}
\scriptsize
\begin{{tabular}}{{p{{5.1cm}}rrrp{{4.6cm}}}}
\toprule
Quantity & Used & Check & Difference & How it was checked \\
\midrule
""" + "\n".join(f"{r.quantity} & {sg(r.value_used, 4)} & {sg(r.independent_check, 4)} & {sci(r.difference)} & {r.method_of_check} \\\\"
                for _, r in VER.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

{WORD[int(((VER.difference.abs() < 5e-10) & ~VER.stochastic).sum())]} of the {WORD[len(VER)].lower()} quantities agree exactly, and
{WORD[int(((VER.difference.abs() >= 5e-10) & ~VER.stochastic).sum())].lower()} more agree to
{sci(VER[~VER.stochastic].difference.abs().max(), 0)}, which is the rounding applied when values are written to disk.
The remaining check is a permutation importance, which is a random quantity rather than a
fixed one: a second set of thirty permutations gives
{VER.loc[VER.stochastic, 'independent_check'].iloc[0]:.4f} against
{VER.loc[VER.stochastic, 'value_used'].iloc[0]:.4f}, a difference well inside the standard
deviation of a single permutation reported in Table~\ref{{tab:res_importance}}.
What this establishes is that the baselines are driven and scored correctly, not that the
libraries themselves are correct: the forest, the tree, logistic regression and k means are the
scikit learn implementations at the versions pinned in Table~\ref{{tab:environment}},
McNemar's test was checked against statsmodels 0.15.0, and the signed rank statistic and the
relaxation bound were recomputed by hand.

\subsection{{Execution Times}}

Table~\ref{{tab:res_timing}} gives the cost of fitting and applying each model on the
chronological partition, measured on the machine of Table~\ref{{tab:environment}} as the median of
three repeats.

\begin{{table}}[htbp]
\centering
\caption[Execution times]{{\textbf{{Time to fit each model on the {th(2234)} training readings and to
assign the {th(795)} test readings.}}}}
\label{{tab:res_timing}}
\small
\begin{{tabular}}{{lrr}}
\toprule
Model & Fit (seconds) & Assign 795 readings (milliseconds) \\
\midrule
""" + "\n".join(f"{r.model} & {r.fit_seconds:.3f} & " + ('' if pd.isna(r.predict_ms) else f"{r.predict_ms:.2f}") + " \\\\" for _, r in TIM.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

""" + (rf"""The complete pipeline, from the raw workbook to every figure, table and the text of this
chapter, takes {STEPS.seconds.sum() / 60:.1f} minutes on the same machine. Its three longest steps are
{TOP3}.
""" if STEPS is not None else "") + rf"""Assignment is what a deployed node would run, and at
{TIM.predict_ms.min():.2f} to {TIM.predict_ms.max():.0f} milliseconds for {th(795)} readings none of these models is a
constraint on an hourly reporting interval. The forest is the most expensive of the three to fit
and to apply, by two orders of magnitude, which is a further consideration against it given the
performance reported in Section~\ref{{sec:res_classifier}}.
""")


def budrow(r):
    return (f"{int(r.restarts)} & {r.objective_mean:.3f} & {r.objective_sd:.3f} & {r.objective_best:.3f} & "
            f"{r.objective_worst:.3f} & {r.gap_to_best_known_mean_percent:.4f} & "
            f"{r.gap_to_best_known_worst_percent:.4f} & {int(r.runs_within_a_hundredth_of_a_percent)} & "
            f"{r.seconds:.3f} \\\\")
B1 = BUD[BUD.restarts == 1].iloc[0]; B30 = BUD[BUD.restarts == 30].iloc[0]; B10 = BUD[BUD.restarts == 10].iloc[0]
A(rf"""
\section{{Solution Quality and the Optimality Gap}}
\label{{sec:res_quality}}

The states are the partition that minimises the within cluster sum of squares, written here as
the objective. Minimising it exactly is NP hard \cite{{aloise2009}}, so the pipeline does what
is standard: it runs Lloyd's algorithm \cite{{lloyd1982}} from the careful random start of
k means++ \cite{{arthur2007}}, thirty times, and keeps the best result. No exact optimum is
therefore available to compare against, and the quality of the solution has to be bracketed from
both sides.

From above, the objective is compared with the best value found by a far longer search, two
thousand restarts, which took {Q['best_known_search_seconds']:.1f} seconds and reached
{Q['best_known']:.3f}. The solution reported throughout this chapter has an objective of
{Q['pipeline_objective']:.3f}, which is {Q['pipeline_gap_to_best_known_percent']:.4f} percent above it,
and no run of the {WORD[RB['n_seeds']].lower()} in Table~\ref{{tab:res_matrix}} is further above it
than that. The long search returns the same four named
states, so the extra effort buys a fractionally lower objective and not a different answer.

From below, the relaxation of Ding and He \cite{{ding2004}} gives a value no partition into four
groups can beat: the total scatter of the record, {Q['total_scatter']:.1f}, minus
{th(Q['n'])} times the three largest eigenvalues of its covariance matrix, which is
{Q['lower_bound']:.1f}. The true optimum therefore lies somewhere between that bound and
{Q['best_known']:.3f}. The bound is loose, as relaxation bounds for this problem generally are,
and the distance to it, {Q['pipeline_gap_to_lower_bound_percent']:.0f} percent, should not be read
as the distance to the optimum; it is reported because it is the only guarantee available, and
because a reader is entitled to know that no tight one exists.

\begin{{table}}[htbp]
\centering
\caption[What a restart budget buys]{{\textbf{{The objective under each restart budget, over
twenty seeds.}} Gap is the distance above the best known value, in percent. The pipeline uses
thirty restarts.}}
\label{{tab:res_budget}}
\small
\begin{{tabular}}{{rrrrrrrrr}}
\toprule
Restarts & Mean & Standard deviation & Best & Worst & Gap, mean & Gap, worst & Within 0.01 percent & Seconds \\
\midrule
""" + "\n".join(budrow(r) for _, r in BUD.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_quality.png}}
\caption[What a restart budget buys]{{\textbf{{Restarts against the quality of the
solution.}} (a) The objective under each budget, with the standard deviation over twenty seeds
and the best known value marked. (b) The same budgets as cost against quality, each point
labelled with its number of restarts.}}
\label{{fig:res_quality}}
\end{{figure}}

Table~\ref{{tab:res_budget}} and Figure~\ref{{fig:res_quality}} show where the effort stops
paying. A single restart is {B1.gap_to_best_known_mean_percent:.2f} percent above the best known
value on average and {B1.gap_to_best_known_worst_percent:.1f} percent in its worst seed, which is a
visibly different partition, not a rounding difference. Ten restarts bring every seed within a
hundredth of a percent. The thirty the pipeline uses cost {B30.seconds:.3f} seconds and sit
{B30.gap_to_best_known_mean_percent:.4f} percent above the best known value, so the choice fixed in
Section~\ref{{sec:clustering}} is comfortably past the point of diminishing returns, and the
restart budget is not a quantity any conclusion in this chapter is sensitive to.
""")

PW = SC['power_law']
def scrow(r):
    return (f"{th(r.readings)} & {'record' if r.source == 'record' else 'replicated'} & "
            f"{r.kmeans_seconds:.3f} & {r.forest_fit_seconds:.3f} & {r.logistic_fit_seconds:.3f} & "
            f"{r.forest_assign_ms:.1f} \\\\")
def vrow(r):
    return (f"{int(r.nodes)} & {r.period} & {th(r.readings)} & {r.kmeans_seconds:.1f} & "
            f"{r.forest_fit_seconds:.1f} & {r.refit_seconds:.1f} & "
            f"{'yes' if r.inside_an_hourly_cycle else 'no'} \\\\")
V5Y = VIAB[(VIAB.nodes == 100) & (VIAB.period == 'five years')].iloc[0]
A(rf"""
\section{{Scalability and the Limits of the Approach}}
\label{{sec:res_scaling}}

Section~\ref{{sec:res_verification}} reports the cost of one run on the record as it stands.
This section asks how that cost behaves as the problem grows, and where the approach would stop
being practical. Three directions are measured: the number of readings, the number of nodes and
the number of states. Beyond the size of the record itself the readings are replicated with
small random jitter, which says nothing about soil and is used only to extend the curve; those
rows are marked as replicated throughout. Table~\ref{{tab:res_scaling}} and
Figure~\ref{{fig:res_scaling}} report the measurements.

\begin{{table}}[htbp]
\centering
\caption[Cost against the size of the record]{{\textbf{{Time to identify the states and to fit
each classifier, against the number of readings.}} Rows marked replicated are the record
repeated with jitter, used only to extend the curve. Assignment is the time to assign the
{th(795)} readings of the test week.}}
\label{{tab:res_scaling}}
\small
\begin{{tabular}}{{rlrrrr}}
\toprule
Readings & Source & States (s) & Forest fit (s) & Logistic fit (s) & Assign 795 (ms) \\
\midrule
""" + "\n".join(scrow(r) for _, r in SCAL.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_scaling.png}}
\caption[How the pipeline scales]{{\textbf{{Cost against problem size.}} (a) Against the
number of readings, on logarithmic axes; filled points are the record, open points are
replicated readings. (b) Against the number of states. (c) Against the number of nodes.}}
\label{{fig:res_scaling}}
\end{{figure}}

Above about {th(SC['fitted_above'])} readings, where the fixed cost of building four hundred
trees is no longer the dominant term, a power law fitted to the measurements gives an exponent
of {PW['kmeans_seconds']['exponent']:.2f} for identifying the states and
{PW['forest_fit_seconds']['exponent']:.2f} for fitting the forest. Both are at most linear in the
number of readings, which is what the algorithms predict: each iteration of Lloyd's algorithm
touches every reading once per state, and the trees are grown to a fixed depth. Below that size
the forest is dominated by its own fixed cost, which is why it takes
{SCAL[SCAL.source == 'record'].forest_fit_seconds.iloc[0]:.2f} seconds on
{th(SCAL[SCAL.source == 'record'].readings.iloc[0])} readings and only
{SCAL.forest_fit_seconds.iloc[-1]:.1f} seconds on {th(SCAL.readings.iloc[-1])}. Memory is not a
constraint at these sizes either: the largest fit measured, {th(SC['largest_measured'])} readings,
took the whole process to {SC['peak_memory_mb']:.0f} MB. The number of states and the number of
nodes both change the cost by less than a factor of four over the ranges tried.

\begin{{table}}[htbp]
\centering
\caption[Estimated cost of a complete refit]{{\textbf{{Estimated time for a complete refit at
deployment sizes.}} Readings assume hourly reporting from every node for the period shown. Times
are extrapolated from the fitted power law, on the machine of Table~\ref{{tab:environment}}.}}
\label{{tab:res_viability}}
\small
\begin{{tabular}}{{rlrrrrc}}
\toprule
Nodes & Period & Readings & States (s) & Forest fit (s) & Refit (s) & Inside an hourly cycle \\
\midrule
""" + "\n".join(vrow(r) for _, r in VIAB.iterrows()) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

Table~\ref{{tab:res_viability}} answers the practical question. A complete refit of one hundred
nodes reporting hourly for five years, {th(V5Y.readings)} readings, is estimated at
{V5Y.refit_seconds:.0f} seconds, well inside the hour between readings, and assignment of a single
reading stays at the microseconds of Section~\ref{{sec:res_verification}} whatever the size of
the training record. Within the range this study can speak to, computation is therefore not the
binding constraint, and the system does not need the cloud server that comparable deployments
in Chapter Two assume.

What does bind is data, and the evidence for that is in the same measurements from the other
direction. The four states of the training period form from as few as
{th(int(SNOD.readings.iloc[0]))} readings on two nodes, and from the first
{pc(SLEN.share_of_training_period.iloc[0])} percent of the training period, a little over
{th(int(SLEN.readings.iloc[0]))} readings: by that measure the record is already long enough for
the states to appear. It is not long enough for the questions the states raise. One month cannot
separate a recurring condition from a trend, as Section~\ref{{sec:res_states}} reports for the
drier states; a node that never experiences a condition cannot confirm it, as Node 04 shows for
the saturated state; and no amount of additional computation repairs either. The limits of this
approach are limits of the record and of the instrument, not of the algorithm.
""")

A(rf"""
\section{{Forecasting Against Persistence}}
\label{{sec:res_forecast}}

Table~\ref{{tab:res_forecast}} and Figure~\ref{{fig:res_forecast}} compare the tuned
forest regressors of Section~\ref{{sec:forecast}} with the persistence baseline and
with the value at the same hour on the previous day, on the test week.

\begin{{table}}[htbp]
\centering
\caption[Forecasting against persistence]{{\textbf{{Forecasting error on the test
week.}} Mean absolute error in percent moisture or pH units. Skill is relative to
persistence, in percent; a negative value means the forest is worse than assuming no
change.}}
\label{{tab:res_forecast}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Quantity & Horizon (h) & Leaf & Forest & Persistence & Previous day & Skill \\
\midrule
""" + "\n".join(frow('moist', h, '') for h in [1, 3, 6, 12, 24]) + "\n\\addlinespace\n" +
"\n".join(frow('ph', h, '') for h in [1, 3, 6, 12, 24]) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figs/ch4_forecast.png}}
\caption[Forecasting error by horizon]{{\textbf{{Forecasting error by horizon}} on
the test week.}}
\label{{fig:res_forecast}}
\end{{figure}}

For moisture the forest is worse than persistence at every horizon, by between
{-100 * fr[fr.target == 'moist'].skill.max():.0f} and {-100 * fr[fr.target == 'moist'].skill.min():.0f} percent. For pH it is better at
one hour and at twelve, by {100 * fchk('ph', 1).skill:.0f} and {100 * fchk('ph', 12).skill:.0f} percent, and worse
at the other horizons, so there is no consistent advantage across horizons. Forecasting
moisture six hours ahead on a withheld node, the forest is worse than persistence on
four of the five nodes, by up to {-100 * fl.skill.min():.0f} percent. The previous day's value
is worse than persistence at every horizon below twenty four hours, where the two
forecasts coincide, which is consistent with the absence of any daily cycle in valid
moisture readings reported in Appendix~\ref{{app:sensitivity}}. Because the minimum leaf size was tuned on the
validation window for each quantity and horizon, the negative result does not arise
from an untuned model. The finding supports the scope stated in Chapter One: over a record of this
length, these features do not anticipate the next reading better than the assumption
that it will not change, which is consistent with soil moisture changing slowly, and
the system characterises the current state instead of forecasting a future one.
""")

A(rf"""
\section{{The Rule Base Applied to the Record}}
\label{{sec:res_rules}}

The expert review of Section~\ref{{sec:expertvalidation}} has not yet been conducted.
The review instrument has been prepared, and its outcome, the answer to the third
research question, will be reported when the responses are received; no agreement
statistic is reported here. What can be reported now is how often the conditions of
each rule in Table~\ref{{tab:rules}} occur in the record, which bears on what the
reviewers are asked to judge.

\begin{{table}}[htbp]
\centering
\caption[Rule conditions in the record]{{\textbf{{Occurrence of each rule's condition
in the record}}, using the reference states.}}
\label{{tab:res_rules}}
\small
\begin{{tabular}}{{lp{{11.5cm}}}}
\toprule
Rule & Occurrence in the record \\
\midrule
R1 & {rules['saturated_episodes']} saturated episodes, {rules['saturated_episodes_over_1h']} of them longer than one hour and none longer than {rules['saturated_longest_h']}; median recovery {TS['recovery']['median']:.1f} hours in the {TS['recovery']['n']} episodes that recovered within twelve hours \\
R2 & saturated readings on at most {rules['r2_max_days_in_a_week']} days of any calendar week at one node, reached twice, both at Node 01 \\
R3 & one run of either drier state longer than twenty four hours, of {rules['r3_longest_drier_run_h']} hours \\
R4 & the moist and cooler state, {pc(M_.share, 1)} percent of readings \\
R5 & never: the longest run of the lower pH state lasts {N['longest_run']['Drier / lower pH']} hours, against the forty eight required, and the {rules['r5_valid_readings_ph_below_5_5']} readings between pH 3 and 5.5 all belong to the saturated state \\
R6 & Node 04, whose median conductivity of {rules['r6_ec_median_by_node']['Node-04']:.3f} dS m$^{{-1}}$ lies below the {rules['r6_ec_median_by_node']['Node-01']:.3f} of every other node \\
R7 & {N['out_of_range_analysis']} out of range readings in the analysis record, {N['out_of_range']} of the {th(3030)} retained, Appendix~\ref{{app:data}}; no moment at which all nodes step together \\
R8 & the first twenty four hours of each of the five nodes \\
\bottomrule
\end{{tabular}}
\end{{table}}

Two findings from Table~\ref{{tab:res_rules}} bear directly on the review. Rule R5
would never have fired. No run of the lower pH state lasts the forty eight hours its
condition requires, and of the {N['ph_below_5_5_all']} readings below its pH threshold,
{N['ph_below_5_5_all'] - rules['r5_valid_readings_ph_below_5_5']} lie below the valid minimum of 3 and the remaining
{rules['r5_valid_readings_ph_below_5_5']} all belong to the saturated state. The rule also depends on the lower pH
state, which Section~\ref{{sec:res_lono}} found to be the least secure of the four. Its condition is therefore better expressed on the drier state together with a
measured pH, and the reviewers should be invited to judge that revision. Rule R1
depends on a recovery that took a median of {TS['recovery']['median']:.0f} hour where it occurred, and that did not
occur within twelve hours in {N['recovery_episodes_total'] - TS['recovery']['n']} of the {N['recovery_episodes_total']} episodes, and which reflects the
behaviour of the probe at its ceiling, and its threshold should be treated as provisional, as
Section~\ref{{sec:rules}} already states.
""")

A(rf"""
\section{{Discussion}}
\label{{sec:discussion}}

\textbf{{What the states are.}} The per node normalisation does what it was designed
to do: it removes the location effect that dominated the raw readings, and the
resulting states are nearly independent of the sensor that produced them. Two
conditions meet all three requirements of the definition in
Section~\ref{{sec:background}}: a moist and cooler condition, whose runs average
{f(TS['run_hours']['Moist / cooler'], 1)} hours, and a drier condition, whose runs average {f(N['run_merged_drier'], 1)} hours when its
two pH divisions are counted as one. Both recur on every withheld node, both are stable under resampling, with mean
Jaccard coefficients of {VD['Moist / cooler'].mean_jaccard:.3f} and between
{min(VD['Drier / higher pH'].mean_jaccard, VD['Drier / lower pH'].mean_jaccard):.3f} and
{max(VD['Drier / higher pH'].mean_jaccard, VD['Drier / lower pH'].mean_jaccard):.3f} for its two pH
divisions, and the moist and cooler state and both pH divisions of the drier state are
held for about twice as long as records that vary as smoothly as the readings but
contain no states. The saturated
cluster recurs wherever a node produces readings far enough from its own normal,
and is interpretable, but it lasts one to two hours, so it is an event
rather than a state in the sense of that definition; it is retained because the rule
base must flag it, and it is reported as a sensor event. The division of the drier
condition by pH is weaker: it persists, but {pc(W_.share_from_3nov)} percent of its readings
fall in the last eleven days of the record, it follows a decline in pH at three nodes, and it does not form when
Node 05 is withheld or when the test week is excluded. With a single month
of data, a slow change over the month and a recurring condition cannot be told apart,
and the state is reported as provisional. The saturated state is secure as an event
but not as a measurement: {N['sat_at_ceiling']} of its {th(S_.n)} readings are the probe at its
ceiling, and its pH values are not credible. What the evidence establishes about
all four is a property of the record of the five nodes taken together.
Section~\ref{{sec:res_recognisability}} shows that the readings of a single node,
clustered on their own, do not reproduce the grouping, for the reason given there;
no claim is therefore made that one node analysed in isolation would recover
them.

\textbf{{Why logistic regression outperforms the forest.}} The states are defined by
the k means algorithm, which assigns each reading to its nearest centre, and the
boundary between two such regions is a plane. Logistic regression draws planes, and so
matches the shape of the partition it is asked to reproduce; the nearest centroid rule,
which draws exactly these planes, reproduces it without error. A forest approximates an
oblique plane by a staircase of splits parallel to the feature axes, and its errors
concentrate where that approximation is poorest, on the boundary between the two
largest states, which is where Section~\ref{{sec:res_classifier}} finds them. The
advantage of the forest, its ability to represent irregular boundaries, confers
nothing when the boundaries are planes by construction. Chapter Three raised this
possibility when it chose logistic regression as a baseline, but adopted the forest on
the strength of benchmarking evidence for tree ensembles on tabular data
\cite{{grinsztajn2022}} and of the ecological comparison in which a forest outperformed
logistic regression and three other classifiers \cite{{cutler2007}}. Neither finding is
contradicted here. Both concern labels that come from outside the feature space, a
species present or absent, a soil class recorded in the field; the labels in this study
are produced by a procedure whose decision boundaries are planes in the same space the
classifier sees, which is the condition under which a linear model is not merely
competitive but correctly specified \cite{{breiman2001}}. The result is consistent with
the design of the study. Whether it holds wherever a classifier is trained to reproduce
a k means partition is not something one deployment can settle.

\textbf{{What the inputs contribute.}} Section~\ref{{sec:res_inputs}} measures what each
input is worth, and two of its results bear on the interpretation rather than on the
models. The first is that pH carries real weight: withholding it from the clustering moves
{pc(ABLK[ABLK.removed == 'pH'].iloc[0].share_of_readings_reassigned, 1)} percent of readings
to a different state, and it is the input that separates the two drier states from each
other. That is consistent with the place pH occupies in the literature of Chapter Two,
where it is the strongest single correlate of bacterial community composition across soils
\cite{{lauber2009}} and across a liming gradient within one field \cite{{rousk2010}}. The
qualification of Section~\ref{{sec:res_states}} still applies: the pH values here are
uncalibrated readings from a probe whose saturated readings are not credible, so the
division is reported as provisional whatever its weight in the clustering. The second is
that the change in moisture over three hours, not the moisture level, is the input the
classifier can least afford to lose, and that the rapid drainage state of the training
period exists only because that input is present. A state defined by a rate of change is a
state about drainage, which is the quantity Chapter Two associates with aeration and with
the switch between aerobic and anaerobic processes \cite{{linn1984}}; it is not a state
about how wet the soil is.

\textbf{{Consequences for the choice of classifier.}} Section~\ref{{sec:classifier}}
gives two reasons for using a classifier and not the centroid rule. The first,
that a classifier estimates probabilities, holds for logistic regression at least as
well as for the forest: on the test week its log loss is
{fin['logistic_regression']['test_log_loss']:.3f} against the forest's {fin['random_forest']['test_log_loss']:.3f}. The second, that a
classifier can be retrained on labels that are not a k means partition, such as
labels revised by the expert panel or confirmed in the laboratory, remains the case
for the forest, because such labels need not be separated by planes; it cannot,
however, be tested until such labels exist. On the evidence of this deployment,
logistic regression is the better model for assigning the present states, and it lost
less than the forest under the one change of normalisation baseline that was tried. The forest remains the appropriate
candidate for labels whose boundaries are irregular. Whether the thesis should
continue to present the Random Forest as the system's classifier is for the research
team and its adviser to settle on the evidence reported here.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.78\textwidth]{{figs/ch4_tradeoff.png}}
\caption[Cost against quality of assignment]{{\textbf{{What each model costs to apply, against
how well it assigns.}} Filled circles are the test week, open squares the mean over the five
withheld nodes. The horizontal scale is the time to assign one reading, on a logarithmic
scale.}}
\label{{fig:res_tradeoff}}
\end{{figure}}

\textbf{{Trade offs.}} Four choices in this pipeline trade one thing against another, and
the measurements settle three of them. Figure~\ref{{fig:res_tradeoff}} shows the first of them
at a glance. The forest against logistic regression: the forest
costs {TIM.set_index('model').loc['Random Forest', 'fit_seconds'] / TIM.set_index('model').loc['Logistic regression', 'fit_seconds']:.0f} times as long to fit and
{TIM.set_index('model').loc['Random Forest', 'predict_ms'] / TIM.set_index('model').loc['Logistic regression', 'predict_ms']:.0f} times as long to assign a reading, and on this record it
scores lower, so there is no trade off in its favour to weigh, only a reason to keep it in
reserve for labels that planes cannot separate. The restart budget: thirty restarts cost
{BUD[BUD.restarts == 30].iloc[0].seconds:.2f} seconds and land
{BUD[BUD.restarts == 30].iloc[0].gap_to_best_known_mean_percent:.4f} percent above the best known
objective, where a single restart costs {BUD[BUD.restarts == 1].iloc[0].seconds:.3f} seconds and
risks a solution {BUD[BUD.restarts == 1].iloc[0].gap_to_best_known_worst_percent:.0f} percent worse,
so the saving is not worth taking. The number of states: on the training period the objective
falls monotonically as states are added, from {SSTA.set_index('states').loc[3, 'inertia']:.0f} at
three to {SSTA.set_index('states').loc[6, 'inertia']:.0f} at six, so a better fit is always
available and is not the criterion; interpretability is, and it fails at six. The fourth is not settled by
measurement: the normalisation baseline computed from the whole record gives higher scores
than one computed from past readings only, and only the second is available to a deployed
node, so the honest figure is the lower one and Section~\ref{{sec:res_baseline}} reports
both.

\textbf{{Results that were not expected.}} Five results ran against the expectation set out
in Chapters One and Three, and each is reported where it occurs rather than smoothed over.
Logistic regression beat the Random Forest under every partition and in all
{WORD[RB['n_seeds']].lower()} repeated runs, which is the result the paragraph above explains and which the
title of this thesis does not anticipate. The change in moisture over one hour, included in
the feature set to capture short term drainage, contributes nothing measurable to either
model and can be removed without changing the states. Temperature, included as context
rather than as a defining quantity, turns out to carry more weight in the classifier than
pH does. The single classification tree improves when inputs are taken away from it, which
is a symptom of the depth limit rather than of the data. And the random partition that
Chapter Three rejected on principle would have raised the tree's macro F1 by
{KFOLD.loc['decision_tree', 'macro_f1_mean'] - LSUM.loc['decision_tree', 'macro_f1_mean']:.2f}, enough to
make the weakest classifier in this study look like a credible one, which is the clearest
possible demonstration of why that design was rejected.

\textbf{{Limits on these results.}} Five limits apply beyond those stated in
Section~\ref{{sec:residualbias}}. The record covers one month, which is too short to
separate a trend from a recurring condition. The chronological scores depend on a
normalisation baseline that included the test week, and fall for the forest when the
baseline is computed from past readings only. The rare states are represented by a handful of readings in each window,
so that a single reading can move macro F1 on the validation window by several
hundredths, and on a single withheld node by as much as a quarter. And consecutive readings are correlated,
so the McNemar tests overstate the evidence, and the leave one node out comparison
has only five folds. Finally, the states are a property of the inputs as much as of the
soil: withholding pH and temperature together leaves a partition the naming rule still
calls by the same four names while
{pc(ABLK[ABLK.removed == 'pH and temperature'].iloc[0].share_of_readings_reassigned, 0)} percent of
readings change state, so the names in this chapter should be read as labels for a
particular procedure applied to a particular set of inputs.

\textbf{{Implications for deployment.}} The normalisation baseline of
each node should be recomputed as readings accumulate, and the state set should be
refitted periodically and checked against the transfer criterion before its states
are acted upon. The saturated state should be reported as a sensor event requiring
inspection, as rule R7 already does for the out of range readings that make up
most of it, and its pH readings should not be used. And the pH condition of rule R5 should rest on a measured pH
rather than on the provisional lower pH state.
""")

A(rf"""
\section{{Chapter Summary}}

Per node normalisation reduced the dependence of the states on sensor location from a
normalised mutual information of {N['raw_nmi_k4']:.2f} to {f(c_p.nmi_with_node, 2)}, which pooled standardisation
does not achieve. No candidate number of states met the transfer criterion in full.
Every state held together when the record was resampled, with mean Jaccard
coefficients between {min(VD[st].mean_jaccard for st in ORDER):.3f} and
{max(VD[st].mean_jaccard for st in ORDER):.3f} under resampling of whole days, the
lowest belonging to the provisional pH division. The prediction strength of each state on a
withheld node, reported in Section~\ref{{sec:res_recognisability}}, reaches the
published threshold in {numw(PS_REACH)} of the {numw(PS_TOTAL)} combinations of a state
and a fold and in no state in every fold, while the statistic that procedure
defines, the minimum over the states of a fold, exceeds its reference level in
{numw(len(BEAT))} of the five folds; neither comparison is used to retain or remove a state, for the
reasons Section~\ref{{sec:lono}} states.
Four states were retained on stated grounds: a saturated event, a moist and cooler
state, and two drier states divided by pH, of which the division is provisional while
the drier condition itself persists and recurs on every withheld node. Tuned on a
held out validation window, the Random Forest reproduced the training period states on
the test week with a macro F1 of {ch.loc['random_forest', 'macro_f1']:.3f}, with every error between the
two adjacent large states, but logistic regression performed significantly better,
with a macro F1 of {ch.loc['logistic_regression', 'macro_f1']:.3f}, and lost less than the forest when the
normalisation baseline was computed from past readings only, which is consistent with the
states being separated by planes. That ordering held in all {WORD[RB['n_seeds']].lower()} repeated
runs, and the paired signed rank test over those runs rejects equality at
{sci(WIL.set_index('comparison').loc['random_forest vs logistic_regression', 'p_holm'])} after
correction. Of the five inputs, the change in moisture over one hour contributes nothing
measurable and can be removed without changing the states, while the change over three hours is
the input the states depend on. The clustering sits
{Q['pipeline_gap_to_best_known_percent']:.4f} percent above the best objective found by a search
sixty six times longer, and a complete refit for a hundred nodes reporting hourly for five years
is estimated at {VIAB[(VIAB.nodes == 100) & (VIAB.period == 'five years')].iloc[0].refit_seconds / 60:.0f}
minutes, so computation is not what limits this system; the length of the record is. Forecasting did not improve on persistence for moisture at any horizon, and improved
on it for pH at two of five, so no consistent gain was found. The expert review is pending, and the rule base analysis
identifies rules R1 and R5 as those the reviewers should examine most closely.
""")

TXT = ''.join(L)
TXT = TXT.replace('\\begin{tabular}', '\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}').replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}')
open(OUT, 'w').write(TXT); print('wrote', OUT)
