import subprocess, sys, time, platform, hashlib, os

XLSX = sys.argv[1] if len(sys.argv) > 1 else 'data/One_Month_Calibrated_Data.xlsx'
LOG  = 'pipeline_run_log.txt'

STEPS = [
    ('0. Extract the usable readings from the workbook',      ['extract_data.py', XLSX]),
    ('1. Is the clustering just recovering the sensor locations?', ['cluster_test.py']),
    ('2. Build the location free features',                    ['features.py']),
    ('3. How many states?',                                    ['pick_k.py']),
    ('4. Leave one node out transfer test (k = 4, seed 0)',    ['lono.py', '0', '4']),
    ('5. Hyperparameter tuning on the validation window',      ['tune_models.py']),
    ('6. Classifier and state transitions',                    ['classifier_transitions.py']),
    ('7. Baseline comparison and statistical tests',           ['compare_models.py']),
    ('8. Forecasting against the no change baseline',          ['forecast_test.py']),
    ('9. Nutrient channels versus conductivity',               ['npk_table.py']),
    ('10. Exploratory data analysis for Chapter 3',            ['eda_numbers.py']),
    ('11. Chapter 3 figures (thesis Figures 3 and 4)',                                   ['charts.py']),
    ('12. Supplementary transfer figure',                                         ['chart_lono.py']),
]

def sha(path):
    try:
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
    except OSError:
        return 'not found'

out = []
def w(s=''):
    print(s)
    out.append(s)

w('=' * 78)
w('SOIL CONDITION STATE ANALYSIS: FULL PIPELINE RUN')
w('=' * 78)
w(f'Run at        : {time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
w(f'Python        : {platform.python_version()} on {platform.system()}')
for mod in ['pandas', 'numpy', 'sklearn', 'scipy', 'matplotlib']:
    try:
        w(f'{mod:<14}: {__import__(mod).__version__}')
    except Exception as e:
        w(f'{mod:<14}: not importable ({e})')
w(f'Input file    : {XLSX}')
w(f'Input sha256  : {sha(XLSX)} (first 16 hex)')
w()
w('Every script below fixes its random seed. Two runs of this file on the same')
w('workbook must agree line for line except for the timestamp above.')
w()

failed = []
for title, cmd in STEPS:
    w()
    w('-' * 78)
    w(title)
    w('-' * 78)
    w(f'$ python3 {" ".join(cmd)}')
    w()
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    for line in r.stdout.rstrip('\n').split('\n'):
        w(line)
    if r.returncode != 0:
        failed.append(title)
        w()
        w('*** THIS STEP FAILED ***')
        for line in r.stderr.rstrip('\n').split('\n')[-25:]:
            w('    ' + line)
    w()
    w(f'[completed in {time.time()-t0:.1f} s]')

w()
w('=' * 78)
w('OUTPUT FILES')
w('=' * 78)
for f in sorted(['soil_clean.csv', 'soil_features.csv', 'soil_states_k4.csv',
                 'lono_summary_seed0_k4.csv', 'forecast_results.csv', 'forecast_tuning.csv',
                 'forecast_lono.csv', 'best_configs.json', 'tuning_random_forest.csv',
                 'tuning_decision_tree.csv', 'tuning_logistic_regression.csv',
                 'comparison_chronological.csv', 'comparison_lono_folds.csv',
                 'comparison_clustering.csv', 'comparison_tests.json',
                 'npk_ec_evidence.csv', 'table_npk_ec.tex',
                 'fig1_clusters_vs_nodes.png', 'fig2_what_changes_over_time.png',
                 'fig3_leave_one_node_out.png']):
    size = os.path.getsize(f) if os.path.exists(f) else 0
    w(f'  {f:<36} {size:>10,} bytes   sha {sha(f)}')

w()
w('=' * 78)
w('FAILED STEPS: ' + (', '.join(failed) if failed else 'none'))
w('=' * 78)

open(LOG, 'w').write('\n'.join(out) + '\n')
print(f'\nwrote {LOG}')
