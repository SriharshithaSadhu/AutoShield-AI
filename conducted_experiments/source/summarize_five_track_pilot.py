"""Summarize the shared pilot; preserve incomplete-track status."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from five_track_common import probabilities

ROOT=Path(__file__).resolve().parent
P=ROOT/'results/five_track_pilot'
d=json.loads((P/'results.json').read_text())
lines=['# Five-track development pilot','',
       'Three seeds, six fixed configurations per seed. This is a development bank, not a completed optimizer study or novelty claim.', '',
       '| Seed | Model | Raw NLL | Calibrated NLL | Eval FPR at 0.1% target | Recall | Conformal coverage / mean size | Desktop batch-one p95 ms |',
       '|---|---|---|---|---|---|---|---|']
for run in d['runs']:
    for r in run['candidates']:
        op=r['operating_points']['0.001'];cm=r['conformal']
        lines.append(f"| {run['seed']} | {r['name']} | {r['evaluation_raw']['nll']:.4f} | {r['evaluation_calibrated']['nll']:.4f} | {op['fpr']:.3%} | {op['recall']:.2%} | {cm['coverage']:.2%} / {cm['mean_set_size']:.3f} | {r['desktop_batch1_ms']['p95']:.3f} |")
lines+=['','## Tuning selections and CSE transfer','','FPR/recall below use frozen source-calibrated thresholds, with no adaptation or target labels in calibration. Original-CSE infiltration label limitations apply.','',
        '| Seed | Selection rule | Model | CSE pooled FPR | CSE pooled recall |','|---|---|---|---|---|']
for run in d['runs']:
    for rule,name in run['selections'].items():
        ops=[w['arms'][name]['operating_points']['0.001'] for w in run['transfer']]
        nb=sum(m['benign'] for m in ops);na=sum(m['attack'] for m in ops)
        lines.append(f"| {run['seed']} | {rule} | {name} | {sum(m['fp'] for m in ops)/nb:.2%} | {sum(m['tp'] for m in ops)/na:.2%} |")
lines+=['','## Explanation sensitivity','','Native TreeSHAP explains raw LightGBM margins, not calibrated probabilities. The fixed 64-row source probe is descriptive, not a selection criterion.','']
for s in d['explanation_stability']:
    lines.append(f"- {s['model']}, seeds {s['seeds']}: top-10 Jaccard {s['top10_jaccard']:.3f}.")
lines+=['','## What remains','','- Track 1: Pi latency/memory/power and compiled-backend comparisons pending hardware/backend validation.',
        '- Track 2: Finite-bank calibration/conformal pilot completed; objective-integrated AutoML and confirmation remain.',
        '- Track 3: Frozen transfer screen completed; corrected-CSE validity and warm-start adaptation remain.',
        '- Track 4: Four-objective candidate archive computed; adversarial threat-model validation and optimizer comparisons remain. Natural protocol recall is not adversarial robustness.',
        '- Track 5: Six configurations and native TreeSHAP sensitivity completed; explanation-aware selection and wider comparisons remain.',
        '', 'Grouped development partitions do not prove exchangeability or temporal generalization. Conformal coverage is empirical, not a validated guarantee under shift. Timing excludes flow extraction. No real-device or energy measurements yet.']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
bundle=joblib.load(P/'lgb_small_seed42.joblib')
x=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv',nrows=32)[bundle['features']].replace([np.inf,-np.inf],np.nan).to_numpy()
np.savez_compressed(P/'lgb_small_seed42_fixture.npz',x=x,expected=probabilities(bundle,x))
print('\n'.join(lines))
