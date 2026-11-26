"""Analyze completed seed artifacts, including causal versus oracle disagreement."""
import json
from pathlib import Path
import numpy as np
from run_archive_joint_feasibility import summarize, metrics
from run_recent_target_score_diagnostic import threshold

P = Path(__file__).resolve().parent/'results/recent_target_score_diagnostic'
out = []
diagnoses = []
audited = 0
for path in sorted(P.glob('seed*.json')):
    d = json.loads(path.read_text())
    for row in d['records']:
        assert row['train_max_ns'] < row['calibration_min_ns']
        assert row['calibration_max_ns'] < row['evaluation_min_ns']
        assert not set(row['train_chunks']) & set(row['calibration_chunks'])
        with np.load(P/f'seed{d["seed"]}_chunk{row["chunk"]}.npz') as z:
            for name,arm in row['arms'].items():
                assert np.isfinite(z[name]).all()
                assert arm['default'] == metrics(z['y'],z[name]>.5)
                for point in arm['operating_points']:
                    t = threshold(z[name+'_calibration'],z['calibration_y'],point['cap'])
                    assert t == point['threshold']
                    assert point['causal'] == metrics(z['y'],z[name]>t)
                    ot = threshold(z[name],z['y'],point['cap'])
                    assert ot == point['oracle_threshold']
                    if ot is not None:
                        assert point['oracle'] == metrics(z['y'],z[name]>ot)
                        assert point['oracle']['fp'] <= point['cap']*point['oracle']['benign']
                    audited += 1
    for name in ('source','early_target','recent_target'):
        for ci,cap in enumerate((.001,.01)):
            arms = [w['arms'][name] for w in d['records']]
            for kind in ('causal','oracle'):
                rows = [a['operating_points'][ci][kind] for a in arms]
                out.append(dict(seed=d['seed'],model=name,cap=cap,kind=kind,
                                **summarize([r for r in rows if r is not None],cap,.5)))
        if name == 'recent_target':
            for row in d['records']:
                a = row['arms'][name]
                for point in a['operating_points']:
                    c,o = point['causal'],point['oracle']
                    if o is None or min(c['benign'],c['attack']) < 100:
                        continue
                    feasible = lambda m: m['fp']/m['benign'] <= point['cap'] and m['tp']/m['attack'] >= .5
                    diagnoses.append(dict(seed=d['seed'],date=row['date'],chunk=row['chunk'],cap=point['cap'],
                                          causal_feasible=feasible(c),oracle_feasible=feasible(o),
                                          causal_fpr=c['fp']/c['benign'],causal_recall=c['tp']/c['attack'],
                                          oracle_recall=o['tp']/o['attack']))
    print('seed',d['seed'],'recent fit CPU',sum(w['arms']['recent_target']['fit_cpu_s'] for w in d['records']))
print(json.dumps(out,indent=2))
(P/'aggregate.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(P/'window_diagnoses.json').write_text(json.dumps(diagnoses,indent=2),encoding='utf-8')
print('Verified score-derived operating points:',audited)
