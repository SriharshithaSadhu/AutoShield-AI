"""Development diagnostic; fixed chronological stages, no future-label routing."""
import hashlib
import json
import pickle
from pathlib import Path
from time import process_time

import numpy as np
import pandas as pd

from experiment_protocol import make_leakage_safe_pipeline
from research_runner import make_estimator

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data/cse_replay_chunk_cache_v2'


def metrics(y, p):
    return dict(n=len(y), benign=int((y == 0).sum()), attack=int((y == 1).sum()),
                fp=int(((y == 0) & p).sum()), tp=int(((y == 1) & p).sum()))


def summarize(rows, cap, floor):
    supported = [r for r in rows if min(r['benign'], r['attack']) >= 100]
    def good(r):
        return r['fp'] / r['benign'] <= cap and r['tp'] / r['attack'] >= floor
    nb, na = sum(r['benign'] for r in rows), sum(r['attack'] for r in rows)
    return dict(windows=len(rows), supported=len(supported), joint_pass=sum(map(good, supported)),
                fpr=sum(r['fp'] for r in rows) / nb if nb else None,
                recall=sum(r['tp'] for r in rows) / na if na else None,
                worst_supported_fpr=max((r['fp']/r['benign'] for r in supported), default=None))


def main():
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    chunks = manifest['chunks'][::8]  # deterministic, label-independent development subsample
    dates = sorted({c['capture_date'] for c in chunks})
    assert len(dates) >= 6
    features = manifest['features']
    windows = []
    for c in chunks:
        with np.load(CACHE / c['file']) as z:
            windows.append(dict(date=c['capture_date'], chunk=c['global_chunk'],
                                x=z['x'], y=z['y'], lo=int(z['timestamps_ns'].min()),
                                hi=int(z['timestamps_ns'].max())))
    phases = [[w for w in windows if w['date'] in ds]
              for ds in [dates[:2], dates[2:3], dates[3:4], dates[4:]]]
    assert all(phases)
    assert all(max(w['hi'] for w in a) < min(w['lo'] for w in b)
               for a, b in zip(phases, phases[1:]))
    archive_path = ROOT / 'results/iotid20_binary_common77_mopso_seed42.json'
    archive = json.loads(archive_path.read_text())
    configs = sorted([r for r in archive['pareto_archive'] if r['feasibility']['feasible']],
                     key=lambda r: -r['metrics']['weighted_f1'])[:4]
    source = pd.read_csv(ROOT / 'data/iotid20_binary_common77_dev.csv')
    tx = np.concatenate([w['x'] for w in phases[0]])
    ty = np.concatenate([w['y'] for w in phases[0]])
    results = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        ix = rng.choice(len(ty), min(40000, len(ty)), replace=False)
        models, metadata = [], []
        for domain in ('source', 'target_prefix'):
            for i, row in enumerate(configs):
                model = make_leakage_safe_pipeline(make_estimator('lightgbm', row['config'], seed, 1), random_state=seed)
                x, y = (source[features], source['is_attack']) if domain == 'source' else (pd.DataFrame(tx[ix], columns=features), ty[ix])
                start = process_time()
                model.fit(x, y)
                metadata.append(dict(name=f'{domain}_{i}', train_rows=len(y), fit_cpu_s=process_time()-start,
                                     bytes=len(pickle.dumps(model)), config=row['config']))
                models.append(model)
        # Same prefix data, simple fixed LightGBM baseline; no target AutoML search.
        baseline = make_leakage_safe_pipeline(make_estimator('lightgbm', {'n_estimators':100,'num_leaves':31}, seed, 1), random_state=seed)
        start = process_time()
        baseline.fit(pd.DataFrame(tx[ix], columns=features), ty[ix])
        metadata.append(dict(name='target_fixed_reference', train_rows=len(ix), fit_cpu_s=process_time()-start, bytes=len(pickle.dumps(baseline))))
        models.append(baseline)
        scored = []
        for phase in phases[1:]:
            scored.append([(w, [m.predict_proba(pd.DataFrame(w['x'], columns=features))[:, 1] for m in models]) for w in phase])
        experiments = []
        for cap in (.001, .01):
            thresholds = []
            for i in range(len(models)):
                benign = np.concatenate([s[i][w['y']==0] for w,s in scored[0]])
                assert len(benign) > 0
                thresholds.append(float(np.quantile(benign, 1-cap, method='higher')))
            records = []
            for phase in scored[1:]:
                records.append([[dict(date=w['date'], chunk=w['chunk'], **metrics(w['y'], s[i]>thresholds[i])) for w,s in phase] for i in range(len(models))])
            for floor in (.1, .5, .8):
                selection = [summarize(r, cap, floor) for r in records[0]]
                evaluation = [summarize(r, cap, floor) for r in records[1]]
                # Reserve selection from prefix only; default index zero is source primary.
                eligible = [i for i in range(1,len(models)) if selection[i]['joint_pass'] > 0]
                chosen = max(eligible, key=lambda i:(selection[i]['joint_pass'], selection[i]['recall'], -metadata[i]['bytes'])) if eligible else None
                experiments.append(dict(cap=cap, recall_floor=floor, selected=chosen,
                                        thresholds=thresholds, selection=selection, evaluation=evaluation,
                                        records=records[1]))
        results.append(dict(seed=seed, models=metadata, experiments=experiments))
        print('finished seed', seed, flush=True)
    out = dict(status='exploratory development, not confirmation',
               limitations=['Previously inspected CSE stream', 'Every eighth chunk; not full replay',
                            'No future risk guarantee; empirical calibration', 'No online reserve switching tested',
                            'Source archive fixed at seed42; seeds vary fitting and prefix row sampling',
                            'Source and target arms have different label exposure; not matched-budget superiority'],
               protocol=dict(train=dates[:2], calibration=dates[2:3], selection=dates[3:4], evaluation=dates[4:],
                             caps=[.001,.01], recall_floors=[.1,.5,.8], sampling='every eighth chunk, no label filtering'),
               archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
               manifest_sha256=hashlib.sha256((CACHE/'manifest.json').read_bytes()).hexdigest(), results=results)
    (ROOT/'results/archive_joint_feasibility.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
