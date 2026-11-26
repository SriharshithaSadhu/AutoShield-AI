"""Causal sampled-stream diagnostic. Oracle thresholds are analysis only."""
import hashlib
import json
import pickle
from pathlib import Path
from time import process_time, perf_counter

import numpy as np
import pandas as pd

from experiment_protocol import make_leakage_safe_pipeline
from research_runner import make_estimator
from run_archive_joint_feasibility import metrics, summarize

ROOT = Path(__file__).resolve().parent
CACHE = ROOT/'data/cse_replay_chunk_cache_v2'
OUT = ROOT/'results/recent_target_score_diagnostic'
CAPS = (.001, .01)


def threshold(scores, labels, cap):
    """Lowest observed benign boundary admitting <=floor(cap*n) false alarms.

    Predictions must use strict >, including when scores tie.
    """
    benign = np.sort(scores[labels == 0])
    if not len(benign):
        return None
    return float(benign[-1-int(np.floor(cap*len(benign)))])


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((CACHE/'manifest.json').read_text())
    windows = []
    for c in manifest['chunks'][::8]:
        path = CACHE/c['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == c['sha256']
        with np.load(path) as z:
            windows.append(dict(chunk=c['global_chunk'], date=c['capture_date'], x=z['x'], y=z['y'],
                                lo=int(z['timestamps_ns'].min()), hi=int(z['timestamps_ns'].max())))
    dates = sorted({w['date'] for w in windows})
    features = manifest['features']
    early = [w for w in windows if w['date'] in dates[:2]]
    ex, ey = np.concatenate([w['x'] for w in early]), np.concatenate([w['y'] for w in early])
    source = pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    config = dict(n_estimators=100, num_leaves=31)
    def fit(x,y,seed):
        model = make_leakage_safe_pipeline(make_estimator('lightgbm',config,seed,1),random_state=seed)
        start = process_time()
        model.fit(pd.DataFrame(x,columns=features), y)
        return model, process_time()-start
    all_results = []
    for seed in (42,43,44):
        rng = np.random.default_rng(seed)
        ei = rng.choice(len(ey),min(40000,len(ey)),replace=False)
        anchor = ei[:min(10000,len(ei))]
        sm, sc = fit(source[features],source['is_attack'],seed)
        em, ec = fit(ex[ei],ey[ei],seed)
        records = []
        for wi,w in enumerate(windows):
            if w['date'] < dates[4]:
                continue
            cal = windows[wi-2:wi]
            train = [p for p in windows[max(0,wi-5):wi-2] if p['date'] not in dates[:2]]
            assert train and max(p['hi'] for p in train+early) < min(p['lo'] for p in cal)
            assert max(p['hi'] for p in cal) < w['lo']
            rx,ry = np.concatenate([p['x'] for p in train]),np.concatenate([p['y'] for p in train])
            ri = rng.choice(len(ry),min(30000,len(ry)),replace=False)
            rx,ry = np.concatenate([ex[anchor],rx[ri]]),np.concatenate([ey[anchor],ry[ri]])
            rm,rc = fit(rx,ry,seed)
            cx,cy = np.concatenate([p['x'] for p in cal]),np.concatenate([p['y'] for p in cal])
            assert (cy==0).sum() > 0
            row = dict(chunk=w['chunk'],date=w['date'], train_chunks=[p['chunk'] for p in train],
                       calibration_chunks=[p['chunk'] for p in cal], recent_train_rows=len(ry),
                       calibration_rows=len(cy), calibration_benign=int((cy==0).sum()),
                       train_max_ns=max(p['hi'] for p in train+early), calibration_min_ns=min(p['lo'] for p in cal),
                       calibration_max_ns=max(p['hi'] for p in cal), evaluation_min_ns=w['lo'], arms={})
            arrays = dict(y=w['y'],calibration_y=cy)
            for name,model,cost in [('source',sm,0),('early_target',em,0),('recent_target',rm,rc)]:
                start = perf_counter()
                cp = model.predict_proba(pd.DataFrame(cx,columns=features))[:,1]
                cal_seconds = perf_counter()-start
                start = perf_counter()
                scores = model.predict_proba(pd.DataFrame(w['x'],columns=features))[:,1]
                elapsed = perf_counter()-start
                arrays[name]=scores
                arrays[name+'_calibration']=cp
                arm = dict(fit_cpu_s=cost,model_bytes=len(pickle.dumps(model)),
                           calibration_score_seconds=cal_seconds,inference_seconds=elapsed,
                           default=metrics(w['y'], scores>.5), operating_points=[])
                for cap in CAPS:
                    causal_t = threshold(cp,cy,cap)
                    # Only the analysis branch accesses current labels; it cannot affect fitting/calibration.
                    oracle_t = threshold(scores,w['y'],cap)
                    arm['operating_points'].append(dict(cap=cap,threshold=causal_t,
                        causal=metrics(w['y'],scores>causal_t),
                        oracle_threshold=oracle_t,
                        oracle=metrics(w['y'],scores>oracle_t) if oracle_t is not None else None))
                row['arms'][name]=arm
            np.savez_compressed(OUT/f'seed{seed}_chunk{w["chunk"]}.npz',**arrays)
            records.append(row)
        result = dict(seed=seed, initial_fit_cpu_s=dict(source=sc,early_target=ec),records=records)
        (OUT/f'seed{seed}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        all_results.append(result)
        print('completed seed',seed,'windows',len(records),flush=True)
    out = dict(status='development diagnostic; no novel method or confirmation',
               protocol=dict(seeds=[42,43,44],config=config,sample_stride=8,early_dates=dates[:2],
                             evaluation_dates=dates[4:],training='10k early anchor + up to 30k from preceding three sampled windows, excluding two calibration windows',
                             calibration='two most recent sampled windows; labels released after each window',
                             caps=CAPS,recall_floor=.5,minimum_each_class=100,
                             caveats=['sampled windows are not adjacent full-stream chunks','recorded flow timestamp order, not exporter arrival',
                                      'same stream across seeds','retrospective oracle is not deployable','target label use differs from source',
                                      'single-thread CPU timing; no equal-CPU superiority claim']),
               manifest_sha256=hashlib.sha256((CACHE/'manifest.json').read_bytes()).hexdigest(),results=all_results)
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Recent-target score diagnostic','',json.dumps(out['protocol']), '',
           '| Seed | Model | Threshold | Cap | Pooled FPR | Pooled recall | Joint passes (recall >=50%) |',
           '|---|---|---|---|---|---|---|']
    for result in all_results:
        for name in ('source','early_target','recent_target'):
            for ci,cap in enumerate(CAPS):
                for kind in ('causal','oracle'):
                    rs=[r['arms'][name]['operating_points'][ci][kind] for r in result['records']]
                    s=summarize([r for r in rs if r is not None],cap,.5)
                    lines.append(f"| {result['seed']} | {name} | {kind} | {cap:.1%} | {s['fpr']:.4%} | {s['recall']:.3%} | {s['joint_pass']}/{s['supported']} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
