"""Exact grouping and optimistic empirical recall bound, not a generalization bound."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
P=ROOT/'results/recent_target_score_diagnostic'
manifest=json.loads((ROOT/'data/cse_replay_chunk_cache_v2/manifest.json').read_text())
diagnosis=json.loads((P/'failure_diagnosis.json').read_text())
records=[]
for row in diagnosis['records']:
    if row['seed']!=42:
        continue
    chunk=manifest['chunks'][row['chunk']]
    assert chunk['global_chunk']==row['chunk']
    with np.load(ROOT/'data/cse_replay_chunk_cache_v2'/chunk['file']) as z:
        x,y=z['x'],z['y']
    frame=pd.DataFrame(x).replace([np.inf,-np.inf],np.nan)
    columns=list(frame.columns)
    frame['target']=y
    groups=frame.groupby(columns,dropna=False,sort=False)['target'].agg(['sum','count'])
    a=groups['sum'].to_numpy(dtype=int)
    b=groups['count'].to_numpy(dtype=int)-a
    mixed=(a>0)&(b>0)
    conflicting_rows=int((a[mixed]+b[mixed]).sum())
    assert conflicting_rows==row['conflicting_vector_rows']
    assert int(np.minimum(a,b).sum())==row['empirical_conflict_min_errors']
    bounds=[]
    for cap in (.001,.01):
        # All pure-attack groups are free. Fractional relaxation of mixed-group
        # knapsack is an upper bound for any deterministic vector-only classifier.
        budget=int(np.floor(cap*(y==0).sum()))
        remaining=float(budget)
        tp=float(a[b==0].sum())
        for j in np.argsort(-(a[mixed]/b[mixed])):
            aj,bj=a[mixed][j],b[mixed][j]
            portion=min(1.,remaining/bj)
            tp+=portion*aj
            remaining-=portion*bj
            if remaining<=0:
                break
        bounds.append(dict(cap=cap,fp_budget=budget,optimistic_fractional_recall_upper_bound=tp/int((y==1).sum())))
    records.append(dict(chunk=row['chunk'],date=row['date'],n=len(y),attack=int((y==1).sum()),
                        unique_vectors=len(groups),conflicting_groups=int(mixed.sum()),conflicting_rows=conflicting_rows,
                        conflicting_attack_rows=int(a[mixed].sum()),minimum_total_errors=int(np.minimum(a,b).sum()),bounds=bounds))
out=dict(scope='Finite observed window, deterministic function of these 77 canonicalized features; not a population or future-risk guarantee.',
         canonicalization='Nonfinite values mapped to missing, consistent with existing preprocessing; missing patterns remain distinct from finite values.',
         method='Exact pandas groupby over all 77 columns, no hash-only identity decision. Fractional mixed-group knapsack gives optimistic recall upper bounds.',records=records)
(P/'feature_label_ambiguity.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
lines=['# Exact feature/label ambiguity audit','',out['scope'],'',out['method'],'',
       '| Chunk | Conflicting rows / all rows | Attack rows in mixed groups | Minimum total errors | Optimistic recall bound at 0.1% FPR |',
       '|---|---|---|---|---|']
for r in records:
    lines.append(f"| {r['chunk']} | {r['conflicting_rows']}/{r['n']} | {r['conflicting_attack_rows']}/{r['attack']} | {r['minimum_total_errors']} | {r['bounds'][0]['optimistic_fractional_recall_upper_bound']:.2%} |")
lines+=['','The bound permits hindsight knowledge of every vector group and fractional selection of mixed groups. It is optimistic and does not establish achievable performance. It does not apply to classifiers using additional context, identifiers, history, or randomized per-flow decisions.',
        '', 'Conflicting labels do not by themselves prove annotation errors: flows may differ in omitted context. No records or labels were altered.']
(P/'FEATURE_LABEL_AMBIGUITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
