"""Post-hoc failure diagnosis; no fitting, routing, or performance claims."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'data/cse_replay_chunk_cache_v2'
OUT=ROOT/'results/recent_target_score_diagnostic'


def main():
    manifest=json.loads((CACHE/'manifest.json').read_text())
    chunks=manifest['chunks'][::8]
    byid={c['global_chunk']:c for c in chunks}
    replay=json.loads((OUT/'results.json').read_text())
    labels_path=OUT/'raw_label_composition.json'
    composition={}
    for filename in dict.fromkeys(c['capture_file'] for c in chunks):
        selected=[c for c in chunks if c['capture_file']==filename]
        bounds=np.array([c['timestamp_min_ns'] for c in selected])
        ends=np.array([c['timestamp_max_ns'] for c in selected])
        counters=[Counter() for c in selected]
        for frame in pd.read_csv(ROOT/'data/cse_cic_ids2018/raw'/filename,
                                 usecols=['Timestamp','Label'],chunksize=200000,low_memory=False):
            ts=pd.to_datetime(frame['Timestamp'].astype(str).str.strip(),format='%d/%m/%Y %H:%M:%S',errors='coerce').to_numpy(dtype='datetime64[ns]').astype('int64')
            labs=frame['Label'].astype(str).str.strip().to_numpy()
            idx=np.searchsorted(bounds,ts,side='right')-1
            valid=(idx>=0)&(labs!='Label')
            valid &= ts <= ends[np.maximum(idx,0)]
            for i in np.unique(idx[valid]):
                counters[i].update(labs[valid & (idx==i)].tolist())
        for c,counts in zip(selected,counters):
            assert sum(counts.values())==c['rows'],(c,counts)
            assert sum(v for k,v in counts.items() if k.lower()=='benign')==c['benign']
            composition[str(c['global_chunk'])]=dict(counts)
        print('matched label counts',filename,flush=True)
    labels_path.write_text(json.dumps(composition,indent=2),encoding='utf-8')
    early_dates=replay['protocol']['early_dates']
    early=Counter()
    for c in chunks:
        if c['capture_date'] in early_dates:
            early.update(composition[str(c['global_chunk'])])
    records=[]
    for result in replay['results']:
        for row in result['records']:
            c=byid[row['chunk']]
            if min(c['benign'],c['attack'])<100:
                continue
            with np.load(OUT/f'seed{result["seed"]}_chunk{row["chunk"]}.npz') as z:
                y,s=z['y'],z['recent_target']
                cy,cs=z['calibration_y'],z['recent_target_calibration']
            training=Counter(early)
            for ci in row['train_chunks']:
                training.update(composition[str(ci)])
            target_labels=composition[str(row['chunk'])]
            unseen=[lab for lab in target_labels if lab.lower()!='benign' and training[lab]==0]
            r=dict(seed=result['seed'],chunk=row['chunk'],date=row['date'],labels=target_labels,
                   available_training_pool_labels=dict(training),absent_from_training_pool=unseen,
                   auc=float(roc_auc_score(y,s)),
                   score_quantiles={str(k):np.quantile(s[y==k],[0,.5,.9,.99,.999,1]).tolist() for k in (0,1)},
                   calibration_benign_quantiles=np.quantile(cs[cy==0],[.5,.9,.99,.999,1]).tolist(),
                   operating_points=row['arms']['recent_target']['operating_points'])
            if result['seed']==42:
                with np.load(CACHE/c['file']) as z:
                    x=z['x']
                univariate=[]
                for j,name in enumerate(manifest['features']):
                    v=x[:,j].astype(float).copy()
                    finite=np.isfinite(v)
                    v[~finite]=np.median(v[finite]) if finite.any() else 0
                    a=float(roc_auc_score(y,v))
                    univariate.append(dict(feature=name,absolute_auc=max(a,1-a),direction=1 if a>=.5 else -1))
                r['descriptive_top_univariate']=sorted(univariate,key=lambda v:-v['absolute_auc'])[:8]
                # Exact observed feature-vector collisions, not an estimate of Bayes error.
                frame=pd.DataFrame(x).replace([np.inf,-np.inf],np.nan)
                h=pd.util.hash_pandas_object(frame,index=False).to_numpy()
                table=pd.crosstab(h,y)
                r['conflicting_vector_rows']=int(table.loc[(table[0]>0)&(table[1]>0)].sum().sum())
                r['empirical_conflict_min_errors']=int(table.min(axis=1).sum())
            records.append(r)
    out=dict(status='post-hoc descriptive development diagnosis',
             limitations=['Training exposure counts refer to the available pool, not the sampled 40k rows; absence is definitive, presence is not exact sampled exposure.',
                          'Family composition aligned by complete timestamp intervals and binary/count equality, not per-row family matching.',
                          'Univariate AUC direction and ranking use evaluation labels; not deployable feature selection or a validated classifier.',
                          'Exact vector collision counts use 64-bit hashes; nonfinite values canonicalized; not a Bayes error bound.',
                          'Capture labels are publisher annotations, not independently verified ground truth.'],records=records)
    (OUT/'failure_diagnosis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Failure-window diagnosis','',
           'Post-hoc development analysis. Family exposure is the available training pool, not exact sampled rows.', '',
           '| Seed | Chunk/date | Attack labels | Absent from training pool | Score AUC | Oracle recall at 0.1% FPR |',
           '|---|---|---|---|---|---|']
    for r in records:
        o=r['operating_points'][0]['oracle']
        attacks=', '.join(f'{k}: {v}' for k,v in r['labels'].items() if k.lower()!='benign')
        lines.append(f"| {r['seed']} | {r['chunk']} / {r['date']} | {attacks} | {', '.join(r['absent_from_training_pool']) or 'none'} | {r['auc']:.4f} | {o['tp']/o['attack']:.3%} |")
    lines+=['','## Descriptive feature separation (seed 42 only)','','Evaluation-label-selected AUCs are diagnostic only. They must not be reported as validated model performance.','',
            '| Chunk | Best raw feature | Direction-free AUC | Rows in conflicting feature vectors | Empirical minimum errors in those vectors |',
            '|---|---|---|---|---|']
    for r in records:
        if r['seed']==42:
            v=r['descriptive_top_univariate'][0]
            lines.append(f"| {r['chunk']} | {v['feature']} | {v['absolute_auc']:.4f} | {r['conflicting_vector_rows']} | {r['empirical_conflict_min_errors']} |")
    lines+=['','## Limits','',*['- '+s for s in out['limitations']]]
    (OUT/'FAILURE_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
