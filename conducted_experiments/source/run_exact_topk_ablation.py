"""Train-only top-k and exact additive-score baseline on saved development partitions."""
import json
from pathlib import Path
from time import perf_counter,process_time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from audit_base_objectives import exact_mask
from five_track_common import probabilities,quality,benign_threshold,confusion,conformal_thresholds,conformal_metrics

ROOT=Path(__file__).resolve().parent
BANK=ROOT/'results/five_track_pilot'
OUT=ROOT/'results/exact_topk_ablation'


def main():
    OUT.mkdir(exist_ok=True)
    bank=json.loads((BANK/'results.json').read_text())
    features=bank['features']
    data=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    x=data[features].replace([np.inf,-np.inf],np.nan).to_numpy();y=data.is_attack.to_numpy(dtype=int)
    runs=[]
    for previous in bank['runs']:
        seed=previous['seed']
        with np.load(BANK/f'splits_seed{seed}.npz') as z:
            tr,pc,tu,rc,ev=[z[k] for k in ('train','probcal','tune','riskcal','evaluation')]
        base=joblib.load(BANK/f'lgb_small_seed{seed}.joblib')
        weights=base['pipeline'][-1].feature_importances_.astype(float)
        order=np.argsort(-weights,kind='stable')
        variants={f'top{k}':order[:k] for k in (8,16,32)}
        variants['exact_additive']=np.flatnonzero(exact_mask(weights))
        records=[];bundles={}
        for name,idx in variants.items():
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                               lgb.LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,verbosity=-1,random_state=seed))
            start=process_time();pipe.fit(x[tr][:,idx],y[tr]);cpu=process_time()-start
            p=np.clip(pipe.predict_proba(x[pc][:,idx])[:,1],1e-12,1-1e-12)
            cal=LogisticRegression(C=1e6,max_iter=1000).fit(np.log(p/(1-p)).reshape(-1,1),y[pc])
            bundle=dict(pipeline=pipe,calibrator=cal,features=features,feature_indices=idx.tolist(),name=name,seed=seed)
            tp=probabilities(bundle,x[tu]);rp=probabilities(bundle,x[rc]);ep=probabilities(bundle,x[ev])
            thresholds={str(c):benign_threshold(y[rc],rp,c) for c in (.001,.01)}
            qs=conformal_thresholds(y[rc],rp)
            bundle.update(thresholds=thresholds,conformal_thresholds=qs)
            path=OUT/f'{name}_seed{seed}.joblib';joblib.dump(bundle,path)
            assert np.allclose(probabilities(joblib.load(path),x[ev[:32]]),ep[:32],atol=1e-12,rtol=0)
            for _ in range(5):probabilities(bundle,x[tu[:1]])
            times=[]
            for i in range(100):
                start=perf_counter();probabilities(bundle,x[tu[i:i+1]]);times.append((perf_counter()-start)*1000)
            reduced_groups=pd.util.hash_pandas_object(pd.DataFrame(x[:,idx]),index=False).to_numpy()
            overlap=float(np.isin(reduced_groups[ev],reduced_groups[tr]).mean())
            records.append(dict(name=name,k=len(idx),features=[features[i] for i in idx],fit_cpu_s=cpu,
                                artifact_bytes=path.stat().st_size,p95_batch1_ms=float(np.quantile(times,.95)),
                                tune=quality(y[tu],tp),evaluation=quality(y[ev],ep),
                                operating_points={c:confusion(y[ev],ep,t) for c,t in thresholds.items()},
                                conformal=conformal_metrics(y[ev],ep,qs),projected_evaluation_overlap_with_train=overlap))
            bundles[name]=bundle
            print('finished',seed,name,len(idx),flush=True)
        full=next(r for r in previous['candidates'] if r['name']=='lgb_small')
        # Selection compares tuning NLL only; all arms share split identities.
        selection=min([(r['name'],r['tune']['nll']) for r in records]+[('full77',full['tune_calibrated']['nll'])],key=lambda t:t[1])[0]
        n=int((y[rc]==0).sum())
        support=dict(benign_calibration_n=n,zero_fp_one_sided_95_upper=1-.05**(1/n),
                     iid_minimum_n_for_zero_fp_95_bound={str(c):int(np.ceil(np.log(.05)/np.log(1-c))) for c in (.001,.01)},
                     caution='Illustrative iid fixed-test binomial support calculation, not a guarantee for data-selected thresholds or dependent flows')
        runs.append(dict(seed=seed,records=records,full77=full,tuning_selected=selection,calibration_support=support))
    out=dict(status='paired development ablation; no new confirmation',runs=runs)
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Exact top-k ablation and calibration support','',
           'Same saved partitions and small LightGBM configuration; importance ranking learned on training only. All results remain development evidence.', '',
           '| Seed | Model | Features | Evaluation NLL | FPR at 0.1% target | Recall | Artifact KiB | Batch-one p95 ms | Projected evaluation/train overlap |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        f=r['full77'];op=f['operating_points']['0.001']
        lines.append(f"| {r['seed']} | full77 | 77 | {f['evaluation_calibrated']['nll']:.4f} | {op['fpr']:.3%} | {op['recall']:.2%} | {f['artifact_bytes']/1024:.1f} | {f['desktop_batch1_ms']['p95']:.3f} | 0% (original grouping) |")
        for a in r['records']:
            op=a['operating_points']['0.001']
            lines.append(f"| {r['seed']} | {a['name']} | {a['k']} | {a['evaluation']['nll']:.4f} | {op['fpr']:.3%} | {op['recall']:.2%} | {a['artifact_bytes']/1024:.1f} | {a['p95_batch1_ms']:.3f} | {a['projected_evaluation_overlap_with_train']:.2%} |")
    lines+=['','## Selection and support','']
    for r in runs:
        s=r['calibration_support']
        lines.append(f"- Seed {r['seed']}: tuning selects {r['tuning_selected']}; calibration has {s['benign_calibration_n']} benign rows. Even zero false positives gives an illustrative iid 95% upper bound of {s['zero_fp_one_sided_95_upper']:.3%}.")
    lines+=['','At least 2,995 independent benign examples with zero errors are needed for the simple fixed-test one-sided 95% binomial bound to be <=0.1%; 299 for <=1%. This does not certify our adaptively selected thresholds or grouped/dependent data.',
            '', 'Reduced projections can produce repeated model inputs across the original full-feature-separated partitions. Their overlap rates are disclosed; this is not a newly projection-grouped evaluation. Subsequent confirmatory evaluation must account for repeated traffic and temporal/site separation.',
            '', 'Exact additive-score selection optimizes the base score, not predictive performance. Artifact size includes preprocessing and calibration. The full77 timing came from the previous run and is not a paired hardware speed comparison; repeat benchmarking on-device before speed claims.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
