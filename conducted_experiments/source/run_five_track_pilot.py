"""Bounded shared evidence bank; see FIVE_TRACK_EXTENSION_PROTOCOL.md."""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter,process_time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from five_track_common import probabilities,quality,confusion,benign_threshold,conformal_thresholds,conformal_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/five_track_pilot'


def main():
    OUT.mkdir(exist_ok=True)
    source_path=ROOT/'data/iotid20_binary_common77_dev.csv'
    df=pd.read_csv(source_path)
    manifest=json.loads((ROOT/'data/cse_replay_chunk_cache_v2/manifest.json').read_text())
    features=manifest['features']
    x=df[features].replace([np.inf,-np.inf],np.nan).to_numpy()
    y=df.is_attack.to_numpy(dtype=int)
    groups=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
    runs=[]
    probe=x[:64] # fixed label-free explanation probe, never a selection metric
    explanation_vectors={}
    for seed in (42,43,44):
        cv=StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=seed)
        parts=[test for _,test in cv.split(x,y,groups)]
        for i,a in enumerate(parts):
            for b in parts[i+1:]: assert not set(groups[a])&set(groups[b])
        train=np.concatenate(parts[:2]); probcal,tune,riskcal,evaluation=parts[2:]
        assert all(len(np.unique(y[z]))==2 for z in [train,probcal,tune,riskcal,evaluation])
        np.savez_compressed(OUT/f'splits_seed{seed}.npz',train=train,probcal=probcal,tune=tune,riskcal=riskcal,evaluation=evaluation)
        models={
            'lgb_small':lgb.LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,verbosity=-1,random_state=seed),
            'lgb_large':lgb.LGBMClassifier(n_estimators=150,num_leaves=31,n_jobs=1,verbosity=-1,random_state=seed),
            'xgboost':xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.1,n_jobs=1,tree_method='hist',eval_metric='logloss',random_state=seed),
            'random_forest':RandomForestClassifier(n_estimators=100,max_depth=16,n_jobs=1,random_state=seed),
            'extra_trees':ExtraTreesClassifier(n_estimators=100,max_depth=16,n_jobs=1,random_state=seed),
            'logistic':LogisticRegression(C=1,max_iter=1500,random_state=seed)}
        records=[]; bundles={}
        for name,model in models.items():
            stages=[SimpleImputer(strategy='median',keep_empty_features=True)]
            if name=='logistic': stages.append(StandardScaler())
            pipeline=make_pipeline(*stages,model)
            start=process_time();pipeline.fit(x[train],y[train]);fit_cpu=process_time()-start
            cp=np.clip(pipeline.predict_proba(x[probcal])[:,1],1e-12,1-1e-12)
            calibrator=LogisticRegression(C=1e6,max_iter=1000).fit(np.log(cp/(1-cp)).reshape(-1,1),y[probcal])
            bundle=dict(pipeline=pipeline,calibrator=calibrator,features=features,name=name,seed=seed)
            tp=probabilities(bundle,x[tune]);rawtp=probabilities(bundle,x[tune],False)
            rp=probabilities(bundle,x[riskcal]);ep=probabilities(bundle,x[evaluation]);raw=probabilities(bundle,x[evaluation],False)
            qs=conformal_thresholds(y[riskcal],rp)
            thresholds={str(cap):benign_threshold(y[riskcal],rp,cap) for cap in (.001,.01)}
            bundle.update(thresholds=thresholds,conformal_thresholds=qs)
            path=OUT/f'{name}_seed{seed}.joblib';joblib.dump(bundle,path)
            loaded=joblib.load(path)
            assert np.allclose(probabilities(loaded,x[evaluation[:32]]),ep[:32],rtol=0,atol=1e-12)
            # Batch-one whole preprocessing + model + probability calibration.
            for _ in range(5): probabilities(bundle,x[tune[:1]])
            times=[]
            for i in range(60):
                start=perf_counter();probabilities(bundle,x[tune[i:i+1]]);times.append((perf_counter()-start)*1000)
            subgroup=[]
            protocol=x[tune,features.index('Protocol')]
            for value in np.unique(protocol[np.isfinite(protocol)]):
                mask=protocol==value
                attack=mask&(y[tune]==1)
                if attack.sum()>=30:
                    subgroup.append(dict(protocol=float(value),attacks=int(attack.sum()),recall=float((tp[attack]>=.5).mean())))
            record=dict(name=name,fit_cpu_s=fit_cpu,artifact_bytes=path.stat().st_size,
                        desktop_batch1_ms=dict(p50=float(np.median(times)),p95=float(np.quantile(times,.95))),
                        tune_raw=quality(y[tune],rawtp),tune_calibrated=quality(y[tune],tp),
                        evaluation_raw=quality(y[evaluation],raw),evaluation_calibrated=quality(y[evaluation],ep),
                        operating_points={cap:confusion(y[evaluation],ep,t) for cap,t in thresholds.items()},
                        conformal=conformal_metrics(y[evaluation],ep,qs),
                        riskcal_class_counts={str(c):int((y[riskcal]==c).sum()) for c in (0,1)},
                        natural_protocol_subgroups=subgroup)
            if name.startswith('lgb'):
                transformed=pipeline[:-1].transform(probe)
                start=perf_counter();contrib=model.predict(transformed,pred_contrib=True);seconds=perf_counter()-start
                margins=model.predict(transformed,raw_score=True)
                assert np.allclose(contrib.sum(axis=1),margins,atol=1e-7)
                importance=np.abs(contrib[:,:-1]).mean(axis=0)
                explanation_vectors[(seed,name)]=importance
                record['explanation']=dict(seconds_64_rows=seconds,additivity_max_error=float(np.max(np.abs(contrib.sum(axis=1)-margins))),
                                            top10=[features[i] for i in np.argsort(-importance)[:10]],scope='raw uncalibrated model margin on fixed probe')
            records.append(record);bundles[name]=bundle
            print('fit',seed,name,flush=True)
        # Selection does not access final risk calibration or evaluation metrics.
        confidence=max(records,key=lambda r:r['tune_raw']['mean_confidence'])['name']
        nll=min(records,key=lambda r:r['tune_calibrated']['nll'])['name']
        f1=max(records,key=lambda r:r['tune_calibrated']['macro_f1'])['name']
        # Four objectives: F1 loss, proper score, artifact bytes, worst natural-protocol miss rate.
        vectors=np.array([[1-r['tune_calibrated']['macro_f1'],r['tune_calibrated']['nll'],r['artifact_bytes'],
                           1-min((s['recall'] for s in r['natural_protocol_subgroups']),default=0)] for r in records])
        nondominated=[records[i]['name'] for i in range(len(records)) if not any(np.all(vectors[j]<=vectors[i]) and np.any(vectors[j]<vectors[i]) for j in range(len(records)) if i!=j)]
        transfer=[]
        for c in manifest['chunks'][::8]:
            if c['capture_date']<'2018-02-22':continue
            path=ROOT/'data/cse_replay_chunk_cache_v2'/c['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
            with np.load(path) as z: xx,yy=z['x'],z['y']
            xx=np.where(np.isfinite(xx),xx,np.nan)
            arms={}
            for name in sorted({confidence,nll,f1}):
                b=bundles[name];p=probabilities(b,xx)
                arms[name]=dict(operating_points={cap:confusion(yy,p,t) for cap,t in b['thresholds'].items()},
                               conformal=conformal_metrics(yy,p,b['conformal_thresholds']))
            transfer.append(dict(chunk=c['global_chunk'],date=c['capture_date'],arms=arms))
        result=dict(seed=seed,partition_sizes={k:len(v) for k,v in dict(train=train,probcal=probcal,tune=tune,riskcal=riskcal,evaluation=evaluation).items()},
                    candidates=records,selections=dict(confidence=confidence,calibrated_nll=nll,calibrated_f1=f1),
                    four_objective_archive=nondominated,transfer=transfer)
        (OUT/f'seed{seed}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');runs.append(result)
        print('completed seed',seed,flush=True)
    stability=[]
    for name in ('lgb_small','lgb_large'):
        for a,b in ((42,43),(42,44),(43,44)):
            aa=set(np.argsort(-explanation_vectors[(a,name)])[:10]);bb=set(np.argsort(-explanation_vectors[(b,name)])[:10])
            stability.append(dict(model=name,seeds=[a,b],top10_jaccard=len(aa&bb)/len(aa|bb)))
    out=dict(status='development pilot, all five tracks remain incomplete',platform=platform.platform(),
             source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),features=features,runs=runs,explanation_stability=stability)
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
