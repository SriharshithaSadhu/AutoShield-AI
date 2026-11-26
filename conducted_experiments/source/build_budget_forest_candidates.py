"""Exploratory second-stage bank after matched pilot failure; no claim of fresh confirmation."""
import json
from pathlib import Path
from time import process_time,perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from five_track_common import probabilities,quality

ROOT=Path(__file__).resolve().parent
B=ROOT/'results/five_track_pilot'
OUT=ROOT/'results/budget_forest_candidates'


def main():
    OUT.mkdir(exist_ok=True)
    bank=json.loads((B/'results.json').read_text())
    df=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    x=df[bank['features']].replace([np.inf,-np.inf],np.nan).to_numpy();y=df.is_attack.to_numpy(dtype=int)
    allrows=[]
    for run in bank['runs']:
        seed=run['seed']
        with np.load(B/f'splits_seed{seed}.npz') as z:tr,pc,tu=[z[k] for k in ('train','probcal','tune')]
        models={f'rf{n}':RandomForestClassifier(n_estimators=n,max_depth=16,n_jobs=1,random_state=seed) for n in (40,60)}
        models.update({f'et{n}':ExtraTreesClassifier(n_estimators=n,max_depth=16,n_jobs=1,random_state=seed) for n in (20,30)})
        rows=[]
        for name,m in models.items():
            pipeline=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),m)
            start=process_time();pipeline.fit(x[tr],y[tr]);cpu=process_time()-start
            p=np.clip(pipeline.predict_proba(x[pc])[:,1],1e-12,1-1e-12)
            calibrator=LogisticRegression(C=1e6,max_iter=1000).fit(np.log(p/(1-p)).reshape(-1,1),y[pc])
            b=dict(pipeline=pipeline,calibrator=calibrator,name=name,seed=seed,features=bank['features'])
            path=OUT/f'{name}_seed{seed}.joblib';joblib.dump(b,path)
            assert np.allclose(probabilities(b,x[tu[:32]]),probabilities(joblib.load(path),x[tu[:32]]),atol=1e-12)
            times=[]
            for i in range(65):
                start=perf_counter();probabilities(b,x[tu[i:i+1]]);dt=(perf_counter()-start)*1000
                if i>=5:times.append(dt)
            r=dict(seed=seed,name=name,artifact_bytes=path.stat().st_size,fit_cpu_s=cpu,
                   tune_raw=quality(y[tu],probabilities(b,x[tu],False)),
                   tune_calibrated=quality(y[tu],probabilities(b,x[tu])),
                   desktop_batch1_ms=dict(p50=float(np.median(times)),p95=float(np.quantile(times,.95))),
                   path=str(path))
            rows.append(r);print(seed,name,'MiB',r['artifact_bytes']/1048576,flush=True)
        allrows.append(dict(seed=seed,candidates=rows))
    out=dict(status='exploratory bank expansion, training/calibration/tuning only',runs=allrows)
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
