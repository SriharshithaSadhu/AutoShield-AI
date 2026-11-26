"""Reconstruct all source evaluation metrics from saved deployment artifacts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from five_track_common import probabilities,quality,confusion,conformal_metrics,conformal_thresholds

ROOT=Path(__file__).resolve().parent
P=ROOT/'results/five_track_pilot'


def main():
    results=json.loads((P/'results.json').read_text())
    data=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    x=data[results['features']].replace([np.inf,-np.inf],np.nan).to_numpy();y=data.is_attack.to_numpy(dtype=int)
    groups=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
    checked=0
    for run in results['runs']:
        with np.load(P/f'splits_seed{run["seed"]}.npz') as split:
            indices={k:split[k] for k in split.files}
        joined=np.concatenate(list(indices.values()))
        assert len(joined)==len(x) and len(np.unique(joined))==len(x)
        names=list(indices)
        for i,a in enumerate(names):
            for b in names[i+1:]:assert not set(groups[indices[a]])&set(groups[indices[b]])
        ev=indices['evaluation'];rc=indices['riskcal']
        for record in run['candidates']:
            bundle=joblib.load(P/f'{record["name"]}_seed{run["seed"]}.joblib')
            ep=probabilities(bundle,x[ev]);rp=probabilities(bundle,x[rc])
            assert np.isfinite(ep).all() and ((ep>=0)&(ep<=1)).all()
            got=quality(y[ev],ep)
            for key,v in got.items():assert np.isclose(v,record['evaluation_calibrated'][key],atol=1e-12)
            for cap,t in bundle['thresholds'].items():
                assert confusion(y[ev],ep,t)==record['operating_points'][cap]
                cal=confusion(y[rc],rp,t)
                assert cal['fp']<=float(cap)*cal['benign']
            assert conformal_metrics(y[ev],ep,bundle['conformal_thresholds'])==record['conformal']
            assert conformal_thresholds(y[rc],rp)==bundle['conformal_thresholds']
            checked+=1
    out=dict(verified_bundles=checked,checks=['partition exhaustiveness/disjointness','feature-group separation',
            'reloaded probability metrics','empirical calibration FPR caps','evaluation confusion counts','conformal thresholds and metrics'])
    (P/'audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
