"""Reconstruct top-k and expanded-calibration evidence from saved artifacts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from five_track_common import probabilities,quality,confusion,benign_threshold

ROOT=Path(__file__).resolve().parent
B=ROOT/'results/five_track_pilot'


def main():
    bank=json.loads((B/'results.json').read_text())
    data=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    x=data[bank['features']].replace([np.inf,-np.inf],np.nan).astype(float).to_numpy();y=data.is_attack.to_numpy(dtype=int)
    top=json.loads((ROOT/'results/exact_topk_ablation/results.json').read_text())
    count=0
    for run in top['runs']:
        with np.load(B/f'splits_seed{run["seed"]}.npz') as z:ev=z['evaluation']
        for r in run['records']:
            bundle=joblib.load(ROOT/f'results/exact_topk_ablation/{r["name"]}_seed{run["seed"]}.joblib')
            p=probabilities(bundle,x[ev]);q=quality(y[ev],p)
            for k,v in q.items():assert np.isclose(v,r['evaluation'][k],atol=1e-12)
            for cap,t in bundle['thresholds'].items():assert confusion(y[ev],p,t)==r['operating_points'][cap]
            count+=1
    exdir=ROOT/'results/expanded_benign_calibration'
    expanded=json.loads((exdir/'results.json').read_text())
    with np.load(exdir/'benign_partitions.npz') as z:
        cx,ex=z['calibration_x'],z['evaluation_x']
        ch,eh=z['calibration_hashes'],z['evaluation_hashes']
    dh=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
    assert not set(ch)&set(eh) and not (set(ch)|set(eh))&set(dh)
    points=0
    for run in bank['runs']:
        with np.load(B/f'splits_seed{run["seed"]}.npz') as z:ev=z['evaluation']
        ax=x[ev][y[ev]==1]
        for r in run['candidates']:
            bundle=joblib.load(B/f'{r["name"]}_seed{run["seed"]}.joblib')
            cp,ep,ap=probabilities(bundle,cx),probabilities(bundle,ex),probabilities(bundle,ax)
            for record in expanded['records']:
                if record['seed']!=run['seed'] or record['model']!=r['name']:continue
                t=record['threshold']
                assert int((ep>t).sum())==record['fp'] and int((ap>t).sum())==record['tp']
                if record['policy']=='expanded_calibration':
                    assert t==benign_threshold(np.zeros(len(cp),dtype=int),cp,record['cap'])
                points+=1
    result=dict(verified_topk_bundles=count,verified_calibration_operating_points=points,
                expanded_benign_groups_disjoint=True,excluded_all_original_development_groups=True)
    (ROOT/'results/extension_audit_20261111.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
