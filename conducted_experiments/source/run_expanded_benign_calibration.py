"""Use already-local full source benign flows, excluding all prior development vectors."""
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from five_track_common import probabilities,benign_threshold,confusion

ROOT=Path(__file__).resolve().parent
P=ROOT/'results/five_track_pilot'
OUT=ROOT/'results/expanded_benign_calibration'


def main():
    OUT.mkdir(exist_ok=True)
    bank=json.loads((P/'results.json').read_text())
    features=bank['features']
    path=ROOT/'data/provenance/IoTID20_official_original.csv'
    cols=pd.read_csv(path,nrows=0).columns
    canonical=lambda s:re.sub(r'[^a-z0-9]','',s.casefold())
    mapping={f:[c for c in cols if canonical(c)==canonical(f)] for f in features}
    assert all(len(v)==1 for v in mapping.values())
    source=pd.read_csv(path,usecols=[v[0] for v in mapping.values()]+['Label','Cat'],low_memory=False)
    benign=source.Label.str.strip().str.casefold().eq('normal')
    assert np.array_equal(benign,source.Cat.str.strip().str.casefold().eq('normal'))
    x=source.loc[benign,[mapping[f][0] for f in features]].replace([np.inf,-np.inf],np.nan).astype(float).to_numpy()
    dev=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    dx=dev[features].replace([np.inf,-np.inf],np.nan).astype(float).to_numpy();dy=dev.is_attack.to_numpy(dtype=int)
    h=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
    dh=pd.util.hash_pandas_object(pd.DataFrame(dx),index=False).to_numpy()
    keep=~np.isin(h,dh)
    x,h=x[keep],h[keep]
    cal=(h%2)==0;ev=~cal
    assert not set(h[cal])&set(h[ev]) and not set(h)&set(dh)
    # Deterministic group assignment; repeated rows retained and their counts disclosed.
    np.savez_compressed(OUT/'benign_partitions.npz',calibration_x=x[cal],evaluation_x=x[ev],calibration_hashes=h[cal],evaluation_hashes=h[ev])
    records=[]
    for run in bank['runs']:
        seed=run['seed']
        with np.load(P/f'splits_seed{seed}.npz') as z: idx=z['evaluation']
        attacks=dx[idx][dy[idx]==1]
        for row in run['candidates']:
            name=row['name'];b=joblib.load(P/f'{name}_seed{seed}.joblib')
            cp=probabilities(b,x[cal]);ep=probabilities(b,x[ev]);ap=probabilities(b,attacks)
            for cap in (.001,.01):
                threshold=benign_threshold(np.zeros(len(cp),dtype=int),cp,cap)
                for policy,t in [('original_small_calibration',b['thresholds'][str(cap)]),('expanded_calibration',threshold)]:
                    fp=int((ep>t).sum());tp=int((ap>t).sum());n=len(ep)
                    upper=float(beta.ppf(.95,fp+1,n-fp)) if fp<n else 1.
                    records.append(dict(seed=seed,model=name,cap=cap,policy=policy,threshold=t,
                                        benign_validation_n=n,fp=fp,fpr=fp/n,iid_one_sided_95_upper=upper,
                                        original_eval_attack_n=len(ap),tp=tp,recall=tp/len(ap)))
    out=dict(status='source-only development diagnostic; new benign partition is not independent-network confirmation',
             counts=dict(original_benign=int(benign.sum()),excluded_development_overlap=int((~keep).sum()),
                         calibration_rows=int(cal.sum()),calibration_unique=int(len(np.unique(h[cal]))),
                         evaluation_rows=int(ev.sum()),evaluation_unique=int(len(np.unique(h[ev])))),records=records,
             limits=['All prior development feature-vector groups excluded from new benign partitions.',
                     'Only threshold changes; frozen models and probability calibration unchanged.',
                     'Benign validation and attack recall come from different source partitions; no pooled F1 or artificial prevalence claim.',
                     'Repeated and correlated source traffic remains; iid bounds are descriptive, not certified risk guarantees.',
                     'No CSE or target labels used; no temporal or cross-network claim.'])
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Expanded benign calibration','',json.dumps(out['counts']),'',
           '| Seed | Model | Calibration | Target FPR | Benign validation FPR | Attack recall |',
           '|---|---|---|---|---|---|']
    for r in records:
        if r['cap']==.001:
            lines.append(f"| {r['seed']} | {r['model']} | {r['policy']} | 0.1% | {r['fpr']:.4%} | {r['recall']:.2%} |")
    lines+=['','## Limits','',*['- '+s for s in out['limits']]]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
