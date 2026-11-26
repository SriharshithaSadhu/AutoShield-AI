"""Known order-statistic tolerance baseline; development evidence, not a new guarantee."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binom
from threadpoolctl import threadpool_limits
from five_track_common import probabilities,benign_threshold

ROOT=Path(__file__).resolve().parent
BANK=ROOT/'results/five_track_pilot'
OUT=ROOT/'results/tolerance_threshold_audit'


def tolerance_threshold(scores,alpha,delta):
    """Strict > prediction; largest allowed calibration exceedance count.

    For a fixed score and iid benign calibration, binom.cdf(k,n,alpha)
    bounds the probability that the selected threshold has population FPR>alpha.
    Ties make strict-threshold classification conservative.
    """
    s=np.sort(np.asarray(scores));n=len(s)
    if not 0<alpha<1 or not 0<delta<1:raise ValueError('invalid alpha/delta')
    if n==0 or binom.cdf(0,n,alpha)>delta:
        return dict(threshold=None,k=None,status='insufficient_support')
    lo,hi=0,n-1
    while lo<hi:
        mid=(lo+hi+1)//2
        if binom.cdf(mid,n,alpha)<=delta:lo=mid
        else:hi=mid-1
    assert binom.cdf(lo,n,alpha)<=delta
    assert lo==n-1 or binom.cdf(lo+1,n,alpha)>delta
    return dict(threshold=float(s[n-lo-1]),k=lo,status='supported_under_iid_assumptions',
                binomial_tail=float(binom.cdf(lo,n,alpha)))


def main():
    OUT.mkdir(exist_ok=True)
    bank=json.loads((BANK/'results.json').read_text())
    source=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    x=source[bank['features']].replace([np.inf,-np.inf],np.nan).to_numpy();y=source.is_attack.to_numpy(dtype=int)
    with np.load(ROOT/'results/expanded_benign_calibration/benign_partitions.npz') as z:
        cx,ex=z['calibration_x'],z['evaluation_x']
    records=[]
    for run in bank['runs']:
        seed=run['seed']
        with np.load(BANK/f'splits_seed{seed}.npz') as z:idx=z['evaluation']
        ax=x[idx][y[idx]==1]
        for candidate in run['candidates']:
            name=candidate['name'];b=joblib.load(BANK/f'{name}_seed{seed}.joblib')
            cp,ep,ap=probabilities(b,cx),probabilities(b,ex),probabilities(b,ax)
            np.savez_compressed(OUT/f'{name}_seed{seed}_scores.npz',calibration=cp,benign_validation=ep,attack_validation=ap)
            for alpha in (.001,.01):
                policies={'empirical':dict(threshold=benign_threshold(np.zeros(len(cp),dtype=int),cp,alpha),k=int(np.floor(alpha*len(cp))),status='empirical_only'),
                          'tolerance_pointwise':tolerance_threshold(cp,alpha,.05),
                          'tolerance_family18':tolerance_threshold(cp,alpha,.05/18)}
                for policy,details in policies.items():
                    t=details['threshold']
                    if t is None:
                        records.append(dict(seed=seed,model=name,cap=alpha,policy=policy,**details));continue
                    records.append(dict(seed=seed,model=name,cap=alpha,policy=policy,**details,
                                        calibration_exceedances=int((cp>t).sum()),
                                        benign_n=len(ep),fp=int((ep>t).sum()),fpr=float((ep>t).mean()),
                                        attack_n=len(ap),tp=int((ap>t).sum()),recall=float((ap>t).mean())))
    summary=[]
    for cap in (.001,.01):
        for policy in ('empirical','tolerance_pointwise','tolerance_family18'):
            rows=[r for r in records if r['cap']==cap and r['policy']==policy and r['threshold'] is not None]
            summary.append(dict(cap=cap,policy=policy,models=len(rows),cap_pass=sum(r['fpr']<=cap for r in rows),
                                joint_pass=sum(r['fpr']<=cap and r['recall']>=.5 for r in rows),
                                median_recall=float(np.median([r['recall'] for r in rows])),
                                median_fpr=float(np.median([r['fpr'] for r in rows]))))
    out=dict(status='exploratory source-only comparison of established threshold baselines',records=records,summary=summary,
             caveats=['Previously inspected validation sets; no fresh confirmation.',
                      'Scores frozen before expanded calibration; no validation labels used to select thresholds.',
                      'IID benign assumption is not established: repeated/correlated flows and group-hash splitting remain.',
                      'Family18 uses Bonferroni delta/18 for a fixed cap; no joint claim across both caps or arbitrary future model search.',
                      'Counts across 18 models are correlated observations, not independent statistical replications.',
                      'Separate benign and attack validation partitions; no pooled prevalence-dependent metric.',
                      'This is an existing Neyman-Pearson/order-statistic method, not a new algorithm or drift guarantee.'])
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Tolerance-threshold development audit','','All thresholds use only the expanded benign calibration partition. Guarantees require iid assumptions not established here.','',
           '| FPR target | Policy | Models under cap | Models under cap and >=50% recall | Median recall | Median FPR |',
           '|---|---|---|---|---|---|']
    for r in summary:
        lines.append(f"| {r['cap']:.1%} | {r['policy']} | {r['cap_pass']}/{r['models']} | {r['joint_pass']}/{r['models']} | {r['median_recall']:.2%} | {r['median_fpr']:.4%} |")
    lines+=['','## Limits','',*['- '+s for s in out['caveats']]]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
