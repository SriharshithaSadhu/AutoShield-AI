"""Paired resampling of observed chronological blocks, conditional on each fit.

Intervals are exploratory sensitivity summaries, not future-risk guarantees.
Blocks are sampled-period blocks, not independent real captures or networks.
"""
import json
import numpy as np
from run_network_context_study import O

def main():
    rows=json.loads((O/'results.json').read_text())['results'];out=[]
    rng=np.random.default_rng(20261121);draw=rng.multinomial(20,np.full(20,.05),size=5000)
    for policy in ['shared_window','shared_pooled','network_window','network_pooled']:
        for env in ['ton_future','unsw_future','ton_stress']:
            for seed in [42,43,44]:
                rr=[r for r in rows if r['policy']==policy and r['environment']==env and r['seed']==seed]
                base=next(r for r in rr if r['variant']=='physical14')
                attacks=np.array([w['attacks'] for w in base['windows']]);benign=np.array([w['benign'] for w in base['windows']])
                denom_a=draw@attacks;denom_b=draw@benign
                if not (denom_a>0).all() or not (denom_b>0).all():continue
                for r in rr:
                    tp=np.array([a['tp']-b['tp'] for a,b in zip(r['windows'],base['windows'])]);fp=np.array([a['fp']-b['fp'] for a,b in zip(r['windows'],base['windows'])])
                    delta_r=(draw@tp)/denom_a;delta_f=(draw@fp)/denom_b
                    out.append(dict(variant=r['variant'],seed=seed,environment=env,policy=policy,
                        recall_delta=r['recall']-base['recall'],fpr_delta=r['fpr']-base['fpr'],
                        paired_block_recall_delta_95_range=np.quantile(delta_r,[.025,.975]).tolist(),
                        paired_block_fpr_delta_95_range=np.quantile(delta_f,[.025,.975]).tolist()))
    (O/'paired_block_sensitivity.json').write_text(json.dumps(dict(scope=__doc__,replicates=5000,results=out,
        caveats=['Conditional on a fixed trained model/seed; training seeds are not pooled as independent networks.',
                 'Dependent/nonstationary chronological blocks can invalidate bootstrap confidence interpretations.',
                 'No multiple-comparison significance claim; no selection using these intervals.']),indent=2))
    print('COMPLETE paired block sensitivity',len(out),'comparisons')
if __name__=='__main__':main()
