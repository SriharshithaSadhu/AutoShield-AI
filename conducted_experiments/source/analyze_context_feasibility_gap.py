"""Label-assisted development score feasibility, not deployable operating points."""
import json,math
import numpy as np
from run_network_context_study import O,D,development_inputs,holdouts
from run_multienvironment_training import weights

def min_fpr_at_recall(y,s,target=.5):
    a=np.sort(s[y==1])[::-1];needed=int(np.ceil(target*len(a)))
    t=np.nextafter(a[needed-1],-np.inf);pred=s>t
    return dict(threshold=float(t),recall=float(pred[y==1].mean()),fpr=float(pred[y==0].mean()))

def main():
    h=holdouts();train,cal,sel=development_inputs();results=json.loads((O/'results.json').read_text())['results'];out=[]
    for r in results:
        if r['environment']!='ton_stress':continue
        with np.load(O/(r['identity']+'_'+r['policy']+'_scores.npz')) as z:s=z['ton_stress']
        point=min_fpr_at_recall(h['ton_stress']['y'],s)
        out.append(dict(identity=r['identity'],variant=r['variant'],seed=r['seed'],policy=r['policy'],
            min_empirical_fpr_at_50percent_recall=point['fpr'],realized_recall_at_that_threshold=point['recall'],
            fpr_budget_multiple=point['fpr']/.001,
            oracle_recall_at_001=r['ranking']['posthoc_threshold_oracle_recall']['0.001']))
    w,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(train['env'],train['y'])]))
    groups=[]
    for e in [0,1]:
        for y in [0,1]:
            ix=(train['env']==e)&(train['y']==y)
            groups.append(dict(environment=e,label=y,rows=int(ix.sum()),total_weight=float(w[ix].sum()),
                weight_per_row=float(w[ix][0]),family_counts={str(k):int(v) for k,v in zip(*np.unique(train['family'][ix],return_counts=True))}))
    families={name:{str(k):int(v) for k,v in zip(*np.unique(d['family'][d['y']==1],return_counts=True))}
              for name,d in [('ton_training', {k:v[train['env']==0] for k,v in train.items()}),('ton_selection',sel['ton']),('ton_stress',h['ton_stress'])]}
    minimum_n=math.ceil(math.log(.025)/math.log1p(-.001))
    prefix=json.loads((O/'prefix_coverage_control/data_audit.json').read_text())
    n=prefix['matching_benign_calibration_rows']
    audit=dict(scope=__doc__,results=out,source_weight_groups=groups,
        weighted_training_effective_sample_size=float(w.sum()**2/(w*w).sum()),
        effective_sample_size_scope='Descriptive importance-weight ESS, not a guarantee or count of independent network events.',
        ton_family_coverage=families,
        prefix_calibration_feasibility=dict(observed_benign=n,minimum_zero_exceedance_support=minimum_n,
            alpha=.001,delta=.025,zero_fp_one_sided_upper=1-.025**(1/n),
            assumptions='Fixed score, iid/exchangeable benign calibration; temporal drift may violate these assumptions.',
            decision='Insufficient support. No threshold deployed and no FPR target relaxed.'),
        caveat='Posthoc score diagnostics use development labels only. They are not calibration choices or evidence of future feasibility.')
    (O/'feasibility_gap_audit.json').write_text(json.dumps(audit,indent=2))
    print('COMPLETE feasibility audit; best posthoc FPR for >=50% recall:',min(r['min_empirical_fpr_at_50percent_recall'] for r in out))
if __name__=='__main__':main()
