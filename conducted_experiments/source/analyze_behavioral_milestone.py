"""Post-evaluation diagnostics; never feeds training, selection or calibration."""
import json,hashlib,datetime
from pathlib import Path
from collections import defaultdict
import numpy as np,joblib
from behavioral_observation import FEATURE_NAMES
from behavioral_models import scores
from run_behavioral_comparison import R,D,U,O,load_part,representation,inputs

def paired_block_sensitivity(full):
    reference=next(r for r in full if r['name']=='flow14_reference');a=np.array([[w[k] for k in ['benign','attacks','fp','tp']] for w in reference['flow_windows']],float)
    n=len(a);rng=np.random.default_rng(20261123);length=10;draw=[]
    # Resample contiguous100k-flow blocks as a descriptive dependence check.
    # Not an IID confidence guarantee for this nonstationary network trace.
    for _ in range(1000):
        starts=rng.integers(0,n-length+1,size=int(np.ceil(n/length)));draw.append(np.concatenate([np.arange(s,s+length) for s in starts])[:n])
    out=[]
    for r in full:
        assert [w['window'] for w in r['flow_windows']]==[w['window'] for w in reference['flow_windows']]
        b=np.array([[w[k] for k in ['benign','attacks','fp','tp']] for w in r['flow_windows']],float);values=[]
        for ids in draw:
            x=a[ids].sum(axis=0);y=b[ids].sum(axis=0);values.append([y[3]/y[1]-x[3]/x[1],y[2]/y[0]-x[2]/x[0]])
        v=np.array(values)
        out.append(dict(method=r['name'],delta_recall=r['recall']-reference['recall'],delta_fpr=r['fpr']-reference['fpr'],
            delta_recall_resample_p025_p975=np.quantile(v[:,0],[.025,.975]).tolist(),delta_fpr_resample_p025_p975=np.quantile(v[:,1],[.025,.975]).tolist()))
    return dict(reference='flow14_reference',replicates=1000,block_flows=100000,results=out,
        caveat='Descriptive paired moving-block resampling of this fixed development trace. Nonstationarity invalidates an unconditional stationarity/IID coverage claim. Not independent networks or seed uncertainty.')

def diagnostics():
    f=json.loads((D/'manifest.json').read_text())['features'];train,cal,_=inputs();stress=load_part(D,'evaluation','ton');choices=json.loads((O/'selections.json').read_text())
    dist=[]
    for j,name in enumerate(FEATURE_NAMES+['log_global_record_rate10s','log_global_record_rate60s']):
        a=cal['ton']['behavior'][:,j];b=stress['behavior'][stress['y']==0,j]
        dist.append(dict(feature=name,calibration_benign_q01_q50_q99=np.quantile(a,[.01,.5,.99]).tolist(),
            stress_benign_q01_q50_q99=np.quantile(b,[.01,.5,.99]).tolist(),
            stress_benign_fraction_above_calibration_q99=float((b>np.quantile(a,.99)).mean())))
    in_sample=[];importance=[];family_coverage={}
    for env in [0,1]:family_coverage['ton' if env==0 else 'unsw']=sorted(np.unique(train['family'][(train['env']==env)&(train['y']==1)]).tolist())
    for c in choices:
        if c['seed']!=42:continue
        chosen=c['selected']['shared_pooled'];m=joblib.load(R/chosen['artifact']);z=representation(train,c['method'],f);s=scores(m,z)
        # Deliberately in-sample; evidence that added history encodes patterns,
        # never reported as unbiased detection performance or novelty.
        for env in [0,1]:
            for fam in np.unique(train['family'][(train['env']==env)&(train['y']==1)]):
                ix=(train['env']==env)&(train['family']==fam)&(train['y']==1)
                in_sample.append(dict(method=c['method'],environment='ton' if env==0 else 'unsw',family=str(fam),n=int(ix.sum()),
                    in_sample_recall=float((s[ix]>chosen['thresholds']['shared']).mean())))
        if c['method']=='quantile_tail_anomaly':continue
        from multienvironment_features import PHYSICAL_NAMES
        names=PHYSICAL_NAMES if c['method']=='flow14' else PHYSICAL_NAMES+FEATURE_NAMES if c['method']=='hybrid38' else FEATURE_NAMES+(['log_global_record_rate10s','log_global_record_rate60s'] if c['method']=='behavior26_global_ablation' else [])
        gain=m.booster_.feature_importance(importance_type='gain');order=np.argsort(-gain)
        importance.append(dict(method=c['method'],top10=[dict(feature=names[i],gain_fraction=float(gain[i]/max(1,gain.sum()))) for i in order[:10]],
            caveat='Training split gain, descriptive only; correlated-feature gain is not a causal effect.'))
    groups={
        'source_history_absent':stress['behavior'][:,22]==0,
        'source_history_present':stress['behavior'][:,22]!=0,
        'any_contact_count_capped':np.any(stress['behavior'][:,16:20]>=np.float32(np.log1p(64)),axis=1),
        'no_contact_count_capped':np.all(stress['behavior'][:,16:20]<np.float32(np.log1p(64)),axis=1),
        'any_behavior_above_calibration_benign_q99':np.any(stress['behavior'][:,:24]>np.quantile(cal['ton']['behavior'][:,:24],.99,axis=0),axis=1),
        'ton_fit_family_name_present':np.isin(stress['family'],family_coverage['ton']),
        'ton_fit_family_name_absent':~np.isin(stress['family'],family_coverage['ton'])}
    group_out=[]
    for c in choices:
        if c['seed']!=42:continue
        chosen=c['selected']['shared_pooled']
        with np.load(O/(c['identity']+'_shared_pooled_scores.npz')) as z:s=z['ton_stress']
        pred=s>chosen['thresholds']['shared']
        for name,mask in groups.items():
            a=mask&(stress['y']==1);b=mask&(stress['y']==0);tp=int(pred[a].sum());fp=int(pred[b].sum())
            group_out.append(dict(method=c['method'],group=name,attacks=int(a.sum()),benign=int(b.sum()),tp=tp,fp=fp,
                recall=tp/int(a.sum()) if a.any() else None,fpr=fp/int(b.sum()) if b.any() else None))
    return dict(training_attack_family_coverage=family_coverage,feature_shift=dist,in_sample_family_separation=in_sample,feature_importance=importance,
        stress_subgroups=group_out,subgroup_caveat='Post-evaluation descriptive associations only; no causal attribution. Family-name absence in ToN is not absence of related techniques across network taxonomies. Q99 union is not a calibrated OOD test.')

def sensor_accounting():
    data=json.loads((O/'sensor_sensitivity_results.json').read_text());records=[]
    for r in data['results']:
        x=dict(r);jitter=r['scenario']=='bounded_export_jitter_1000ms'
        x['right_censored_probe_rows']=r['all_probe_rows']-r['observed_probe_rows'] if jitter else 0
        x['unobserved_probe_rows']=0 if jitter else r['all_probe_rows']-r['observed_probe_rows']
        if jitter:
            x['legacy_unconditional_field_interpretation']='Includes six administrative boundary-censored probes in denominator; do not interpret as unobserved traffic.'
            x['coverage_inclusive_recall']=None
        else:x['coverage_inclusive_recall']=x['unconditional_recall_including_unobserved']
        records.append(x)
    (O/'sensor_sensitivity_accounting_v2.json').write_text(json.dumps(dict(results=records,
        scope='Clarifies observation loss versus right censoring; original v1 results retained without alteration. Conditional recall/FPR unchanged.'),indent=2))

def main():
    full=json.loads((O/'full_replay_results.json').read_text())['results'];data=diagnostics();data['paired_block_sensitivity']=paired_block_sensitivity(full)
    data['evaluated_at']=datetime.datetime.now().astimezone().isoformat();data['scope']=__doc__
    (O/'post_evaluation_diagnostics.json').write_text(json.dumps(data,indent=2));sensor_accounting();print('COMPLETE post-evaluation diagnostics',flush=True)
if __name__=='__main__':main()
