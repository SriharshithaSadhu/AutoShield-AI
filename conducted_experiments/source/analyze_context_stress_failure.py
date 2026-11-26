"""Development-only error decomposition of the frozen physical14 baseline.

Posthoc thresholds and attributions are diagnostics, never deployment policies
or causal proof. No protected rows and no feature redesign from these labels.
"""
from pathlib import Path
import json
import numpy as np
import joblib
from scipy.stats import ks_2samp,wasserstein_distance
from threadpoolctl import threadpool_limits
from multienvironment_features import transform,PHYSICAL_NAMES
from network_context_features import service_hints,SERVICE_NAMES
from run_network_context_study import R,D,U,O,OLD,part,development_inputs,holdouts
from run_multienvironment_training import metrics,ranking_metrics

def summary(y,score,threshold):
    pred=score>threshold;a=int((y==1).sum());b=int((y==0).sum())
    tp=int(pred[y==1].sum());fp=int(pred[y==0].sum())
    return dict(rows=len(y),attacks=a,benign=b,tp=tp,fp=fp,recall=tp/a if a else None,
                fpr=fp/b if b else None,score_quantiles=np.quantile(score,[0,.1,.5,.9,.99,1]).tolist())

def grouped(d,s,t,f):
    h=service_hints(d['x'],d['ports'],f)
    result={}
    for category in range(10):
        ix=h['category']==category
        if not ix.any():continue
        name='unknown_or_ambiguous' if category==0 else SERVICE_NAMES[category-1]
        result[name]=summary(d['y'][ix],s[ix],t)
        result[name]['source_service_hint_fraction']=float(h['source_service_hint'][ix].mean())
    return result

def main():
    train,cal,selection=development_inputs();hold=holdouts();f=json.loads((D/'manifest.json').read_text())['features']
    choice=json.loads((OLD/'physical14_environment_balanced_seed42_selection.json').read_text())['selected']
    m=joblib.load(R/choice['artifact']);threshold=choice['thresholds']['max_environment']
    allparts={**{f'{e}_calibration':d for e,d in cal.items()},**hold};report={}
    train_ton=part(D,'train');seen=set(np.unique(train_ton['family'][train_ton['y']==1]))
    ref=transform(cal['ton']['x'],f,'physical14')
    ref_scores=m.booster_.predict(ref,num_threads=1)
    for name,d in allparts.items():
        x=transform(d['x'],f,'physical14');scores=m.booster_.predict(x,num_threads=1)
        pred=scores>threshold;benign=d['y']==0;attacks=~benign
        item=dict(overall=summary(d['y'],scores,threshold),by_service=grouped(d,scores,threshold,f),
            benign_zero_duration_fraction=float((d['x'][benign,f.index('FLOW_DURATION_MILLISECONDS')]==0).mean()),
            benign_oneway_fraction=float((d['x'][benign,f.index('OUT_PKTS')]==0).mean()),
            benign_tcp_fraction=float((d['x'][benign,f.index('PROTOCOL')]==6).mean()),
            calibration_benign_discrimination={})
        for family in np.unique(d['family'][attacks]):
            ix=attacks&(d['family']==family)
            item['calibration_benign_discrimination'][str(family)]=dict(n=int(ix.sum()),
                seen_in_ton_training=bool(family in seen),
                recall=float(pred[ix].mean()),score_quantiles=np.quantile(scores[ix],[0,.1,.5,.9,.99,1]).tolist())
        if name.startswith('ton_') and attacks.any():
            seen_mask=attacks&np.isin(d['family'],list(seen));unseen=attacks&~seen_mask
            item['seen_family']=summary(d['y'][seen_mask],scores[seen_mask],threshold) if seen_mask.any() else None
            item['unseen_family']=summary(d['y'][unseen],scores[unseen],threshold) if unseen.any() else None
        if name=='ton_stress':
            # Optimal monotone threshold constrained by this period's benign labels.
            sorted_benign=np.sort(scores[benign])[::-1];k=int(np.floor(.001*benign.sum()))
            oracle=float(sorted_benign[k]);diagnostic_pred=scores>oracle
            item['undeployable_oracle']=dict(threshold=oracle,
                **summary(d['y'],scores,oracle),
                family={str(fam):dict(n=int(((d['family']==fam)&attacks).sum()),
                    recall=float(diagnostic_pred[(d['family']==fam)&attacks].mean()))
                    for fam in np.unique(d['family'][attacks])},
                scope='Period-label-assisted diagnostic; threshold is never used by the experiment policies.')
            item['feature_shift_benign']=[]
            for i,feature in enumerate(PHYSICAL_NAMES):
                a=ref[:,i];b=x[benign,i];a=a[np.isfinite(a)];b=b[np.isfinite(b)]
                if not len(a) or not len(b):continue
                scale=float(np.quantile(a,.75)-np.quantile(a,.25))
                item['feature_shift_benign'].append(dict(feature=feature,calibration_median=float(np.median(a)),
                    stress_median=float(np.median(b)),ks=float(ks_2samp(a,b).statistic),
                    wasserstein=float(wasserstein_distance(a,b)),
                    wasserstein_per_calibration_iqr=float(wasserstein_distance(a,b)/scale) if scale else None))
            fp=np.flatnonzero(benign&pred);tn=np.flatnonzero(benign&~pred);rng=np.random.default_rng(42)
            fp=rng.choice(fp,min(1000,len(fp)),False);tn=rng.choice(tn,min(1000,len(tn)),False)
            contrib=m.booster_.predict(x[np.r_[fp,tn]],pred_contrib=True,num_threads=1)
            score_raw=m.booster_.predict(x[np.r_[fp,tn]],raw_score=True,num_threads=1)
            assert np.allclose(contrib.sum(axis=1),score_raw,atol=1e-8)
            item['descriptive_tree_contributions']=dict(fp_rows=len(fp),tn_rows=len(tn),
                values=[dict(feature=feat,fp_mean=float(contrib[:len(fp),i].mean()),
                    tn_mean=float(contrib[len(fp):,i].mean())) for i,feat in enumerate(PHYSICAL_NAMES)],
                scope='Exact additive TreeSHAP raw-score explanations on sampled FP/TN. Association, not causal attribution.')
        report[name]=item
    (O/'baseline_stress_diagnosis.json').write_text(json.dumps(dict(scope=__doc__,baseline_artifact=choice['artifact'],
        threshold=threshold,ton_training_attack_families=sorted(seen),partitions=report),indent=2))
    print('COMPLETE baseline stress diagnosis',report['ton_stress']['overall'],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
