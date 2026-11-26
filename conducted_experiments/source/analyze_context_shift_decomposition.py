"""Descriptive service/protocol standardization; no causal attribution claim."""
import json
import numpy as np,joblib
from run_network_context_study import R,D,O,OLD,development_inputs,holdouts
from network_context_features import service_hints
from multienvironment_features import transform

def main():
    _,cal,_=development_inputs();stress=holdouts()['ton_stress'];f=json.loads((D/'manifest.json').read_text())['features']
    selected=json.loads((OLD/'physical14_environment_balanced_seed42_selection.json').read_text())['selected']
    m=joblib.load(R/selected['artifact']);threshold=selected['thresholds']['max_environment'];groups={}
    for name,d in [('calibration',cal['ton']),('stress',stress)]:
        good=d['y']==0;x=d['x'][good];p=d['ports'][good];h=service_hints(x,p,f)
        pred=m.booster_.predict(transform(x,f,'physical14'),num_threads=1)>threshold
        protocol=x[:,f.index('PROTOCOL')].astype(int);key=np.array([f'{a}:{b}' for a,b in zip(h['category'],protocol)])
        groups[name]={k:dict(n=int((key==k).sum()),fp=int(pred[key==k].sum())) for k in np.unique(key)}
    n0=sum(v['n'] for v in groups['calibration'].values());n1=sum(v['n'] for v in groups['stress'].values());rows=[]
    for key in sorted(set(groups['calibration'])|set(groups['stress'])):
        a=groups['calibration'].get(key,dict(n=0,fp=0));b=groups['stress'].get(key,dict(n=0,fp=0))
        w0=a['n']/n0;w1=b['n']/n1;r0=a['fp']/a['n'] if a['n'] else 0;r1=b['fp']/b['n'] if b['n'] else 0
        rows.append(dict(service_protocol=key,calibration=a,stress=b,calibration_weight=w0,stress_weight=w1,
            calibration_fpr=r0 if a['n'] else None,stress_fpr=r1 if b['n'] else None,
            composition_component=(w1-w0)*r0,
            within_supported_group_component=w1*(r1-r0) if a['n'] else 0,
            unsupported_group_component=w1*r1 if not a['n'] else 0))
    fpr0=sum(v['fp'] for v in groups['calibration'].values())/n0;fpr1=sum(v['fp'] for v in groups['stress'].values())/n1
    components={k:sum(r[k] for r in rows) for k in ['composition_component','within_supported_group_component','unsupported_group_component']}
    assert abs(sum(components.values())-(fpr1-fpr0))<1e-12
    (O/'stress_shift_decomposition.json').write_text(json.dumps(dict(scope=__doc__,calibration_fpr=fpr0,stress_fpr=fpr1,
        components=components,unsupported_benign_mass=sum(r['stress_weight'] for r in rows if r['calibration_fpr'] is None),groups=rows,
        caveat='Observed standardization identity on development samples. Service hints are fallible. Rates have sampling error; no causal mechanism or future performance is established.'),indent=2))
    print('COMPLETE descriptive FPR shift decomposition',fpr0,fpr1,components)
if __name__=='__main__':main()
