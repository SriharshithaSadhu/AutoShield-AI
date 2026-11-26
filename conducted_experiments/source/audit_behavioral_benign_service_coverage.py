"""Training-only raw service-field census plus fixed-sample shift diagnostics.

Ports are service proxies, not application confirmation. No features or models
are changed by this audit; protected network never read.
"""
import json
from collections import Counter
from time import perf_counter
import numpy as np,pandas as pd
from run_behavioral_comparison import D,O,inputs,load_part

def main():
    wall=perf_counter();counts=Counter();n=0;rows=0;stop=json.loads((D/'manifest.json').read_text())['boundaries'][1]
    for frame in pd.read_csv(D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv',usecols=['Label','PROTOCOL','L4_DST_PORT'],nrows=stop,chunksize=250000):
        b=frame[frame.Label==0];n+=len(b);rows+=len(frame)
        for (protocol,port),v in b.groupby(['PROTOCOL','L4_DST_PORT']).size().items():counts[(int(protocol),int(port))]+=int(v)
    assert rows==13760005 and n==10495670
    train,cal,_=inputs();stress=load_part(D,'evaluation','ton');features=json.loads((D/'manifest.json').read_text())['features'];pi=features.index('PROTOCOL')
    # Actual fitting metadata is sampled using exactly the input-pool RNG.
    original=load_part(D,'train','ton');rng=np.random.default_rng(20261121);ix=np.sort(rng.choice(len(original['y']),150000,False));original={k:v[ix] for k,v in original.items()}
    samples={'ton_fit':original,'ton_calibration':cal['ton'],'ton_stress':stress};out=[]
    chosen=next(c for c in json.loads((O/'selections.json').read_text()) if c['identity']=='hybrid38_seed42')['selected']['shared_pooled']
    with np.load(O/'hybrid38_seed42_shared_pooled_scores.npz') as z:prediction=z['ton_stress']>chosen['thresholds']['shared']
    for label,protocol,port in [('tcp_dst21',6,21),('tcp_dst80',6,80),('tcp_dst443',6,443),('udp_dst53',17,53)]:
        record=dict(service_proxy=label,raw_training_prefix_benign=counts[(protocol,port)],raw_training_prefix_benign_fraction=counts[(protocol,port)]/n)
        for name,d in samples.items():
            mask=(d['y']==0)&(d['x'][:,pi]==protocol)&(d['metadata']['port']==port);record[name+'_benign']=int(mask.sum())
            if name=='ton_stress':record['hybrid_stress_fp']=int(prediction[mask].sum());record['hybrid_stress_conditional_fpr']=float(prediction[mask].mean()) if mask.any() else None
        out.append(record)
    (O/'benign_service_coverage_audit.json').write_text(json.dumps(dict(scope=__doc__,raw_training_rows=rows,raw_training_benign=n,results=out,wall_s=perf_counter()-wall,
        interpretation='Distinguishes absence in original raw training, sparse fitting exposure and calibration-period coverage. Conditional service FPR is diagnostic and never replaces overall FPR.'),indent=2));print('COMPLETE benign service census',round(perf_counter()-wall,2),out,flush=True)
if __name__=='__main__':main()
