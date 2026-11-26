"""Posthoc development control isolating features from fit-selection effects.

Reuses each variant's predeclared step0 model and shared calibration threshold.
No new fitting or threshold tuning. Not fresh confirmation.
"""
import json,datetime
import numpy as np,joblib
from run_network_context_study import R,D,O,holdouts
from network_context_features import transform
from run_multienvironment_training import metrics,ranking_metrics

def main():
    b=O/'fixed_fit_control';b.mkdir(exist_ok=True);protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
        scope=__doc__,step=0,calibration='Identical fixed historical benign calibration, common maximum-environment threshold.',
        fitting_budget='One reused matched fit per feature/seed; no optimization.')
    (b/'protocol.json').write_text(json.dumps(protocol,indent=2));hold=holdouts();features=json.loads((D/'manifest.json').read_text())['features'];out=[]
    for c in json.loads((O/'selections.json').read_text()):
        fixed=c['candidates'][0];model=joblib.load(R/fixed['artifact']);threshold=fixed['thresholds']['shared']
        for environment,d in hold.items():
            x=transform(d['x'],d['ports'],features,c['variant']);s=model.booster_.predict(x,num_threads=1)
            out.append(dict(identity=c['identity'],variant=c['variant'],seed=c['seed'],environment=environment,policy='fixed_step0_shared',
                ranking=ranking_metrics(d['y'],s),**metrics(d['y'],s,threshold,d['family'],windows=20)))
    (b/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2))
    print('COMPLETE reused fixed-fit feature control',len(out),'points')
if __name__=='__main__':main()
