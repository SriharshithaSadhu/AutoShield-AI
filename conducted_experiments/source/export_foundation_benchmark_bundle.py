"""Export research-only native LightGBM artifacts and verify score parity."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,joblib
from lightgbm import Booster
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;O=R/'deployment/raspberry_pi_foundation'
def main():
 O.mkdir(parents=True,exist_ok=True);D=R/'data/nf_ton_iot_v3/detection_foundation';rows={};manifest=json.loads((D/'manifest.json').read_text())
 with np.load(D/'train.npz') as z:x=z['x'][:1000]
 context=np.load(D/'train_causal_context.npy')[:1000];inputs={}
 for enabled,name in ((False,'baseline46'),(True,'context50')):
  source=R/'results/causal_context_ablation'/f'model_context{enabled}_balancedFalse_seed42.joblib';model=joblib.load(source);p=O/(name+'.txt');model.booster_.save_model(str(p));native=Booster(model_file=str(p));xx=np.ascontiguousarray(np.column_stack((x,context)) if enabled else x)
  old=model.booster_.predict(xx,num_threads=1);new=native.predict(xx,num_threads=1);error=float(np.max(np.abs(old-new)));assert error<1e-12
  inputs[name]=xx;rows[name]=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),features=manifest['features']+(['log1p_source_completed_flows_prior_1s','log1p_destination_completed_flows_prior_1s','log1p_source_distinct_destination_ports_prior_1s','log1p_total_completed_flows_prior_1s'] if enabled else []),score_export_max_abs_error=error,deployment_approved=False)
 np.savez_compressed(O/'benchmark_inputs.npz',**inputs);shutil.copyfile(R/'causal_flow_context.py',O/'causal_flow_context.py')
 (O/'manifest.json').write_text(json.dumps(dict(scope='Research benchmark only. Both models fail target detection/FPR requirements; these are not approved production IDS models. No Pi measurements are bundled.',models=rows),indent=2));print(rows)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
