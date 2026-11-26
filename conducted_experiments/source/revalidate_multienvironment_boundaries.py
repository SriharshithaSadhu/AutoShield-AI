"""Revalidate cached fitted candidates after timestamp-only boundary correction."""
from pathlib import Path
import json,shutil,datetime,numpy as np,joblib
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,D,U,read,metrics,ton_temporal_boundaries
from multienvironment_features import transform
ROOT=R/'results/multienvironment_2026-11-21'
def main():
 cal,stop=ton_temporal_boundaries();assert cal==60000
 tune=read(D/'tune.npz');us=read(U/'selection.npz');selection={'ton':{k:v[cal:stop] for k,v in tune.items()},'unsw':us};features=json.loads((D/'manifest.json').read_text())['features'];x={mode:{e:transform(v['x'],features,mode) for e,v in selection.items()} for mode in ['raw46','physical14','harmonized14']};count=0
 for directory in (ROOT,ROOT/'harmonization'):
  backup=directory/'pre_boundary_correction';backup.mkdir(exist_ok=True)
  for p in sorted(directory.glob('*_selection.json')):
   original=json.loads(p.read_text());shutil.copyfile(p,backup/p.name)
   mode=original['representation']
   for c in original['candidates']:
    m=joblib.load(R/c['artifact']);v={e:metrics(selection[e]['y'],m.booster_.predict(xx,num_threads=1),c['thresholds']['max_environment'],selection[e]['family']) for e,xx in x[mode].items()};c['validation']=v;c['worst_fpr']=max(z['worst_window_fpr'] for z in v.values());c['worst_recall']=min(z['worst_window_recall'] for z in v.values())
   eligible=[v for v in original['candidates'] if v['bytes']<=1572864 and v['p95_ms']<=5]
   chosen=max(eligible or original['candidates'],key=lambda v:(v['worst_fpr']<=.001,-max(0,v['worst_fpr']-.001),v['worst_recall'],-v['step']))
   original['selected']=chosen;original['development_operating_point_feasible']=bool(eligible) and chosen['worst_fpr']<=.001 and chosen['worst_recall']>=.5;original['boundary_correction']='Selection ends at full completion-time group; no held-out outcomes had been scored.'
   p.write_text(json.dumps(original,indent=2),encoding='utf-8');count+=1;print('revalidated',p.name,flush=True)
 report=dict(corrected_at=datetime.datetime.now().astimezone().isoformat(),old_selection_stop=130000,new_selection_stop=stop,calibration_stop=cal,revalidated_choices=count,reason='Equal-time group crossed selection/holdout split. Corrected with timestamp metadata only before held-out scoring; saved fits reused.',protected_labels_used=False)
 (ROOT/'boundary_correction.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
