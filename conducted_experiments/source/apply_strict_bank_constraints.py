"""Apply strict static AutoML constraints using validation and resource files only."""
from pathlib import Path
import json,re
from constrained_candidate_selection import select
R=Path(__file__).resolve().parent;O=R/'results/representation_selection'
def main():
 protocol=json.loads((O/'protocol.json').read_text());resources=json.loads((O/'isolated_desktop_resources_v2.json').read_text());profiles={}
 for p in resources:
  match=re.search(r'candidate_seed(\d+)_id(\d+)',Path(p['path']).stem)
  if match:profiles.setdefault(int(match[1]),{})[int(match[2])]=p
 rows=[]
 for seed in protocol['seeds']:
  bank=[json.loads((O/f'candidate_seed{seed}_id{ci}.json').read_text()) for ci in range(len(protocol['bank']))]
  choices={rule:select(bank,profiles.get(seed,{}),rule) for rule in ('pooled','worst_window','family_empirical','generalization_bound')}
  rows.append(dict(seed=seed,choices=choices));print(seed,{k:v['selected'] for k,v in choices.items()})
 (O/'strict_constraints.json').write_text(json.dumps(dict(scope='No final calibration/evaluation data loaded. Static selection uses fixed development scorecards and desktop native inference/process-peak profiles. These are observed constraints, not a future-distribution guarantee or Raspberry Pi approval.',runs=rows),indent=2))
if __name__=='__main__':main()
