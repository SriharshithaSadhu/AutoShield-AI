"""Capture-cluster paired bootstrap for the fixed IoTID20-to-CSE transfer replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results'/'iotid20_cse_transfer_by_capture_seed42.csv'
OUT=ROOT/'results'/'iotid20_cse_transfer_capture_cluster_bootstrap.json'
REPORT=ROOT/'results'/'IOTID20_CSE_TRANSFER_CAPTURE_CLUSTER_BOOTSTRAP.md'
B=50_000
SEED=20261109
ARMS={'default_periodic':'reference_default_periodic_refit','default_static':'reference_default_no_adaptation','feasible_periodic':'mopso_best_feasible_periodic_refit','feasible_static':'mopso_best_feasible_no_adaptation'}
COMPS=[('Default periodic - default static','default_periodic','default_static'),('Feasible periodic - feasible static','feasible_periodic','feasible_static'),('Feasible periodic - default periodic','feasible_periodic','default_periodic')]
METRICS=['weighted_f1','macro_f1','attack_recall','false_positive_rate']
def main():
 d=pd.read_csv(SRC)
 if d.capture_file.nunique()!=10 or len(d)!=40: raise RuntimeError('Unexpected capture table shape')
 piv={m:d.pivot(index='capture_file',columns='arm',values=m) for m in METRICS}
 labels=d.drop_duplicates('capture_file').set_index('capture_file')['attack_prevalence']
 rng=np.random.default_rng(SEED); results=[]
 for title,a,b in COMPS:
  for metric in METRICS:
   deltas=(piv[metric][ARMS[a]]-piv[metric][ARMS[b]]).dropna()
   if metric=='attack_recall': deltas=deltas.loc[labels.loc[deltas.index]>0]
   vals=deltas.to_numpy(float); n=len(vals)
   means=vals[rng.integers(0,n,size=(B,n))].mean(axis=1)
   results.append({'comparison':title,'arm_a':ARMS[a],'arm_b':ARMS[b],'metric':metric,'capture_n':n,'mean_delta':float(vals.mean()),'median_delta':float(np.median(vals)),'ci95_percentile':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],'captures_positive':int((vals>0).sum()),'captures_negative':int((vals<0).sum()),'capture_values':{str(k):float(v) for k,v in deltas.items()}})
 out={'source_csv':'results/iotid20_cse_transfer_by_capture_seed42.csv','unit':'capture-level paired bootstrap; 10 capture clusters, attack recall uses attack-bearing captures only','bootstrap_replicates':B,'seed':SEED,'results':results,'limitations':'Intervals describe capture heterogeneity in this single inspected CSE collection for one model seed. Captures are not established as an IID sample of networks; these are descriptive cluster-bootstrap intervals, not independent confirmation or population-level significance.'}
 OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
 lines=['# Capture-cluster uncertainty for IoTID20 to CSE transfer','', '**Purpose:** describe whether the transfer/adaptation effects are consistent across the ten CSE capture files. Paired resampling uses complete capture files as clusters; within-capture flows are never treated as independent bootstrap units. This is one explored collection and one model seed, so the intervals do not support population-level inference.','', 'Mean paired effect (arm A minus arm B). FPR and recall differences are also shown in percentage points. Percentile intervals resample capture IDs with replacement 50,000 times. Recall excludes the no-attack capture.','', '| Comparison | Outcome | Captures | Mean delta | Median delta | 95% capture-bootstrap interval | Improved captures |','|---|---|---:|---:|---:|---:|---:|']
 for r in results:
  factor=100 if r['metric'] in ('attack_recall','false_positive_rate') else 1
  unit=' pp' if factor==100 else ''
  lo,hi=r['ci95_percentile']
  lines.append(f"| {r['comparison']} | {r['metric']} | {r['capture_n']} | {r['mean_delta']*factor:+.4f}{unit} | {r['median_delta']*factor:+.4f}{unit} | [{lo*factor:+.4f}, {hi*factor:+.4f}]{unit} | {r['captures_positive']}/{r['capture_n']} |")
 lines += ['', '## Interpretation','', 'These are paired differences on the same capture files and are more defensible than row-level confidence intervals for this replay. They still do not establish that the ten capture files are exchangeable or representative of other networks. The stream schedule and dataset have already been explored; this analysis is an uncertainty description, not a fresh confirmatory test. The one-seed limitation remains. A confidence interval that excludes zero describes only this capture-level resampling scheme, not population-level superiority.','', 'Reproduction: `research_refs/bootstrap_iotid20_cse_capture_effects.py`. Input: `results/iotid20_cse_transfer_by_capture_seed42.csv`. Machine-readable output: `results/iotid20_cse_transfer_capture_cluster_bootstrap.json`.']
 REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(REPORT)
 for r in results:
  if r['metric'] in ('attack_recall','false_positive_rate','macro_f1'):
   print(r['comparison'],r['metric'],r['capture_n'],r['mean_delta'],r['ci95_percentile'],r['captures_positive'],r['captures_negative'])
if __name__=='__main__': main()
