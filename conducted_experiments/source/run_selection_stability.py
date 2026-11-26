"""Exploratory grouped-bootstrap selector; no new statistical guarantee."""
import json
from pathlib import Path
from time import process_time
import numpy as np
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/matched_selection_budget_forests'
OUT=ROOT/'results/selection_stability'

def main():
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT/'results/expanded_benign_calibration/benign_partitions.npz') as z:
        hashes=z['calibration_hashes'];hashes=hashes[hashes%4==0]
    groups,inv=np.unique(hashes,return_inverse=True)
    # Same grouped resamples for all candidate models and seeds. No final data.
    rng=np.random.default_rng(20261111)
    weights=[np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[inv] for _ in range(200)]
    decisions=[];start=process_time()
    for seed in (42,43,44):
        source=json.loads((SOURCE/f'decisions_seed{seed}.json').read_text())
        for decision in source:
            cap=decision['cap'];candidates=[]
            for candidate in decision['candidates']:
                name=candidate['name']
                with np.load(SOURCE/f'{name}_seed{seed}_scores.npz') as z:
                    benign=z['selection_benign'];attack=np.sort(z['tuning_attack'])
                assert len(benign)==len(hashes)
                recalls=[]
                for weight in weights:
                    threshold=tolerance_threshold(np.repeat(benign,weight),cap,.05)['threshold']
                    recalls.append(float(1-np.searchsorted(attack,threshold,side='right')/len(attack)))
                candidates.append(dict(name=name,bytes=candidate['bytes'],original_tuning_recall=candidate['tuning_recall'],
                    bootstrap_recall_q10=float(np.quantile(recalls,.1)),bootstrap_recall_median=float(np.median(recalls)),
                    bootstrap_zero_recall_rate=float(np.mean(np.asarray(recalls)==0))))
            chosen=max(candidates,key=lambda r:(r['bootstrap_recall_q10'],-r['bytes'],r['name']))['name']
            decisions.append(dict(seed=seed,cap=cap,selected=chosen,candidates=candidates))
            print(seed,cap,chosen,flush=True)
    # Persist decisions before joining final outcomes. Prior viewed outcomes mean exploratory.
    (OUT/'decisions.json').write_text(json.dumps(decisions,indent=2))
    source=json.loads((SOURCE/'results.json').read_text());results=[]
    for d in decisions:
        run=next(r for r in source['runs'] if r['seed']==d['seed'])
        result=next(r for r in run['candidate_evaluation'] if r['name']==d['selected'] and r['cap']==d['cap'])
        old=next(r for r in run['selected_evaluation'] if r['method']=='recall_constrained' and r['cap']==d['cap'])
        results.append(dict(seed=d['seed'],**result,original_selector=old))
    output=dict(status='exploratory development heuristic; not preregistered, not a coverage guarantee',
        bootstrap_replicates=200,bootstrap_unit='identical-feature hash groups',selection_quantile=.1,
        selection_cpu_seconds=process_time()-start,results=results)
    (OUT/'results.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
