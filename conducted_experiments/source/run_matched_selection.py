"""Matched finite-bank selector comparison; see dated protocol."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from five_track_common import probabilities
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent
BANK=ROOT/'results/five_track_pilot'
OUT=ROOT/'results/matched_selection'
LIMIT=1.5*1024*1024


def main(include_budget_forests=False):
    global OUT
    if include_budget_forests:
        OUT=ROOT/'results/matched_selection_budget_forests'
    OUT.mkdir(exist_ok=True)
    bank=json.loads((BANK/'results.json').read_text())
    if include_budget_forests:
        extra=json.loads((ROOT/'results/budget_forest_candidates/results.json').read_text())
        for run in bank['runs']:
            run['candidates'].extend(next(r['candidates'] for r in extra['runs'] if r['seed']==run['seed']))
    df=pd.read_csv(ROOT/'data/iotid20_binary_common77_dev.csv')
    x=df[bank['features']].replace([np.inf,-np.inf],np.nan).to_numpy();y=df.is_attack.to_numpy(dtype=int)
    with np.load(ROOT/'results/expanded_benign_calibration/benign_partitions.npz') as z:
        cx,h,ex,eh=z['calibration_x'],z['calibration_hashes'],z['evaluation_x'],z['evaluation_hashes']
    select=h%4==0;final=h%4==2
    assert np.all(select|final)
    assert not set(h[select])&set(h[final]) and not set(h)&set(eh)
    results=[]
    for run in bank['runs']:
        seed=run['seed']
        with np.load(BANK/f'splits_seed{seed}.npz') as z:tu,ev=z['tune'],z['evaluation']
        tune_attack=x[tu][y[tu]==1];eval_attack=x[ev][y[ev]==1]
        eligible=[r for r in run['candidates'] if r['artifact_bytes']<=LIMIT]
        scores={}
        for r in eligible:
            name=r['name'];b=joblib.load(r.get('path',BANK/f'{name}_seed{seed}.joblib'))
            start=perf_counter()
            scores[name]=dict(selection_benign=probabilities(b,cx[select]),tuning_attack=probabilities(b,tune_attack))
            scores[name]['selection_scoring_seconds']=perf_counter()-start
        decisions=[]
        for cap in (.001,.01):
            candidate=[]
            for r in eligible:
                s=scores[r['name']]
                t=tolerance_threshold(s['selection_benign'],cap,.05)
                assert t['threshold'] is not None
                candidate.append(dict(name=r['name'],bytes=r['artifact_bytes'],threshold=t,
                                      tuning_recall=float((s['tuning_attack']>t['threshold']).mean())))
            recall=max(candidate,key=lambda r:(r['tuning_recall'],-r['bytes'],r['name']))['name']
            selected=dict(recall_constrained=recall,
                confidence=max(eligible,key=lambda r:r['tune_raw']['mean_confidence'])['name'],
                nll=min(eligible,key=lambda r:r['tune_calibrated']['nll'])['name'],
                f1=max(eligible,key=lambda r:r['tune_calibrated']['macro_f1'])['name'],fixed_small='lgb_small')
            decisions.append(dict(cap=cap,candidates=candidate,selected=selected))
        # Persist choices before reading any final-calibration/evaluation scores.
        (OUT/f'decisions_seed{seed}.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
        final_results=[]
        for r in eligible:
            name=r['name'];b=joblib.load(r.get('path',BANK/f'{name}_seed{seed}.joblib'))
            s=scores[name]
            s.update(final_benign=probabilities(b,cx[final]),evaluation_benign=probabilities(b,ex),evaluation_attack=probabilities(b,eval_attack))
            np.savez_compressed(OUT/f'{name}_seed{seed}_scores.npz',**s)
            for cap in (.001,.01):
                threshold=tolerance_threshold(s['final_benign'],cap,.05)
                t=threshold['threshold'];assert t is not None
                fp=int((s['evaluation_benign']>t).sum());tp=int((s['evaluation_attack']>t).sum())
                fpr=fp/len(ex);recall=tp/len(eval_attack)
                final_results.append(dict(name=name,cap=cap,threshold=threshold,fp=fp,benign_n=len(ex),tp=tp,attack_n=len(eval_attack),
                                          fpr=fpr,recall=recall,cap_pass=fpr<=cap,joint_pass=fpr<=cap and recall>=.5))
        selected_results=[]
        for decision in decisions:
            cap=decision['cap']
            for method,name in decision['selected'].items():
                r=next(r for r in final_results if r['name']==name and r['cap']==cap)
                selected_results.append(dict(method=method,**r))
            rr=[r for r in final_results if r['cap']==cap]
            selected_results.append(dict(method='uniform_random_expectation',name='uniform eligible model draw',cap=cap,
                                         fpr=float(np.mean([r['fpr'] for r in rr])),recall=float(np.mean([r['recall'] for r in rr])),
                                         cap_pass_probability=float(np.mean([r['cap_pass'] for r in rr])),
                                         joint_pass_probability=float(np.mean([r['joint_pass'] for r in rr]))))
        results.append(dict(seed=seed,eligible=[r['name'] for r in eligible],excluded=[r['name'] for r in run['candidates'] if r['artifact_bytes']>LIMIT],
                            shared_bank_fit_cpu_s=sum(r['fit_cpu_s'] for r in run['candidates']),decisions=decisions,candidate_evaluation=final_results,selected_evaluation=selected_results))
    out=dict(status='development finite-bank comparison; no optimizer, novelty or independent confirmation claim',
             exploratory_budget_forest_expansion=include_budget_forests,
             limits=dict(artifact_bytes=LIMIT,primary_fpr_cap=.001,sensitivity_cap=.01,min_recall=.5,delta=.05),
             partitions=dict(selection_benign=int(select.sum()),final_calibration_benign=int(final.sum()),evaluation_benign=len(ex)),runs=results)
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Matched model-selection comparison','',json.dumps(out['partitions']), '',
           'All selectors share a <=1.5 MiB eligible bank, tuning labels, and the same independent-of-selection final threshold rule. Source development only; iid assumptions are not established.', '',
           '| Seed | FPR cap | Selector | Selected model | Validation FPR | Attack recall | Joint pass (>=50% recall) |',
           '|---|---|---|---|---|---|---|']
    for run in results:
        for r in run['selected_evaluation']:
            joint=str(r['joint_pass']) if 'joint_pass' in r else f"probability {r['joint_pass_probability']:.2f}"
            lines.append(f"| {run['seed']} | {r['cap']:.1%} | {r['method']} | {r['name']} | {r['fpr']:.4%} | {r['recall']:.2%} | {joint} |")
    lines+=['','Random rows are averages over choosing one model uniformly, not ensemble predictions. Their mean FPR is not a safety guarantee. Previously inspected validation data remain development data. Separate benign/attack partitions mean no pooled F1. Decision artifacts are written before final scoring.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-budget-forests',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.include_budget_forests)
