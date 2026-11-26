"""Reconstruct stored decisions and all evaluation counts from saved scores."""
import json
from pathlib import Path
import numpy as np
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent
def main():
    checks=0
    with np.load(ROOT/'results/expanded_benign_calibration/benign_partitions.npz') as z:
        h=z['calibration_hashes'];eh=z['evaluation_hashes']
    selection=set(h[h%4==0]);final=set(h[h%4==2]);evaluation=set(eh)
    assert not selection&final and not selection&evaluation and not final&evaluation
    for folder in ('matched_selection','matched_selection_budget_forests'):
        path=ROOT/'results'/folder;data=json.loads((path/'results.json').read_text())
        for run in data['runs']:
            decisions=json.loads((path/f"decisions_seed{run['seed']}.json").read_text())
            assert decisions==run['decisions']
            for decision in decisions:
                winner=max(decision['candidates'],key=lambda r:(r['tuning_recall'],-r['bytes'],r['name']))['name']
                assert winner==decision['selected']['recall_constrained']
                for c in decision['candidates']:
                    assert c['bytes']<=1.5*1024*1024
                    with np.load(path/f"{c['name']}_seed{run['seed']}_scores.npz") as z:
                        t=tolerance_threshold(z['selection_benign'],decision['cap'],.05)['threshold']
                        assert t==c['threshold']['threshold']
                        assert float((z['tuning_attack']>t).mean())==c['tuning_recall']
            for r in run['candidate_evaluation']:
                with np.load(path/f"{r['name']}_seed{run['seed']}_scores.npz") as z:
                    assert len(z['final_benign'])==int((h%4==2).sum())
                    assert len(z['evaluation_benign'])==len(eh)
                    t=tolerance_threshold(z['final_benign'],r['cap'],.05)['threshold']
                    assert t==r['threshold']['threshold']
                    assert int((z['evaluation_benign']>t).sum())==r['fp']
                    assert int((z['evaluation_attack']>t).sum())==r['tp']
                    assert r['fpr']==r['fp']/r['benign_n'] and r['recall']==r['tp']/r['attack_n']
                    assert r['joint_pass']==(r['fpr']<=r['cap'] and r['recall']>=.5)
                    checks+=1
    out=dict(status='passed',reconstructed_operating_points=checks,selection_unique_groups=len(selection),
             final_calibration_unique_groups=len(final),evaluation_unique_groups=len(evaluation),
             scope='saved scores, decisions, count reconstruction and partition disjointness; not a generalization test')
    (ROOT/'results/matched_selection/verification.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
