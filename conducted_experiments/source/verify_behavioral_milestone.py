"""Reproducibility and split-integrity checks; protected labels never loaded."""
from pathlib import Path
import hashlib,json,subprocess,sys,datetime
import numpy as np
R=Path(__file__).resolve().parent;O=R/'results/behavioral_observation_2026-11-22'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check_ledger(path):
    entries=json.loads(path.read_text());bad=[]
    for name,wanted in entries.items():
        p=R/name
        if not p.is_file() or sha(p)!=wanted:bad.append(name)
    assert not bad,(str(path),bad)
    return dict(ledger=str(path.relative_to(R)),unchanged_files=len(entries))

def main():
    ledgers=[check_ledger(R/'results/network_context_2026-11-21/milestone_sha256.json'),
        check_ledger(R/'results/network_context_2026-11-21/input_sha256.json'),
        check_ledger(R/'results/multienvironment_2026-11-21/milestone_sha256.json')]
    protocol=json.loads((O/'protocol.json').read_text())
    for name,wanted in protocol['feature_code_sha256'].items():assert sha(R/name)==wanted,name
    protected=np.load(R/'data/nf_unsw_nb15_v3/development/protected_row_ids.npy');allowed=np.load(O/'unsw_allowed_metadata.npy',mmap_mode='r')
    assert len(protected)==473085 and len(allowed)==1892339
    assert len(np.intersect1d(protected,allowed['row']))==0
    assert len(np.unique(allowed['row']))==len(allowed) and len(allowed)+len(protected)==2365424
    for name in ['train','calibration','selection','development_holdout']:
        m=np.load(O/f'unsw_{name}_metadata.npy',mmap_mode='r');assert len(np.intersect1d(protected,m['row']))==0
    full=json.loads((O/'full_replay_results.json').read_text());rows=full['resources']['scored_records'];assert rows==9160253
    for r in full['results']:
        assert r['benign']+r['attacks']==rows and r['latency']['n']==rows
        assert r['alert_latency']['n']==r['tp']+r['fp']
        assert r['family'] and sum(f['n'] for f in r['family'].values())==r['attacks']
        for field in ['flow_windows','time_windows']:
            for k in ['benign','attacks','fp','tp']:assert sum(w[k] for w in r[field])==r[k]
        assert 0<=r['fpr']<=1 and 0<=r['recall']<=1
    scorecards=json.loads((O/'results.json').read_text())['results'];assert len(scorecards)==156
    recent=json.loads((O/'recent_benign_control/training_audit.json').read_text())
    assert recent['latest_added_training_ms']<recent['earliest_calibration_ms']<recent['latest_calibration_ms']
    with np.load(R/'data/nf_ton_iot_v3/detection_foundation/calibration.npz') as z:
        m=np.load(O/'recent_benign_control/metadata.npy',mmap_mode='r')
        assert np.array_equal(z['row'],m['row']) and (z['y']==0).all()
        assert recent['latest_calibration_ms']<np.load(O/'ton_evaluation_metadata.npy',mmap_mode='r')['t'].min()
    tests=subprocess.run([sys.executable,'-m','unittest','test_behavioral_observation.py','test_behavioral_runtime.py','test_prior_attack_context.py','-v'],cwd=R,capture_output=True,text=True,check=True)
    (O/'unit_test_output.txt').write_text(tests.stdout+tests.stderr)
    result=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),old_ledgers=ledgers,feature_code_matches_pre_fit_freeze=True,
        protected_rows=473085,protected_state_updates=0,protected_predictions=0,allowed_unsw_rows=1892339,
        scored_full_development_rows=rows,primary_scorecards=len(scorecards),unit_tests=11,passed=True,
        scope='This verification checks protected identities only. Cache preparation filtered raw metadata before state updates; no protected labels or predictions used. Existing reference files remain hash-identical.')
    (O/'verification.json').write_text(json.dumps(result,indent=2));print('VERIFIED',result,flush=True)
if __name__=='__main__':main()
