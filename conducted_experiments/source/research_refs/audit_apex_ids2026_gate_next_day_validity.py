from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research_refs'/'run_apex_ids2026_update_gate_hash_sensitivity.py'
spec=importlib.util.spec_from_file_location('apex_gate',RUN); gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
CACHE=ROOT/'results'/'apex_ids2026_update_gate_cache'
OUT=ROOT/'results'/'apex_ids2026_gate_next_day_validity.json'
FEATURES=gate.FEATURE_NAMES

def predict_counts(model,x,y,threshold):
    scores=model.booster_.predict(x,num_threads=4).astype(np.float32)
    tn,fp,fn,tp=map(int,gate.rates_from_rows(y,scores,threshold))
    return {'tn':tn,'fp':fp,'fn':fn,'tp':tp,'benign_support':tn+fp,'attack_support':fn+tp,
            'fpr':fp/(tn+fp) if tn+fp else None,'recall':tp/(fn+tp) if fn+tp else None}

def main():
  allrows=[]
  dates=list(pd.date_range(gate.REPLAY_START,gate.REPLAY_END,freq='D').strftime('%Y-%m-%d'))
  for seed in (42,43,44):
    gate.SEED=seed
    # Recreate the same prefix reservoir and initial classifier.
    rng=np.random.default_rng(seed); samples={}; ys={}; counts={}; train_rows=0
    from collections import defaultdict
    counts=defaultdict(int)
    for date in pd.date_range(gate.TRAIN_START,gate.TRAIN_END,freq='D').strftime('%Y-%m-%d'):
      f=pq.read_table(CACHE/f'{date}.parquet').to_pandas()
      x=f[FEATURES].to_numpy(dtype=np.float32); y=f['label_binary'].to_numpy(dtype=np.int8)
      day=np.full(len(y),int(pd.Timestamp(date).tz_localize('UTC').value//gate.DAY_NS),dtype=np.int64)
      train_rows+=len(y); gate.reservoir_add(samples,ys,counts,rng,x,y,day)
    x0,y0=gate.stack_reservoir(samples,ys); del samples,ys
    inc=gate.model(); inc.fit(x0,y0,feature_name=FEATURES)
    calx=[]; caly=[]; cals=[]
    for date in pd.date_range(gate.CAL_START,gate.CAL_END,freq='D').strftime('%Y-%m-%d'):
      f=pq.read_table(CACHE/f'{date}.parquet').to_pandas(); xx=f[FEATURES].to_numpy(dtype=np.float32); yy=f['label_binary'].to_numpy(dtype=np.int8)
      calx.append(xx);caly.append(yy);cals.append(inc.booster_.predict(xx,num_threads=4).astype(np.float32))
    threshold=gate.threshold_from_dev(np.concatenate(cals),np.concatenate(caly))
    saved=json.loads((ROOT/'results'/f'apex_ids2026_update_gate_hash_sensitivity_seed{seed}.json').read_text())
    expected=saved['decisions']; replay_inc=inc
    for i,date in enumerate(dates[:-1]):
      f=pq.read_table(CACHE/f'{date}.parquet').to_pandas()
      x=f[FEATURES].to_numpy(dtype=np.float32);y=f['label_binary'].to_numpy(dtype=np.int8);bins=f['bin_id'].to_numpy(dtype=np.int64)
      val_bins={int(b) for b in np.unique(bins) if gate.validation_residue(int(b))==0}
      isval=np.fromiter((int(b) in val_bins for b in bins),count=len(bins),dtype=bool)
      incscore=replay_inc.booster_.predict(x,num_threads=4).astype(np.float32)
      rng2=np.random.default_rng(seed+i+17); xxrows=[]; yyrows=[]
      for lab in (0,1):
        ids=np.flatnonzero((~isval)&(y==lab))
        if len(ids)>gate.CAP: ids=rng2.choice(ids,gate.CAP,replace=False)
        xxrows.append(x[ids]);yyrows.append(y[ids])
      cx,cy=np.vstack(xxrows),np.concatenate(yyrows)
      cand=gate.model();cand.fit(np.vstack([x0,cx]),np.concatenate([y0,cy]),feature_name=FEATURES)
      cs=cand.booster_.predict(x,num_threads=4).astype(np.float32)
      vkeys,vbc=gate.conf_by_bin(cs[isval]>threshold,y[isval],bins[isval]); ikeys,ibc=gate.conf_by_bin(incscore[isval]>threshold,y[isval],bins[isval]); imap={int(k):c for k,c in zip(ikeys,ibc)}
      ibaligned=np.vstack([imap[int(k)] for k in vkeys])
      bounds=gate.bootstrap_bounds(vbc,ibaligned,seed+10_000+i)
      accept=bounds['fpr_ucb_95']<=gate.FPR_CAP and bounds['recall_gain_lcb_95']>0
      exp=expected[i]
      if bool(exp['accepted_for_next_day'])!=bool(accept): raise RuntimeError(f'seed {seed} gate mismatch {date}: recomputed={accept}, saved={exp["accepted_for_next_day"]}')
      nv=gate.metrics(vbc.sum(axis=0)); iv=gate.metrics(ibaligned.sum(axis=0))
      nextdate=dates[i+1]; nf=pq.read_table(CACHE/f'{nextdate}.parquet').to_pandas(); nx=nf[FEATURES].to_numpy(dtype=np.float32);ny=nf['label_binary'].to_numpy(dtype=np.int8)
      cand_next=predict_counts(cand,nx,ny,threshold); inc_next=predict_counts(replay_inc,nx,ny,threshold)
      allrows.append({'seed':seed,'proposal_date':date,'evaluation_date':nextdate,'accepted':bool(accept),
        'val_fpr_candidate':nv['fpr'],'val_fpr_incumbent':iv['fpr'],'val_fpr_delta':nv['fpr']-iv['fpr'],
        'val_recall_candidate':nv['recall'],'val_recall_incumbent':iv['recall'],'val_recall_gain':nv['recall']-iv['recall'],
        'val_fpr_ucb_95':bounds['fpr_ucb_95'],'val_recall_gain_lcb_95':bounds['recall_gain_lcb_95'],
        'next_fpr_candidate':cand_next['fpr'],'next_fpr_incumbent':inc_next['fpr'],'next_fpr_delta':cand_next['fpr']-inc_next['fpr'],
        'next_recall_candidate':cand_next['recall'],'next_recall_incumbent':inc_next['recall'],'next_recall_gain':cand_next['recall']-inc_next['recall'],
        'next_candidate_confusion':cand_next,'next_incumbent_confusion':inc_next})
      if accept: replay_inc=cand
      print(f'seed={seed} {date}->{nextdate} accept={accept} val dFPR={nv["fpr"]-iv["fpr"]:+.3%} next dFPR={cand_next["fpr"]-inc_next["fpr"]:+.3%} next dRec={cand_next["recall"]-inc_next["recall"]:+.3%}',flush=True)
    print(f'seed {seed}: verified {len(dates)-1} decisions against saved output',flush=True)
  OUT.write_text(json.dumps({'status':'retrospective exploratory diagnostic; not confirmation','seeds':[42,43,44],
    'comparison_unit':'candidate proposal and its immediately following calendar day','rows':allrows},indent=2))
  # aggregate descriptive summary, keeping model seeds clustered by proposal day
  lines=['# APEX next-day update-gate validity diagnostic','', '**Status: retrospective exploratory analysis on an already-inspected development stream.** Next-day labels are used only to score proposals after the gate decision; this analysis cannot confirm generalization or validate a future-risk guarantee. The reconstructed gate decisions were checked against each saved run.', '', '## Per-seed outcomes','', '| Seed | Proposals scored next day | Accepted | Accepted candidates with higher next-day FPR | Accepted with lower next-day recall | Rejected candidates with higher next-day FPR | Rejected with lower next-day recall |','|---:|---:|---:|---:|---:|---:|---:|']
  for seed in (42,43,44):
    rr=[r for r in allrows if r['seed']==seed]; aa=[r for r in rr if r['accepted']]; jj=[r for r in rr if not r['accepted']]
    lines.append(f"| {seed} | {len(rr)} | {len(aa)} | {sum(r['next_fpr_delta']>0 for r in aa)}/{len(aa)} | {sum(r['next_recall_gain']<0 for r in aa)}/{len(aa)} | {sum(r['next_fpr_delta']>0 for r in jj)}/{len(jj)} | {sum(r['next_recall_gain']<0 for r in jj)}/{len(jj)} |")
  lines += ['', '## Interpretation boundary','', 'The table is a post-hoc, one-day-ahead diagnostic. Proposals and seeds share one stream, so counts are descriptive rather than independent Bernoulli trials. A candidate can be rejected for failing either of two validation rules; rejected candidates are still scored counterfactually on the next day, which is not information available to the controller. No method is tuned from these outcomes.', '', 'Machine-readable per-proposal record: `apex_ids2026_gate_next_day_validity.json`; reproduction script: `research_refs/audit_apex_ids2026_gate_next_day_validity.py`.']
  (ROOT/'results'/'APEX_IDS2026_GATE_NEXT_DAY_VALIDITY.md').write_text('\n'.join(lines)+'\n')
  print('wrote',OUT,flush=True)
if __name__=='__main__': main()
