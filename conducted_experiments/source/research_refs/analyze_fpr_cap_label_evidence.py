"""Exact one-sided binomial evidence requirements for benign FPR caps."""
from pathlib import Path
import json
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1]
alpha=.05

def ucb(x,n):
 if x>=n:return 1.0
 return float(beta.ppf(1-alpha,x+1,n-x))

def min_n(cap,x,max_n=2_000_000):
 lo=max(x+1,1); hi=lo
 while hi<max_n and ucb(x,hi)>cap: hi*=2
 hi=min(hi,max_n)
 if ucb(x,hi)>cap:return None
 while lo<hi:
  mid=(lo+hi)//2
  if ucb(x,mid)<=cap:hi=mid
  else:lo=mid+1
 return lo

def main():
 caps=[.001,.0025,.005,.01]
 required=[]
 for cap in caps:
  for x in range(5):required.append({'fpr_cap':cap,'observed_false_positives':x,'minimum_benign_iid_n':min_n(cap,x)})
 supports=[40,80,160,300,500,1000,2000,2995,5000,10000]
 evidence=[{'benign_support':n,'false_positives':x,'fpr_point':x/n,'one_sided_95pct_exact_ucb':ucb(x,n)} for n in supports for x in (0,1,5) if x<=n]
 out={'confidence':.95,'formula':'one-sided Clopper-Pearson upper confidence bound: BetaQuantile(0.95; x+1, n-x)','assumption':'IID Bernoulli benign decisions, or exchangeability sufficient for this binomial model; not valid as a guarantee under serial dependence or arbitrary drift','required_sample_sizes':required,'examples':evidence}
 (ROOT/'results'/'FPR_CAP_LABEL_EVIDENCE_FEASIBILITY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 lines=['# Label evidence required for low-FPR acceptance','', '## Exact binomial reference', '', 'For `x` false positives among `n` benign validation examples, the one-sided 95% Clopper–Pearson upper bound is `BetaQuantile(0.95; x+1, n-x)`. This requires independent Bernoulli outcomes (or a defensible equivalent exchangeability assumption). It is a reference feasibility calculation—not a valid temporal-stream guarantee when flows are dependent or the distribution drifts.', '', '### Minimum independent benign validation count to certify a cap', '', '| FPR cap | 0 observed FP | 1 observed FP | 2 observed FP | 3 observed FP | 4 observed FP |', '|---:|---:|---:|---:|---:|---:|']
 for cap in caps:
  vals=[min_n(cap,x) for x in range(5)]
  lines.append('| '+f'{cap:.2%}'+' | '+' | '.join(f'{v:,}' if v is not None else 'not reached' for v in vals)+' |')
 lines += ['', 'At the project’s 0.1% FPR cap, **zero observed false positives still require at least 2,995 independent benign examples** for a one-sided 95% exact binomial upper bound to fall at or below the cap. One observed false positive requires appreciably more. If a comparator uses at most 160 validation labels split evenly by class, it has only about 80 benign examples; even with zero false positives the corresponding IID upper bound is about 3.68%, far above 0.1%.', '', '## Practical interpretation for this project', '', '- A small-label promotion gate can compare candidate quality, but it cannot support a 0.1% per-flow FPR claim from a few hundred benign labels under exact binomial reasoning.', '- The project’s flow stream is serially dependent and nonstationary. Treating thousands of rows as IID would overstate evidence; dependence-aware blocks reduce effective information, and the effective number of independent blocks must be justified.', '- The APEX gate’s block bootstrap estimates a conditional validation uncertainty under its chosen five-minute-block resampling design. It is **not** a distribution-free or arbitrary-drift guarantee. Its same-stream retrospective next-day diagnostic found one accepted next-day FPR above 0.1%, as documented separately.', '- Fair budget comparisons must state both total labeled examples and benign effective evidence, along with the risk unit (per-flow FPR versus fraction of time blocks with any false alarm). These are different estimands.', '', 'Reproduction: `research_refs/analyze_fpr_cap_label_evidence.py`. Machine-readable values: `results/FPR_CAP_LABEL_EVIDENCE_FEASIBILITY.json`.']
 (ROOT/'results'/'FPR_CAP_LABEL_EVIDENCE_FEASIBILITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('0.1% cap:',[min_n(.001,x) for x in range(5)])
 print('80 benign, 0 FP upper:',ucb(0,80))
 print('80 benign, 1 FP upper:',ucb(1,80))
if __name__=='__main__':main()
