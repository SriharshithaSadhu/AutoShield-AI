"""Save exploratory outcome table, tail-tie diagnostic and scientific figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/selection_stability'
def main():
    results=json.loads((OUT/'results.json').read_text())
    decisions=json.loads((OUT/'decisions.json').read_text())
    for d in decisions:
        winner=max(d['candidates'],key=lambda r:(r['bootstrap_recall_q10'],-r['bytes'],r['name']))['name']
        assert winner==d['selected']
    diagnostic={}
    with np.load(ROOT/'results/matched_selection_budget_forests/rf40_seed43_scores.npz') as z:
        maximum=max(z['selection_benign'].max(),z['final_benign'].max())
        for k in ('selection_benign','final_benign','evaluation_benign','tuning_attack','evaluation_attack'):
            diagnostic[k]=dict(n=len(z[k]),max_score_ties=int((z[k]==maximum).sum()))
        diagnostic['maximum_score']=float(maximum)
    d=next(d for d in decisions if d['seed']==43 and d['cap']==.001)
    diagnostic['selection_bootstrap']=next(c for c in d['candidates'] if c['name']=='rf40')
    (OUT/'forest_tail_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    lines=['# Calibration-stability selection: exploratory result','',
        'Follow-up motivated by the observed seed-43 forest failure. This is development evidence, not a locked confirmation or a novel guarantee.',
        '200 shared bootstrap resamples of selection-only identical-feature groups; select maximum lower-decile constrained tuning recall; independent final calibration remains unchanged. Bootstrap assumes useful group resampling and does not recreate independent networks.', '',
        '| Seed | Cap | Point selector recall / FPR | Stability selector | Recall / FPR | Joint pass |',
        '|---|---|---|---|---|---|']
    for r in results['results']:
        old=r['original_selector']
        lines.append(f"| {r['seed']} | {r['cap']:.1%} | {old['recall']:.2%} / {old['fpr']:.4%} | {r['name']} | {r['recall']:.2%} / {r['fpr']:.4%} | {r['joint_pass']} |")
    lines+=['',
        'Primary 0.1% cap: stability avoids the seed-43 all-negative model (0% to 49.88% recall), but loses recall on seed 42 and still has zero joint successes across three seeds. No overall superiority claim.',
        'Sensitivity 1% cap: all three selections satisfy the empirical cap with recall 78.87%, 81.48%, 79.93%. These are not primary-endpoint successes.',
        'Mechanism: RF40 seed43 has 2 maximum-score benign rows in selection versus 3 in final calibration; final calibration permits k=2 exceedances, so the strict threshold becomes the maximum. It then rejects every attack, including 2,196 evaluation attacks tied at that maximum. Probability calibration cannot separate tied raw scores.',
        f"Selection CPU cost: {results['selection_cpu_seconds']:.2f} seconds, excluding candidate training and earlier score generation.", '',
        '## Related work and claim boundary',
        '- Known NP threshold method: https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/',
        '- Model selection/calibration reuse is established: https://arxiv.org/abs/2408.07066',
        '- Tuning bias: https://proceedings.mlr.press/v267/zeng25e.html',
        'A bootstrap stability penalty is an exploratory engineering hypothesis. These references do not establish its novelty; a specific literature comparison and fresh protocol are required.', '',
        'Replication gate: evaluate several fixed selection/final-calibration group partitions without changing this quantile. Outcomes, when available, follow below.']
    if (OUT/'partition_results.json').exists():
        replication=json.loads((OUT/'partition_results.json').read_text())
        lines+=['','## Five additional group partitions completed',
                'Fixed partition seeds 100–104, unchanged 200-replicate lower-decile selector. These 15 cases reuse the same development data and are not independent network experiments.',
                '', '| Method | Joint successes | Zero-recall choices | Mean recall |', '|---|---|---|---|']
        for r in replication['summary']:
            lines.append(f"| {r['method']} | {r['joint_pass']}/{r['n']} | {r['zero_recall']} | {r['mean_recall']:.2%} |")
        lines+=['','Decision: the initial collapse avoidance did not replicate reliably. Do not promote this bootstrap rule as a solution or cherry-pick its one passing configuration. Investigate discrete score-tail support and unseen rare benign groups before further optimization. The primary target remains unmet.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,cap in zip(axes,(.001,.01)):
        rr=[r for r in results['results'] if r['cap']==cap];xx=np.arange(3)
        ax.bar(xx-.18,[r['original_selector']['recall']*100 for r in rr],.36,label='Point constrained recall')
        ax.bar(xx+.18,[r['recall']*100 for r in rr],.36,label='Bootstrap lower-decile recall')
        ax.axhline(50,color='black',ls=':',lw=1);ax.set_xticks(xx,['42','43','44']);ax.set_xlabel('Model / split seed')
        ax.set_ylim(0,100);ax.set_title(f'FPR cap {cap:.1%}');ax.set_ylabel('Attack recall (%)')
    axes[0].legend(fontsize=8);fig.suptitle('Development evidence: primary joint target still unmet')
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    print(json.dumps(diagnostic,indent=2))
if __name__=='__main__':main()
