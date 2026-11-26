"""Postprocess complete raw-prefix counts; preserves original FPR target."""
import json,sqlite3,math,argparse
from pathlib import Path
from run_network_context_study import O
from observability_relaxation import upper_tp,lower_fp_for_tp,rational

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default=str(O/'full_training_text_audit'));args=ap.parse_args()
    b=Path(args.directory);result=json.loads((b/'results.json').read_text())
    conn=sqlite3.connect('file:'+str((b/'groups.sqlite').resolve())+'?mode=ro',uri=True)
    pairs=conn.execute('SELECT scanning,benign FROM groups WHERE scanning>0 AND benign>0').fetchall();conn.close()
    a=[x[0] for x in pairs];v=[x[1] for x in pairs];free=result['free_attack_only_tp'];n0=result['benign'];n1=result['scanning'];rows=[]
    for cap in [0,.0001,.001,.005,.01,.05]:
        budget=int(math.floor(cap*n0));upper,slope,dual=upper_tp(a,v,budget,free)
        rows.append(dict(fpr_reference=cap,fp_budget=budget,upper_tp=rational(upper),
            upper_recall=float(upper/n1),dual_slope=rational(slope),dual_objective=rational(dual)))
    cost=lower_fp_for_tp(a,v,math.ceil(.5*n1),free)
    assert result['optimal_tp']<=next(r['upper_tp']['value'] for r in rows if r['fpr_reference']==.001)
    out=dict(scope=__doc__,results=rows,minimum_fp_fractional_lower_bound_for_50percent_scanning=rational(cost),
        minimum_fpr_fractional_lower_bound_for_50percent_scanning=float(cost/n0),
        caveat='All envelopes are optimistic empirical label-assisted relaxations, not achieved detector performance. The0.1% deployment target is unchanged. A deterministic vector classifier cannot beat the fractional relaxation; bounds do not apply to richer observation inputs or arbitrary future traffic.')
    (b/'fractional_envelope.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
