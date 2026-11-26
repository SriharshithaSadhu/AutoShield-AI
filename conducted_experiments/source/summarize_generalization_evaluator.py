"""Audit causal boundaries and compare outcomes from shared selection runs."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/generalization_evaluator"

def main():
 data=json.loads((OUT/"results.json").read_text())
 lines=["# Generalization-aware candidate selection","",
 "All rules use the same four configurations and the same development fits. Final calibration and evaluation were read only after selections were saved.",
 "The larger dataset and final evaluation rows were inspected in earlier studies, so this is development evidence.",
 "", "| Seed | Rule | Config | FPR | Recall | Artifact MiB | Diagnostic only |",
 "|---|---|---:|---:|---:|---:|---|"]
 records=[]
 for run in data["runs"]:
  decisions=json.loads((OUT/f'decisions_seed{run["seed"]}.json').read_text())
  assert decisions["selected"]==run["selected"]
  for rule,outcome in run["outcomes"].items():
   lines.append(f'| {run["seed"]} | {rule} | {run["selected"][rule]} | {outcome["fpr"]:.3%} | {outcome["recall"]:.2%} | {outcome["bytes"]/1024**2:.3f} | {run["diagnostic_only"]} |')
   records.append(dict(seed=run["seed"],rule=rule,**outcome))
 lines+=["","Parameter seeds alone do not create independent replications when tree subsampling is disabled. Report seed ranges descriptively.",
         "A zero minimum held-family score ties all candidates and cannot support a claim of improved selection.",
         "Desktop latency reflects this machine and competing workloads; process RSS is observed total memory, not isolated peak allocation."]
 for rule in ("pooled","worst_window","family_aware"):
  rr=[r for r in records if r["rule"]==rule]
  lines.append(f'{rule}: FPR range {min(r["fpr"] for r in rr):.3%}–{max(r["fpr"] for r in rr):.3%}; recall range {min(r["recall"] for r in rr):.2%}–{max(r["recall"] for r in rr):.2%}.')
 (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print("\n".join(lines))

if __name__=="__main__":
 main()
