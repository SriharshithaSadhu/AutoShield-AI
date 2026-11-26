"""Compare selection rules from a shared development candidate table."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"results/generalization_evaluator"

def choose(candidates,rule):
 def rank(row):
  counts=row["development_counts"]
  pooled_fpr=sum(r["fp"] for r in counts)/max(1,sum(r["benign"] for r in counts))
  fpr=pooled_fpr if rule=="pooled" else row["worst_fpr"]
  objective=row["pooled_recall"] if rule=="pooled" else (
      row["worst_recall"] if rule=="worst_window" else row["held_family_min"])
  return (fpr<=.001,-max(0,fpr-.001),objective,-row["config_id"])
 viable=[r for r in candidates if r["resources"]["bytes"]<=1.5*1024**2 and
         r["resources"]["batch_one_p95_ms"]<=5]
 return max(viable,key=rank)["config_id"]

def main():
 rows=[]
 for file in sorted(ROOT.glob("decisions_seed*.json")):
  data=json.loads(file.read_text())
  rows.append(dict(seed=int(file.stem.split("seed")[1]),
                   selected={rule:choose(data["candidates"],rule)
                             for rule in ("pooled","worst_window","family_aware")},
                   family_support={r["config_id"]:r["held_family"] for r in data["candidates"]}))
 (ROOT/"selection_rule_audit.json").write_text(json.dumps(rows,indent=2))
 print(rows)

if __name__=="__main__":
 main()
