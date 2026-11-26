"""Summarize joint feasibility and development Pareto tradeoffs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/detection_foundation"

def main():
 data=json.loads((OUT/"results.json").read_text())
 resources={r["name"]:r for r in data["candidates"]}
 lines=["# Chronological detection foundation comparison","",
 "Fixed sampled training, tuning, final calibration and evaluation partitions on explored NF-v3.",
 "Rows with repeated float32 feature vectors are counted separately in results.json.",
 "", "| Model | Calibration | Cap | FPR | Recall | F1 | Worst sampled block FPR | Size MiB | Joint feasible |",
 "|---|---|---:|---:|---:|---:|---:|---:|---|"]
 archive=[]
 for row in data["operating_points"]:
  cost=resources[row["name"]]
  feasible=(row["fpr"]<=row["cap"] and row["recall"]>=.5 and cost["bytes"]<=1.5*1024**2
            and cost["batch_one_p95_ms"]<=5)
  lines.append(f'| {row["name"]} | {row["calibration_policy"]} | {row["cap"]:.1%} | {row["fpr"]:.3%} | {row["recall"]:.2%} | {row["f1"]:.3f} | {row["worst_sample_block_fpr"]:.2%} | {cost["bytes"]/1024**2:.3f} | {feasible} |')
  if row["cap"]==.001 and row["calibration_policy"]=="pooled":
   archive.append(dict(name=row["name"],objectives=[1-row["recall"],row["fpr"],cost["bytes"]/1024**2]))
 nondominated=[r for r in archive if not any(
  all(a<=b for a,b in zip(other["objectives"],r["objectives"])) and
  any(a<b for a,b in zip(other["objectives"],r["objectives"]))
  for other in archive)]
 lines+=["",f'Tuning-selected resource-feasible candidate: {data["selected"]}.',
 "These seven fixed configurations are a diagnostic candidate bank, not an optimizer comparison.",
 "Worst sampled blocks contain equal numbers of sampled rows and can have different clock duration.",
 "Latency is descriptive desktop batch-one inference with current competing workload.",
 "The blockwise threshold comparator uses five consecutive benign calibration blocks and a union adjustment.",
 "Neither calibration method establishes an FPR guarantee under temporal distribution shift.",
 "",f'Primary pooled operating-point nondominated models: {[r["name"] for r in nondominated]}.']
 (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 (OUT/"pareto.json").write_text(json.dumps(nondominated,indent=2),encoding="utf-8")
 print("\n".join(lines))

if __name__=="__main__":
 main()
