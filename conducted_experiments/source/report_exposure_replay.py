"""Reconcile delayed-exposure metrics and report matched policy outcomes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    run = json.loads(args.result.read_text())
    frozen = run["chunks"]["frozen"]
    delay = run["label_delay_chunks"]
    lines = [f"# Policy comparison at label delay {delay}", "",
             "| Policy | False alarms | FPR | Recall | Missed attacks | Worst five-chunk FPR |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy, rows in run["chunks"].items():
        assert [r["chunk"] for r in rows] == [r["chunk"] for r in frozen]
        summary = run["summary"][policy]
        for metric in ("fp", "tp", "attacks", "benign"):
            assert sum(r[metric] for r in rows) == summary[metric]
        if policy in run["exposure"]:
            increments = [max(0, r["fp"] - f["fp"]) for r, f in zip(rows, frozen, strict=True)]
            state = run["exposure"][policy]
            assert max(sum(increments[max(0, i-delay):i+1]) for i in range(len(rows))) <= state["peak"] <= state["budget"]
        windows = (rows[i:i+5] for i in range(len(rows)-4))
        worst = max((sum(r["fp"] for r in w)/sum(r["benign"] for r in w)
                     for w in windows if any(r["benign"] for r in w)), default=0)
        lines.append(f'| {policy} | {summary["fp"]:,} | {summary["fpr"]:.3%} | {summary["recall"]:.2%} | '
                     f'{summary["attacks"]-summary["tp"]:,} | {worst:.3%} |')
    approvals = [e for e in run["lifecycle"] if e["policy"]=="rollback" and e["action"]=="approve"]
    lines += ["", f"Rollback approvals: {len(approvals)}. Asserted pending exposure caps against realized incremental false alarms.",
              "These are exploratory replay observations. The exposure bound is relative to frozen and does not constrain its false alarms."]
    args.result.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
