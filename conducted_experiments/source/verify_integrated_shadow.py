import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"

def main():
    report = []
    for path in sorted(ROOT.glob("integrated_shadow*.json")):
        if path.name == "integrated_shadow_verification.json":
            continue
        run = json.loads(path.read_text())
        delay = run["label_delay_chunks"]
        tag = "locked" if delay == 0 else f"delay{delay}"
        baseline = json.loads((ROOT / f"original_core6_lifecycle_{tag}.json").read_text())
        for policy, rows in baseline["chunks"].items():
            assert run["chunks"][policy] == rows, (path, policy)
        for policy, rows in run["chunks"].items():
            if not policy.endswith("_shadow"):
                continue
            for row, frozen, source in zip(rows, run["chunks"]["frozen"],
                                          run["chunks"][run.get("shadow_source", "gated")], strict=True):
                assert row in (frozen, source)
            counts = {key: sum(row[key] for row in rows) for key in ("fp", "tp", "attacks", "benign")}
            assert all(run["summary"][policy][key] == value for key, value in counts.items())
            report.append(dict(run=path.name,delay=delay,policy=policy,**counts,
                               cost=run["shadow_cost"], switches=[e["chunk"] for e in run["shadow_events"]
                                                                if e["policy"] == policy]))
    (ROOT / "integrated_shadow_verification.json").write_text(json.dumps(report, indent=2))
    print(f"Verified {len(report)} policy runs against baseline and chunk counts.")

if __name__ == "__main__":
    main()
