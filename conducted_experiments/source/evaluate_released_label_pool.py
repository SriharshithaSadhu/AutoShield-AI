"""Development replay of causal routing among concurrently scored policy arms."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"
POOL = ("frozen", "rollback", "lease", "threshold_only")


def evaluate(run, window, margin, defer=False):
    delay = run["label_delay_chunks"]
    arms = run["chunks"]
    current = "frozen"
    chosen = []
    switches = []
    deferred = dict(attacks=0, benign=0, chunks=0)
    for i, row in enumerate(arms["frozen"]):
        end = i - delay
        if end >= window:
            history = {k: arms[k][end-window:end] for k in POOL}
            benign = sum(r["benign"] for r in history["frozen"])
            attacks = sum(r["attacks"] for r in history["frozen"])
            if benign >= 3000 and attacks >= 100:
                fp = {k: sum(r["fp"] for r in rows) for k, rows in history.items()}
                tp = {k: sum(r["tp"] for r in rows) for k, rows in history.items()}
                allowed = [k for k in POOL if fp[k]*1000 <= benign and tp[k] >= tp["frozen"]]
                best = max(allowed, key=lambda k: (tp[k], -fp[k])) if allowed else (
                    "defer" if defer else "frozen")
                if best != current and (current not in allowed or tp[best]-tp[current] >= margin*attacks):
                    switches.append(dict(chunk=row["chunk"], source=best,
                                         latest_label_chunk=history["frozen"][-1]["chunk"]))
                    current = best
        if current == "defer":
            deferred["chunks"] += 1
            deferred["attacks"] += row["attacks"]
            deferred["benign"] += row["benign"]
            chosen.append(dict(row, fp=0, tp=0))
        else:
            chosen.append(arms[current][i])
    counts = {k: sum(r[k] for r in chosen) for k in ("fp", "tp", "attacks", "benign")}
    return dict(**counts, fpr=counts["fp"]/counts["benign"], recall=counts["tp"]/counts["attacks"],
                worst_fpr=max(r["fp"]/r["benign"] for r in chosen if r["benign"]),
                deferred=deferred, switches=switches)


def main():
    results = {}
    for lag, tag in [(0, "locked"), (1, "delay1"), (2, "delay2"), (4, "delay4")]:
        run = json.loads((ROOT / f"original_core6_lifecycle_{tag}.json").read_text())
        results[lag] = {f"{window}/{margin}": evaluate(run, window, margin)
                        for window, margin in [(2, 0), (2, .01), (5, .01)]}
        results[lag]["2/0.01/defer"] = evaluate(run, 2, .01, defer=True)
    (ROOT / "released_label_pool_diagnostic.json").write_text(json.dumps(results, indent=2))
    for lag, variants in results.items():
        print(lag, [(k, v["fp"], round(v["recall"], 4), round(v["worst_fpr"], 4),
                     len(v["switches"])) for k, v in variants.items()])

if __name__ == "__main__":
    main()
