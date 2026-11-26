"""Audit fallback feasibility after the first observable breach of each tenure."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("replay", type=Path)
args = parser.parse_args()
data = json.loads(args.replay.read_text(encoding="utf-8"))
chunks = data.get("chunk_metrics", data.get("chunk_rows"))
baseline_path = args.replay.with_name(args.replay.name.replace("_shadow_previous", "").replace(".partial", ""))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
fields = ("tn", "fp", "fn", "tp")
def signature(replay):
    return [(row["chunk"], tuple(row["arms"][arm][key]
                                 for arm in ("fpr_recall", "no_update") for key in fields))
            for row in replay.get("chunk_metrics", replay.get("chunk_rows"))]
if signature(data) != signature(baseline)[:len(chunks)]:
    raise SystemExit("Shadow replay differs from saved baseline confusion counts")
eligible_chunks = 0
viable_counts = {"previous": 0, "frozen": 0}
promotions = [d["after_chunk"] for d in data.get("candidate_decisions", data.get("decisions"))
              if d["decisions"]["fpr_recall"]["accept"]]
for start, end in zip(promotions, promotions[1:] + [len(chunks)]):
    seen_breach = False
    for row in chunks[start:end]:
        active = row["arms"]["fpr_recall"]
        if seen_breach and active["false_positive_rate"] > .001 and active["tp"] + active["fn"] > 0:
            eligible_chunks += 1
            for name, alternate in (("previous", row["previous_shadow"]),
                                    ("frozen", row["arms"]["no_update"])):
                positives = active["tp"] + active["fn"]
                viable = (positives > 0 and alternate["false_positive_rate"] <= .001
                          and active["false_positive_rate"] > .001
                          and alternate["tp"] >= active["tp"] - .01 * positives)
                if viable:
                    viable_counts[name] += 1
                    print(json.dumps(dict(promotion=start, chunk=row["chunk"], fallback=name,
                                          fp_saved=active["fp"] - alternate["fp"],
                                          tp_delta=alternate["tp"] - active["tp"])))
        seen_breach |= active["false_positive_rate"] > .001

print(json.dumps(dict(completed_chunks=len(chunks), eligible_chunks=eligible_chunks, viable_counts=viable_counts)))
