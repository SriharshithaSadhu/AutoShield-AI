"""Score an ordered CSV of completed flows using a bundled frozen model."""

import argparse, csv, hashlib, ipaddress, json
from pathlib import Path
import joblib
from autoshield.runtime import BehavioralRuntime, CompletedFlow, RAW_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--model", choices=["flow", "hybrid"], default="flow")
    parser.add_argument("--arrival-column", default="FLOW_END_MILLISECONDS")
    args = parser.parse_args()
    spec = json.loads((ROOT / "models/manifest.json").read_text())["models"][args.model]
    path = ROOT / "models" / spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Model checksum mismatch")
    runtime = BehavioralRuntime(joblib.load(path), spec["method"], spec["threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    alerts = 0
    batch = []
    with args.input.open(newline="") as source, args.output.open(
        "w", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.writer(target)
        writer.writerow(["row", "arrival_ms", "score", "alert"])

        def flush():
            nonlocal total, alerts, batch
            if not batch:
                return
            result = runtime.score_batch(batch)
            for record, score, alert in zip(batch, result["scores"], result["alerts"]):
                writer.writerow([total, record.arrival_ms, float(score), int(alert)])
                total += 1
                alerts += int(alert)
            batch = []

        for row in reader:
            record = CompletedFlow(
                int(row[args.arrival_column]),
                int(ipaddress.IPv4Address(row["IPV4_SRC_ADDR"])),
                int(ipaddress.IPv4Address(row["IPV4_DST_ADDR"])),
                int(row["L4_DST_PORT"]),
                tuple(float(row[key]) for key in RAW_FIELDS),
            )
            if batch and record.arrival_ms - batch[0].arrival_ms >= 1000:
                flush()
            batch.append(record)
            if len(batch) == 1000:
                flush()
        flush()
    print(
        f"Scored {total} completed flows; emitted {alerts} alerts; saved {args.output}"
    )


if __name__ == "__main__":
    main()
