"""Build a causal NF-ToN-IoT-v3 stream ordered by flow completion time.

The verified source CSV is ordered by flow start. Because every flow ends at or
after its start, a start-time watermark safely releases complete end-time groups
without sorting the 5.3 GB table in memory or creating multiple full-size runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nf_ton_iot_v3"
SOURCE = DATA / "NF-ToN-IoT-v3.csv"
OUTPUT = DATA / "NF-ToN-IoT-v3_endtime_ordered.csv"
AUDIT = ROOT / "results" / "nf_ton_iot_v3_endtime_ordered_manifest.json"


def serialize_record(row: list[str]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(row)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {OUTPUT}; use --force after review")
    DATA.mkdir(parents=True, exist_ok=True)
    partial_fd, partial_name = tempfile.mkstemp(prefix=".NF-ToN-IoT-v3_endtime_",
                                                suffix=".partial", dir=DATA)
    partial = Path(partial_name)
    started = time.time()
    digest = hashlib.sha256()
    rows_seen = rows_emitted = 0
    invalid_rows = start_inversions = negative_intervals = end_inversions = 0
    max_duration_ms = max_buffer_rows = max_equal_end_group = 0
    previous_start = previous_output_end = None
    source_label_counts: Counter[str] = Counter()
    emitted_label_counts: Counter[str] = Counter()
    sequence = 0
    current_start = None
    # Heap entries retain source sequence as the stable tie-breaker for equal ends.
    pending: list[tuple[int, int, str, str]] = []

    print(f"Source: {SOURCE}", flush=True)
    print("Emitting records only after the complete equal-start group is ingested.", flush=True)

    try:
        with SOURCE.open("r", encoding="utf-8-sig", newline="") as src, os.fdopen(partial_fd, "wb") as dst:
            reader = csv.reader(src)
            header_fields = next(reader)
            header = serialize_record(header_fields).encode("utf-8")
            dst.write(header)
            digest.update(header)
            normalized = [name.strip() for name in header_fields]
            start_i = normalized.index("FLOW_START_MILLISECONDS")
            end_i = normalized.index("FLOW_END_MILLISECONDS")
            label_i = normalized.index("Label")

            def emit_group(watermark: int | None = None, drain_all: bool = False) -> None:
                nonlocal rows_emitted, previous_output_end, end_inversions, max_equal_end_group
                while pending and (drain_all or pending[0][0] <= watermark):
                    group_end = pending[0][0]
                    same_end = []
                    while pending and pending[0][0] == group_end:
                        same_end.append(heapq.heappop(pending))
                    max_equal_end_group = max(max_equal_end_group, len(same_end))
                    for end_ms, _seq, label, record in same_end:
                        if previous_output_end is not None and end_ms < previous_output_end:
                            end_inversions += 1
                        previous_output_end = end_ms
                        payload = record.encode("utf-8")
                        dst.write(payload)
                        digest.update(payload)
                        rows_emitted += 1
                        emitted_label_counts[label] += 1

            for row in reader:
                rows_seen += 1
                try:
                    if len(row) != len(header_fields):
                        raise ValueError("wrong number of fields")
                    start_ms = int(float(row[start_i]))
                    end_ms = int(float(row[end_i]))
                    label_num = int(float(row[label_i]))
                    if label_num not in (0, 1):
                        raise ValueError("non-binary label")
                    if previous_start is not None and start_ms < previous_start:
                        start_inversions += 1
                    if end_ms < start_ms:
                        negative_intervals += 1
                    max_duration_ms = max(max_duration_ms, end_ms - start_ms)
                    previous_start = start_ms
                    label = str(label_num)
                    source_label_counts[label] += 1
                    if current_start is None:
                        current_start = start_ms
                    elif start_ms > current_start:
                        # All rows at current_start were read before advancing the watermark.
                        emit_group(watermark=current_start)
                        current_start = start_ms
                    record = serialize_record(row)
                    heapq.heappush(pending, (end_ms, sequence, label, record))
                    sequence += 1
                    max_buffer_rows = max(max_buffer_rows, len(pending))
                except (ValueError, IndexError, OverflowError):
                    invalid_rows += 1
                    raise RuntimeError(f"Invalid input row {rows_seen}")
                if rows_seen % 1_000_000 == 0:
                    print(f"read={rows_seen:,} emitted={rows_emitted:,} pending={len(pending):,} "
                          f"elapsed_s={time.time()-started:.1f}", flush=True)

            emit_group(drain_all=True)
            dst.flush()

        if rows_seen != rows_emitted:
            raise RuntimeError(f"Row count changed: read {rows_seen:,}, emitted {rows_emitted:,}")
        if emitted_label_counts != source_label_counts:
            raise RuntimeError(f"Label counts changed: {emitted_label_counts} != {source_label_counts}")
        if start_inversions or negative_intervals or end_inversions or invalid_rows:
            raise RuntimeError(
                f"Integrity failure: start inversions={start_inversions}, "
                f"negative intervals={negative_intervals}, end inversions={end_inversions}, "
                f"invalid rows={invalid_rows}"
            )

        file_size = partial.stat().st_size
        output_sha256 = digest.hexdigest()
        os.replace(partial, OUTPUT)
        manifest = {
            "date_utc": datetime.now(timezone.utc).date().isoformat(),
            "source": SOURCE.relative_to(ROOT).as_posix(),
            "output": OUTPUT.relative_to(ROOT).as_posix(),
            "sort_key": "FLOW_END_MILLISECONDS ascending; source row order breaks equal-end ties",
            "method": "bounded start-time watermark with min-heap; equal-start source groups are ingested before advancing watermark",
            "rows": rows_seen,
            "columns": len(header_fields),
            "source_label_counts": dict(source_label_counts),
            "output_label_counts": dict(emitted_label_counts),
            "publisher_start_time_inversions": start_inversions,
            "output_end_time_inversions": end_inversions,
            "negative_end_minus_start_rows": negative_intervals,
            "invalid_rows": invalid_rows,
            "max_flow_duration_ms": max_duration_ms,
            "pending_buffer_high_water_rows": max_buffer_rows,
            "max_equal_end_timestamp_group_rows": max_equal_end_group,
            "output_size_bytes": file_size,
            "output_sha256": output_sha256,
            "elapsed_seconds": round(time.time() - started, 3),
            "causal_interpretation": "A full flow-final vector is scored no earlier than its end time. Because the source is verified start-time ordered and end >= start, no future source row can have an end at or before the released start watermark. Equal-end rows are emitted atomically.",
            "limitations": [
                "This is a chronological testbed campaign stream, not production concept drift or an independent dataset-family confirmation.",
                "Chunk-level labels must still be withheld until all predictions for that chunk have been produced.",
                "The publisher's timestamp-plus-five-tuple labeling provenance is documented by the authors, not independently re-annotated from raw packets."
            ]
        }
        AUDIT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
    finally:
        if partial.exists():
            partial.unlink()


if __name__ == "__main__":
    main()
