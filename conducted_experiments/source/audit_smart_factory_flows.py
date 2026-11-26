"""Integrity, timing, and support audit for TU Wien smart-factory flow CSVs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "tuwien_smart_factory"
MANIFEST = DATA / "download_manifest.json"
OUTPUT = DATA / "audit.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET TimeZone='UTC'")
    files_out = []
    all_hashes = {}
    for item in manifest["files"]:
        path = DATA / item["filename"]
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        if path.stat().st_size != item["actual_bytes"] or digest != item["sha256"]:
            raise RuntimeError(f"Integrity mismatch: {path.name}")
        all_hashes[path.name] = digest
        source = str(path).replace("'", "''")
        labels = con.execute(f"""
            SELECT coalesce(Label, '<NULL>') AS label, count(*) AS n
            FROM read_csv_auto('{source}', header=true) GROUP BY 1 ORDER BY 1
        """).fetchall()
        summary = con.execute(f"""
            SELECT count(*) AS n,
                   min(flowStartMilliseconds) AS min_start,
                   max(flowStartMilliseconds) AS max_start,
                   min(flowDurationMilliseconds) AS min_duration,
                   max(flowDurationMilliseconds) AS max_duration,
                   count(*) FILTER (WHERE flowStartMilliseconds IS NULL) AS null_start,
                   count(*) FILTER (WHERE flowDurationMilliseconds IS NULL) AS null_duration,
                   count(*) FILTER (WHERE flowDurationMilliseconds < 0) AS negative_duration,
                   count(*) FILTER (WHERE sourceIPAddress IS NULL OR destinationIPAddress IS NULL) AS null_endpoint,
                   count(*) FILTER (WHERE sourceTransportPort IS NULL OR destinationTransportPort IS NULL) AS null_port,
                   count(DISTINCT SequenceNumber) AS unique_sequence,
                   min(SequenceNumber) AS min_sequence, max(SequenceNumber) AS max_sequence
            FROM read_csv_auto('{source}', header=true)
        """).fetchone()
        # Source ordering descents, using row_number ordering as stored.
        regressions = con.execute(f"""
            WITH x AS (
                SELECT SequenceNumber, flowStartMilliseconds,
                       lag(flowStartMilliseconds) OVER (ORDER BY SequenceNumber) AS prev
                FROM read_csv_auto('{source}', header=true)
            ) SELECT count(*) FILTER (WHERE flowStartMilliseconds < prev) FROM x
        """).fetchone()[0]
        # Flow-start interval summaries per UTC hour expose whether each attack file
        # actually contains both labels at useful temporal resolution.
        hourly = con.execute(f"""
            SELECT strftime(to_timestamp(flowStartMilliseconds / 1000.0), '%Y-%m-%d %H:00'),
                   coalesce(Label, '<NULL>'), count(*)
            FROM read_csv_auto('{source}', header=true)
            GROUP BY 1, 2 ORDER BY 1, 2
        """).fetchall()
        # Detect exact repeated flow records and tuple/time collisions.
        dup = con.execute(f"""
            SELECT coalesce(sum(n), 0) AS repeated_rows, count(*) AS repeated_keys
            FROM (
                SELECT hash(flowStartMilliseconds, flowDurationMilliseconds,
                            sourceIPAddress, destinationIPAddress,
                            sourceTransportPort, destinationTransportPort,
                            protocolIdentifier) AS key, count(*) AS n
                FROM read_csv_auto('{source}', header=true)
                GROUP BY ALL HAVING count(*) > 1
            )
        """).fetchone()
        chunk_support = {}
        for chunk_size in (5_000, 10_000, 20_000):
            chunk = con.execute(f"""
                WITH x AS (
                    SELECT Label,
                           row_number() OVER (ORDER BY flowStartMilliseconds, SequenceNumber)-1 AS rn
                    FROM read_csv_auto('{source}', header=true)
                ), y AS (
                    SELECT floor(rn/{chunk_size}) AS chunk,
                           count(*) FILTER (WHERE Label='BENIGN') AS benign,
                           count(*) FILTER (WHERE Label<>'BENIGN') AS attack
                    FROM x GROUP BY 1
                )
                SELECT count(*) AS chunks,
                       count(*) FILTER (WHERE benign>0 AND attack>0) AS mixed,
                       count(*) FILTER (WHERE benign=0 AND attack>0) AS attack_only,
                       count(*) FILTER (WHERE attack=0) AS benign_only,
                       min(benign) AS min_benign, max(attack) AS max_attack
                FROM y
            """).fetchone()
            chunk_support[str(chunk_size)] = {
                "chunks": int(chunk[0]), "mixed_class_chunks": int(chunk[1]),
                "attack_only_chunks": int(chunk[2]), "benign_only_chunks": int(chunk[3]),
                "min_benign_rows": int(chunk[4]), "max_attack_rows": int(chunk[5]),
            }
        files_out.append({
            "filename": path.name,
            "rows": int(summary[0]),
            "start_min_unix_ms": int(summary[1]),
            "start_max_unix_ms": int(summary[2]),
            "duration_min_ms": int(summary[3]),
            "duration_max_ms": int(summary[4]),
            "null_start_rows": int(summary[5]),
            "null_duration_rows": int(summary[6]),
            "negative_duration_rows": int(summary[7]),
            "null_endpoint_rows": int(summary[8]),
            "null_port_rows": int(summary[9]),
            "unique_sequence_numbers": int(summary[10]),
            "sequence_min": int(summary[11]),
            "sequence_max": int(summary[12]),
            "source_order_start_descents": int(regressions),
            "label_counts": {str(k): int(v) for k, v in labels},
            "hourly_label_counts": [
                {"hour": row[0], "label": row[1], "rows": int(row[2])}
                for row in hourly
            ],
            "duplicate_tuple_time_groups": int(dup[1]),
            "rows_in_duplicate_tuple_time_groups": int(dup[0]),
            "chunk_label_support": chunk_support,
            "sha256_verified": digest,
        })
        print(f"{path.name}: rows={summary[0]:,}, labels={dict(labels)}, descents={regressions:,}", flush=True)

    result = {
        "dataset": "Malware communication in smart factories: A network traffic data set",
        "record_doi": manifest["record_doi"],
        "license": manifest["license"],
        "total_rows": sum(x["rows"] for x in files_out),
        "all_download_sha256_verified": True,
        "predictor_columns_for_initial_audit": [
            "flowDurationMilliseconds", "protocolIdentifier", "packetTotalCount"
        ],
        "excluded_columns": {
            "SequenceNumber": "file-local record order; not a predictor",
            "flowStartMilliseconds": "temporal ordering only",
            "sourceIPAddress": "endpoint identity leakage; one attacker address is known to appear in stealth training data",
            "destinationIPAddress": "endpoint identity leakage",
            "sourceTransportPort": "possible attack/campaign proxy; use only as a named ablation",
            "destinationTransportPort": "service/attack proxy; use only as a named ablation",
            "Label": "target",
        },
        "causal_scoring_time": "flowStartMilliseconds + flowDurationMilliseconds, subject to verification of exported flow aggregation semantics",
        "files": files_out,
        "limitations": [
            "Real smart-factory traffic and human-led penetration tests, but attacks are controlled and only four labeled attack days exist.",
            "The public flow feature list is very small (nine columns including sequence, timestamp, endpoints, ports, protocol, packet count, and label); it is not compatible with the CICFlowMeter feature map used by the current project.",
            "Two separate attack scenarios have separate benign training weeks and must not be concatenated as one continuous deployment stream.",
            "Only flow-level labels are available in the downloaded tables; validation must examine attack label timing and repeated tuple/time flows.",
            "This is one factory and one cell segment; any result supports a controlled real-factory case study, not broad operational generalization.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
