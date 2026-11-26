"""Post-hoc descriptive diagnosis of the two APEX temporal error clusters.

Ports are included only as diagnostic strata because dst_port participates in
the publisher's attack-label rule; no model is retrained and no port feature is
added to the primary baseline.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_apex_ids2026_no_update_baseline import (
    BATCH, DAY_NS, FIVE_MIN_NS, DATA, MODEL_OUT, ROOT, make_features, utc_event_ns,
)

OUT = ROOT / "results" / "apex_ids2026_failure_attribution_20261107.json"
REPORT = ROOT / "results" / "APEX_IDS2026_ERROR_CLUSTER_DIAGNOSIS.md"
READ_COLS = ["flow_start", "duration_s", "protocol", "packets", "bytes", "tos",
             "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
             "label_binary", "dst_port", "attack_type", "attack_category", "mitre_technique",
             "mitre_tactic", "confidence", "evidence_source"]
TARGET_BINS = {
    "worst_fpr": int(pd.Timestamp("2026-08-01T08:05:00Z").value // FIVE_MIN_NS),
    "lowest_recall": int(pd.Timestamp("2026-07-29T20:20:00Z").value // FIVE_MIN_NS),
}
FOCAL_DATES = {"2026-07-29", "2026-08-01"}


def iter_focus_batches(pf: pq.ParquetFile):
    date_col = pf.schema.names.index("date")
    first, last = pd.Timestamp("2026-07-28").date(), pd.Timestamp("2026-08-02").date()
    for rg in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(rg).column(date_col).statistics
        if stats is None or not stats.has_min_max:
            yield from pf.iter_batches(batch_size=BATCH, columns=READ_COLS,
                                       use_threads=True, row_groups=[rg])
            continue
        if stats.max < first:
            continue
        if stats.min > last:
            break
        yield from pf.iter_batches(batch_size=BATCH, columns=READ_COLS,
                                   use_threads=True, row_groups=[rg])


def summarize_features(rows: pd.DataFrame) -> dict:
    numeric = ["duration_s", "packets", "bytes", "tos"]
    out = {"rows": int(len(rows))}
    for name in numeric:
        x = pd.to_numeric(rows[name], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x):
            out[name] = {"median": float(np.median(x)),
                         "p10": float(np.quantile(x, .1)),
                         "p90": float(np.quantile(x, .9))}
    for name in ["protocol", "dst_port", "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
                 "attack_type", "attack_category", "mitre_technique", "mitre_tactic", "confidence", "evidence_source"]:
        vals = rows[name].astype("string").fillna("<NULL>").value_counts().head(10)
        out[name + "_top10"] = {str(k): int(v) for k, v in vals.items()}
    return out


def main() -> None:
    pf = pq.ParquetFile(DATA)
    model = lgb.Booster(model_file=str(MODEL_OUT))
    threshold = 0.94419264793396
    retained = []
    daily = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    row_count = 0
    for batch in iter_focus_batches(pf):
        frame = batch.to_pandas()
        event_ns = utc_event_ns(frame)
        day_ids = event_ns // DAY_NS
        dates = pd.to_datetime(day_ids * DAY_NS, unit="ns", utc=True).strftime("%Y-%m-%d")
        take = np.isin(dates, list(FOCAL_DATES)) & (event_ns >= 0)
        if not take.any():
            continue
        part = frame.loc[take].reset_index(drop=True)
        ns = event_ns[take]
        dates = dates[take]
        x, _ = make_features(part)
        scores = model.predict(x, num_threads=4)
        pred = scores > threshold
        y = pd.to_numeric(part["label_binary"], errors="coerce").to_numpy(dtype=np.int8)
        for day in FOCAL_DATES:
            m = dates == day
            if not m.any():
                continue
            p, yy = pred[m], y[m] == 1
            c = np.array([np.count_nonzero(~p & ~yy), np.count_nonzero(p & ~yy),
                          np.count_nonzero(~p & yy), np.count_nonzero(p & yy)], dtype=np.int64)
            daily[day] += c
        bins = ns // FIVE_MIN_NS
        keep = np.zeros(len(part), dtype=bool)
        for target in TARGET_BINS.values():
            keep |= np.abs(bins - target) <= 6
        if keep.any():
            diag = part.loc[keep, ["duration_s", "protocol", "packets", "bytes", "tos", "flag_syn", "flag_ack",
                                   "flag_fin", "flag_rst", "flag_psh", "flag_urg", "label_binary", "dst_port",
                                   "attack_type", "attack_category", "mitre_technique", "mitre_tactic",
                                   "confidence", "evidence_source"]].copy()
            diag["event_bin"] = bins[keep]
            diag["score"] = scores[keep]
            diag["prediction"] = pred[keep].astype(np.int8)
            retained.append(diag)
        row_count += int(take.sum())
        print(f"diagnostic focal-date rows scored: {row_count:,}", flush=True)

    rows = pd.concat(retained, ignore_index=True)
    rows["label_binary"] = pd.to_numeric(rows["label_binary"], errors="coerce").astype(np.int8)
    bins_out = {}
    for label, target in TARGET_BINS.items():
        x = rows[rows["event_bin"] == target].copy()
        x["outcome"] = np.where((x["label_binary"] == 0) & (x["prediction"] == 1), "false_positive",
                        np.where((x["label_binary"] == 1) & (x["prediction"] == 0), "false_negative",
                        np.where(x["label_binary"] == 1, "true_positive", "true_negative")))
        bins_out[label] = {
            "bin_start_utc": pd.to_datetime(target * FIVE_MIN_NS, unit="ns", utc=True).isoformat(),
            "confusion": {"tn": int(((x.label_binary == 0) & (x.prediction == 0)).sum()),
                          "fp": int(((x.label_binary == 0) & (x.prediction == 1)).sum()),
                          "fn": int(((x.label_binary == 1) & (x.prediction == 0)).sum()),
                          "tp": int(((x.label_binary == 1) & (x.prediction == 1)).sum())},
            "by_outcome": {outcome: summarize_features(group)
                           for outcome, group in x.groupby("outcome", sort=True)},
            "adjoining_30min_summary": {
                "before": summarize_features(rows[(rows.event_bin >= target - 6) & (rows.event_bin < target)]),
                "after": summarize_features(rows[(rows.event_bin > target) & (rows.event_bin <= target + 6)]),
            },
        }
    importance = model.feature_importance(importance_type="gain")
    feature_importance = sorted([{"feature": n, "gain": float(v)}
                                 for n, v in zip(model.feature_name(), importance)],
                                key=lambda z: z["gain"], reverse=True)
    result = {
        "source_result": "results/apex_ids2026_no_update_seed43.json",
        "model": str(MODEL_OUT.relative_to(ROOT)),
        "threshold": threshold,
        "diagnostic_only_ports": True,
        "method": "Rescore only the two focal UTC dates; identify rows by flow_start + duration_s; ports are post-hoc strata only.",
        "focal_date_rows_scored": row_count,
        "daily_confusion": {d: {"tn": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "tp": int(c[3])}
                            for d, c in sorted(daily.items())},
        "target_bins": bins_out,
        "model_gain_importance": feature_importance,
        "interpretation_limit": "Descriptive subgroup associations only. Port is part of label construction; association is not causal and does not authorize port predictors in the primary model.",
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# APEX-IDS2026 error-cluster diagnosis — 2026-11-07", "",
        "This is a descriptive post-hoc inspection of two already-reported bins. It does not alter the model or threshold. Ports are diagnostic-only because destination port participates in the publisher's verified-attack label rule.", "",
        f"Rescored {row_count:,} rows from the two focal UTC dates with the fixed seed-43 model and frozen threshold {threshold:.6f}.", "",
        "## Focal bins", "",
    ]
    for label, content in bins_out.items():
        report.extend([f"### {label}: {content['bin_start_utc']}", "",
                       f"Confusion counts: `{json.dumps(content['confusion'], sort_keys=True)}`.", "",
                       "Counts/quantiles by prediction outcome and post-hoc port/protocol strata are in the JSON artifact.", ""])
    report.extend(["## Model-level gain importance", "",
                   "```text", *[f"{x['feature']}: {x['gain']:.3f}" for x in feature_importance], "```", "",
                   "## Limits", "",
                   "These associations do not establish drift causality. Destination-port patterns are especially confounded with the label rule. The high-confidence subset is selected retrospectively, the benign construction used global attacker-IP filtering, and this is one ISP/perimeter. Use the diagnostics to decide what controlled development experiments and new confirmation source are justified; do not claim that the model learned a general attack concept.", "",
                   "Raw breakdown: `results/apex_ids2026_failure_attribution_20261107.json`."])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
