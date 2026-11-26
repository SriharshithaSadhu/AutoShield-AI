"""Profile candidate NF-ToN-IoT-v3 predictors before temporal modeling."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "nf_ton_iot_v3" / "NF-ToN-IoT-v3_endtime_ordered.csv"
OUT = ROOT / "results" / "nf_ton_iot_v3_model_schema_audit.json"
CHUNK_ROWS = 500_000

EXCLUDED = {
    "IPV4_SRC_ADDR": "network identity; may memorize hosts/campaign endpoints",
    "IPV4_DST_ADDR": "network identity; may memorize hosts/campaign endpoints",
    "L4_SRC_PORT": "endpoint identity; excluded from this behavior-only baseline",
    "L4_DST_PORT": "endpoint identity; excluded from this behavior-only baseline",
    "FLOW_START_MILLISECONDS": "chronology metadata, not a model predictor",
    "FLOW_END_MILLISECONDS": "flow completion chronology metadata, not a model predictor",
    "Label": "binary outcome target",
    "Attack": "attack-family target/metadata; direct label leakage if used as predictor",
    "DNS_QUERY_ID": "transaction identifier; excluded to avoid identifier memorization",
}
CATEGORICAL = {
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
}

header = pd.read_csv(SOURCE, nrows=0).columns.tolist()
missing_exclusions = sorted(set(EXCLUDED) - set(header))
if missing_exclusions:
    raise ValueError(f"Expected excluded fields missing from data: {missing_exclusions}")
features = [name for name in header if name not in EXCLUDED]
numeric_usecols = features + ["Label"]

stats = {
    col: {"missing": 0, "non_numeric": 0, "positive_infinity": 0,
          "negative_infinity": 0, "finite": 0, "zero": 0, "negative": 0,
          "minimum": None, "maximum": None}
    for col in features
}
category_counts: dict[str, Counter[str]] = {col: Counter() for col in CATEGORICAL if col in features}
label_counts: Counter[str] = Counter()
rows = 0

for chunk in pd.read_csv(SOURCE, usecols=numeric_usecols, chunksize=CHUNK_ROWS,
                         low_memory=False, na_values=["", "NA", "NaN", "nan"]):
    rows += len(chunk)
    labels = pd.to_numeric(chunk["Label"], errors="coerce")
    label_counts.update(labels.value_counts(dropna=False).to_dict())
    for col in features:
        raw = chunk[col]
        values = pd.to_numeric(raw, errors="coerce")
        missing = values.isna()
        converted_missing = raw.notna() & missing
        arr = values.to_numpy(dtype=np.float64, na_value=np.nan)
        positive_inf = np.isposinf(arr)
        negative_inf = np.isneginf(arr)
        finite_mask = np.isfinite(arr)
        finite_values = arr[finite_mask]
        item = stats[col]
        item["missing"] += int(missing.sum())
        item["non_numeric"] += int(converted_missing.sum())
        item["positive_infinity"] += int(positive_inf.sum())
        item["negative_infinity"] += int(negative_inf.sum())
        item["finite"] += int(finite_mask.sum())
        item["zero"] += int((finite_values == 0).sum())
        item["negative"] += int((finite_values < 0).sum())
        if finite_values.size:
            lo, hi = float(finite_values.min()), float(finite_values.max())
            item["minimum"] = lo if item["minimum"] is None else min(item["minimum"], lo)
            item["maximum"] = hi if item["maximum"] is None else max(item["maximum"], hi)
        if col in category_counts:
            cleaned = raw.astype("string").fillna("<MISSING>").str.strip()
            category_counts[col].update(cleaned.value_counts(dropna=False).to_dict())
    print(f"profiled_rows={rows:,}", flush=True)

for col, item in stats.items():
    item["constant_or_single_valued"] = item["minimum"] == item["maximum"]
    item["missing_fraction"] = item["missing"] / rows if rows else None
    item["nonfinite_count"] = item["positive_infinity"] + item["negative_infinity"]

result = {
    "source": SOURCE.relative_to(ROOT).as_posix(),
    "rows": rows,
    "total_input_columns": len(header),
    "excluded_fields_and_reasons": EXCLUDED,
    "candidate_feature_count": len(features),
    "candidate_features": features,
    "categorical_code_fields_tentatively_typed_as_categorical": sorted(category_counts),
    "binary_label_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
    "feature_statistics": stats,
    "categorical_value_counts": {
        col: {value: int(count) for value, count in sorted(counter.items())}
        for col, counter in category_counts.items()
    },
    "limitations": [
        "This is a schema/data-quality profile, not a feature-selection or leakage-proof certificate.",
        "Numeric code fields are marked as categorical candidates from their documented meaning; the final model protocol must freeze these types.",
        "Flow-level labels follow the publisher's documented join and have not been independently re-annotated here."
    ]
}
OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "rows": rows,
    "candidate_feature_count": len(features),
    "label_counts": result["binary_label_counts"],
    "fields_with_missing": [k for k, v in stats.items() if v["missing"]],
    "fields_with_non_numeric": [k for k, v in stats.items() if v["non_numeric"]],
    "fields_with_nonfinite": [k for k, v in stats.items() if v["nonfinite_count"]],
    "constant_features": [k for k, v in stats.items() if v["constant_or_single_valued"]],
    "output": OUT.relative_to(ROOT).as_posix(),
}, indent=2))
