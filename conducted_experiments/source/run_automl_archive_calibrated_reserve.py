"""Compare reserve selection rules on sampled chronological CSE windows."""
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiment_protocol import make_leakage_safe_pipeline
from research_runner import make_estimator

root = Path(__file__).resolve().parent
cache = root / "data/cse_replay_chunk_cache_v2"
manifest = json.loads((cache / "manifest.json").read_text())
archive = json.loads((root / "results/iotid20_binary_common77_mopso_seed42.json").read_text())
candidates = sorted((r for r in archive["pareto_archive"] if r["feasibility"]["feasible"]),
                    key=lambda r: -r["metrics"]["weighted_f1"])[:4]
features = manifest["features"]
source = pd.read_csv(root / "data/iotid20_binary_common77_dev.csv")
models = []
for row in candidates:
    model = make_leakage_safe_pipeline(make_estimator("lightgbm", row["config"], 42, 1), random_state=42)
    model.fit(source[features], source["is_attack"])
    models.append(model)


windows = []
for chunk in manifest["chunks"][::8]:
    with np.load(cache / chunk["file"]) as data:
        x, y = data["x"], data["y"]
    if min(int((y == label).sum()) for label in (0, 1)) < 100:
        continue
    frame = pd.DataFrame(x, columns=features)
    windows.append((chunk["capture_file"], y, [m.predict_proba(frame)[:, 1] for m in models]))

dates = list(dict.fromkeys(w[0] for w in windows))
calibration = [w for w in windows if w[0] in dates[:3]]
evaluation = [w for w in windows if w[0] in dates[3:]]
thresholds = []
for i in range(len(models)):
    benign = np.sort(np.concatenate([scores[i][y == 0] for _, y, scores in calibration]))
    thresholds.append(float(benign[-1-int(.001*len(benign))]))
records = []
for capture, y, scores in evaluation:
    record = []
    for score, threshold in zip(scores, thresholds):
        pred = score > threshold
        record.append(dict(fpr=float(pred[y==0].mean()), recall=float(pred[y==1].mean())))
    records.append(dict(capture=capture, models=record))
failures = [r for r in records if r["models"][0]["fpr"] > .001]
summary = [dict(cap_windows=sum(r["models"][i]["fpr"] <= .001 for r in records),
                mean_recall=float(np.mean([r["models"][i]["recall"] for r in records])),
                repairs=sum(r["models"][i]["fpr"] <= .001 and
                            r["models"][i]["recall"] >= r["models"][0]["recall"]-.01 for r in failures))
           for i in range(len(models))]
out = dict(calibration_dates=dates[:3], evaluation_dates=dates[3:], thresholds=thresholds,
           windows=len(records), primary_failures=len(failures), summary=summary, records=records)
(root / "results/automl_archive_calibrated_reserve.json").write_text(json.dumps(out,indent=2), encoding="utf-8")
print({k:v for k,v in out.items() if k!="records"})
