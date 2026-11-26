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

def measure(model, frame, y):
    start = perf_counter()
    prediction = model.predict(frame)
    elapsed = perf_counter() - start
    return [int(((prediction == 1) & (y == label)).sum()) / int((y == label).sum())
            for label in (0, 1)] + [elapsed]

windows = []
for chunk in manifest["chunks"][::8]:
    with np.load(cache / chunk["file"]) as data:
        x, y = data["x"], data["y"]
    if min(int((y == label).sum()) for label in (0, 1)) < 100:
        continue
    frame = pd.DataFrame(x, columns=features)
    windows.append(dict(capture=chunk["capture_file"], scores=[measure(m, frame, y) for m in models]))
dates = list(dict.fromkeys(w["capture"] for w in windows))
selection = [w for w in windows if w["capture"] in dates[:3]]
evaluation = [w for w in windows if w["capture"] in dates[3:]]
def coverage(windows, reserve):
    failures = [w for w in windows if w["scores"][0][0] > .001]
    return dict(failures=len(failures), repairs=sum(
        w["scores"][reserve][0] <= .001 and w["scores"][reserve][1] >= w["scores"][0][1] - .01
        for w in failures))
chosen = max(range(1,len(models)), key=lambda i: coverage(selection,i)["repairs"])
out = dict(selection_dates=dates[:3], evaluation_dates=dates[3:], selected=chosen,
           sizes=[len(pickle.dumps(m)) for m in models], candidates=candidates,
           selection=[coverage(selection,i) for i in range(1,len(models))],
           evaluation=[coverage(evaluation,i) for i in range(1,len(models))], windows=windows)
(root / "results/automl_archive_reserve_pilot.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(out["selection"], out["evaluation"], chosen)
