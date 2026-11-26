"""Frozen same-capture baseline on chronological training/calibration/evaluation thirds."""
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from run_tolerance_threshold_audit import tolerance_threshold

ROOT = Path(__file__).resolve().parent


def main():
    with np.load(ROOT / "data/cse_corrected/replay_core6/stream.npz") as stream:
        features, labels, offsets = (stream[key] for key in ("features", "labels", "chunk_offsets"))
    chunks = len(offsets) - 1
    train_end, calibration_end = offsets[chunks // 3], offsets[2 * chunks // 3]
    train_labels = labels[:train_end]
    benign = np.flatnonzero(train_labels == 0)
    sample = np.random.default_rng(42).choice(benign, size=min(100_000, len(benign)), replace=False)
    train = np.sort(np.r_[sample, np.flatnonzero(train_labels == 1)])
    model = LGBMClassifier(n_estimators=60, num_leaves=15, random_state=42, n_jobs=1, verbosity=-1)
    model.fit(features[train], labels[train])
    calibration = features[train_end:calibration_end][labels[train_end:calibration_end] == 0]
    threshold = tolerance_threshold(model.predict_proba(calibration)[:, 1], .001, .05)["threshold"]
    assert threshold is not None
    evaluation = labels[calibration_end:]
    predicted = model.predict_proba(features[calibration_end:])[:, 1] > threshold
    fp = int(predicted[evaluation == 0].sum())
    tp = int(predicted[evaluation == 1].sum())
    report = dict(training_rows=len(train), calibration_benign=len(calibration), threshold=threshold,
                  fp=fp, tp=tp, benign=int((evaluation == 0).sum()), attacks=int(evaluation.sum()),
                  fpr=fp/int((evaluation == 0).sum()), recall=tp/int(evaluation.sum()),
                  scope="Same corrected capture, frozen six-field model. No iid or future-traffic guarantee.")
    path = ROOT / "results/corrected_cse_gate/core6_frozen_baseline.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
