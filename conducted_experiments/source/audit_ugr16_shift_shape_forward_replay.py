"""Forward-time audit of public UGR'16 shift-shape signals.

This is an exploratory, day-batched monitor experiment on the author-maintained
UGR'16 v1 per-minute matrices.  It intentionally avoids random window-level CV:
the first five attack-bearing days train the monitor and all later days are held
out.  The signal family follows Amiri's MIT-licensed drift-aware-nids code, but
the chronological split and strict per-day windowing are a separate audit.

This predicts whether a 10-minute aggregate window contains a labeled attack;
it is not a flow-level IDS result and does not establish that an adaptation gate
improves a downstream detector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_refs" / "drift-aware-nids" / "data" / "ugr16"
OUT_JSON = ROOT / "results" / "ugr16_shift_shape_forward_replay.json"
OUT_CSV = ROOT / "results" / "ugr16_shift_shape_forward_replay_windows.csv"
ATTACK_LABELS = [
    "labeldos", "labelscan11", "labelscan44", "labelnerisbotnet",
    "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyspam",
]
WINDOW = 10
STEP = 5
TRAIN_DAYS = 5
SEED = 42
BOOTSTRAPS = 2000
SHAPE = ["concentration", "top_feature_share", "abruptness"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_windows(x_test: pd.DataFrame, y_test: pd.DataFrame,
                  mu: np.ndarray, sd: np.ndarray, features: list[str]) -> pd.DataFrame:
    raw = np.log1p(x_test[features].to_numpy(dtype=np.float64))
    dev = (raw - mu) / sd
    attack_minute = y_test[ATTACK_LABELS].gt(0).any(axis=1).to_numpy()
    times = pd.to_datetime(x_test["Row"].astype(str), format="%Y%m%d%H%M", errors="raise")
    day_arr = times.dt.date.to_numpy()
    rows: list[dict] = []

    # Reset at each date so no test window crosses a day boundary and no
    # overlapping window can straddle the train/test day cutoff.
    for day in pd.unique(day_arr):
        day_idx = np.flatnonzero(day_arr == day)
        if len(day_idx) != 1440:
            raise ValueError(f"Expected a complete 1440-minute day for {day}, got {len(day_idx)}")
        day_dev = dev[day_idx]
        day_attack = attack_minute[day_idx]
        norm = np.linalg.norm(day_dev, axis=1)
        jump = np.abs(np.diff(norm, prepend=norm[0]))
        jump_scale = np.median(jump) + 1e-9
        n_feat = day_dev.shape[1]

        for start in range(0, len(day_idx) - WINDOW + 1, STEP):
            block = day_dev[start:start + WINDOW]
            squared = block ** 2
            row_energy = squared.sum(axis=1)
            participation = (row_energy ** 2) / (
                row_energy ** 2 / n_feat + (squared ** 2).sum(axis=1) + 1e-12
            )
            abs_block = np.abs(block)
            rows.append({
                "time": times.iloc[day_idx[start + WINDOW // 2]].isoformat(),
                "day": str(day),
                "magnitude": float(norm[start:start + WINDOW].mean()),
                "concentration": float(1.0 - np.clip((participation.mean() - 1) / (n_feat - 1), 0, 1)),
                "top_feature_share": float((abs_block.max(axis=1) / (abs_block.sum(axis=1) + 1e-12)).mean()),
                "abruptness": float(jump[start:start + WINDOW].max() / jump_scale),
                "is_attack": int(day_attack[start:start + WINDOW].any()),
            })
    return pd.DataFrame(rows)


def day_bootstrap_delta(y: np.ndarray, groups: np.ndarray,
                        shape_score: np.ndarray, magnitude_score: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    days = np.unique(groups)
    by_day = {d: np.flatnonzero(groups == d) for d in days}
    gaps = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([by_day[d] for d in chosen])
        if np.unique(y[idx]).size == 2:
            gaps.append(roc_auc_score(y[idx], shape_score[idx]) - roc_auc_score(y[idx], magnitude_score[idx]))
    if not gaps:
        raise ValueError("No valid day-block bootstrap resamples")
    return [float(x) for x in np.percentile(gaps, [2.5, 97.5])]


def main() -> None:
    xtr_path = DATA / "UGR16v1.Xtrain.csv"
    xte_path = DATA / "UGR16v1.Xtest.csv"
    ytr_path = DATA / "UGR16v1.Ytrain.csv"
    yte_path = DATA / "UGR16v1.Ytest.csv"
    paths = [xtr_path, xte_path, ytr_path, yte_path]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("Run the pinned reference repository's experiments/fetch_ugr16.py first")

    x_train = pd.read_csv(xtr_path)
    x_test = pd.read_csv(xte_path)
    y_train = pd.read_csv(ytr_path)
    y_test = pd.read_csv(yte_path)
    for split_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        if len(x) != len(y) or not x["Row"].equals(y["Row"]):
            raise ValueError(f"X/Y {split_name} timestamps are not row-aligned")
    missing_labels = sorted(set(ATTACK_LABELS) - set(y_train.columns))
    if missing_labels:
        raise ValueError(f"Missing attack labels: {missing_labels}")
    if y_train[ATTACK_LABELS].gt(0).any().any():
        raise ValueError("Training interval is not discrete-attack-free")

    features = [c for c in x_train.columns if c != "Row"]
    tr_log = np.log1p(x_train[features].to_numpy(dtype=np.float64))
    mu = tr_log.mean(axis=0)
    sd = tr_log.std(axis=0)
    sd[sd == 0] = 1.0
    windows = build_windows(x_test, y_test, mu, sd, features)
    days = np.sort(windows["day"].unique())
    train_days = set(days[:TRAIN_DAYS])
    train = windows[windows["day"].isin(train_days)].copy()
    test = windows[~windows["day"].isin(train_days)].copy()
    if train["is_attack"].nunique() != 2 or test["is_attack"].nunique() != 2:
        raise ValueError("Chronological split must contain both classes in train and test")

    specs = {
        "magnitude_only": ["magnitude"],
        "shape_only": SHAPE,
        "all_signals": ["magnitude", *SHAPE],
    }
    predictions = {}
    summary = {}
    ytr = train["is_attack"].to_numpy()
    yte = test["is_attack"].to_numpy()
    for name, cols in specs.items():
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        model.fit(train[cols].to_numpy(), ytr)
        score = model.predict_proba(test[cols].to_numpy())[:, 1]
        predictions[name] = score
        summary[name] = {
            "roc_auc": float(roc_auc_score(yte, score)),
            "average_precision": float(average_precision_score(yte, score)),
            "brier_score": float(brier_score_loss(yte, score)),
        }

    groups = test["day"].to_numpy()
    ci = day_bootstrap_delta(yte, groups, predictions["shape_only"], predictions["magnitude_only"])
    test = test.copy()
    for name, score in predictions.items():
        test[f"score_{name}"] = score
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(OUT_CSV, index=False)
    payload = {
        "experiment": "UGR16v1 forward-time shift-signal audit",
        "source": "research_refs/drift-aware-nids, MIT code, pinned commit 7521f12bb7762767c0324b546d3630c719cc77df; feature files from josecamachop/UGR16_FeatureData (upstream data license is not explicit in its README; do not redistribute)",
        "method": "10-minute windows, 5-minute hop, reset per day; first five test days train monitor; all subsequent days held out; benign baseline standardization from attack-free UGR train minutes",
        "not_a_claim": "This is an aggregate-window attack-presence monitor, not flow-level IDS or proof of downstream adaptation benefit.",
        "parameters": {"window_minutes": WINDOW, "hop_minutes": STEP, "train_days": TRAIN_DAYS, "seed": SEED, "bootstrap_replicates": BOOTSTRAPS},
        "input_sha256": {p.name: sha256(p) for p in paths},
        "counts": {
            "train_minutes": len(x_train), "test_minutes": len(x_test),
            "train_days": sorted(train_days), "test_days": sorted(set(test["day"])),
            "train_windows": len(train), "test_windows": len(test),
            "train_positive_windows": int(ytr.sum()), "test_positive_windows": int(yte.sum()),
            "test_attack_days": sorted(test.loc[test.is_attack.eq(1), "day"].unique()),
            "test_benign_only_days": sorted(set(test["day"]) - set(test.loc[test.is_attack.eq(1), "day"])),
        },
        "metrics": summary,
        "shape_minus_magnitude_roc_auc": {
            "point_difference": summary["shape_only"]["roc_auc"] - summary["magnitude_only"]["roc_auc"],
            "day_block_bootstrap_95pct_ci": ci,
        },
        "output_windows_csv": str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
