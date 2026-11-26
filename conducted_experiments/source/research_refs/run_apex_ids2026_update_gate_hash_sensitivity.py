"""Post-result hash-split sensitivity replay for the APEX update-gate pilot.

Development-only: never reads the previously opened final interval.
"""
from __future__ import annotations

import json
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "apex_ids2026" / "apex_ids2026_hc_subset.parquet"
SEED = int(os.environ.get("APEX_GATE_SEED", "43"))
OUT = ROOT / "results" / f"apex_ids2026_update_gate_hash_sensitivity_seed{SEED}.json"
REPORT = ROOT / "results" / f"APEX_IDS2026_UPDATE_GATE_HASH_SENSITIVITY_SEED{SEED}.md"
CACHE_DIR = ROOT / "results" / "apex_ids2026_update_gate_cache"
BATCH = 400_000
CAP = 10_000
FPR_CAP = 0.001
BOOTSTRAPS = 2_000
MIN_SUPPORT = 100
FIVE_MIN_NS = 300_000_000_000
DAY_NS = 86_400_000_000_000
TRAIN_START, TRAIN_END = "2026-06-21", "2026-07-05"
CAL_START, CAL_END = "2026-07-06", "2026-07-12"
REPLAY_START, REPLAY_END = "2026-07-13", "2026-07-23"
COLS = ["date", "flow_start", "duration_s", "protocol", "packets", "bytes", "tos",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg", "label_binary"]
NUMERIC = ["duration_s", "packets", "bytes", "tos", "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg"]
FEATURE_NAMES = NUMERIC + ["proto_tcp", "proto_udp", "proto_icmp", "proto_other"]


def features(frame: pd.DataFrame) -> np.ndarray:
    arr = []
    for name in NUMERIC:
        arr.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan))
    proto = frame["protocol"].astype("string").str.upper()
    for name in ("TCP", "UDP", "ICMP"):
        arr.append((proto == name).fillna(False).to_numpy(dtype=np.float32))
    arr.append((~proto.isin(["TCP", "UDP", "ICMP"])).fillna(True).to_numpy(dtype=np.float32))
    return np.column_stack(arr).astype(np.float32, copy=False)


def event_time(frame: pd.DataFrame) -> np.ndarray:
    start = pd.to_datetime(frame["flow_start"], utc=True, errors="coerce")
    duration = pd.to_numeric(frame["duration_s"], errors="coerce")
    return (start + pd.to_timedelta(duration, unit="s")).astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", n_estimators=200, num_leaves=31,
        learning_rate=0.05, max_depth=-1, min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, n_jobs=4, deterministic=True, force_col_wise=True, verbosity=-1)


def rowgroups(pf: pq.ParquetFile, start: str, end: str):
    di = pf.schema.names.index("date")
    first, last = (pd.Timestamp(start).date() - pd.Timedelta(days=1),
                   pd.Timestamp(end).date() + pd.Timedelta(days=1))
    for rg in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(rg).column(di).statistics
        if stats and stats.has_min_max:
            if stats.max < first:
                continue
            if stats.min > last:
                break
        yield from pf.iter_batches(batch_size=BATCH, columns=COLS, row_groups=[rg], use_threads=True)


def reservoir_add(samples, ys, counts, rng, x, y, keys):
    for day, label in np.unique(np.column_stack([keys, y]), axis=0):
        k = (int(day), int(label))
        ids = np.flatnonzero((keys == day) & (y == label))
        xx, yy = x[ids], y[ids]
        counts[k] += len(ids)
        if k in samples:
            xx = np.vstack([samples[k], xx]); yy = np.concatenate([ys[k], yy])
        if len(yy) > CAP:
            keep = rng.choice(len(yy), CAP, replace=False)
            xx, yy = xx[keep], yy[keep]
        samples[k], ys[k] = xx, yy


def stack_reservoir(samples, ys):
    return np.vstack(list(samples.values())), np.concatenate(list(ys.values()))


def threshold_from_dev(scores, labels):
    benign = np.asarray(scores)[np.asarray(labels) == 0]
    benign.sort()
    allowed = int(np.floor(FPR_CAP * len(benign)))
    return float(benign[max(0, len(benign) - allowed - 1)])


def conf_by_bin(pred: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys, inv = np.unique(bins, return_inverse=True)
    out = np.zeros((len(keys), 4), dtype=np.int64)
    for j, flag in enumerate((~pred & (y == 0), pred & (y == 0), ~pred & (y == 1), pred & (y == 1))):
        out[:, j] = np.bincount(inv, weights=flag.astype(np.int64), minlength=len(keys)).astype(np.int64)
    return keys, out


def rates_from_rows(y, scores, threshold):
    p = scores > threshold
    tn = int(np.count_nonzero(~p & (y == 0))); fp = int(np.count_nonzero(p & (y == 0)))
    fn = int(np.count_nonzero(~p & (y == 1))); tp = int(np.count_nonzero(p & (y == 1)))
    return np.array([tn, fp, fn, tp], dtype=np.int64)


def metrics(c):
    tn, fp, fn, tp = map(int, c)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "benign_support": tn + fp,
            "attack_support": fn + tp, "fpr": fp / (tn + fp) if tn + fp else None,
            "recall": tp / (fn + tp) if fn + tp else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def bootstrap_bounds(bin_conf, incumbent_conf, seed):
    # Resample every validation time block. A block can inform FPR, recall, or both.
    cand, inc = bin_conf, incumbent_conf
    rng = np.random.default_rng(seed)
    if len(cand) < 5:
        return {"fpr_ucb_95": 1.0, "recall_gain_lcb_95": -1.0, "blocks": int(len(cand))}
    picks = rng.integers(0, len(cand), size=(BOOTSTRAPS, len(cand)))
    cc, ic = cand[picks].sum(axis=1), inc[picks].sum(axis=1)
    cfpr = cc[:, 1] / np.maximum(1, cc[:, 0] + cc[:, 1])
    crec = cc[:, 3] / np.maximum(1, cc[:, 2] + cc[:, 3])
    irec = ic[:, 3] / np.maximum(1, ic[:, 2] + ic[:, 3])
    return {"fpr_ucb_95": float(np.quantile(cfpr, .95)),
            "recall_gain_lcb_95": float(np.quantile(crec - irec, .05)), "blocks": int(len(cand))}


def prepare_feature_cache(pf: pq.ParquetFile):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expected = list(pd.date_range(TRAIN_START, REPLAY_END, freq="D").strftime("%Y-%m-%d"))
    paths = [CACHE_DIR / f"{d}.parquet" for d in expected]
    cache_valid = all(p.exists() and p.stat().st_size > 0 for p in paths)
    if cache_valid:
        try:
            cache_valid = all(pq.ParquetFile(p).metadata.num_rows > 0 for p in paths)
            cache_valid = cache_valid and all(pq.ParquetFile(p).schema.names == FEATURE_NAMES + ["label_binary", "bin_id"] for p in paths)
        except Exception:
            cache_valid = False
    t0w, t0c = time.perf_counter(), time.process_time()
    if cache_valid:
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        print(f"reusing complete approved-feature cache: {rows:,} rows", flush=True)
        return rows, time.perf_counter() - t0w, time.process_time() - t0c

    writers, rows = {}, 0
    for batch in rowgroups(pf, TRAIN_START, REPLAY_END):
        f = batch.to_pandas(); ev = event_time(f)
        dates = pd.to_datetime((ev // DAY_NS) * DAY_NS, unit="ns", utc=True).strftime("%Y-%m-%d")
        mask = (ev >= 0) & (dates >= TRAIN_START) & (dates <= REPLAY_END)
        if not mask.any():
            continue
        ff = f.loc[mask]
        xx = features(ff)
        yy = pd.to_numeric(ff["label_binary"], errors="coerce").to_numpy(dtype=np.int8)
        bb, day_dates = ev[mask] // FIVE_MIN_NS, dates[mask]
        rows += len(yy)
        for date in np.unique(day_dates):
            m = day_dates == date
            payload = pd.DataFrame(xx[m], columns=FEATURE_NAMES)
            payload["label_binary"], payload["bin_id"] = yy[m], bb[m]
            table = pa.Table.from_pandas(payload, preserve_index=False)
            if str(date) not in writers:
                writers[str(date)] = pq.ParquetWriter(CACHE_DIR / f"{date}.parquet", table.schema, compression="zstd")
            writers[str(date)].write_table(table)
        if rows and rows % 5_000_000 < len(yy):
            print(f"cached {rows:,} rows", flush=True)
    for writer in writers.values():
        writer.close()
    missing = [d for d, p in zip(expected, paths) if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Incomplete development cache; missing/empty dates: {missing}")
    print(f"feature cache complete: {rows:,} rows across {len(writers)} days", flush=True)
    return rows, time.perf_counter() - t0w, time.process_time() - t0c


def validation_residue(bin_id: int) -> int:
    digest = hashlib.blake2b(str(int(bin_id)).encode("ascii"), digest_size=8, person=b"APEX5MIN").digest()
    return int.from_bytes(digest, "big") % 5


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(DATA)
    wall0, cpu0 = time.perf_counter(), time.process_time()

    # One source scan: cache only approved predictors, endpoint, and time-bin IDs.
    cache_rows, cache_elapsed_wall, cache_elapsed_cpu = prepare_feature_cache(pf)
    cache_reused = cache_elapsed_wall < 1.0

    # Sample the initial prefix from cached dates, then fit the fixed reference.
    rng = np.random.default_rng(SEED)
    samples, sy, counts = {}, {}, defaultdict(int)
    train_rows = 0
    for date in pd.date_range(TRAIN_START, TRAIN_END, freq="D").strftime("%Y-%m-%d"):
        path = CACHE_DIR / f"{date}.parquet"
        if not path.exists(): continue
        f = pq.read_table(path).to_pandas()
        x, y = f[FEATURE_NAMES].to_numpy(dtype=np.float32), f["label_binary"].to_numpy(dtype=np.int8)
        d = np.full(len(y), int(pd.Timestamp(date).tz_localize("UTC").value // DAY_NS), dtype=np.int64)
        train_rows += len(y); reservoir_add(samples, sy, counts, rng, x, y, d)
    x0, y0 = stack_reservoir(samples, sy)
    fit_start_wall, fit_start_cpu = time.perf_counter(), time.process_time()
    incumbent = model(); incumbent.fit(x0, y0, feature_name=FEATURE_NAMES)
    initial_model_fit_wall = time.perf_counter() - fit_start_wall
    initial_model_fit_cpu = time.process_time() - fit_start_cpu
    del samples, sy

    # Calibration week is scored once; its pooled 0.1% threshold remains frozen.
    dev_scores, dev_labels = [], []
    for date in pd.date_range(CAL_START, CAL_END, freq="D").strftime("%Y-%m-%d"):
        path = CACHE_DIR / f"{date}.parquet"
        if not path.exists(): continue
        f = pq.read_table(path).to_pandas()
        xx, yy = f[FEATURE_NAMES].to_numpy(dtype=np.float32), f["label_binary"].to_numpy(dtype=np.int8)
        dev_scores.append(incumbent.booster_.predict(xx, num_threads=4).astype(np.float32)); dev_labels.append(yy)
    threshold = threshold_from_dev(np.concatenate(dev_scores), np.concatenate(dev_labels))
    dev_conf = rates_from_rows(np.concatenate(dev_labels), np.concatenate(dev_scores), threshold)
    del dev_scores, dev_labels

    no_update = periodic = gated = incumbent
    results, decisions = {"no_update": [], "periodic": [], "gated": []}, []
    evaluation_score_wall_total = evaluation_score_cpu_total = 0.0
    replay_start_wall, replay_start_cpu = time.perf_counter(), time.process_time()
    ordered_days = [d for d in pd.date_range(REPLAY_START, REPLAY_END, freq="D").strftime("%Y-%m-%d")
                    if (CACHE_DIR / f"{d}.parquet").exists()]
    for i, date in enumerate(ordered_days):
        frame = pq.read_table(CACHE_DIR / f"{date}.parquet").to_pandas()
        x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
        y = frame["label_binary"].to_numpy(dtype=np.int8, copy=True)
        bins = frame["bin_id"].to_numpy(dtype=np.int64, copy=True)
        unique_bins = np.unique(bins)
        # Locked split rule: stable hash of each five-minute event-time bin;
        # exactly one of five hash residues is held out, independent of labels.
        val_bins = {int(b) for b in unique_bins if validation_residue(int(b)) == 0}
        is_val = np.fromiter((int(b) in val_bins for b in bins), count=len(bins), dtype=bool)
        rng2 = np.random.default_rng(SEED + i + 17)
        cx, cy = [], []
        for lab in (0, 1):
            inds = np.flatnonzero((~is_val) & (y == lab))
            if len(inds) > CAP: inds = rng2.choice(inds, CAP, replace=False)
            cx.append(x[inds]); cy.append(y[inds])
        cx, cy = np.vstack(cx), np.concatenate(cy)

        scoring0w, scoring0c = time.perf_counter(), time.process_time()
        inc_scores = gated.booster_.predict(x, num_threads=4).astype(np.float32)
        day_scores = {"no_update": no_update.booster_.predict(x, num_threads=4).astype(np.float32),
                      "periodic": periodic.booster_.predict(x, num_threads=4).astype(np.float32),
                      "gated": inc_scores}
        # The full day's predictions are made before labels arrive. The 20% held-out
        # blocks control promotion only; they remain part of retrospective day metrics.
        eval_mask = np.ones(len(y), dtype=bool)
        for name, scores in day_scores.items():
            cc = rates_from_rows(y[eval_mask], scores[eval_mask], threshold)
            keys, bc = conf_by_bin(scores[eval_mask] > threshold, y[eval_mask], bins[eval_mask])
            results[name].append({"date": date, "confusion": metrics(cc), "bin_ids": keys.tolist(), "bin_confusion": bc.tolist()})

        candidate = model()
        fit0w, fit0c = time.perf_counter(), time.process_time()
        candidate.fit(np.vstack([x0, cx]), np.concatenate([y0, cy]), feature_name=FEATURE_NAMES)
        candidate_fit_wall, candidate_fit_cpu = time.perf_counter() - fit0w, time.process_time() - fit0c
        candidate_scoring0w, candidate_scoring0c = time.perf_counter(), time.process_time()
        cand_scores = candidate.booster_.predict(x, num_threads=4).astype(np.float32)
        candidate_scoring_wall = time.perf_counter() - candidate_scoring0w
        candidate_scoring_cpu = time.process_time() - candidate_scoring0c
        evaluation_score_wall_total += time.perf_counter() - scoring0w
        evaluation_score_cpu_total += time.process_time() - scoring0c
        vkeys, vbc = conf_by_bin(cand_scores[is_val] > threshold, y[is_val], bins[is_val])
        ikeys, ibc = conf_by_bin(inc_scores[is_val] > threshold, y[is_val], bins[is_val])
        ib_map = {int(k): c for k, c in zip(ikeys, ibc)}
        ib_aligned = np.vstack([ib_map[int(k)] for k in vkeys]) if len(vkeys) else np.zeros_like(vbc)
        bounds = bootstrap_bounds(vbc, ib_aligned, SEED + 10_000 + i)
        accept = bounds["fpr_ucb_95"] <= FPR_CAP and bounds["recall_gain_lcb_95"] > 0
        decisions.append({"date_labels_released": date, "candidate_validation": metrics(vbc.sum(axis=0)),
            "incumbent_validation": metrics(ib_aligned.sum(axis=0)), **bounds,
            "accepted_for_next_day": bool(accept), "chunk_rows": int(len(y)),
            "candidate_train_rows": int(len(cy)), "candidate_validation_rows": int(np.count_nonzero(is_val)),
            "candidate_fit_wall_s": candidate_fit_wall, "candidate_fit_cpu_s": candidate_fit_cpu,
            "candidate_scoring_wall_s": candidate_scoring_wall, "candidate_scoring_cpu_s": candidate_scoring_cpu,
            "candidate_model_bytes": len(candidate.booster_.model_to_string().encode("utf-8"))})
        periodic = candidate
        if accept: gated = candidate
        print(f"{date}: n={len(y):,}; val FPR UCB={bounds['fpr_ucb_95']:.5%}; "
              f"recall-gain LCB={bounds['recall_gain_lcb_95']:.5%}; promote={accept}", flush=True)
        del frame, x, y, bins, inc_scores, cand_scores, candidate, cx, cy, day_scores
    replay_wall, replay_cpu = time.perf_counter(), time.process_time()

    def summarize(records):
        c = np.sum([np.array([r["confusion"][k] for k in ("tn", "fp", "fn", "tp")], dtype=np.int64) for r in records], axis=0)
        bins_conf = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        for r in records:
            for k, v in zip(r["bin_ids"], r["bin_confusion"]): bins_conf[int(k)] += np.asarray(v, dtype=np.int64)
        eligible = [v for v in bins_conf.values() if v[0]+v[1] >= MIN_SUPPORT and v[2]+v[3] >= MIN_SUPPORT]
        fprs = [v[1]/(v[0]+v[1]) for v in eligible]
        return {"pooled": metrics(c), "daily": [{"date": r["date"], **r["confusion"]} for r in records],
            "eligible_five_min_bins": len(eligible), "bins_over_cap": int(sum(x > FPR_CAP for x in fprs)),
            "share_bins_over_cap": float(np.mean(np.asarray(fprs) > FPR_CAP)) if fprs else None,
            "bin_fpr_median": float(np.median(fprs)) if fprs else None,
            "bin_fpr_p90": float(np.quantile(fprs, .9)) if fprs else None,
            "bin_fpr_p95": float(np.quantile(fprs, .95)) if fprs else None,
            "worst_bin_fpr": float(max(fprs)) if fprs else None}
    out = {"protocol": "APEX_IDS2026_UPDATE_GATE_DEVELOPMENT_PROTOCOL_2026-11-07.md; post-result sensitivity",
        "validation_block_assignment": {"hash": "BLAKE2b", "input": "ASCII decimal UTC five-minute event-time bin ID", "digest_size_bytes": 8, "person": "APEX5MIN", "held_out_residue": 0, "modulus": 5}, "dataset_sha256": "7e7855964ddc8061804fa48acfd6c2a3e1891462466654f9294cdab58744730c",
        "seed": SEED, "split": {"train": [TRAIN_START, TRAIN_END], "calibration": [CAL_START, CAL_END], "replay": [REPLAY_START, REPLAY_END]},
        "threshold": threshold, "initial_train_rows_seen": train_rows, "initial_training_sample": int(len(y0)),
        "calibration": metrics(dev_conf), "results": {k: summarize(v) for k, v in results.items()},
        "decisions": decisions, "timing": {"feature_cache_wall_s": float(cache_elapsed_wall),
          "feature_cache_cpu_s": float(cache_elapsed_cpu), "feature_cache_reused": bool(cache_reused),
          "feature_cache_rows": int(cache_rows),
          "initial_training_scan_wall_s": 0.0, "initial_training_scan_cpu_s": 0.0,
          "initial_model_fit_wall_s": float(initial_model_fit_wall),
          "initial_model_fit_cpu_s": float(initial_model_fit_cpu),
          "candidate_fit_total_wall_s": float(sum(d["candidate_fit_wall_s"] for d in decisions)),
          "candidate_fit_total_cpu_s": float(sum(d["candidate_fit_cpu_s"] for d in decisions)),
          "candidate_scoring_total_wall_s": float(sum(d["candidate_scoring_wall_s"] for d in decisions)),
          "candidate_scoring_total_cpu_s": float(sum(d["candidate_scoring_cpu_s"] for d in decisions)),
          "policy_scoring_total_wall_s": float(evaluation_score_wall_total),
          "policy_scoring_total_cpu_s": float(evaluation_score_cpu_total),
          "development_replay_wall_s": float(replay_wall-replay_start_wall), "run_wall_s": float(replay_wall-wall0),
          "run_cpu_s": float(replay_cpu-cpu0), "replay_cpu_s": float(replay_cpu-replay_start_cpu)},
        "caveat": "Exploratory single-seed development result; no future-risk guarantee; no blind confirmation."}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(out)


def write_report(out):
    lines = [f"# APEX-IDS2026 update-gate hash-split sensitivity — seed {SEED}", "",
        "This is a development-only, single-seed replay on earlier APEX dates; it is not confirmation, a future-risk guarantee, or evidence of general IDS performance.", "",
        f"Frozen threshold: {out['threshold']:.8f}; initial training rows seen: {out['initial_train_rows_seen']:,}; sampled initial training rows: {out['initial_training_sample']:,}.", "",
        "## Policy outcomes", "", "| Policy | Pooled FPR | Pooled recall | Supported 5-min bins over 0.1% FPR | Worst supported-bin FPR |", "|---|---:|---:|---:|---:|"]
    for name, title in (("no_update", "No update"), ("periodic", "Daily periodic"), ("gated", "Block-validation gate")):
        s = out["results"][name]
        lines.append(f"| {title} | {s['pooled']['fpr']:.5%} | {s['pooled']['recall']:.3%} | {s['bins_over_cap']}/{s['eligible_five_min_bins']} ({s['share_bins_over_cap']:.2%}) | {s['worst_bin_fpr']:.3%} |")
    accepted = sum(d["accepted_for_next_day"] for d in out["decisions"])
    sizes = [d["candidate_model_bytes"] for d in out["decisions"]]
    t = out["timing"]
    lines += ["", f"Gate decisions accepted {accepted}/{len(out['decisions'])} candidates. Each proposal used at most 20,000 current-day training labels (10,000 per class) plus 300,000 initial training rows. Held-out validation block labels informed promotion. Candidate model text size ranged from {min(sizes):,} to {max(sizes):,} bytes (median {int(np.median(sizes)):,}).", "", "## Daily outcomes", "", "| Date | No-update FPR | No-update recall | Gated FPR | Gated recall | Gate decision for next day |", "|---|---:|---:|---:|---:|---|"]
    by_date = {name: {r["date"]: r for r in out["results"][name]["daily"]} for name in ("no_update", "gated")}
    for d in out["decisions"]:
        date = d["date_labels_released"]
        a, b = by_date["no_update"][date], by_date["gated"][date]
        lines.append(f"| {date} | {a['fpr']:.5%} | {a['recall']:.3%} | {b['fpr']:.5%} | {b['recall']:.3%} | {'accept' if d['accepted_for_next_day'] else 'reject'} |")
    lines += ["", "## Compute record", "", f"Candidate fits: {len(out['decisions'])}; total candidate-fit wall/CPU time {t['candidate_fit_total_wall_s']:.1f}/{t['candidate_fit_total_cpu_s']:.1f} s. Candidate scoring took {t['candidate_scoring_total_wall_s']:.1f} wall s. Scoring the deployed-policy predictions took {t['policy_scoring_total_wall_s']:.1f} wall s. Initial fit took {t['initial_model_fit_wall_s']:.1f} wall s. The {t['run_wall_s']/60:.1f}-minute policy run reused a feature cache; the initial source-to-cache scan is not included in that timing.", "", "## Interpretation", "", "All current-day predictions use the pre-day model; labels from a day can only affect the next day's model. Candidate selection uses held-out five-minute blocks from the labeled day. The block bootstrap resamples all validation time blocks, but it remains a development diagnostic rather than a distribution-free or future-risk guarantee. Repeated decisions, temporal dependence, single-site sampling, selected labels, and event-time proxy remain material limitations. Results are not confirmation: the APEX source and final interval have already been inspected, and this comparison itself is an exploratory policy-selection result.", "", "## Split sensitivity and provenance", "", "This post-result sensitivity uses deterministic BLAKE2b over the ASCII decimal UTC five-minute event-time bin ID, digest size 8 bytes, personalization `APEX5MIN`; residue 0 modulo 5 is validation and residues 1–4 are candidate training. The exact hash function was not specified in the locked protocol, so this explicit choice tests sensitivity; it does not erase the earlier runner deviation, which used a seeded permutation. The hash sensitivity is not blind confirmation.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
