"""Matched chronological NF-v3 feature ablation with calibrated operating points."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from run_tolerance_threshold_audit import tolerance_threshold

ROOT = Path(__file__).resolve().parent
CORE = ["PROTOCOL", "FLOW_DURATION_MILLISECONDS", "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES"]
EXCLUDE = {"FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR",
           "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "Label", "Attack", "DNS_QUERY_ID"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--later-training", action="store_true")
    parser.add_argument("--common-calibration", action="store_true")
    args = parser.parse_args()
    source = ROOT / "data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv"
    with np.load(ROOT / "data/nf_ton_iot_v3/core6_development/stream.npz") as z:
        offsets, labels = z["chunk_offsets"], z["labels"]
    n = len(offsets)-1
    train_end, eval_start = (int(offsets[k]) for k in (n//2 if args.later_training else n//3, 2*n//3))
    rng = np.random.default_rng(42)
    train_labels = labels[:train_end]
    train = np.sort(np.concatenate([rng.choice(np.flatnonzero(train_labels==c), 100000, replace=False) for c in (0,1)]))
    calibration_start = int(offsets[n//2]) if args.common_calibration else train_end
    calibration = np.sort(rng.choice(np.flatnonzero(labels[calibration_start:eval_start]==0)+calibration_start,
                                     200000, replace=False))
    columns = [c for c in pd.read_csv(source,nrows=0).columns if c not in EXCLUDE]
    feature_sets = {"core6": CORE, "rich": columns}
    positions = {k:[columns.index(c) for c in cols] for k,cols in feature_sets.items()}
    models = {}
    fit_seconds = {}
    score_seconds = {k:0. for k in positions}
    thresholds = {}
    metrics = {k:dict(fp=0,tp=0,attacks=0,benign=0,families={},blocks=[]) for k in positions}
    train_x, cal_x = [], []
    offset = 0
    for frame in pd.read_csv(source,usecols=columns+["Label","Attack"],chunksize=100000):
        stop = offset+len(frame)
        x = frame[columns].to_numpy(dtype=np.float32)
        y = frame.Label.to_numpy()
        assert np.array_equal(y, labels[offset:stop])
        for indices, buffers in ((train,train_x),(calibration,cal_x)):
            lo,hi = np.searchsorted(indices,[offset,stop])
            if hi>lo:
                buffers.append(x[indices[lo:hi]-offset])
        if stop >= eval_start and not models:
            tx,cx = np.concatenate(train_x),np.concatenate(cal_x)
            for name, ids in positions.items():
                start = perf_counter()
                model = LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,random_state=42,verbosity=-1)
                model.fit(tx[:,ids],labels[train])
                fit_seconds[name] = perf_counter()-start
                models[name] = model
                thresholds[name] = {str(cap):tolerance_threshold(model.predict_proba(cx[:,ids])[:,1],cap,.05)
                                    for cap in (.001,.01)}
            metrics = {(name,cap):dict(fp=0,tp=0,attacks=0,benign=0,families={},blocks=[])
                       for name in positions for cap in (".001",".01")}
            print("Fit and calibration complete",flush=True)
        if stop > eval_start:
            first = max(0,eval_start-offset)
            labels_now = y[first:]
            families = frame.Attack.to_numpy()[first:]
            for name, ids in positions.items():
                start = perf_counter()
                scores = models[name].predict_proba(x[first:,ids])[:,1]
                score_seconds[name] += perf_counter()-start
                for cap in (".001",".01"):
                    pred = scores > thresholds[name][str(float(cap))]["threshold"]
                    item = metrics[name,cap]
                    counts = dict(fp=int((pred & (labels_now==0)).sum()),tp=int((pred & (labels_now==1)).sum()),
                                  attacks=int(labels_now.sum()),benign=int((labels_now==0).sum()))
                    for key,value in counts.items():item[key]+=value
                    item["blocks"].append(counts)
                    for family in np.unique(families[labels_now==1]):
                        mask = (families==family)&(labels_now==1)
                        pair = item["families"].setdefault(str(family),[0,0])
                        pair[0]+=int(mask.sum())
                        pair[1]+=int((mask&pred).sum())
        offset=stop
    assert offset==len(labels)
    result={}
    for (name,cap),item in metrics.items():
        assert item["benign"]+item["attacks"]==len(labels)-eval_start
        fp,tp,benign,attacks=(item[k] for k in ("fp","tp","benign","attacks"))
        result[f"{name}/{cap}"]=dict(**item,fpr=fp/benign,recall=tp/attacks,
            precision=tp/max(1,tp+fp),f1=2*tp/max(1,tp+fp+attacks),
            worst_block_fpr=max(b["fp"]/b["benign"] for b in item["blocks"] if b["benign"]),
            model_text_bytes=len(models[name].booster_.model_to_string().encode()),
            fit_seconds=fit_seconds[name],score_seconds=score_seconds[name])
    output=ROOT/"results"/("nf_ton_later_training_feature_ablation.json" if args.later_training
                          else "nf_ton_early_common_calibration.json" if args.common_calibration
                          else "nf_ton_matched_feature_ablation.json")
    output.write_text(json.dumps(dict(scope="Previously explored development collection",train_rows=len(train),
        calibration_benign=len(calibration),train_end=train_end,calibration_start=calibration_start,eval_start=eval_start,
        features=feature_sets,thresholds=thresholds,
        results=result),indent=2),encoding="utf-8")
    for key,item in result.items():print(key,{k:item[k] for k in ("fpr","recall","f1","worst_block_fpr","model_text_bytes")})

if __name__=="__main__":
    main()
