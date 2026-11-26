import argparse
import json,time
from pathlib import Path
import numpy as np
from lightgbm import LGBMClassifier
from run_tolerance_threshold_audit import tolerance_threshold

def fit(x,y,seed):
    benign=np.flatnonzero(y==0)
    sample=np.random.default_rng(seed).choice(benign,min(100000,len(benign)),replace=False)
    ix=np.sort(np.r_[sample,np.flatnonzero(y==1)])
    return LGBMClassifier(n_estimators=60,num_leaves=15,random_state=42,n_jobs=1,verbosity=-1).fit(x[ix],y[ix])

def scores(model,x):
    return model.predict_proba(x)[:,1]

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, default=Path("data/cse_corrected/replay_core6"))
parser.add_argument("--output", type=Path, default=Path("results/corrected_cse_gate/adaptation_comparison.json"))
parser.add_argument("--evaluation-start-chunk", type=int, help="Zero-based chunk index after the training third.")
parser.add_argument("--schedule-offset", type=int, choices=range(10), default=0,
                    help="Delay the first scheduled update by this many chunks.")
parser.add_argument("--label-delay", type=int, choices=range(11), default=0,
                    help="Extra chunks before labels are released, shared by all policies.")
parser.add_argument("--shadow-safeguards", action="store_true")
parser.add_argument("--shadow-source", choices=("gated", "rollback", "lease"), default="gated")
args = parser.parse_args()
if args.shadow_safeguards:
    from statistics import median
    from collections import deque
    from river.drift import ADWIN
with np.load(args.cache / "stream.npz") as z:
    x,y,o=[z[k] for k in ('features','labels','chunk_offsets')]
    capture_days = z["capture_day"] if "capture_day" in z else None
n=len(o)-1
start=2*n//3 if args.evaluation_start_chunk is None else args.evaluation_start_chunk
if not n//3 < start < n:
    parser.error("evaluation start must follow the training third and precede stream end")
a,b=o[n//3],o[start]
if len(np.unique(y[:a])) != 2 or not (y[a:b] == 0).any() or len(np.unique(y[b:])) != 2:
    raise SystemExit("Chronological thirds lack the class support required for this comparison.")
initial=fit(x[:a],y[:a],42)
t=tolerance_threshold(scores(initial,x[a:b][y[a:b]==0]),.001,.05)['threshold']
models={k:(initial,t) for k in ('frozen','periodic','gated','threshold_only','recall_guard','rollback','lease')}
records={k:[] for k in models}
safeguards = ("rolling_shadow", "adwin_shadow", "bootstrap_reference_shadow",
              "adwin_subblock_shadow") if args.shadow_safeguards else ()
records.update({key: [] for key in safeguards})
capture_counts = {k: {} for k in records}
shadow_events = []
shadow_cost = dict(scoring_seconds=0., active_frozen_scoring_seconds=0.,
                   active_candidate_scoring_seconds=0., routing_seconds=0.,
                   peak_prediction_bytes=0, active_chunks=0)
shadow_model = initial
events=[]
lifecycle = []
active_updates = {}
lease_chunks = 10
monitor_policies = ("rollback", "lease")
for i in range(start,n):
    left,right=o[i:i+2]
    yy=y[left:right]
    if capture_days is not None:
        days = capture_days[left:right]
        day_masks = [(int(day), days == day) for day in np.unique(days)]
    chunk_scores = {}
    predictions = {}
    scoring_costs = {}
    for key,(model,threshold) in models.items():
        model_id = id(model)
        score_start = time.perf_counter()
        if model_id not in chunk_scores:
            chunk_scores[model_id] = scores(model, x[left:right])
            scoring_costs[model_id] = time.perf_counter() - score_start
        predictions[key] = chunk_scores[model_id] > threshold
        if safeguards and key in ("frozen", args.shadow_source) and (key == "frozen" or model is not initial):
            shadow_cost["scoring_seconds"] += time.perf_counter() - score_start
    if safeguards:
        started = time.perf_counter()
        candidate = models[args.shadow_source][0]
        if candidate is not shadow_model:
            shadow_model = candidate
            history = deque(maxlen=3)
            reference = []
            detector = ADWIN(delta=.002, clock=1)
            subblock_detector = ADWIN(delta=.002, clock=1)
            latched = dict.fromkeys(safeguards, False)
        frozen_pred, candidate_pred = predictions["frozen"], predictions[args.shadow_source]
        if candidate is not initial:
            shadow_cost["active_chunks"] += 1
            shadow_cost["active_frozen_scoring_seconds"] += scoring_costs[id(initial)]
            shadow_cost["active_candidate_scoring_seconds"] += scoring_costs[id(candidate)]
            rate = float(candidate_pred.mean())
            shadow_cost["peak_prediction_bytes"] = max(shadow_cost["peak_prediction_bytes"],
                frozen_pred.nbytes + candidate_pred.nbytes + chunk_scores[id(initial)].nbytes
                + chunk_scores[id(candidate)].nbytes)
            prior = detector.estimation
            detector.update(rate)
            subblock_increase = False
            for block_start in range(0, len(candidate_pred), 1000):
                block = candidate_pred[block_start:block_start+1000]
                if len(block) < 1000:
                    continue
                block_rate = float(block.mean())
                block_prior = subblock_detector.estimation
                subblock_detector.update(block_rate)
                subblock_increase |= subblock_detector.drift_detected and block_rate > block_prior
            triggers = {
                "rolling_shadow": len(history) == 3 and rate - median(history) > .10,
                "adwin_shadow": detector.drift_detected and rate > prior,
                "adwin_subblock_shadow": subblock_increase,
                "bootstrap_reference_shadow": len(reference) == 3 and rate - median(reference) > .10,
            }
            for key, triggered in triggers.items():
                if triggered and not latched[key]:
                    latched[key] = True
                    shadow_events.append(dict(policy=key, chunk=i, candidate_alert_rate=rate))
            history.append(rate)
            if len(reference) < 3:
                reference.append(rate)
        for key in safeguards:
            predictions[key] = frozen_pred if candidate is not initial and latched[key] else candidate_pred
        shadow_cost["routing_seconds"] += time.perf_counter() - started
    for key, pred in predictions.items():
        records[key].append(dict(chunk=i,benign=int((yy==0).sum()),attacks=int(yy.sum()),
            fp=int((pred & (yy==0)).sum()),tp=int((pred & (yy==1)).sum())))
        if capture_days is not None:
            for day, mask in day_masks:
                counts = capture_counts[key].setdefault(day, dict(benign=0, attacks=0, fp=0, tp=0))
                counts["benign"] += int((mask & (yy == 0)).sum())
                counts["attacks"] += int((mask & (yy == 1)).sum())
                counts["fp"] += int((mask & pred & (yy == 0)).sum())
                counts["tp"] += int((mask & pred & (yy == 1)).sum())
    released = i - args.label_delay
    if released >= start:
        for policy in monitor_policies:
            state = active_updates.get(policy)
            if state is None:
                continue
            observed = records[policy][released-start]
            deployed = state["deployed"]
            supported = observed["benign"] >= 3000
            breach = supported and 1000 * observed["fp"] > observed["benign"]
            if released >= deployed and supported:
                state["evidence"].append(observed)
            reason = None
            if policy == "rollback" and released >= deployed and breach:
                reason = "observed_cap_breach"
            if policy == "lease" and i + 1 == state["expires"]:
                recent = state["evidence"][-2:]
                renew = len(recent) == 2 and all(1000*r["fp"] <= r["benign"] for r in recent)
                lifecycle.append(dict(policy=policy, action="renew" if renew else "expire",
                                      after_chunk=i, evidence_chunks=[r["chunk"] for r in recent]))
                if renew:
                    state["expires"] += lease_chunks
                    state["evidence"].clear()
                else:
                    reason = "lease_not_renewed"
            if reason:
                models[policy] = state["previous"]
                del active_updates[policy]
                lifecycle.append(dict(policy=policy, action="restore", after_chunk=i,
                                      observed_chunk=released, reason=reason))
    since_start = released - start + 1
    if since_start < 10 + args.schedule_offset or (since_start - args.schedule_offset) % 10 or i == n-1:
        continue
    # Fit on six released chunks, calibrate on the following two,
    # then check on the final two released chunks before deployment.
    lo,mid,valid=o[released-9],o[released-3],o[released-1]
    labeled_end = o[released+1]
    calibration_benign = x[mid:valid][y[mid:valid] == 0]
    guard_candidates = {}
    threshold_seconds = None
    if len(calibration_benign) >= 2995:
        before_threshold = time.perf_counter()
        threshold_only = tolerance_threshold(scores(initial, calibration_benign), .001, .05)["threshold"]
        models["threshold_only"] = (initial, threshold_only)
        guard_model = models["recall_guard"][0]
        guard_threshold = threshold_only
        if guard_model is not initial:
            guard_threshold = tolerance_threshold(scores(guard_model, calibration_benign), .001, .05)["threshold"]
        guard_candidates["recalibrate"] = (guard_model, guard_threshold)
        threshold_seconds = time.perf_counter() - before_threshold
    event = dict(after_chunk=i, threshold_only_seconds=threshold_seconds)
    yy = y[valid:labeled_end]
    if len(np.unique(y[lo:mid])) < 2 or len(calibration_benign) < 2995:
        event["skipped"] = "insufficient_fit_or_calibration_support"
    else:
        before = time.perf_counter()
        candidate=fit(x[lo:mid],y[lo:mid],42)
        threshold=tolerance_threshold(scores(candidate,x[mid:valid][y[mid:valid]==0]),.001,.05)['threshold']
        yy=y[valid:labeled_end]
        pred=scores(candidate,x[valid:labeled_end])>threshold
        incumbent,old_t=models['gated']
        old=scores(incumbent,x[valid:labeled_end])>old_t
        # Require empirical benign cap in each validation chunk and a pooled recall gain.
        cap_pass=all((pred[s:e] & (yy[s:e]==0)).sum() <= .001*(yy[s:e]==0).sum()
            for s,e in [(0,o[released]-valid),(o[released]-valid,labeled_end-valid)])
        gain=int((pred & (yy==1)).sum())-int((old & (yy==1)).sum())
        accept=bool(cap_pass and gain>0)
        models['periodic']=(candidate,threshold)
        if accept:
            models['gated']=(candidate,threshold)
        event.update(dict(after_chunk=i,accepted=accept,cap_pass=bool(cap_pass),validation_tp_gain=gain,
            validation_attacks=int(yy.sum()),shared_candidate_and_gate_seconds=time.perf_counter()-before,
            threshold_only_seconds=threshold_seconds))
        if cap_pass and int(yy.sum()) > 0:
            for policy in monitor_policies:
                if policy in active_updates:
                    continue
                previous = models[policy]
                previous_scores = scores(previous[0], x[valid:labeled_end])
                previous_tp = int(((previous_scores > previous[1]) & (yy == 1)).sum())
                if int((pred & (yy == 1)).sum()) > previous_tp:
                    models[policy] = (candidate, threshold)
                    active_updates[policy] = dict(previous=previous, deployed=i+1,
                                                  expires=i+1+lease_chunks, evidence=[])
                    lifecycle.append(dict(policy=policy, action="approve", after_chunk=i,
                                          last_released_chunk=released))
        guard_candidates["refit"] = (candidate, threshold)
    guard_start = time.perf_counter()
    attack_count = int(yy.sum())
    incumbent, incumbent_threshold = models["recall_guard"]
    incumbent_predictions = scores(incumbent, x[valid:labeled_end]) > incumbent_threshold
    incumbent_fp = int(incumbent_predictions[yy == 0].sum())
    incumbent_tp = int(incumbent_predictions[yy == 1].sum())
    action, best_fp = "keep", incumbent_fp
    validation_chunks = [slice(0, o[released]-valid), slice(o[released]-valid, labeled_end-valid)]
    if attack_count >= 100 and all((yy[part] == 0).sum() >= 3000 for part in validation_chunks):
        for name, (model, cutoff) in guard_candidates.items():
            prediction = scores(model, x[valid:labeled_end]) > cutoff
            fp = int(prediction[yy == 0].sum())
            tp = int(prediction[yy == 1].sum())
            cap_met = all(1000 * prediction[part][yy[part] == 0].sum() <= (yy[part] == 0).sum()
                          for part in validation_chunks)
            if cap_met and 100 * (incumbent_tp - tp) <= attack_count and fp < best_fp:
                action, best_fp = name, fp
    if action != "keep":
        models["recall_guard"] = guard_candidates[action]
    event.update(guard_action=action, guard_validation_attacks=attack_count,
                 guard_validation_fp_reduction=incumbent_fp-best_fp,
                 guard_decision_seconds=time.perf_counter()-guard_start)
    events.append(event)
summary={}
for key,rr in records.items():
    total={k:sum(r[k] for r in rr) for k in ('fp','tp','benign','attacks')}
    summary[key]=dict(**total,fpr=total['fp']/total['benign'],recall=total['tp']/total['attacks'],
        worst_chunk_fpr=max(r['fp']/r['benign'] for r in rr if r['benign']),
        chunks_over_cap=sum(r['fp']>.001*r['benign'] for r in rr))
if args.cache.name == "replay_core6" and args.evaluation_start_chunk is None:
    assert summary['frozen']['fp']==1527 and summary['frozen']['tp']==8679
report=dict(scope='Exploratory chronological replay. Empirical gate, no risk guarantee.',
    shadow_events=shadow_events,shadow_cost=shadow_cost,shadow_source=args.shadow_source,
    input_cache=str(args.cache),capture_counts=capture_counts,
    train_end_chunk=n//3,evaluation_start_chunk=start,schedule_offset=args.schedule_offset,
    schedule='Every ten evaluation chunks; six fit, two threshold calibration, two gate validation; deploy next chunk.',
    guard_rule=dict(minimum_attacks=100,minimum_benign_per_chunk=3000,
                    max_empirical_recall_loss=.01,validation_chunk_fpr_cap=.001),
    label_delay_chunks=args.label_delay,lease_chunks=lease_chunks,
    lifecycle=lifecycle,summary=summary,events=events,chunks=records)
args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(dict(summary=summary,events=events),indent=2))
