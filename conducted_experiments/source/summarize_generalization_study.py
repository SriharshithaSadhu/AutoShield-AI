"""Rebuild the evidence report from completed result files."""

from pathlib import Path

import json,numpy as np

R=Path(__file__).resolve().parent;O=R/'results/generalization_study_2026-11-19'

def read(path):return json.loads((R/path).read_text())

def pct(x):return f'{100*x:.6f}%' if 0<x<1e-5 else f'{100*x:.3f}%'

def span(values):return f'{pct(min(values))}–{pct(max(values))}'

def main():

 O.mkdir(exist_ok=True);raw=read('results/detection_foundation/raw_conflict_audit.json');context=read('results/detection_foundation/causal_context_validation.json')

 lines=['# AutoShield AI: generalization-aware foundation study — 19 November 2026','',

 '## Status','',

 'The raw conflict cause is resolved. A reusable evaluator, causal history extractor, matched feature experiments, fixed-bank selection rules, and a published adaptive-window mean-estimation baseline were implemented. A positive detection or novel algorithmic contribution is not established. No fallback/review-allocation controller was added.','',

 'All NF-v3 outcomes here are development evidence on a previously explored network. Candidate choices exclude final calibration/evaluation in their execution path; that does not restore independent confirmation after earlier dataset exploration.','',

 '## Feature-label conflicts','',

 f"All {raw['cached_conflict_rows']:,} training rows in {raw['cached_conflicting_vectors']} conflicting vectors remain conflicting under raw exporter text and float64 numeric values. None of these rows changes any feature value under float32 conversion.",

 '','| Label/family | Conflicting training rows |','|---|---:|']

 for k,v in raw['families'].items():lines.append(f'| {k} | {v:,} |')

 lines+=['', 'Each conflicting vector spans different flow times; full time/endpoint/port context yields zero conflicting groups in this training audit. This indicates aggregate-feature aliasing across distinct flows. It does not prove all underlying annotations are correct. Identifiers and timestamps were used for provenance only.',

 '','Training-conflict vectors also appear in tune (38,893 rows / 47 vectors), calibration (23,235 / 42), and evaluation (49,613 / 89). No difficult row was removed. Of the conflict rows, 116,587 contain at most two packets and 115,864 have 1 ms duration.','',

 '## Causal representation experiment','',

 'Four log1p features count prior-one-second source completions, destination completions, distinct destination ports per source, and global completions. Endpoint values serve as transient counting keys. Only strictly earlier completion timestamps contribute; simultaneous events are excluded. No label is used for extraction.',

 f"Training aliasing decreases from {context['baseline']['rows']:,} rows / {context['baseline']['vectors']} vectors to {context['augmented']['rows']} rows / {context['augmented']['vectors']} vectors. Cache parity was bit-exact on {context['reference_compared_rows']:,} training rows, and 2,000 independent reference checks cover ordering and expiry.",

 '', 'This improved separability does not transfer to the later period. Both local-only and global-only history fail, so removing the global traffic-rate feature is insufficient. The context-model gain is concentrated in unstable history variables (about 47.1% of split gain in seed 42). Post-evaluation distribution diagnostics are explanatory and were not used to redesign frozen features.','',

 '## Matched detection ablations','',

 'Three stochastic tree seeds, fixed train/calibration/evaluation indices, identical 100-tree model settings. All 72 operating points fail pooled FPR <= their declared cap together with attack recall >=50% (caps 0.1% and 1%). The table below shows the primary 0.1% target; ranges are seed ranges, not population confidence intervals.',

 '', '| Representation | Family weights | Calibration | FPR range | Recall range | Highest 20-block FPR |','|---|---|---|---|---|---|']

 rows=[]

 for folder in ('causal_context_ablation','causal_context_components'):rows+=read('results/'+folder+'/results.json')['results']

 for c,b,label in [(False,False,'46 fields'),(False,True,'46 fields'),(True,False,'46 + history'),(True,True,'46 + history'),('local',False,'46 + local history'),('global',False,'46 + global history')]:

  for policy in ('pooled','worst_block'):

   rr=[v for v in rows if v['context']==c and v['family_balanced']==b and v['cap']==.001 and v['policy']==policy]

   lines.append(f'| {label} | {b} | {policy} | {span([v["fpr"] for v in rr])} | {span([v["recall"] for v in rr])} | {pct(max(v["worst_window_fpr"] for v in rr))} |')

 lines+=['','The worst-block calibrator uses five earlier benign blocks and a union-adjusted order-statistic threshold. Its conservative operating points often suppress nearly all unseen-family detections. Its assumed iid/exchangeability conditions do not establish a future-traffic guarantee. Equal-sized sampled blocks are not full-capture time windows; separate 60s/600s diagnostics record support and explicitly label sampled counts.','',

 'A post-hoc evaluation-label oracle over all 18 fitted score functions reaches at most 20.720% recall at 0.1% FPR (37.248% at 1% FPR). This diagnostic is undeployable and was not used to select thresholds. It shows that monotonic probability recalibration or a different scalar threshold cannot meet 50% recall for these scores on this period; model ranking/generalization must improve.', '',

 '## Shared candidate-bank selection','',

 'Four shared candidates: two fixed LightGBM configurations crossed with 46 versus 50 fields. Five seeds use actual row/feature subsampling. Fits use the first 180,000 training samples; the remaining training prefix supplies proxy benign calibration. Three later tune windows score chronological generalization. Final benign calibration and final-period labels are excluded from selection.',

 '', 'Internal family-transfer folds remove a complete attack family from fitting within the known training prefix. They are group-transfer diagnostics, not chronological forecasts. Earlier scanning-only-prefix experiments are preserved in generalization_evaluator and v2; they cannot isolate removal of families already absent from that prefix. v3 fixes coverage but its rules still choose the same candidate.','',

 'The generalization score combines minimum family-transfer and temporal recall lower bounds, with Bonferroni allocation over the fixed bank/groups. These are heuristic selection scores under unverified independent-Bernoulli assumptions, not certified lower bounds on arbitrary unseen-family deployment recall.','',

 '| Rule | Seeds | Later FPR range | Later recall range | Under 0.1% FPR and >=50% recall |','|---|---:|---|---|---:|']

 bank=read('results/representation_selection/results.json')

 for rule in ('pooled','worst_window','family_empirical','generalization_bound'):

  rr=[run['outcomes'][str(run['decisions'][rule]['id'])] for run in bank['runs']]

  lines.append(f'| {rule} | {len(rr)} | {span([v["fpr"] for v in rr])} | {span([v["recall"] for v in rr])} | {sum(v["fpr"]<=.001 and v["recall"]>=.5 for v in rr)} |')

 lines+=['', 'The family-aware rules have zero minimum held-family recall and fail the development recall requirement. Their chosen fits are best-effort diagnostics, not approved deployments. A strict selector returns no eligible model when any required constraint or resource measurement is missing. Seed variation within one capture does not establish independent-network uncertainty.','']

 arw=R/'results/representation_selection/arw_results.json'

 if arw.exists():

  rr=json.loads(arw.read_text())['runs'];lines+=['## Published estimation baseline','',f'Algorithm 1 of Han et al. (ICML 2024) is implemented using their empirical Bernstein and bias-proxy equations. Its estimate-based IDS selection adaptation yields later FPR {span([v["fpr"] for v in rr])} and recall {span([v["recall"] for v in rr])}. Their full pairwise tournament was not reproduced. This baseline uses the same bank, information and validation labels.','']

 full=R/'results/context_full_period/results.json'

 if full.exists():

  d=json.loads(full.read_text());lines+=['## Frozen full-period inference','',f"All {d['evaluated_rows']:,} final-period flows were scored in actual batches of up to 1,000 after processing earlier unlabelled history. This is an expanded development check, not an independent network. Model hashes and thresholds were frozen before this run.",'','| Fixed model / calibration | FPR | Recall | False positives | Missed attacks | Worst 20,000-flow FPR |','|---|---|---|---:|---:|---|']

  for name,v in d['stats'].items():

   w=d['flow_windows'][name].values();worst=max(k['fp']/k['benign'] for k in w if k['benign'])

   lines.append(f"| {name} | {pct(v['fp']/v['benign'])} | {pct(v['tp']/v['attacks'])} | {v['fp']:,} | {v['attacks']-v['tp']:,} | {pct(worst)} |")

  lines+=['', 'Full-period resource measurements: '+json.dumps(d['resources'])+'. Batch compute includes history, two model scores, four fixed thresholds and metric bookkeeping; parsing is separate. Exporter lag, network transport, queue waiting and alert delivery were not measured. Completion ordering is an offline replay proxy.','']

 strict=R/'results/representation_selection/strict_constraints.json'

 if strict.exists():

  sr=json.loads(strict.read_text())['runs'];lines+=['## Strict resource and detection feasibility','', 'All 20 candidate profiles were measured in separate desktop processes. Native one-row p95 was 0.022–0.054 ms, serialized size 0.106–0.257 MiB, and conservative peak resident working set 171–173 MiB. These are component measurements under the observed desktop workload, not Raspberry Pi or end-to-end alert latency.', '', 'After applying the unchanged 0.1% FPR, 50% recall, 1.5 MiB model, 5 ms native inference and 512 MiB resident-memory constraints, pooled/worst-window validation selects candidate 3 in all five seeds. Family-transfer rules return no qualified candidate in all five seeds. The later-period failure of the pooled choices remains unchanged. Resident-memory caps do not bound virtual address space or committed memory.', '']

 resource=read('results/detection_foundation/causal_context_desktop_resources.json');lines+=['## Resource implementation','',f"Bit-compatible history extraction processed {resource['flows_per_wall_second']:,.0f} flows/s in a 200,000-flow desktop component benchmark ({resource['mean_cpu_us_per_flow']:.2f} CPU microseconds/flow). The full unlabelled trace contains at most 17,721 strictly prior-one-second completions and 457 simultaneous completions.",

 '', 'deployment/raspberry_pi_foundation contains native text models, hashes, cached inference inputs and a hardware-checking benchmark. Exported scores matched original scores exactly on 1,000 vectors. The --require-pi guard correctly refused this desktop. No actual Pi measurements exist. Models are marked research-only because their detection operating points fail.',

 '', 'Isolated process benchmarks include conservative OS peak working set. Early wrapper timing under concurrent fitting is preserved as load-sensitive evidence; native/component timing and fitting-free reruns are reported separately. Observed process RSS is not a model-only allocation or proof of device feasibility.','',

 '## Contribution boundary and next experiment','',

 'Group DRO (Sagawa et al., ICLR 2020), DomainBed (Gulrajani and Lopez-Paz, ICLR 2021), IoTGeM (Kostas et al., Computer Networks 2025), and temporal selection (Han et al., ICML 2024) already cover the major ingredients. Li and Zhou (Electronics, October 2026) already report historical-threshold budget failures on NF-v3. These results establish neither optimizer superiority nor a new robust-learning algorithm. See GENERALIZATION_PRIOR_ART_2026-11-19.md for primary sources.',

 '', 'The next high-value experiment is a predeclared multi-environment training study with better benign-profile and attack-family coverage, using a compatible legitimate representation, strong conventional baselines, and a genuinely reserved network. Keep the current bank and its failure evidence intact. Expanding optimizer search cannot repair a bank whose family-transfer operating points are infeasible. No MOPSO/random/other-optimizer comparison was added in this session.',

 '', '## Reproduction entry points','',

 'Reusable APIs: candidate_evaluation.py, generalization_metrics.py and constrained_candidate_selection.py. validate_reusable_evaluator.py exactly reproduces a frozen candidate scorecard and checks calibration/chronology rejection.', '',

 '1. audit_raw_feature_conflicts.py, audit_conflict_context.py, audit_collision_details.py, audit_collision_shift.py.',

 '2. build_causal_flow_context.py; test_causal_flow_context.py; validate_causal_context_cache.py.',

 '3. run_causal_context_ablation.py; run_causal_context_components.py.',

 '4. run_representation_selection_five_seeds.py (resumes cached candidates); run_arw_selection_baseline.py.',

 '5. run_context_full_period.py; benchmark_model_resources_v2.py; export_foundation_benchmark_bundle.py.',

 '6. test_generalization_metrics.py; test_adaptive_rolling_mean.py; test_constrained_candidate_selection.py.',

 '7. analyze_context_time_windows.py; plot_generalization_study.py; summarize_generalization_study.py.',

 '', 'Input split hashes, environment versions, fixed configurations, model hashes and script hashes are stored with results. Earlier notebooks, reproduction scripts and negative experiments remain preserved.']

 (O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(O/'REPORT.md')

if __name__=='__main__':main()

