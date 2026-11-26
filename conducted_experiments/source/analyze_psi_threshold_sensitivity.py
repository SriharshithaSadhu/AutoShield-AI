"""Detector-only PSI threshold sensitivity using real calibration and replay scores."""
import json
from pathlib import Path
import numpy as np
from drift_adaptation_experiment import (
    INITIAL_DATE, INITIAL_TRAIN_FRACTION, MAX_INITIAL_TRAIN_ROWS, RAW_DIR,
    CHUNK_ROWS, capture_date, feature_columns, read_sorted_capture,
    stratified_cap, psi_setup, time_chunks, psi_score,
)

source_name=next(p for p in RAW_DIR.glob('*_TrafficForML_CICFlowMeter.csv') if capture_date(p.name)==INITIAL_DATE)
features=feature_columns(source_name)
initial,_=read_sorted_capture(source_name,features)
split=int(len(initial)*INITIAL_TRAIN_FRACTION)
calibration=initial.iloc[split:]
threshold_quantiles=(0.90,0.95,0.99)
results={}
for seed in (42,43):
    train=initial.iloc[:split]
    core_x,core_y=stratified_cap(train,features,MAX_INITIAL_TRAIN_ROWS,seed)
    psi_features,reference=psi_setup(core_x)
    calibration_scores=[psi_score(chunk,psi_features,reference) for chunk in time_chunks(calibration,CHUNK_ROWS)]
    thresholds={str(q):float(np.quantile(calibration_scores,q)) for q in threshold_quantiles}
    saved=json.loads(Path(f'results/cse_cic_ids2018_adaptation_all10_seed{seed}.json').read_text())
    noadapt=saved['summaries']['no_adaptation']['chunk_records']
    actual=saved['summaries']['feature_psi']['alert_records']
    seed_result={'calibration_chunk_count':len(calibration_scores),'calibration_scores':calibration_scores,'thresholds':thresholds,'quantiles':{}}
    for q in threshold_quantiles:
        threshold=thresholds[str(q)]
        streak=0
        alerts=[]
        for row in noadapt:
            streak=streak+1 if float(row['feature_psi'])>threshold else 0
            if streak>=2:
                alerts.append({'stream_chunk':row['stream_chunk'],'capture_file':row['capture_file'],
                    'timestamp':row['end_timestamp'],'contains_attack':row['contains_attack']})
                streak=0
        onset_by_day=[]
        for filename in dict.fromkeys(row['capture_file'] for row in noadapt):
            day=[row for row in noadapt if row['capture_file']==filename]
            first=next((row['first_attack_timestamp'] for row in day if row['first_attack_timestamp']),None)
            first_alert=next((alert for alert in alerts if alert['capture_file']==filename and first and alert['timestamp']>=first),None)
            delay=None
            if first_alert:
                from datetime import datetime
                delay=(datetime.fromisoformat(first_alert['timestamp'])-datetime.fromisoformat(first)).total_seconds()
            onset_by_day.append({'capture_file':filename,'first_attack_timestamp':first,
                'first_alert_timestamp':first_alert['timestamp'] if first_alert else None,'delay_seconds':delay})
        actual_pairs={(a['stream_chunk'],a['capture_file']) for a in actual}
        simulated_pairs={(a['stream_chunk'],a['capture_file']) for a in alerts}
        seed_result['quantiles'][str(q)]={
            'alert_count':len(alerts),
            'benign_only_alert_count':sum(not a['contains_attack'] for a in alerts),
            'capture_days_with_post_onset_alert':sum(x['delay_seconds'] is not None for x in onset_by_day),
            'onset_delay_by_day':onset_by_day,
            'matches_saved_policy_alerts':simulated_pairs==actual_pairs if q==0.95 else None,
        }
    results[str(seed)]=seed_result
out=Path('results/cse_psi_threshold_sensitivity_seeds42_43.json')
out.write_text(json.dumps(results,indent=2),encoding='utf-8')
for seed,d in results.items():
 print('seed',seed,'calibration',d['thresholds'])
 for q,s in d['quantiles'].items():
  print(' q',q,'alerts',s['alert_count'],'benign-only',s['benign_only_alert_count'],'days signalled',s['capture_days_with_post_onset_alert'],'reproduces .95',s['matches_saved_policy_alerts'])
print('saved',out)
