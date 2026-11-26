"""Isolated remeasurement of selected MOPSO/random candidate latency after CSE finishes."""
import json
from pathlib import Path
from deployment_evaluation import DeploymentConstraints, assess_feasibility
from research_runner import evaluate_development_candidates

for dataset in ['cicids2017','iotid20']:
    selected = {}
    associations = {}
    for method, seed in [('mopso',42),('random',42042)]:
        path = Path('results') / f'{dataset}_lightgbm_{method}_dev_seed{seed}.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        candidates = saved['candidate_records'] if method == 'mopso' else saved['candidates']
        records=[]
        for row in candidates:
            if method == 'mopso':
                metrics=row['metrics']
            else:
                a=row['aggregate_metrics']
                metrics={k:a[k]['mean'] for k in ['weighted_f1','macro_f1','top_label_ece','median_latency_ms_per_row']}
                metrics['p95_latency_ms_per_row']=a['p95_latency_ms_per_row']['max']
                metrics['serialized_size_mb']=a['serialized_size_mb']['max']
            records.append((row,metrics))
        for cap in [1.5,2.0,4.0]:
            constraints=DeploymentConstraints(0.95,5.0,cap)
            feasible=[(row,m) for row,m in records if assess_feasibility(m,constraints)['feasible']]
            if not feasible:
                continue
            row,metrics=max(feasible,key=lambda pair:pair[1]['macro_f1'])
            key=json.dumps(row['config'],sort_keys=True,separators=(',',':'))
            selected.setdefault(key,row['config'])
            associations.setdefault(key,[]).append({'method':method,'profile_max_size_mb':cap,
                'development_cv_macro_f1':metrics['macro_f1'],'development_cv_weighted_f1':metrics['weighted_f1']})
    constraints=DeploymentConstraints(0.0,1000.0,1000.0)
    result=evaluate_development_candidates(
        Path('data') / ('CICIDS2017_sample_km.csv' if dataset=='cicids2017' else 'IoTID20_sample_0.05.csv'),
        target_column='Label',family='lightgbm',candidate_configs=list(selected.values()),
        constraints=constraints,outer_splits=5,holdout_fold=0,inner_splits=4,seed=42,
        n_jobs=1,warmup_runs=2,benchmark_runs=10)
    result['selected_config_associations']=[{'config':config,'selected_for':associations[key]}
        for key,config in selected.items()]
    result['purpose']='Isolated, later CPU latency remeasurement of development-selected candidates; do not use as a confirmatory test.'
    result['latency_profile']={'warmup_runs':2,'benchmark_runs':10,'model_jobs':1,
        'competing_project_experiments':'none planned during this run'}
    out=Path('results')/f'{dataset}_selected_candidates_isolated_latency_seed42.json'
    out.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(dataset,'selected',len(selected),'saved',out,flush=True)
