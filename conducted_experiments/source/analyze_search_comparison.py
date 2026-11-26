"""Compare saved MOPSO/random candidate banks over explicit CPU resource profiles."""
import json
from pathlib import Path
from deployment_evaluation import DeploymentConstraints, assess_feasibility, pareto_front

methods = ['mopso', 'random']
profiles = [(0.95, 1.5), (0.95, 2.0), (0.95, 4.0), (0.95, 8.0)]
output = {}
for dataset in ['cicids2017', 'iotid20']:
    banks = {}
    for method in methods:
        suffix = f'{dataset}_lightgbm_mopso_dev_seed42.json' if method == 'mopso' else f'{dataset}_lightgbm_random_dev_seed42042.json'
        path = Path('results') / suffix
        if not path.exists():
            raise FileNotFoundError(path)
        saved = json.loads(path.read_text(encoding='utf-8'))
        candidates = saved['candidate_records'] if method == 'mopso' else saved['candidates']
        records = []
        for candidate in candidates:
            if 'metrics' in candidate:
                metrics = dict(candidate['metrics'])
            else:
                aggregate = candidate['aggregate_metrics']
                metrics = {
                    key: value['mean'] for key, value in aggregate.items()
                    if key in {'weighted_f1', 'macro_f1', 'top_label_ece', 'median_latency_ms_per_row'}
                }
                metrics['p95_latency_ms_per_row'] = aggregate['p95_latency_ms_per_row']['max']
                metrics['serialized_size_mb'] = aggregate['serialized_size_mb']['max']
                metrics['mean_fit_seconds'] = candidate['mean_fit_seconds']
            records.append({'candidate_index': candidate.get('candidate_index'), 'config': candidate['config'], 'metrics': metrics})
        banks[method] = records
    summaries = {}
    for method, records in banks.items():
        rows = []
        for min_f1, max_size in profiles:
            constraints = DeploymentConstraints(min_weighted_f1=min_f1, max_p95_latency_ms_per_row=5.0, max_serialized_size_mb=max_size)
            for record in records:
                record['feasibility'] = assess_feasibility(record['metrics'], constraints)
            feasible = [row for row in records if row['feasibility']['feasible']]
            front = pareto_front(records)
            rows.append({
                'weighted_f1_floor': min_f1,
                'max_size_mb': max_size,
                'max_p95_latency_ms_per_row': 5.0,
                'feasible_candidate_count': len(feasible),
                'pareto_front_count': len(front),
                'best_feasible_weighted_f1': max((r['metrics']['weighted_f1'] for r in feasible), default=None),
                'best_feasible_macro_f1': max((r['metrics']['macro_f1'] for r in feasible), default=None),
                'least_violating_front_count_if_none_feasible': len(front) if not feasible else None,
            })
        latency_rows = []
        for max_latency in (0.025, 0.05, 0.1, 0.2, 1.0):
            constraints = DeploymentConstraints(
                min_weighted_f1=0.95,
                max_p95_latency_ms_per_row=max_latency,
                max_serialized_size_mb=4.0,
            )
            for record in records:
                record['feasibility'] = assess_feasibility(record['metrics'], constraints)
            feasible = [row for row in records if row['feasibility']['feasible']]
            front = pareto_front(records)
            latency_rows.append({
                'weighted_f1_floor': 0.95,
                'max_size_mb': 4.0,
                'max_p95_latency_ms_per_row': max_latency,
                'feasible_candidate_count': len(feasible),
                'pareto_front_count': len(front),
                'best_feasible_weighted_f1': max((r['metrics']['weighted_f1'] for r in feasible), default=None),
                'best_feasible_macro_f1': max((r['metrics']['macro_f1'] for r in feasible), default=None),
            })
        summaries[method] = {
            'candidate_count': len(records),
            'budget_sweep': rows,
            'latency_sweep_at_4mb': latency_rows,
        }
    output[dataset] = summaries
out = Path('results') / 'mopso_vs_random_resource_profiles_seed42.json'
out.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(out)
print(json.dumps(output, indent=2))
