"""Equal-evaluation-budget random-search comparator for the MOPSO runs."""
import json
from pathlib import Path
import numpy as np
from research_runner import evaluate_development_candidates, file_sha256
from deployment_evaluation import DeploymentConstraints
from deployment_evaluation import pareto_front

specs = [
    ('cicids2017', Path('data/CICIDS2017_sample_km.csv')),
    ('iotid20', Path('data/IoTID20_sample_0.05.csv')),
]
constraints = DeploymentConstraints(min_weighted_f1=0.95, max_p95_latency_ms_per_row=5.0, max_serialized_size_mb=1.5)
rng = np.random.default_rng(42042)
for name, data_path in specs:
    mopso_path = Path('results') / f'{name}_lightgbm_mopso_dev_seed42.json'
    if not mopso_path.exists():
        print('waiting for', mopso_path, flush=True)
        continue
    mopso = json.loads(mopso_path.read_text(encoding='utf-8'))
    budget = len(mopso['candidate_records'])
    configs = []
    seen = set()
    while len(configs) < budget:
        config = {
            'n_estimators': int(rng.integers(50, 201)),
            'max_depth': int(rng.integers(5, 101)),
            'learning_rate': float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
            'num_leaves': int(rng.integers(10, 51)),
            'min_child_samples': int(rng.integers(10, 51)),
            'class_weight': 'balanced',
            'verbosity': -1,
        }
        key = json.dumps(config, sort_keys=True, separators=(',', ':'))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    result = evaluate_development_candidates(
        data_path,
        target_column='Label',
        family='lightgbm',
        candidate_configs=configs,
        constraints=constraints,
        outer_splits=5,
        holdout_fold=0,
        inner_splits=4,
        seed=42,
        n_jobs=1,
        warmup_runs=1,
        benchmark_runs=3,
    )
    result['algorithm'] = 'uniform/log-uniform random search; equal candidate-evaluation budget to MOPSO'
    result['seed'] = 42042
    result['budget_matched_to_candidate_count'] = budget
    result['candidate_generation'] = 'n_estimators 50..200, max_depth 5..100, learning_rate log-uniform 0.01..0.3, num_leaves 10..50, min_child_samples 10..50'
    out = Path('results') / f'{name}_lightgbm_random_dev_seed42042.json'
    out.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    print(name, 'random candidates', len(result['candidates']), 'Pareto', len(result['pareto_candidate_indices']), 'saved', out, flush=True)
