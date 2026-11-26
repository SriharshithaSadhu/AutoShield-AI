from pathlib import Path
from research_runner import search_lightgbm_mopso
from deployment_evaluation import DeploymentConstraints

constraints = DeploymentConstraints(min_weighted_f1=0.95, max_p95_latency_ms_per_row=5.0, max_serialized_size_mb=1.5)
for dataset, target, name in [
    (Path('data/CICIDS2017_sample_km.csv'), 'Label', 'cicids2017'),
    (Path('data/IoTID20_sample_0.05.csv'), 'Label', 'iotid20'),
]:
    result = search_lightgbm_mopso(
        dataset,
        target_column=target,
        constraints=constraints,
        output_path=Path('results') / f'{name}_lightgbm_mopso_dev_seed42.json',
        seed=42,
        particles=6,
        iterations=4,
        n_jobs=1,
    )
    print(name, 'evaluated', len(result['candidate_records']), 'archive', len(result['pareto_archive']), 'feasible', sum(x.get('feasibility', {}).get('feasible', False) for x in result['pareto_archive']), flush=True)

