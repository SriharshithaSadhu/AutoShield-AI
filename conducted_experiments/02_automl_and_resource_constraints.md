# AutoML and resource constraints

Implemented Pareto archives, feasibility-first search, repeated seeds and matched optimizer comparisons. Results do not establish consistent MOPSO superiority; measured compute and feasibility accompany quality metrics.

[See results and graph explanations](../results/02_automl_and_resource_constraints.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>

- [analyze_constraint_selection_effect.py](source/analyze_constraint_selection_effect.py)
- [analyze_mopso_constraint_ablation.py](source/analyze_mopso_constraint_ablation.py)
- [analyze_search_comparison.py](source/analyze_search_comparison.py)
- [apply_strict_bank_constraints.py](source/apply_strict_bank_constraints.py)
- [audit_matched_selection.py](source/audit_matched_selection.py)
- [constrained_candidate_selection.py](source/constrained_candidate_selection.py)
- [pareto_mopso.py](source/pareto_mopso.py)
- [research_runner.py](source/research_runner.py)
- [run_arw_selection_baseline.py](source/run_arw_selection_baseline.py)
- [run_automl_archive_calibrated_reserve.py](source/run_automl_archive_calibrated_reserve.py)
- [run_automl_archive_reserve_pilot.py](source/run_automl_archive_reserve_pilot.py)
- [run_matched_selection.py](source/run_matched_selection.py)
- [run_mopso_constraint_ablation.py](source/run_mopso_constraint_ablation.py)
- [run_mopso_sweeps.py](source/run_mopso_sweeps.py)
- [run_random_search_baselines.py](source/run_random_search_baselines.py)
- [run_selection_stability.py](source/run_selection_stability.py)
- [summarize_selection_stability.py](source/summarize_selection_stability.py)
- [test_constrained_candidate_selection.py](source/test_constrained_candidate_selection.py)

</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.
