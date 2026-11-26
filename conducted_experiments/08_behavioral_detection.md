# Causal behavioral detection

Implemented bounded host histories and flow/behavioral/hybrid comparisons. Hybrid recall increases substantially on the full development replay, but false alarms exceed the target. Coverage and sensor-visibility controls are reported separately.

[See results and graph explanations](../results/08_behavioral_detection.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>

- [analyze_behavioral_milestone.py](source/analyze_behavioral_milestone.py)
- [audit_behavioral_benign_service_coverage.py](source/audit_behavioral_benign_service_coverage.py)
- [audit_behavioral_false_alarm_context.py](source/audit_behavioral_false_alarm_context.py)
- [audit_behavioral_family_operating_points.py](source/audit_behavioral_family_operating_points.py)
- [audit_behavioral_training_separation.py](source/audit_behavioral_training_separation.py)
- [behavioral_data.py](source/behavioral_data.py)
- [behavioral_models.py](source/behavioral_models.py)
- [behavioral_observation.py](source/behavioral_observation.py)
- [behavioral_runtime.py](source/behavioral_runtime.py)
- [benchmark_behavioral_deployment.py](source/benchmark_behavioral_deployment.py)
- [build_behavioral_device_bundle.py](source/build_behavioral_device_bundle.py)
- [evaluate_behavioral_sensor_sensitivity.py](source/evaluate_behavioral_sensor_sensitivity.py)
- [freeze_behavioral_protocol.py](source/freeze_behavioral_protocol.py)
- [plot_behavioral_milestone.py](source/plot_behavioral_milestone.py)
- [prepare_behavioral_caches.py](source/prepare_behavioral_caches.py)
- [run_behavioral_benign_group_control.py](source/run_behavioral_benign_group_control.py)
- [run_behavioral_comparison.py](source/run_behavioral_comparison.py)
- [run_behavioral_family_coverage_control.py](source/run_behavioral_family_coverage_control.py)
- [run_behavioral_full_replay.py](source/run_behavioral_full_replay.py)
- [run_behavioral_isolation_control.py](source/run_behavioral_isolation_control.py)
- [run_behavioral_recent_benign_control.py](source/run_behavioral_recent_benign_control.py)
- [run_behavioral_source_transfer.py](source/run_behavioral_source_transfer.py)
- [test_behavioral_observation.py](source/test_behavioral_observation.py)
- [test_behavioral_runtime.py](source/test_behavioral_runtime.py)
- [validate_behavioral_stream_memory.py](source/validate_behavioral_stream_memory.py)
- [verify_behavioral_milestone.py](source/verify_behavioral_milestone.py)

</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.
