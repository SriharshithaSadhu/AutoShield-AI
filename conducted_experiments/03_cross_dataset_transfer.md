# Cross-dataset transfer

Audited feature compatibility and tested source-to-target transfer. Per-capture false alarms and recall reveal failures that pooled scores can conceal.

[See results and graph explanations](../results/03_cross_dataset_transfer.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>

- [analyze_transfer_by_capture.py](source/analyze_transfer_by_capture.py)
- [audit_iotid20_cse_feature_algebra.py](source/audit_iotid20_cse_feature_algebra.py)
- [audit_iotid20_cse_feature_semantics.py](source/audit_iotid20_cse_feature_semantics.py)
- [audit_iotid20_cse_feature_shift.py](source/audit_iotid20_cse_feature_shift.py)
- [audit_iotid20_label_mapping.py](source/audit_iotid20_label_mapping.py)
- [audit_iotid20_split_integrity.py](source/audit_iotid20_split_integrity.py)
- [run_full_iotid20_source_transfer_audit.py](source/run_full_iotid20_source_transfer_audit.py)
- [run_iotid20_binary_end_to_end.py](source/run_iotid20_binary_end_to_end.py)
- [run_iotid20_cse_transfer_replay.py](source/run_iotid20_cse_transfer_replay.py)
- [run_iotid20_exact_dedup_transfer_audit.py](source/run_iotid20_exact_dedup_transfer_audit.py)
- [run_iotid20_time_block_split_sensitivity.py](source/run_iotid20_time_block_split_sensitivity.py)
- [bootstrap_iotid20_cse_capture_effects.py](source/research_refs/bootstrap_iotid20_cse_capture_effects.py)
- [plot_iotid20_cse_capture_risk_tradeoff.py](source/research_refs/plot_iotid20_cse_capture_risk_tradeoff.py)

</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.
