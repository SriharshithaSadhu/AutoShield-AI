# Chronological adaptation

Compared frozen, periodic, drift-triggered and threshold-update policies on ordered traffic. Adaptation can recover detection while introducing capture-level false-alarm risk.

[See results and graph explanations](../results/04_temporal_adaptation.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>

- [analyze_cse_stream_order_sensitivity.py](source/analyze_cse_stream_order_sensitivity.py)
- [analyze_psi_threshold_sensitivity.py](source/analyze_psi_threshold_sensitivity.py)
- [analyze_rccda_cse_policy.py](source/analyze_rccda_cse_policy.py)
- [audit_corrected_cse_capture.py](source/audit_corrected_cse_capture.py)
- [audit_corrected_cse_remote.py](source/audit_corrected_cse_remote.py)
- [audit_cse_cic_ids2018.py](source/audit_cse_cic_ids2018.py)
- [audit_ctu_raspberrypi_temporal.py](source/audit_ctu_raspberrypi_temporal.py)
- [audit_iot_nid_arp_event_temporal_alignment.py](source/audit_iot_nid_arp_event_temporal_alignment.py)
- [audit_smart_factory_flows.py](source/audit_smart_factory_flows.py)
- [audit_wustl_temporal_candidate.py](source/audit_wustl_temporal_candidate.py)
- [download_cicids2017_temporal.py](source/download_cicids2017_temporal.py)
- [download_corrected_cse_member.py](source/download_corrected_cse_member.py)
- [download_cse_cic_ids2018.py](source/download_cse_cic_ids2018.py)
- [download_cse_v3_ranges.py](source/download_cse_v3_ranges.py)
- [download_smart_factory_flow_data.py](source/download_smart_factory_flow_data.py)
- [drift_adaptation_experiment.py](source/drift_adaptation_experiment.py)
- [migrate_cse_replay_cache_timestamps_v2.py](source/migrate_cse_replay_cache_timestamps_v2.py)
- [plot_cse_daily_results.py](source/plot_cse_daily_results.py)
- [plot_threshold_recalibration_tradeoff.py](source/plot_threshold_recalibration_tradeoff.py)
- [plot_wustl_temporal_audit.py](source/plot_wustl_temporal_audit.py)
- [prepare_cse_auxiliary_source.py](source/prepare_cse_auxiliary_source.py)
- [remeasure_cse_equal_budget_refit_outlier.py](source/remeasure_cse_equal_budget_refit_outlier.py)
- [repair_cse_order_result_timestamp_metadata.py](source/repair_cse_order_result_timestamp_metadata.py)
- [run_corrected_replay_adaptation.py](source/run_corrected_replay_adaptation.py)
- [run_cse_equal_update_budget_adaptation.py](source/run_cse_equal_update_budget_adaptation.py)
- [run_cse_stream_order_sensitivity.py](source/run_cse_stream_order_sensitivity.py)
- [run_rccda_inspired_cse.py](source/run_rccda_inspired_cse.py)
- [plot_apex_ids2026_temporal_profile.py](source/research_refs/plot_apex_ids2026_temporal_profile.py)
- [plot_smart_factory_a_to_s_replay.py](source/research_refs/plot_smart_factory_a_to_s_replay.py)
- [run_nf_ton_threshold_only_recalibration.py](source/research_refs/run_nf_ton_threshold_only_recalibration.py)
- [run_smart_factory_temporal_adaptation.py](source/research_refs/run_smart_factory_temporal_adaptation.py)

</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.
