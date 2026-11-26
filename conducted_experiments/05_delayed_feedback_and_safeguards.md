# Delayed feedback and streaming safeguards

Evaluated label delays, approval, rollback, recovery and deadline batching. The work measures harmful updates, missed attacks, switching and delivery cost; it does not establish a universally safe update controller.

[See results and graph explanations](../results/05_delayed_feedback_and_safeguards.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>

- [audit_ctu13_temporal_gate.py](source/audit_ctu13_temporal_gate.py)
- [audit_original_core6_rollback.py](source/audit_original_core6_rollback.py)
- [benchmark_deadline_router.py](source/benchmark_deadline_router.py)
- [deadline_alert_router.py](source/deadline_alert_router.py)
- [evaluate_rolling_alert_safeguards.py](source/evaluate_rolling_alert_safeguards.py)
- [run_iotid20_cse_prefix_vs_fullperiod_threshold_gate.py](source/run_iotid20_cse_prefix_vs_fullperiod_threshold_gate.py)
- [run_iotid20_cse_retraining_gate_pilot.py](source/run_iotid20_cse_retraining_gate_pilot.py)
- [summarize_deadline_recovery.py](source/summarize_deadline_recovery.py)
- [verify_integrated_shadow.py](source/verify_integrated_shadow.py)
- [audit_apex_ids2026_gate_next_day_validity.py](source/research_refs/audit_apex_ids2026_gate_next_day_validity.py)
- [audit_apex_ids2026_temporal_gate.py](source/research_refs/audit_apex_ids2026_temporal_gate.py)
- [audit_nf_ton_promotion_exposure.py](source/research_refs/audit_nf_ton_promotion_exposure.py)
- [audit_nf_ton_shadow_feasibility.py](source/research_refs/audit_nf_ton_shadow_feasibility.py)
- [plot_smart_factory_a_to_s_vbc_comparison.py](source/research_refs/plot_smart_factory_a_to_s_vbc_comparison.py)
- [recompute_promotion_gate_horizons.py](source/research_refs/recompute_promotion_gate_horizons.py)
- [run_apex_ids2026_update_gate_development.py](source/research_refs/run_apex_ids2026_update_gate_development.py)
- [run_apex_ids2026_update_gate_hash_sensitivity.py](source/research_refs/run_apex_ids2026_update_gate_hash_sensitivity.py)
- [run_nf_ton_iot_v3_promotion_gate_pilot.py](source/research_refs/run_nf_ton_iot_v3_promotion_gate_pilot.py)
- [run_smart_factory_vbc_comparison.py](source/research_refs/run_smart_factory_vbc_comparison.py)

</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.
