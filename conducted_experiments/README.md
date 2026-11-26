# Conducted experiments

This index groups the completed research work. Start with a study, inspect its implementation, then open the linked graphs for measured findings. Results include useful improvements and failed hypotheses.

| Study | Work performed and finding | Evidence |
|---|---|---|
| [Baseline reproduction](01_baseline_reproduction.md) | Reproduced benchmark pipelines on CICIDS2017 and IoTID20; compared tree models, balancing, feature selection and tuning. These are benchmark-split reproduction results, not new cross-network performance. | [Graphs](../results/01_baseline_reproduction.md) |
| [AutoML and resource constraints](02_automl_and_resource_constraints.md) | Implemented Pareto archives, feasibility-first search, repeated seeds and matched optimizer comparisons. Results do not establish consistent MOPSO superiority; measured compute and feasibility accompany quality metrics. | [Graphs](../results/02_automl_and_resource_constraints.md) |
| [Cross-dataset transfer](03_cross_dataset_transfer.md) | Audited feature compatibility and tested source-to-target transfer. Per-capture false alarms and recall reveal failures that pooled scores can conceal. | [Graphs](../results/03_cross_dataset_transfer.md) |
| [Chronological adaptation](04_temporal_adaptation.md) | Compared frozen, periodic, drift-triggered and threshold-update policies on ordered traffic. Adaptation can recover detection while introducing capture-level false-alarm risk. | [Graphs](../results/04_temporal_adaptation.md) |
| [Delayed feedback and streaming safeguards](05_delayed_feedback_and_safeguards.md) | Evaluated label delays, approval, rollback, recovery and deadline batching. The work measures harmful updates, missed attacks, switching and delivery cost; it does not establish a universally safe update controller. | [Graphs](../results/05_delayed_feedback_and_safeguards.md) |
| [Budgeted verification](06_budgeted_verification.md) | Compared randomized disagreement review and decision-value allocation under label/reviewer budgets. No reliable general advantage was established; this direction was not promoted as a successful new algorithm. | [Graphs](../results/06_budgeted_verification.md) |
| [Generalization and network context](07_generalization_and_context.md) | Compared single- and multi-environment fitting, robust/group-aware controls and legitimate service/direction context. Later-period performance remains below the joint recall/FPR requirement. | [Graphs](../results/07_generalization_and_context.md) |
| [Causal behavioral detection](08_behavioral_detection.md) | Implemented bounded host histories and flow/behavioral/hybrid comparisons. Hybrid recall increases substantially on the full development replay, but false alarms exceed the target. Coverage and sensor-visibility controls are reported separately. | [Graphs](../results/08_behavioral_detection.md) |
| [Data integrity and observability](09_data_integrity_and_observability.md) | Verified timestamps, cache alignment, feature conventions and label consistency; screened temporal data sources. Empirical feature-collision bounds are scoped to the audited family and fixed records, not population guarantees. | [Graphs](../results/09_data_integrity_and_observability.md) |

## Running the code

- Current inference: `python examples/score_flows.py --help`.
- Current model evaluation: `python examples/evaluate_model.py --help` (training extras required).
- Chronological adaptation example: `python experiments/replay_adaptation.py --help`.
- Historical runner index and source checksums: [catalog.json](catalog.json).

Raw data, packet captures, per-flow caches and vendor repositories are intentionally separate from the portfolio. The source catalog documents the research work; it is not a claim that every historical runner is independently executable without its recorded inputs.
