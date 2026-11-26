# Baseline reproduction

Reproduced benchmark pipelines on CICIDS2017 and IoTID20; compared tree models, balancing, feature selection and tuning. These are benchmark-split reproduction results, not new cross-network performance.

[See results and graph explanations](../results/01_baseline_reproduction.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>


</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.

[CICIDS2017 reproduction notebook](baseline_notebooks/MOO_AutoML_Dataset_1.ipynb) Â· [IoTID20 reproduction notebook](baseline_notebooks/MOO_AutoML_Dataset_2.ipynb)

These are base-paper reproduction artifacts. Original source attribution appears in the root README; their benchmark protocol is distinct from later chronological evaluations.
