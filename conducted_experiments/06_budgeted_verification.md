# Budgeted verification

Compared randomized disagreement review and decision-value allocation under label/reviewer budgets. No reliable general advantage was established; this direction was not promoted as a successful new algorithm.

[See results and graph explanations](../results/06_budgeted_verification.md)

## Implementations

<details>
<summary>Browse the historical scripts for this study</summary>

- [early_verification.py](source/early_verification.py)
- [summarize_early_verification.py](source/summarize_early_verification.py)

- [decision_value_review.py](source/decision_value_review.py)

- [summarize_decision_value_review.py](source/summarize_decision_value_review.py)

</details>

## Reproduction scope

The packaged core under `src/autoshield/` is the clean implementation used by the documented examples. Historical scripts are preserved under `source/` with their original code and workspace-relative paths. To rerun a historical study, treat that directory as its research workspace, provision the licensed datasets/configuration it expects, and follow its entry point. Saved portfolio graphs do not require retraining.
