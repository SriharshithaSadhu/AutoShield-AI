# Cross-dataset transfer: results and graphs

Audited feature compatibility and tested source-to-target transfer. Per-capture false alarms and recall reveal failures that pooled scores can conceal.

[Experiment implementations](../conducted_experiments/03_cross_dataset_transfer.md) · [All results](README.md)

## Transfer threshold-gate feasibility

![Transfer threshold-gate feasibility](graphs/03_cross_dataset_transfer/IOTID20_CSE_THRESHOLD_GATE_TRADEOFF.png)

**What we did:** Evaluated threshold policies against recall and false-positive requirements rather than selecting on F1 alone.

**What the graph establishes:** Tests whether transferred scores support the declared recall and false-positive budgets. Failed operating points remain visible; threshold control alone is insufficient.

**Recorded source:** `IOTID20_CSE_THRESHOLD_GATE_TRADEOFF.png`

## Transfer risk by capture

![Transfer risk by capture](graphs/03_cross_dataset_transfer/IOTID20_CSE_TRANSFER_CAPTURE_RISK_TRADEOFF.png)

**What we did:** Applied source-trained models to target captures and examined capture-specific detection and false-positive outcomes.

**What the graph establishes:** Separates capture-specific recall and false alarms rather than treating pooled performance as deployment safety.

**Recorded source:** `IOTID20_CSE_TRANSFER_CAPTURE_RISK_TRADEOFF.png`

