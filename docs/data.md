# Data and input contract

Raw datasets are not included. Obtain them from their publishers and follow their research-use terms.

- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html): baseline IDS benchmark.
- IoTID20: baseline reproduction dataset; see the [base-paper repository](https://github.com/Western-OC2-Lab/Multi-Objective-Optimization-AutoML-based-Intrusion-Detection-System).
- [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html): chronological adaptation experiments.
- [NF-ToN-IoT-v3](https://doi.org/10.48610/44d7c5e): common-exporter chronological development replay.

## Inference CSV

`examples/score_flows.py` expects ordered completed-flow records with:

```text
FLOW_END_MILLISECONDS, IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT,
PROTOCOL, FLOW_DURATION_MILLISECONDS, IN_PKTS, OUT_PKTS,
IN_BYTES, OUT_BYTES, RETRANSMITTED_IN_PKTS, RETRANSMITTED_OUT_PKTS,
SHORTEST_FLOW_PKT, LONGEST_FLOW_PKT, TCP_FLAGS
```

IPv4 values group histories; they are not classifier predictors. The runtime accepts up to 1,000 records per batch and rejects decreasing arrival timestamps. Records with equal timestamps cannot contribute to each other's historical features. Histories retain at most 4,096 hosts and 64 contacts per host.

CSV replay uses flow completion as an explicitly idealized receipt-time proxy. For actual collector timestamps supply `--arrival-column YOUR_RECEIPT_COLUMN`. Final flow summaries are not available at the first packet. The example is an offline scorer, not a live packet collector or a hard one-second delivery guarantee.

The CSV needs no labels for inference. Output preserves record order and contains score and alarm status. A model alarm is not a verified security incident.

## Training and replay

For `examples/evaluate_model.py`, provide a numeric traffic CSV and target column. Metadata/labels are separated by preprocessing. Inspect the feature contract before using another exporter: matching column names alone does not establish matching definitions or units. Keep any final test set outside model selection.

The adaptation example expects licensed CSE files under `data/cse_cic_ids2018/raw/` and the corresponding `data/cse_cic_ids2018/manifest.json`. It compares frozen, periodic and drift-triggered configurations; current-chunk labels are released after scoring. It is a study runner, not a claim of validated safe adaptation.
