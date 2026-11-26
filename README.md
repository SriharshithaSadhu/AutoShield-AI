# AutoShield AI

**Resource-aware AutoML and causal network intrusion detection.**

AutoShield AI implements model evaluation, multi-objective optimization and streaming inference for network-flow intrusion detection. The engineering focus is the tradeoff between attack detection, false alarms, inference cost and model size.

**For measured performance, see [Results & graphs](results/README.md).**

![Detection and false-alarm tradeoff](results/graphs/overview_detection_tradeoff.png)

## What is implemented

- **Multi-objective AutoML:** seeded MOPSO with an external Pareto archive, feasibility-first selection and resumable checkpoints.
- **Fold-local preprocessing:** normalization and class balancing fitted inside training folds; grouped validation for repeated flow vectors.
- **Causal behavioral inference:** bounded host histories, flow-only/hybrid models and label-free batch scoring.
- **Chronological adaptation experiments:** frozen, periodic and drift-triggered updates with labels released after prediction.
- **Deployment evaluation:** detection metrics, probability calibration, latency and serialized model size.

## Quick start

Python 3.10 or 3.11:

```bash
git clone https://github.com/SriharshithaSadhu/AutoShield-AI.git
cd AutoShield-AI
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/score_flows.py --input /path/to/flows.csv --model flow
```

The scorer writes ordered predictions to `outputs/predictions.csv`. No labels are required. See the [input schema and dataset instructions](docs/data.md); raw data is not bundled.

For model evaluation and adaptation experiments:

```bash
python -m pip install -e ".[training]"
python examples/evaluate_model.py --data /path/to/training.csv --target Label
python experiments/replay_adaptation.py --help
```

## Conducted experiments

[Browse the experiment catalog](conducted_experiments/README.md) for baseline reproduction, optimizer comparisons, transfer, adaptation, delayed feedback, verification, generalization, behavioral detection and data audits. [The results guide](results/README.md) explains every saved graph and what it demonstrates.

## Code map

```text
src/autoshield/
  automl.py          Grouped candidate evaluation and search integration
  optimization.py    Feasibility-first MOPSO and Pareto archive
  preprocessing.py   Fold-local normalization and balancing
  evaluation.py      Quality, calibration and deployment measurements
  observation.py     Bounded, strictly causal host histories
  features.py        Common-exporter physical flow representation
  runtime.py         Validated, label-free batch inference
  scoring.py         Native model scoring helpers
examples/            Runnable inference and evaluation entry points
experiments/         Runnable chronological adaptation example
conducted_experiments/  Organized research studies, historical code and notebooks
tests/               Causality, feature, runtime and optimization checks
models/              Two frozen research models and checksum manifest
results/             All saved experiment graphs, explanations and selected measurements
docs/                Data contract and graph-generation utility
```

## Results at a glance

On a 9.16-million-flow chronological development replay, the flow-only baseline reached 36.51% attack recall at 1.31% FPR; the hybrid reached 90.27% recall at 8.68% FPR. The result exposes a detection/false-alarm tradeoff rather than a deployment-ready operating point. **See the [graphs, resource measurements and evaluation scope](results/README.md).**

This is a working research prototype. Cross-environment generalization remains a limitation; resource measurements were taken on a desktop CPU.

## Research foundation

Built on Yang & Shami's *Toward Autonomous and Efficient Cybersecurity: A Multi-Objective AutoML-based Intrusion Detection System* ([paper](https://doi.org/10.1109/TMLCN.2025.3631379), [original implementation](https://github.com/Western-OC2-Lab/Multi-Objective-Optimization-AutoML-based-Intrusion-Detection-System)). The original work and this project's engineering extensions are distinct. Dataset and upstream software rights remain with their respective owners.
