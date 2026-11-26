# Implementation guide

The project separates reusable implementation, runnable entry points, historical experiments and measured results. The two example scripts are small interfaces to the underlying package.

## Suggested review order

1. [Runtime](src/autoshield/runtime.py): validated records, bounded batches and ordered inference.
2. [Causal observer](src/autoshield/observation.py): how historical state is maintained without future or same-time peer access.
3. [Feature transformation](src/autoshield/features.py): directional flow statistics and missing-value handling.
4. [Optimizer](src/autoshield/optimization.py): feasibility-first comparisons and an external Pareto archive.
5. [AutoML evaluation](src/autoshield/automl.py): grouped model evaluation and search integration.
6. [Tests](tests/): executable checks of the important contracts.

## Core modules and APIs

| Module | Main APIs | Engineering responsibility |
|---|---|---|
| [automl.py](src/autoshield/automl.py) | `evaluate_candidate_cv`, `evaluate_development_candidates`, `search_lightgbm_mopso` | Evaluates configurations on grouped development partitions, constructs classifiers and connects candidate evaluation to search. |
| [optimization.py](src/autoshield/optimization.py) | `Dimension`, `run_pareto_mopso` | Decodes bounded parameters, compares feasibility and objectives, maintains the nondominated archive and saves resumable state. |
| [preprocessing.py](src/autoshield/preprocessing.py) | `numeric_network_features`, `FoldAdaptiveNormalizer`, `HybridSMOTEADASYN`, `make_leakage_safe_pipeline` | Separates targets/metadata and keeps learned normalization and balancing inside fitting folds. |
| [evaluation.py](src/autoshield/evaluation.py) | `DeploymentConstraints`, `evaluate_fitted_model`, `assess_feasibility`, `pareto_front` | Measures quality, calibration, batch prediction cost and serialized size; evaluates declared search constraints. |
| [features.py](src/autoshield/features.py) | `transform` | Converts exporter fields into the physical representation used by the flow and hybrid models. |
| [observation.py](src/autoshield/observation.py) | `BehavioralObservation.observe`, `global_features`, `diagnostics` | Maintains bounded host state and computes strictly prior activity features. The global-rate features belong to the documented ablation. |
| [runtime.py](src/autoshield/runtime.py) | `CompletedFlow`, `BehavioralRuntime.score_batch` | Checks input validity, rejects decreasing arrival times and oversized batches, extracts features, scores a model and returns alarms. |
| [scoring.py](src/autoshield/scoring.py) | `scores`, `QuantileTailAnomaly` | Provides native scoring and a conventional benign-profile anomaly baseline. |

`src/autoshield/__init__.py` exposes the package's inference API. The bundled frozen models are checksum-verified by the CSV example before loading.

## Execution paths

### Completed-flow inference

```text
examples/score_flows.py
  ├─ read the frozen-model manifest and verify its checksum
  ├─ parse ordered CSV records into CompletedFlow objects
  └─ BehavioralRuntime.score_batch
       ├─ validate the complete batch before updating history
       ├─ update/query causal state for behavioral models
       ├─ construct physical or hybrid model inputs
       ├─ call native model scoring
       └─ return scores and thresholded alarms in input order
```

The CSV example writes predictions to a file. Labels are not inference inputs. Collector/network IO and live packet capture are outside this example's service boundary; completion-time replay is explicitly distinguished from actual receipt timestamps.

### Candidate evaluation and search

```text
examples/evaluate_model.py
  └─ evaluate_candidate_cv
       ├─ separate targets and metadata
       ├─ group repeated feature vectors
       ├─ fit preprocessing and classifier inside each fitting fold
       └─ measure quality, calibration and resource cost

search_lightgbm_mopso
  └─ run_pareto_mopso
       ├─ generate bounded configurations
       ├─ evaluate candidates on development partitions
       ├─ compare feasibility and measured objectives
       └─ update the Pareto archive and checkpoint
```

The simple CLI evaluates a fixed configuration; it does not silently launch the expensive search. Search-stage timing is batch-normalized and must not be confused with single-flow delivery latency. Temporal false-positive operating requirements are assessed separately from the optimizer's F1/latency/size feasibility profile.

### Chronological adaptation

[experiments/replay_adaptation.py](experiments/replay_adaptation.py) contains capture reading and ordering, chunk evaluation, drift/error statistics, periodic and triggered update comparisons, and recorded update costs. A chunk is scored before its labels become available to the update policy. Running the study requires its licensed data and manifest; see [the input guide](docs/data.md).

## Historical experiments

The [nine-study experiment index](conducted_experiments/README.md) links the broader research work. Its [source catalog](conducted_experiments/catalog.json) lists 320 preserved research source files, including the experimental runners, controls, data preparation, diagnostics and analysis helpers.

These files retain their original workspace conventions and dependencies. They are research implementations rather than additional public package APIs. Use the study descriptions to identify the data and protocol required for a historical reproduction.

## Tests and evidence

- [Feature contracts](tests/test_feature_contract.py): directional ratios, TCP flags, missing values and duration harmonization.
- [Observer contracts](tests/test_observation.py): strict time ordering, same-time exclusion, memory bounds, reference agreement and query invariance.
- [Runtime contracts](tests/test_runtime.py): frozen-model checksum validation, batch-partition invariance and invalid-input rejection before state mutation.
- [Optimization contracts](tests/test_optimization.py): feasibility-first dominance and bounded parameter decoding.

The tests exercise execution contracts; their small fixtures are not detection-accuracy benchmarks. Actual detector measurements, resource results and all saved experiment graphs are in the [results guide](results/README.md).
