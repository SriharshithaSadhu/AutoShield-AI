"""Evaluate one classifier configuration with grouped development folds."""

import argparse
from pathlib import Path
from autoshield.automl import evaluate_candidate_cv, save_result
from autoshield.evaluation import DeploymentConstraints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", default="Label")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()
    result = evaluate_candidate_cv(
        args.data,
        target_column=args.target,
        family="lightgbm",
        config={"n_estimators": 100, "num_leaves": 15, "verbosity": -1},
        constraints=DeploymentConstraints(),
        n_splits=3,
        n_jobs=1,
    )
    save_result(result, args.output)
    print(f"Saved grouped development evaluation to {args.output}")


if __name__ == "__main__":
    main()
