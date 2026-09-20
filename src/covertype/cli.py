"""Compare classifiers on Covertype under an explicit, shared training budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from covertype import synthetic
from covertype.baseline import MajorityClassifier, NearestCentroid
from covertype.data import CLASS_NAMES, DataError, load_csv, load_uci
from covertype.experiment import (
    cost_of_a_point,
    learning_curve,
    ranking_disagreement,
    table,
)
from covertype.metrics import majority_accuracy
from covertype.models import SPECS, build, feasible

STDLIB_MODELS = {
    "majority": (lambda: MajorityClassifier(), False),
    "centroid": (lambda: NearestCentroid(), False),
    "centroid-scaled": (lambda: NearestCentroid(), True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="covertype",
        description="SVM against random forest on Covertype, at equal training budgets.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="path to covtype.csv")
    source.add_argument(
        "--uci", action="store_true", help="download via ucimlrepo (id 31)"
    )
    parser.add_argument("--rows", type=int, default=20_000, help="synthetic dataset size")
    parser.add_argument(
        "--models",
        default="majority,centroid,centroid-scaled",
        help=f"comma-separated: {', '.join([*STDLIB_MODELS, *SPECS])}",
    )
    parser.add_argument(
        "--train-sizes",
        default="1000,4000",
        help=(
            "comma-separated training-set sizes; the same budget is given to every model"
        ),
    )
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.csv:
            dataset, source = load_csv(args.csv), str(args.csv)
        elif args.uci:
            dataset, source = load_uci(), "UCI id 31"
        else:
            dataset = synthetic.generate(args.rows, seed=args.seed)
            source = f"synthetic ({args.rows:,} rows, seed {args.seed})"
    except FileNotFoundError:
        print(f"file not found: {args.csv}", file=sys.stderr)
        return 2
    except DataError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    counts = dataset.class_counts()
    print(f"source: {source}   rows: {len(dataset):,}   features: {dataset.n_features}")
    print(
        f"imbalance (largest/smallest present class): {dataset.imbalance_ratio():.0f}:1"
    )
    for label, name in CLASS_NAMES.items():
        if counts.get(label):
            print(
                f"  {label} {name:<20}{counts[label]:>9,}  "
                f"{counts[label] / len(dataset):6.2%}"
            )
    print(f"\nmajority-class accuracy: {majority_accuracy(list(dataset.y)):.4f}\n")

    try:
        sizes = [int(s) for s in args.train_sizes.split(",") if s.strip()]
    except ValueError:
        print(
            f"--train-sizes must be integers; got {args.train_sizes!r}", file=sys.stderr
        )
        return 2

    runs = []
    for name in (m.strip() for m in args.models.split(",") if m.strip()):
        if name in STDLIB_MODELS:
            factory, scale = STDLIB_MODELS[name]
        elif name in SPECS:
            for size in sizes:
                ok, why = feasible(name, size)
                if not ok:
                    print(f"refusing {name} at {size:,} rows: {why}", file=sys.stderr)
                    return 2
            factory, scale = (lambda n=name: build(n, seed=args.seed)), False
        else:
            print(f"unknown model {name!r}", file=sys.stderr)
            return 2

        try:
            runs += learning_curve(
                dataset,
                factory,
                sizes,
                test_size=args.test_size,
                scale=scale,
                seed=args.seed,
                name=name,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    print(table(runs))
    print()
    print(ranking_disagreement(runs))

    for name in {run.model for run in runs}:
        curve = [run for run in runs if run.model == name]
        if len(curve) > 1:
            print(f"\n{name}: {cost_of_a_point(curve)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
