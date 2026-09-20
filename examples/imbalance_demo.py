"""Why an accuracy column cannot rank models on this dataset. Offline, no sklearn.

Three demonstrations, all exact:

1. A model that predicts one class for everything scores 48% accuracy and
   abandons six of the seven classes.
2. Accuracy and macro-F1 rank two models in *opposite* order.
3. Standardising the features is worth more than any model choice here — and
   the model that needs it is the SVM, which is routinely compared against a
   random forest that does not.

    python examples/imbalance_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covertype import synthetic
from covertype.baseline import MajorityClassifier, NearestCentroid
from covertype.data import CLASS_NAMES
from covertype.experiment import evaluate, ranking_disagreement, table
from covertype.metrics import majority_accuracy
from covertype.sampling import split, stratified_sample, uniform_sample

TRAIN, TEST = 4000, 4000


def main() -> None:
    dataset = synthetic.generate(30_000, seed=0)
    counts = dataset.class_counts()
    print(f"{len(dataset):,} rows, imbalance {dataset.imbalance_ratio():.0f}:1")
    for label, name in CLASS_NAMES.items():
        print(
            f"  {label} {name:<20}{counts[label]:>8,}{counts[label] / len(dataset):>8.2%}"
        )
    print(f"\nmajority-class accuracy: {majority_accuracy(list(dataset.y)):.4f}")

    sample = split(dataset, train_size=TRAIN, test_size=TEST, seed=0)

    majority = evaluate(lambda: MajorityClassifier(), sample, name="majority")
    centroid = evaluate(lambda: NearestCentroid(), sample, name="centroid")
    scaled = evaluate(
        lambda: NearestCentroid(), sample, scale=True, name="centroid+scaled"
    )

    print(f"\n{'=' * 78}\n1. A model that predicts nothing\n")
    print(majority.report.summary())
    print(
        f"\n   {len(majority.report.abandoned_classes)} of 7 classes never predicted, "
        f"for {majority.accuracy:.1%} accuracy."
    )

    print(f"\n{'=' * 78}\n2. Accuracy and macro-F1 disagree\n")
    print(table([majority, centroid]))
    print()
    print(ranking_disagreement([majority, centroid]))

    print(f"\n{'=' * 78}\n3. Scaling is worth more than the model choice\n")
    print(table([centroid, scaled]))
    print(
        f"\n   standardising: {scaled.accuracy - centroid.accuracy:+.4f} accuracy, "
        f"{scaled.macro_f1 - centroid.macro_f1:+.4f} macro-F1 — same model, same data."
    )

    print(
        f"\n{'=' * 78}\n4. Stratified sampling keeps the rare class in the experiment\n"
    )
    rarest = synthetic.rarest_label()
    for name, draw in (("uniform", uniform_sample), ("stratified", stratified_sample)):
        found = [draw(dataset, 500, seed=s).y.count(rarest) for s in range(8)]
        print(f"   {name:<12}cover type {rarest} in a 500-row draw: {found}")
    print(
        "\n   A draw containing zero examples of a class cannot train or test it, and\n"
        "   two models given different numbers of them are not being compared."
    )


if __name__ == "__main__":
    main()
