"""Two standard-library classifiers, so the experiment runs with no scikit-learn.

Neither is competitive. They are here because the *shape* of the finding — that
accuracy and macro-F1 rank models differently on an imbalanced problem, and that
a model can win on accuracy by abandoning a class — is a property of the metrics
and the data, not of the algorithm. It can be demonstrated exactly, offline, and
tested.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class MajorityClassifier:
    """Always predicts the most common training class.

    Accuracy on Covertype: about 0.49. Balanced accuracy: 1/7 = 0.143. The gap
    between those two numbers is the entire argument for not reporting the first
    one on its own.
    """

    name = "majority"

    def __init__(self) -> None:
        self.label: int | None = None

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> MajorityClassifier:
        if not y:
            raise ValueError("cannot fit on an empty training set")
        counts: dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        # Ties broken by the smaller label, so the model is deterministic.
        self.label = min(counts, key=lambda k: (-counts[k], k))
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> list[int]:
        if self.label is None:
            raise RuntimeError("model is not fitted")
        return [self.label] * len(x)


class NearestCentroid:
    """Assigns each row to the class whose training mean is closest.

    Sensitive to feature scale, which makes it useful for the scaling ablation:
    on unstandardised Covertype, `Horizontal_Distance_To_Roadways` (0-7,000)
    dominates `Slope` (0-66), and the classifier is effectively one-dimensional.
    An RBF SVM has exactly this problem and is usually compared against a random
    forest — which does not, because a tree split is invariant to monotone
    rescaling of a feature.
    """

    name = "nearest centroid"

    def __init__(self) -> None:
        self.centroids: dict[int, list[float]] = {}

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> NearestCentroid:
        if not x:
            raise ValueError("cannot fit on an empty training set")
        if len(x) != len(y):
            raise ValueError(f"x has {len(x)} rows and y has {len(y)} labels")

        sums: dict[int, list[float]] = {}
        counts: dict[int, int] = {}
        width = len(x[0])
        for row, label in zip(x, y, strict=True):
            if len(row) != width:
                raise ValueError("every row must have the same width")
            total = sums.setdefault(label, [0.0] * width)
            for j, value in enumerate(row):
                total[j] += value
            counts[label] = counts.get(label, 0) + 1

        self.centroids = {
            label: [value / counts[label] for value in total]
            for label, total in sums.items()
        }
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> list[int]:
        if not self.centroids:
            raise RuntimeError("model is not fitted")
        labels = sorted(self.centroids)
        predictions = []
        for row in x:
            best, best_distance = labels[0], math.inf
            for label in labels:
                centroid = self.centroids[label]
                distance = sum((a - b) ** 2 for a, b in zip(row, centroid, strict=True))
                if distance < best_distance:
                    best, best_distance = label, distance
            predictions.append(best)
        return predictions


def standardise(
    train: Sequence[Sequence[float]], test: Sequence[Sequence[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    """Scale by the training mean and standard deviation only."""
    if not train:
        raise ValueError("cannot standardise on an empty training set")
    width = len(train[0])
    means = [sum(row[j] for row in train) / len(train) for j in range(width)]
    spreads = []
    for j in range(width):
        variance = sum((row[j] - means[j]) ** 2 for row in train) / max(len(train) - 1, 1)
        spreads.append(math.sqrt(variance) or 1.0)

    def apply(rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(row[j] - means[j]) / spreads[j] for j in range(width)] for row in rows]

    return apply(train), apply(test)
