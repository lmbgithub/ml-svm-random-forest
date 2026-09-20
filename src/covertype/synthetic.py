"""A Covertype-shaped dataset: 7 classes, ~100:1 imbalance, mixed feature scales.

The real download is 75 MB and an RBF SVM on it is a research project. The
findings this repository is about — that accuracy and macro-F1 rank models
differently, that a class can be abandoned almost for free, that an unscaled
distance-based model is effectively one-dimensional — are properties of that
*shape*, and can be produced exactly and offline.
"""

from __future__ import annotations

import random

from covertype.data import Dataset

#: Class shares, close to Covertype's own: two classes are 85% of the rows and
#: the rarest is under 0.5%.
CLASS_SHARES: dict[int, float] = {
    1: 0.365,
    2: 0.487,
    3: 0.062,
    4: 0.0047,
    5: 0.0163,
    6: 0.0299,
    7: 0.0353,
}

COLUMNS = (
    "Elevation",  # 1,800-3,900: dominates any unscaled distance
    "Slope",  # 0-66
    "Horizontal_Distance_To_Roadways",  # 0-7,000
    "Hillshade_Noon",  # 0-255
)


def generate(rows: int = 20_000, *, seed: int = 0) -> Dataset:
    rng = random.Random(seed)
    labels: list[int] = []
    features: list[tuple[float, ...]] = []

    population = list(CLASS_SHARES)
    weights = [CLASS_SHARES[label] for label in population]

    for _ in range(rows):
        label = rng.choices(population, weights)[0]

        # Elevation carries most of the signal, as it does on the real dataset.
        elevation = 2000 + 250 * label + rng.gauss(0, 180)
        # Slope carries a weaker signal on a completely different scale — the
        # feature an unstandardised distance model throws away.
        slope = 10 + 4 * ((label * 3) % 7) + rng.gauss(0, 4)
        roadways = 1500 + rng.gauss(0, 1400)
        hillshade = 220 - 2 * slope + rng.gauss(0, 12)

        labels.append(label)
        features.append((elevation, slope, roadways, hillshade))

    return Dataset(COLUMNS, tuple(features), tuple(labels))


def expected_share(label: int) -> float:
    return CLASS_SHARES[label]


def rarest_label() -> int:
    return min(CLASS_SHARES, key=CLASS_SHARES.get)
