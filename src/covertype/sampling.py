"""Subsampling, which is the comparison — not a preliminary to it.

An RBF SVM on 581,012 rows is not slow, it is impossible: kernel training is
between quadratic and cubic in the number of rows, and the kernel matrix alone
would be about 2.5 TB. Every published SVM-vs-random-forest comparison on this
dataset therefore trains the SVM on a subsample.

The mistake is to subsample for the SVM and train the forest on everything, then
put both accuracies in one table. That table compares two different experiments.
This module makes the sample size an explicit, shared parameter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from covertype.data import Dataset


@dataclass(frozen=True, slots=True)
class Sample:
    train: Dataset
    test: Dataset
    size: int
    stratified: bool

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.test)


def stratified_sample(dataset: Dataset, size: int, *, seed: int = 0) -> Dataset:
    """Draw `size` rows keeping each class's share of the data.

    Stratifying is not about fairness here, it is about the experiment being
    runnable: cover type 4 is 0.47% of the rows, so a uniform sample of 2,000
    contains about nine of them and can easily contain none. A comparison where
    one model saw four examples of a class and the other saw fourteen is not
    measuring the models.
    """
    if size <= 0:
        raise ValueError(f"sample size must be positive; got {size}")
    if size > len(dataset):
        raise ValueError(f"cannot draw {size} rows from {len(dataset)}")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(dataset.y):
        by_class.setdefault(label, []).append(index)

    chosen: list[int] = []
    for _label, indices in sorted(by_class.items()):
        share = len(indices) / len(dataset)
        # At least one row per present class: rounding a 0.47% class down to
        # zero silently removes it from the experiment.
        take = max(1, round(size * share))
        take = min(take, len(indices))
        chosen.extend(rng.sample(indices, take))

    # Rounding per class leaves the total a row or two short of `size`. Top up
    # from the classes that still have rows, so a learning curve's x-axis says
    # what it means: 4,000 has to be 4,000 in every run, or the curve is
    # comparing slightly different budgets.
    if len(chosen) < size:
        taken = set(chosen)
        remaining = [i for i in range(len(dataset)) if i not in taken]
        chosen.extend(rng.sample(remaining, min(size - len(chosen), len(remaining))))

    rng.shuffle(chosen)
    return dataset.subset(chosen[:size])


def uniform_sample(dataset: Dataset, size: int, *, seed: int = 0) -> Dataset:
    """A plain random sample. Here to be compared against, not used."""
    if size <= 0:
        raise ValueError(f"sample size must be positive; got {size}")
    if size > len(dataset):
        raise ValueError(f"cannot draw {size} rows from {len(dataset)}")
    rng = random.Random(seed)
    return dataset.subset(rng.sample(range(len(dataset)), size))


def split(
    dataset: Dataset,
    *,
    train_size: int,
    test_size: int,
    seed: int = 0,
    stratified: bool = True,
) -> Sample:
    """Draw disjoint train and test sets of fixed size.

    Test size is fixed independently of training size, so a learning curve
    varies one thing. Letting the test set grow with the training budget makes
    the noise level change from point to point along the curve.
    """
    if train_size + test_size > len(dataset):
        raise ValueError(
            f"train {train_size} + test {test_size} exceeds the "
            f"{len(dataset)} rows available"
        )

    draw = stratified_sample if stratified else uniform_sample
    pool = draw(dataset, train_size + test_size, seed=seed)
    return Sample(
        train=pool.subset(range(train_size)),
        test=pool.subset(range(train_size, len(pool))),
        size=train_size,
        stratified=stratified,
    )
