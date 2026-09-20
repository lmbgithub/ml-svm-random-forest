"""Equal-budget comparison and learning curves."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from covertype import metrics, sampling
from covertype.baseline import standardise
from covertype.data import CLASS_NAMES, Dataset


@dataclass(frozen=True, slots=True)
class Run:
    model: str
    train_rows: int
    fit_seconds: float
    predict_seconds: float
    report: metrics.Report
    scaled: bool

    @property
    def accuracy(self) -> float:
        return self.report.accuracy

    @property
    def macro_f1(self) -> float:
        return self.report.macro_f1

    def summary(self) -> str:
        abandoned = len(self.report.abandoned_classes)
        return (
            f"{self.model:<18}{self.train_rows:>8,}"
            f"{self.accuracy:>10.4f}{self.report.balanced_accuracy:>10.4f}"
            f"{self.macro_f1:>10.4f}{self.fit_seconds:>9.1f}s"
            f"{abandoned:>12}"
        )


def evaluate(
    model_factory: Callable[[], object],
    sample: sampling.Sample,
    *,
    scale: bool = False,
    name: str | None = None,
) -> Run:
    """Fit on the training half, score on the test half, and time both."""
    x_train = [list(row) for row in sample.train.x]
    x_test = [list(row) for row in sample.test.x]
    if scale:
        x_train, x_test = standardise(x_train, x_test)

    model = model_factory()
    started = time.perf_counter()
    model.fit(x_train, list(sample.train.y))
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predicted = list(model.predict(x_test))
    predict_seconds = time.perf_counter() - started

    return Run(
        model=name or getattr(model, "name", type(model).__name__),
        train_rows=len(sample.train),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        report=metrics.report(list(sample.test.y), predicted, CLASS_NAMES),
        scaled=scale,
    )


def learning_curve(
    dataset: Dataset,
    model_factory: Callable[[], object],
    sizes: Sequence[int],
    *,
    test_size: int = 2000,
    scale: bool = False,
    seed: int = 0,
    name: str | None = None,
) -> list[Run]:
    """One run per training-set size, with the test set held at a fixed size."""
    runs = []
    for size in sizes:
        sample = sampling.split(dataset, train_size=size, test_size=test_size, seed=seed)
        runs.append(evaluate(model_factory, sample, scale=scale, name=name))
    return runs


def table(runs: Sequence[Run]) -> str:
    header = (
        f"{'model':<18}{'train':>8}{'accuracy':>10}{'balanced':>10}"
        f"{'macro-F1':>10}{'fit':>10}{'abandoned':>12}"
    )
    lines = [header, "-" * len(header)]
    lines += [run.summary() for run in runs]
    return "\n".join(lines)


def ranking_disagreement(runs: Sequence[Run]) -> str:
    """Do accuracy and macro-F1 pick the same model?

    When they disagree, the accuracy column is being carried by the two classes
    that are 85% of the data and the table is ranking models by how well they
    predict "lodgepole pine".
    """
    if len(runs) < 2:
        return "one run; nothing to rank"

    by_accuracy = max(runs, key=lambda r: r.accuracy)
    by_macro = max(runs, key=lambda r: r.macro_f1)

    if (
        by_accuracy.model == by_macro.model
        and by_accuracy.train_rows == by_macro.train_rows
    ):
        return (
            f"accuracy and macro-F1 agree: {by_accuracy.model} "
            f"({by_accuracy.train_rows:,} rows)"
        )
    return (
        f"accuracy picks {by_accuracy.model} ({by_accuracy.train_rows:,} rows, "
        f"acc {by_accuracy.accuracy:.4f}, macro-F1 {by_accuracy.macro_f1:.4f})\n"
        f"macro-F1 picks {by_macro.model} ({by_macro.train_rows:,} rows, "
        f"acc {by_macro.accuracy:.4f}, macro-F1 {by_macro.macro_f1:.4f})\n"
        f"The two metrics rank these models differently. On a dataset where two "
        f"classes are 85%\nof the rows, the accuracy column is ranking models by how "
        f"well they predict those two."
    )


def cost_of_a_point(runs: Sequence[Run]) -> str:
    """What the last increment of training data bought, in seconds per point of F1."""
    ordered = sorted(runs, key=lambda r: r.train_rows)
    if len(ordered) < 2:
        return "a learning curve needs at least two sizes"

    first, last = ordered[0], ordered[-1]
    gained = last.macro_f1 - first.macro_f1
    cost = last.fit_seconds - first.fit_seconds
    if abs(gained) < 1e-9:
        return (
            f"{last.train_rows / first.train_rows:.0f}x the training data bought "
            f"no macro-F1 at all, for {cost:+.1f}s of fitting."
        )
    return (
        f"{last.train_rows / first.train_rows:.0f}x the training data bought "
        f"{gained:+.4f} macro-F1 for {cost:+.1f}s of extra fitting "
        f"({cost / (gained * 100):+.1f}s per point of macro-F1)."
    )
