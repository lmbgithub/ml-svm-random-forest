"""Classification metrics for a 7-class problem where two classes are 85% of the data.

Accuracy is nearly useless here and is reported anyway, next to the numbers that
are not. Cover type 4 (cottonwood/willow) is 0.47% of the rows: a model that
never predicts it at all loses half a point of accuracy and can still be the
"best" model in a table sorted by accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: int
    name: str
    support: int
    predicted: int
    true_positive: int

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def never_predicted(self) -> bool:
        """Did the model decline to use this class at all?

        Worth its own flag rather than being inferred from an F1 of zero: a
        class that is never predicted is a modelling decision the accuracy
        column cannot show, and on this dataset it is almost free.
        """
        return self.predicted == 0


@dataclass(frozen=True, slots=True)
class Report:
    per_class: tuple[ClassScore, ...]
    total: int

    @property
    def accuracy(self) -> float:
        return (
            sum(c.true_positive for c in self.per_class) / self.total
            if self.total
            else float("nan")
        )

    @property
    def balanced_accuracy(self) -> float:
        """Mean per-class recall — accuracy if every class were equally common."""
        present = [c for c in self.per_class if c.support]
        if not present:
            return float("nan")
        return sum(c.recall for c in present) / len(present)

    @property
    def macro_f1(self) -> float:
        present = [c for c in self.per_class if c.support]
        if not present:
            return float("nan")
        return sum(c.f1 for c in present) / len(present)

    @property
    def rarest(self) -> ClassScore | None:
        present = [c for c in self.per_class if c.support]
        return min(present, key=lambda c: c.support) if present else None

    @property
    def abandoned_classes(self) -> tuple[ClassScore, ...]:
        return tuple(c for c in self.per_class if c.support and c.never_predicted)

    def summary(self) -> str:
        lines = [
            f"accuracy {self.accuracy:.4f}   balanced {self.balanced_accuracy:.4f}"
            f"   macro-F1 {self.macro_f1:.4f}",
            f"{'class':<20}{'support':>9}{'recall':>9}{'f1':>9}",
        ]
        for score in self.per_class:
            note = "  never predicted" if score.support and score.never_predicted else ""
            lines.append(
                f"{score.name:<20}{score.support:>9}{score.recall:>9.3f}{score.f1:>9.3f}{note}"
            )
        return "\n".join(lines)


def report(
    truth: Sequence[int], predicted: Sequence[int], names: dict[int, str]
) -> Report:
    """Score over every class in `names`, present in this split or not.

    Deriving the class list from the observed labels would drop a class the
    model ignores entirely and raise macro-F1 by averaging over fewer, easier
    classes — rewarding exactly the behaviour the metric exists to punish.
    """
    if len(truth) != len(predicted):
        raise ValueError(f"lengths differ: {len(truth)} and {len(predicted)}")
    if not truth:
        raise ValueError("cannot score an empty evaluation set")

    scores = []
    for label in sorted(names):
        scores.append(
            ClassScore(
                label=label,
                name=names[label],
                support=sum(1 for t in truth if t == label),
                predicted=sum(1 for p in predicted if p == label),
                true_positive=sum(
                    1 for t, p in zip(truth, predicted, strict=True) if t == p == label
                ),
            )
        )
    return Report(tuple(scores), len(truth))


def majority_accuracy(truth: Sequence[int]) -> float:
    """Accuracy of always predicting the most common class.

    On Covertype this is about 0.49, which is the number every reported
    accuracy should be read against.
    """
    if not truth:
        raise ValueError("cannot score an empty evaluation set")
    counts: dict[int, int] = {}
    for label in truth:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) / len(truth)
