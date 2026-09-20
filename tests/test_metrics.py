"""Metrics on a problem where accuracy is carried by two classes."""

import pytest

from covertype.metrics import ClassScore, majority_accuracy, report

NAMES = {1: "a", 2: "b", 3: "rare"}


def test_perfect_prediction():
    result = report([1, 2, 3], [1, 2, 3], NAMES)
    assert result.accuracy == result.macro_f1 == result.balanced_accuracy == 1.0


def test_accuracy_is_hand_checkable():
    assert report([1, 1, 2, 2], [1, 1, 1, 2], NAMES).accuracy == 0.75


def test_balanced_accuracy_is_the_mean_per_class_recall():
    # Class 1 recall 1.0, class 2 recall 0.5, class 3 absent.
    result = report([1, 1, 2, 2], [1, 1, 1, 2], NAMES)
    assert result.balanced_accuracy == pytest.approx(0.75)


def test_a_model_that_predicts_one_class_scores_well_on_accuracy():
    # 90 of class 1, 10 of class 3. This is the whole argument.
    truth = [1] * 90 + [3] * 10
    predicted = [1] * 100
    result = report(truth, predicted, NAMES)
    assert result.accuracy == 0.90
    assert result.balanced_accuracy == pytest.approx(0.5)
    assert result.macro_f1 < 0.5


def test_an_abandoned_class_is_flagged_not_merely_scored_zero():
    truth = [1] * 90 + [3] * 10
    result = report(truth, [1] * 100, NAMES)
    abandoned = result.abandoned_classes
    assert [c.name for c in abandoned] == ["rare"]
    assert abandoned[0].never_predicted


def test_a_class_absent_from_the_split_is_not_counted_as_abandoned():
    # Never predicting a class nobody asked about is not a failure.
    result = report([1, 1], [1, 1], NAMES)
    assert result.abandoned_classes == ()


def test_absent_classes_still_appear_in_the_report():
    # Deriving the class list from observed labels raises macro-F1 by averaging
    # over fewer, easier classes — rewarding the behaviour it should punish.
    result = report([1, 2], [1, 2], NAMES)
    assert [c.name for c in result.per_class] == ["a", "b", "rare"]


def test_macro_f1_ignores_classes_with_no_support():
    assert report([1, 2], [1, 2], NAMES).macro_f1 == 1.0


def test_the_rarest_present_class_is_identified():
    truth = [1] * 90 + [2] * 9 + [3]
    assert report(truth, truth, NAMES).rarest.name == "rare"


def test_precision_of_a_never_predicted_class_is_zero_not_nan():
    score = ClassScore(3, "rare", support=5, predicted=0, true_positive=0)
    assert score.precision == 0.0 and score.f1 == 0.0


def test_f1_of_a_perfectly_predicted_class():
    score = ClassScore(1, "a", support=4, predicted=4, true_positive=4)
    assert score.f1 == 1.0


def test_majority_accuracy_is_the_floor_every_number_is_read_against():
    assert majority_accuracy([1] * 49 + [2] * 51) == 0.51


def test_majority_accuracy_of_nothing_is_refused():
    with pytest.raises(ValueError, match="empty"):
        majority_accuracy([])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="lengths differ"):
        report([1, 2], [1], NAMES)


def test_empty_evaluation_is_refused():
    with pytest.raises(ValueError, match="empty"):
        report([], [], NAMES)


def test_the_summary_marks_abandoned_classes():
    truth = [1] * 9 + [3]
    assert "never predicted" in report(truth, [1] * 10, NAMES).summary()
