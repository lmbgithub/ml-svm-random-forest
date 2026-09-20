"""The standard-library classifiers, and the scaling failure they demonstrate."""

import pytest

from covertype.baseline import MajorityClassifier, NearestCentroid, standardise


def test_the_majority_classifier_predicts_the_most_common_label():
    model = MajorityClassifier().fit([[0.0]] * 5, [1, 1, 1, 2, 3])
    assert model.predict([[9.0], [8.0]]) == [1, 1]


def test_ties_are_broken_deterministically():
    model = MajorityClassifier().fit([[0.0]] * 4, [2, 2, 1, 1])
    assert model.label == 1
    assert MajorityClassifier().fit([[0.0]] * 4, [1, 1, 2, 2]).label == 1


def test_fitting_the_majority_classifier_on_nothing_is_refused():
    with pytest.raises(ValueError, match="empty"):
        MajorityClassifier().fit([], [])


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        MajorityClassifier().predict([[1.0]])
    with pytest.raises(RuntimeError, match="not fitted"):
        NearestCentroid().predict([[1.0]])


def test_the_centroid_classifier_recovers_well_separated_classes():
    x = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
    model = NearestCentroid().fit(x, [1, 1, 2, 2])
    assert model.predict([[0.05, 0.05], [9.9, 9.9]]) == [1, 2]


def test_an_unscaled_feature_with_a_huge_range_dominates():
    # The failure an RBF SVM has and a random forest does not — which is why
    # comparing an unscaled SVM against a forest compares a broken
    # configuration against a working one.
    #
    # Feature 0 spans thousands and is pure noise; feature 1 spans single
    # digits and separates the classes perfectly.
    x = [
        [0.0, 0.0],
        [100.0, 0.2],  # class 1
        [5000.0, 9.0],
        [5100.0, 9.2],  # class 2
    ]
    y = [1, 1, 2, 2]
    unscaled = NearestCentroid().fit(x, y)
    query = [[200.0, 9.1]]  # unambiguously class 2 on feature 1
    scaled_x, scaled_query = standardise(x, query)
    scaled = NearestCentroid().fit(scaled_x, y)
    assert scaled.predict(scaled_query) == [2]
    assert unscaled.predict(query) != [2]


def test_standardising_uses_the_training_statistics_only():
    train, test = standardise([[0.0], [2.0]], [[4.0]])
    assert train[0][0] == pytest.approx(-0.7071, abs=1e-3)
    assert test[0][0] > train[1][0]


def test_a_constant_column_does_not_become_nan():
    scaled, _ = standardise([[5.0], [5.0]], [[5.0]])
    assert all(row[0] == 0.0 for row in scaled)


def test_standardising_an_empty_training_set_is_refused():
    with pytest.raises(ValueError, match="empty"):
        standardise([], [[1.0]])


def test_ragged_rows_are_refused():
    with pytest.raises(ValueError, match="same width"):
        NearestCentroid().fit([[1.0], [2.0, 3.0]], [1, 2])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="rows and y"):
        NearestCentroid().fit([[1.0], [2.0]], [1])
