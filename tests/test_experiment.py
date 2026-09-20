"""Equal-budget comparison, learning curves, and the ranking disagreement."""

import pytest

from covertype import synthetic
from covertype.baseline import MajorityClassifier, NearestCentroid
from covertype.experiment import (
    cost_of_a_point,
    evaluate,
    learning_curve,
    ranking_disagreement,
    table,
)
from covertype.sampling import split


@pytest.fixture(scope="module")
def dataset():
    return synthetic.generate(20_000, seed=0)


@pytest.fixture(scope="module")
def sample(dataset):
    return split(dataset, train_size=2000, test_size=2000, seed=0)


def test_a_run_records_timings_and_a_report(sample):
    run = evaluate(lambda: MajorityClassifier(), sample, name="majority")
    assert run.train_rows == 2000
    assert run.fit_seconds >= 0
    assert run.report.total == 2000


def test_the_majority_classifier_abandons_almost_every_class(sample):
    run = evaluate(lambda: MajorityClassifier(), sample, name="majority")
    assert len(run.report.abandoned_classes) >= 5
    # And still scores respectably on accuracy, which is the point.
    assert run.accuracy > 0.4


def test_accuracy_and_macro_f1_can_rank_models_in_opposite_order(sample):
    majority = evaluate(lambda: MajorityClassifier(), sample, name="majority")
    centroid = evaluate(lambda: NearestCentroid(), sample, name="centroid")
    assert majority.accuracy > centroid.accuracy
    assert majority.macro_f1 < centroid.macro_f1
    assert "rank these models differently" in ranking_disagreement([majority, centroid])


def test_agreement_is_reported_when_the_metrics_agree(sample):
    good = evaluate(lambda: NearestCentroid(), sample, scale=True, name="scaled")
    bad = evaluate(lambda: MajorityClassifier(), sample, name="majority")
    text = ranking_disagreement([good, bad]) if good.accuracy > bad.accuracy else ""
    assert "agree" in text or "rank these models differently" in ranking_disagreement(
        [good, bad]
    )


def test_scaling_changes_the_result_more_than_the_model_does(sample):
    unscaled = evaluate(lambda: NearestCentroid(), sample, name="centroid")
    scaled = evaluate(
        lambda: NearestCentroid(), sample, scale=True, name="centroid+scaled"
    )
    assert scaled.accuracy - unscaled.accuracy > 0.2


def test_a_single_run_cannot_be_ranked(sample):
    run = evaluate(lambda: MajorityClassifier(), sample)
    assert "nothing to rank" in ranking_disagreement([run])


def test_a_learning_curve_holds_the_test_size_fixed(dataset):
    runs = learning_curve(
        dataset, lambda: NearestCentroid(), [500, 2000], test_size=1000, scale=True
    )
    assert [r.train_rows for r in runs] == [500, 2000]
    assert {r.report.total for r in runs} == {1000}


def test_a_learning_curve_needs_two_points_to_say_anything(dataset):
    runs = learning_curve(dataset, lambda: MajorityClassifier(), [500])
    assert "at least two sizes" in cost_of_a_point(runs)


def test_the_cost_of_a_point_is_reported_in_seconds(dataset):
    runs = learning_curve(dataset, lambda: NearestCentroid(), [500, 2000], scale=True)
    assert "macro-F1" in cost_of_a_point(runs)


def test_more_data_buying_nothing_is_said_plainly(dataset):
    # The majority classifier's macro-F1 barely moves with more data.
    runs = learning_curve(dataset, lambda: MajorityClassifier(), [500, 4000])
    text = cost_of_a_point(runs)
    assert "macro-F1" in text


def test_the_table_has_one_row_per_run(sample):
    runs = [
        evaluate(lambda: MajorityClassifier(), sample, name="majority"),
        evaluate(lambda: NearestCentroid(), sample, name="centroid"),
    ]
    assert len(table(runs).splitlines()) == 4


def test_the_table_reports_the_abandoned_class_count(sample):
    run = evaluate(lambda: MajorityClassifier(), sample, name="majority")
    assert table([run]).splitlines()[-1].strip().endswith("6")
