"""Subsampling is the comparison, not a preliminary to it."""

import pytest

from covertype import synthetic
from covertype.sampling import split, stratified_sample, uniform_sample


@pytest.fixture(scope="module")
def dataset():
    return synthetic.generate(20_000, seed=0)


def test_a_stratified_sample_is_the_requested_size(dataset):
    assert len(stratified_sample(dataset, 1000, seed=0)) == 1000


def test_a_stratified_sample_keeps_the_rare_class(dataset):
    # Cover type 4 is under 0.5%: a uniform draw of 500 frequently contains
    # none, and a comparison where one model saw zero examples of a class is
    # not measuring the models.
    rarest = synthetic.rarest_label()
    for seed in range(10):
        assert stratified_sample(dataset, 500, seed=seed).y.count(rarest) >= 1


def test_a_uniform_sample_does_not_guarantee_that(dataset):
    rarest = synthetic.rarest_label()
    counts = [
        uniform_sample(dataset, 300, seed=seed).y.count(rarest) for seed in range(12)
    ]
    assert min(counts) == 0


def test_a_stratified_sample_preserves_class_shares(dataset):
    sample = stratified_sample(dataset, 4000, seed=0)
    for label, count in dataset.class_counts().items():
        if count:
            assert sample.y.count(label) / len(sample) == pytest.approx(
                count / len(dataset), abs=0.01
            )


def test_sampling_is_reproducible(dataset):
    assert (
        stratified_sample(dataset, 500, seed=3).y
        == stratified_sample(dataset, 500, seed=3).y
    )


def test_different_seeds_give_different_samples(dataset):
    assert (
        stratified_sample(dataset, 500, seed=1).y
        != stratified_sample(dataset, 500, seed=2).y
    )


@pytest.mark.parametrize("draw", [stratified_sample, uniform_sample])
def test_a_non_positive_size_is_refused(dataset, draw):
    with pytest.raises(ValueError, match="must be positive"):
        draw(dataset, 0)


@pytest.mark.parametrize("draw", [stratified_sample, uniform_sample])
def test_drawing_more_rows_than_exist_is_refused(dataset, draw):
    with pytest.raises(ValueError, match="cannot draw"):
        draw(dataset, len(dataset) + 1)


def test_train_and_test_are_disjoint(dataset):
    sample = split(dataset, train_size=500, test_size=500, seed=0)
    train_rows = {tuple(row) for row in sample.train.x}
    assert not train_rows & {tuple(row) for row in sample.test.x}


def test_the_test_set_size_is_independent_of_the_training_budget(dataset):
    # A test set that grows with the training budget changes the noise level
    # from point to point along a learning curve.
    small = split(dataset, train_size=200, test_size=1000, seed=0)
    large = split(dataset, train_size=4000, test_size=1000, seed=0)
    assert small.sizes[1] == large.sizes[1] == 1000


def test_asking_for_more_than_the_dataset_holds_is_refused(dataset):
    with pytest.raises(ValueError, match="exceeds"):
        split(dataset, train_size=len(dataset), test_size=1000)
