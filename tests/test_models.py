"""Model specs and the feasibility guard. Building needs scikit-learn; these do not."""

import pytest

from covertype.models import SPECS, build, feasible


def test_the_svms_are_marked_as_needing_scaling():
    assert SPECS["svm-rbf"].needs_scaling
    assert SPECS["svm-linear"].needs_scaling


def test_the_forest_is_not():
    # A tree split is invariant to monotone rescaling of a feature.
    assert not SPECS["random-forest"].needs_scaling


def test_an_rbf_svm_on_the_full_dataset_is_refused_with_a_reason():
    ok, why = feasible("svm-rbf", 581_012)
    assert not ok
    assert "kernel matrix" in why
    assert "GB" in why


def test_a_subsampled_rbf_svm_is_allowed():
    assert feasible("svm-rbf", 20_000)[0]


def test_the_forest_is_allowed_at_full_size():
    assert feasible("random-forest", 581_012)[0]


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown model"):
        feasible("naive-bayes", 100)
    with pytest.raises(ValueError, match="unknown model"):
        build("naive-bayes")


def test_building_without_scikit_learn_says_what_to_install():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="scikit-learn is not installed"):
            build("random-forest")
    else:
        model = build("random-forest", n_estimators=5)
        assert model.name == "random forest"


def test_an_svm_is_wrapped_in_a_scaling_pipeline():
    pytest.importorskip("sklearn")
    model = build("svm-rbf")
    # The scaler lives inside the pipeline, so it is refitted per fold and
    # never sees the test rows.
    assert any("scaler" in step.lower() for step in dict(model.named_steps))
