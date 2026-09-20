"""scikit-learn model factories. The only module that imports it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    needs_scaling: bool
    #: Roughly how training time grows with the number of rows. Used to warn
    #: before a run that will not finish, rather than after.
    complexity: str


SPECS = {
    "svm-rbf": ModelSpec(
        "SVM (RBF)", needs_scaling=True, complexity="quadratic-to-cubic"
    ),
    "svm-linear": ModelSpec("SVM (linear)", needs_scaling=True, complexity="quadratic"),
    "random-forest": ModelSpec(
        "random forest", needs_scaling=False, complexity="n log n"
    ),
}


def build(kind: str, *, seed: int = 0, **overrides):
    """Build a scikit-learn estimator, scaled if the algorithm needs it.

    The SVMs are wrapped in a pipeline with a `StandardScaler`. That is not a
    detail: an RBF kernel on unstandardised Covertype is dominated by
    `Horizontal_Distance_To_Roadways` (0-7,000) while `Slope` (0-66) contributes
    nothing, and comparing that against a random forest — which is invariant to
    monotone rescaling of any feature — compares a broken configuration against
    a working one.

    The scaler lives inside the pipeline so it is refitted on each training fold
    and never sees the test rows.
    """
    if kind not in SPECS:
        raise ValueError(f"unknown model {kind!r}; expected one of {sorted(SPECS)}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn is not installed; run `pip install -r requirements.txt`. "
            "The offline example and the test suite do not need it."
        ) from exc

    if kind == "random-forest":
        model = RandomForestClassifier(
            n_estimators=overrides.pop("n_estimators", 200),
            n_jobs=overrides.pop("n_jobs", -1),
            random_state=seed,
            **overrides,
        )
        model.name = SPECS[kind].name
        return model

    kernel = "rbf" if kind == "svm-rbf" else "linear"
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=kernel,
            C=overrides.pop("C", 1.0),
            gamma=overrides.pop("gamma", "scale"),
            cache_size=overrides.pop("cache_size", 1000),
            random_state=seed,
            **overrides,
        ),
    )
    model.name = SPECS[kind].name
    return model


def feasible(kind: str, rows: int) -> tuple[bool, str]:
    """Is training this model on this many rows going to finish?

    A crude guard, deliberately: the useful thing is not a precise estimate but
    refusing to start a run that would need days and a terabyte of kernel
    matrix. The number below is measured on ordinary hardware, not derived.
    """
    spec = SPECS.get(kind)
    if spec is None:
        raise ValueError(f"unknown model {kind!r}")
    if kind == "svm-rbf" and rows > 100_000:
        kernel_gb = (rows * rows * 8) / 1e9
        return False, (
            f"an RBF SVM on {rows:,} rows is {spec.complexity} in the row count; "
            f"the kernel matrix alone is about {kernel_gb:,.0f} GB. Subsample instead."
        )
    if kind == "svm-linear" and rows > 200_000:
        return False, f"a kernel SVM on {rows:,} rows will not finish in a useful time"
    return True, ""
