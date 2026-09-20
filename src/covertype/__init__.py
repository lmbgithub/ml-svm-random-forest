"""SVM against random forest on Covertype, compared at an equal training budget.

The measurement half — metrics, sampling, the imbalance arithmetic, the
baselines — is standard library, so the finding can be reproduced offline. Only
`models.py` imports scikit-learn and only `data.load_uci` needs the network.
"""

from covertype.data import CLASS_NAMES, Dataset, load_csv, load_uci, parse_csv
from covertype.experiment import (
    Run,
    evaluate,
    learning_curve,
    ranking_disagreement,
    table,
)
from covertype.metrics import Report, majority_accuracy, report
from covertype.sampling import split, stratified_sample, uniform_sample

__all__ = [
    "CLASS_NAMES",
    "Dataset",
    "Report",
    "Run",
    "evaluate",
    "learning_curve",
    "load_csv",
    "load_uci",
    "majority_accuracy",
    "parse_csv",
    "ranking_disagreement",
    "report",
    "split",
    "stratified_sample",
    "table",
    "uniform_sample",
]
