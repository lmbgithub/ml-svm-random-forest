# ml-svm-random-forest

SVM against random forest on UCI Covertype: 581,012 rows, 54 features, 7 classes, and roughly 100:1 class imbalance.

An RBF SVM cannot be trained on 581,012 rows — kernel training is between quadratic and cubic in the row count, and the kernel matrix alone is about 2,700 GB — so every published version of this comparison subsamples for the SVM. The common mistake is to subsample for the SVM, train the forest on everything, and put both accuracies in one table, which compares two different experiments. This project runs the comparison so that the training budget is equal and the numbers mean what the table says they do.

**Standard library only for the measurement half. 82 tests.**

## What the run looks like

```
$ python examples/imbalance_demo.py
30,000 rows, imbalance 111:1
  1 spruce/fir            10,893   36.31%
  2 lodgepole pine        14,516   48.39%
  ...
  4 cottonwood/willow        135    0.45%

majority-class accuracy: 0.4839

==============================================================================
1. A model that predicts nothing

accuracy 0.4839   balanced 0.1429   macro-F1 0.0932
class                 support   recall       f1
spruce/fir               1452    0.000    0.000  never predicted
lodgepole pine           1936    1.000    0.652
cottonwood/willow          19    0.000    0.000  never predicted
...
   6 of 7 classes never predicted, for 48.4% accuracy.

==============================================================================
2. Accuracy and macro-F1 disagree

model                train  accuracy  balanced  macro-F1       fit   abandoned
------------------------------------------------------------------------------
majority             4,000    0.4839    0.1429    0.0932      0.0s           6
centroid             4,000    0.3263    0.3545    0.2655      0.0s           0

accuracy picks majority  (acc 0.4839, macro-F1 0.0932)
macro-F1 picks centroid  (acc 0.3263, macro-F1 0.2655)

==============================================================================
3. Scaling is worth more than the model choice

centroid             4,000    0.3263    0.3545    0.2655
centroid+scaled      4,000    0.8342    0.8005    0.6943

   standardising: +0.5079 accuracy, +0.4288 macro-F1 — same model, same data.

==============================================================================
4. Stratified sampling keeps the rare class in the experiment

   uniform     cover type 4 in a 500-row draw: [0, 1, 4, 2, 2, 1, 3, 3]
   stratified  cover type 4 in a 500-row draw: [2, 2, 2, 2, 2, 2, 2, 2]
```

Read demonstration 2 twice. A model that predicts **one class for everything** wins the accuracy column on this dataset — and a table sorted by accuracy is a table sorted by how well each model predicts lodgepole pine.

## The seven decisions worth discussing

**1. The training budget is a shared parameter, not an implementation detail.** `sampling.split` takes `train_size` and gives it to every model. The SVM and the forest see the same rows. Without that, "the forest is more accurate" is a statement about it having seen fifty times more data.

**2. The run is refused before it starts, with the arithmetic.** `models.feasible("svm-rbf", 581_012)` returns False and says the kernel matrix would be about 2,700 GB. A guard that fires after four hours of swapping is not a guard.

**3. Sampling is stratified, and the reason is not fairness.** Cover type 4 is 0.45% of the rows, so a uniform draw of 500 frequently contains **zero** of them — the fourth demonstration above shows one such draw. A class with no examples cannot be trained or tested, and two models given different numbers of them are not being compared. Each present class is guaranteed at least one row, and the total is topped up to the exact requested size so a learning curve's x-axis means what it says.

**4. Accuracy is reported next to the majority baseline, always.** 48% accuracy on this dataset is the score for predicting "lodgepole pine" forever. `Report.abandoned_classes` names the classes a model declined to use at all — a modelling decision the accuracy column cannot show, and one that costs almost nothing here.

**5. The SVM is wrapped in a scaling pipeline; the forest is not.** An RBF kernel on unstandardised Covertype is dominated by `Horizontal_Distance_To_Roadways` (0–7,000) while `Slope` (0–66) contributes nothing. A random forest is invariant to monotone rescaling of any feature. Comparing an unscaled SVM against a forest compares a broken configuration against a working one — and demonstration 3 prices that mistake at +0.51 accuracy, larger than any difference between the model families. The scaler lives inside the pipeline so it is refitted per fold and never sees test rows.

**6. Collapsing the one-hot soil columns is fine for a contingency table and wrong as a feature.** `invert_one_hot` exists and its docstring says not to feed its output to a model: soil type 17 is not "between" 16 and 18. The same argument applies to the correlation analysis that usually opens this exercise — ranking features by Pearson correlation against `Cover_Type`, a nominal 7-class label, produces numbers whose values depend entirely on the arbitrary integers assigned to the categories. The function raises if the block is not actually one-hot, rather than inventing a category.

**7. The test set size is fixed independently of the training budget.** Letting it grow with the training set changes the noise level from point to point along a learning curve, so the curve stops being a curve in one variable.

## Why a synthetic dataset is the default

The real download is 75 MB and the real SVM run is a research project. The findings here — accuracy and macro-F1 ranking models in opposite order, a class being abandoned almost for free, an unscaled distance model being effectively one-dimensional, a uniform draw missing a class entirely — are properties of the dataset's *shape*: seven classes, ~100:1 imbalance, features spanning four orders of magnitude. `synthetic.generate` reproduces that shape exactly, so every claim above is a test rather than a screenshot.

`--csv` and `--uci` run the identical experiment on the real thing.

## Design

```
src/covertype/
  data.py        CSV loading, class names, the one-hot inversion guard
  sampling.py    stratified and uniform draws, fixed-size train/test split
  metrics.py     accuracy, balanced accuracy, macro-F1, abandoned classes
  baseline.py    two standard-library classifiers + train-only standardisation
  models.py      scikit-learn factories, scaling pipeline, the feasibility guard
  experiment.py  equal-budget runs, learning curves, the ranking disagreement
  synthetic.py   a Covertype-shaped generator
  cli.py         argument parsing
```

Only `models.py` imports scikit-learn and only `data.load_uci` touches the network, so the whole test suite and the offline example run on a clean interpreter in about a second.

## Usage

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
pytest -q
python examples/imbalance_demo.py                        # no downloads, no sklearn
python -m covertype --train-sizes 1000,4000,16000        # synthetic
```

With scikit-learn and the real data:

```bash
pip install -r requirements.txt
python -m covertype --uci --models svm-rbf,random-forest --train-sizes 5000,20000
python -m covertype --csv covtype.csv --models random-forest --train-sizes 100000
```

As a library:

```python
from covertype import load_csv, learning_curve, ranking_disagreement, table
from covertype.models import build

dataset = load_csv("covtype.csv")
runs  = learning_curve(dataset, lambda: build("svm-rbf"), [2000, 8000], name="svm-rbf")
runs += learning_curve(dataset, lambda: build("random-forest"), [2000, 8000], name="forest")
print(table(runs))
print(ranking_disagreement(runs))
```

## Dataset

**UCI Covertype** (id 31) — 581,012 cartographic observations of 30×30 m forest cells in Colorado, 54 features, 7 cover-type classes. Not included; about 11 MB compressed, 75 MB as CSV.

```bash
curl -O https://archive.ics.uci.edu/static/public/31/covertype.zip
unzip covertype.zip && gunzip covtype.data.gz
# the published file has no header row; add one, or use --uci
python -m covertype --csv covtype.csv
```

or let `ucimlrepo` fetch it, which supplies the column names:

```bash
pip install -r requirements.txt
python -m covertype --uci
```

Layout: 10 continuous terrain measurements, then 4 one-hot wilderness-area columns and 40 one-hot soil-type columns, then `Cover_Type` in 1–7. No missing values. Class shares run from 48.8% (lodgepole pine) to 0.47% (cottonwood/willow).

Nothing needs downloading to reproduce the findings: the synthetic generator is the default.

## Not included

- **No hyperparameter search.** A C/gamma grid for the SVM at a fixed budget is a reasonable next step and would not change any conclusion here — the differences this repository is about are larger than tuning gains and appear before the first hyperparameter is chosen.
- **No class weighting or resampling.** `class_weight="balanced"` and SMOTE both change what the model optimises, which is a different experiment. This one measures what the *default* comparison reports.
- **No full-dataset SVM.** It is refused, with the arithmetic, on purpose.
- **No timing claims across machines.** Fit times are printed because the cost/accuracy trade is the point, but they are from one machine and are not a benchmark.
- **No decision-boundary visualisation.** Pretty, and it would say nothing about a 54-dimensional problem that the per-class recall column does not.

## License

MIT — see [LICENSE](LICENSE).
