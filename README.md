# project1_ML_EPFL

- `explore.py`: inspect the data.
- `preprocessing.py`: split, optionally one-hot encode, fill missing values,
  scale, and add bias.
- `implementations.py`: the six required ML methods.
- `boosting.py`: gradient-boosted decision trees written with NumPy.
- `train.py`: prepare the data and call each model's training function.
- `tune_boosting.py`: tuning experiment for the boosted trees (depth,
  learning rate, seed averaging, blending with Ridge).
- `run.py`: retrains the selected model on all labelled rows and writes
  `submission.csv` (our best AIcrowd submission, F1 0.454).
- `report/`: the LaTeX report (`report.tex`, `report.pdf`); its two figures are
  drawn from the saved results with `python3 report/make_figures.py`.
- `run_ridge.py`: the same for our best model without trees (Ridge + one-hot),
  writing `submission_ridge.csv`.

Place the provided CSV files in `dataset/`, which is ignored by Git.

Run from the repository root:

```bash
python3 explore.py
python3 train.py
```

## Reproducing the submission

```bash
python3 run.py
```

`run.py` takes about 15 minutes and needs about 3 GB of memory. It:

1. loads the three CSV files with `load_csv_data`, keeping the test IDs;
2. applies the reviewed feature rules (`clean_known_features`) to the
   labelled and test rows;
3. learns the tree bins from all 328,135 labelled rows and applies them to
   the test rows;
4. trains gradient-boosted trees on all labelled rows with the settings chosen
   on the validation split: depth 3, 800 trees, learning rate 0.05,
   `lambda_=1`, `min_child_weight=1`, subsample 0.5, seed 42;
5. predicts test probabilities, labels a row +1 when its probability is at
   least 0.21 (otherwise -1), and writes `submission.csv` with
   `create_csv_submission`.

Every random step is seeded, so running it again writes the same file. The
depth comes from `results/boosting_depth_comparison.csv` (`train.py`), and the
learning rate and number of trees from `results/tuning/tuning_comparison.csv`
(`tune_boosting.py`: validation F1 0.4518, lowest validation loss at 798 of
800 trees, threshold 0.21). The submitted file scored F1 0.454 and accuracy
0.880 on AIcrowd.

### Tuning experiment

```bash
python3 tune_boosting.py
```

`tune_boosting.py` uses the same validation split as `train.py` and compares
depth 3 and 4 with learning rates 0.1 (up to 400 trees) and 0.05 (up to 800
trees), the average of 3 seeds (42, 43, 44) of the option with the lowest
validation loss, and a blend of that average with Ridge + one-hot (both
models' scores standardized with training-row statistics, Ridge weights 0 to
0.5). Results are in `results/tuning/` and `figures/tuning/`. All differences
were at most 0.003 F1; blending with Ridge never helped.

### Baseline submission without trees

```bash
python3 run_ridge.py
```

`run_ridge.py` writes `submission_ridge.csv` with our best model without
trees: Ridge regression on the reviewed features with one-hot encoding
(`lambda_=1e-5`, score threshold -0.60; validation F1 0.4275 in
`results/feature_comparison.csv`). `prepare_train_test` learns the one-hot
categories, medians, means and standard deviations from all labelled rows and
applies them to the test rows. It uses the same `transform_features` steps as
`prepare_data`, so the test rows get exactly the validated preprocessing.
Ridge has no random steps, so the file is the same on every run.

In `train.py`, set `sub_sample` inside `main()` to `True` to use every
50th labelled row for development, or `False` to use all labelled rows.
Both modes keep the same stratified
80% training / 20% validation approach. Preprocessing statistics are learned
from the training portion only.

Each model has an ordinary function in `train.py`. `main()` always runs the
baseline. The switches `run_feature_comparison`, `run_logistic`, `run_ridge`
and `run_boosting` choose which experiments to run (currently only the
boosted trees, because the other results are already saved):

```python
compare_feature_sets(x_train, y_train, feature_names, ridge_lambdas)
evaluate_baseline(y_fit, y_val)
train_logistic(tx_fit, y_fit, tx_val, y_val, max_iters=5000, gamma=0.01, threshold=0.5)
train_ridge(tx_fit, y_fit, tx_val, y_val, lambdas=ridge_lambdas)
train_boosting(x_fit_raw, y_fit_raw, x_val_raw, y_val_raw, tree_feature_names,
               depths=boosting_depths, n_trees=400, learning_rate=0.1)
```

`ridge_lambdas` is `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]` and
`boosting_depths` is `[1, 3, 5]`. The switches
`use_feature_rules` and `use_one_hot` choose the preprocessing of the arrays
given to the models (currently reviewed rules, without one-hot).

To try another model, write its own function (for example `train_ridge`)
using the same four prepared arrays, then add a direct call in `main()`.
Inside that function, train the model, make -1/+1 predictions, and use
`classification_metrics` and `print_metrics` to inspect its results.

Labels stay -1/+1 outside the logistic training step. The prepared matrices
already contain a bias column. The current `main()` uses 5000 iterations,
step size 0.01, threshold 0.5, and `use_feature_rules=True`.

The terminal shows loading, preprocessing, baseline, training, and evaluation
stages. During training, a progress bar displays completed iterations and the
current training loss. It updates every 20 iterations using the current weights.
Training and validation losses are recorded every 100 iterations in
`results/logistic_learning_curve.csv` and plotted in `figures/logistic_loss.png`.
Validation metrics at 1000 steps and the final step are saved in
`results/logistic_iteration_comparison.csv`.

After training, the same validation probabilities are evaluated at thresholds
0.01 through 0.99, in steps of 0.01. The terminal compares threshold 0.5 with
the tested threshold giving the highest validation F1. Results are saved in
`results/logistic_threshold_comparison.csv` and plotted in
`figures/logistic_thresholds.png`. These are validation scores used for tuning.

Ridge regression is fitted with the normal equations directly on the -1/+1
labels, once for each lambda in the list. Its outputs are linear scores, not
probabilities, so the validation scores are compared at score thresholds
-1.00 through 1.00, in steps of 0.01 (0 is the natural default). For each
lambda, `results/ridge_lambda_comparison.csv` stores the training and
validation MSE (without the penalty), the F1 at threshold 0, and the metrics
at the threshold with the highest validation F1; `figures/ridge_lambda.png`
plots them. The lambda with the highest tuned validation F1 is selected, and
its full threshold comparison is saved in
`results/ridge_threshold_comparison.csv` and `figures/ridge_thresholds.png`.
Both lambda and threshold are tuned on the same validation set.

With `use_one_hot=True`, every column with at most 10 distinct training values
is treated as categorical. It is replaced by one 0/1 column per value seen in
training, plus a 0/1 missing-value column if training had missing values in
it. A value that never appears in training gives zeros in all of that column's
indicators. The categories are learned from the training portion only. Median
filling and standardization then run as before, on all columns. All levels are
kept, so each group of indicators sums to the bias column; Ridge's positive
lambda keeps the solution unique.

`compare_feature_sets` is a small ablation study with Ridge, chosen because it
takes seconds instead of minutes. It uses the same split and lambda list for
four feature sets: original, reviewed rules, original + one-hot, and reviewed
rules + one-hot. For each set, the best lambda and threshold are saved in
`results/feature_comparison.csv` and their validation F1 is plotted in
`figures/feature_comparison.png`.

## Boosted trees

Linear models give each feature one weight, so they cannot learn
thresholds or combinations of features. `boosting.py` implements gradient-
boosted decision trees for the logistic loss (the idea behind XGBoost) with
NumPy only:

- **Data:** the same stratified split and reviewed rules, but raw values:
  no median filling, scaling or one-hot encoding, because trees only compare
  values with thresholds. `split_data` in `preprocessing.py` returns these
  arrays.
- **Bins:** each feature is grouped into at most 64 bins learned from the
  training rows (one bin per value when a feature has fewer than 64 values,
  otherwise quantile bins). Missing values get their own bin (64).
- **Trees:** each tree is grown level by level up to `max_depth`. For every
  node, gradient and hessian sums per bin give the gain of each possible split
  (XGBoost's formula with an L2 penalty `lambda_=1` on leaf values). Missing
  values go to whichever side gives the larger gain. Each child must have a
  hessian sum of at least `min_child_weight=1`. A leaf's value is the Newton
  step `-G / (H + lambda_)`.
- **Boosting:** scores start at the log-odds of the training positive rate.
  Each tree is fitted to the current gradients on a random half of the
  training rows (`subsample=0.5`, seed 42) and added with `learning_rate=0.1`.
  The validation loss after every tree chooses the number of trees; validation
  rows never change a tree.
- **Evaluation:** validation probabilities are compared at thresholds 0.01
  through 0.99, as for logistic regression.

`train_boosting` trains depths 1, 3 and 5 with up to 400 trees each. Depth 1
uses one feature per tree, so it can learn thresholds but no interactions;
deeper trees can combine features. Results:

| File | Contents |
| --- | --- |
| `results/boosting_depth_comparison.csv` | One row per depth: best number of trees, losses, best threshold and its metrics |
| `results/boosting_learning_curve.csv`, `figures/boosting_loss.png` | Training and validation loss after every tree |
| `results/boosting_threshold_comparison.csv`, `figures/boosting_thresholds.png` | All thresholds for the selected depth |
| `results/boosting_feature_importance.csv`, `figures/boosting_feature_importance.png` | Share of the total split gain per feature for the selected model |

The number of trees, the depth and the threshold are all chosen on the same
validation set.
