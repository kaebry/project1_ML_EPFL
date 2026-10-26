"""Candidate submission: gradient-boosted trees of depth 3, learning rate 0.05.

This is a copy of run.py with different settings, writing its own file
(submission_lr005.csv), so run.py and submission.csv stay unchanged until
this candidate is tested on AIcrowd.

The settings below were chosen on the stratified validation split: depth 3
in train.py (results/boosting_depth_comparison.csv), and learning rate 0.05
with 800 trees in tune_boosting.py (results/tuning/tuning_comparison.csv).
Here the model is retrained on all labelled rows with those fixed settings
and predicts x_test. Everything is seeded, so running this script again
writes the same file.
"""
from pathlib import Path

import numpy as np

from boosting import apply_bins, fit_bins, fit_boosting, predict_boosting
from helpers import create_csv_submission, load_csv_data
from implementations import sigmoid
from preprocessing import clean_known_features, load_feature_names

# Settings chosen on validation (validation F1 = 0.4518, lowest loss at 798 trees).
N_TREES = 800
MAX_DEPTH = 3
LEARNING_RATE = 0.05
LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0
SUBSAMPLE = 0.5
SEED = 42
THRESHOLD = 0.21


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "dataset"
    output_path = root / "submission_lr005.csv"

    # Load all labelled rows and the test rows; keep the test IDs in order.
    print("Loading CSV files...", flush=True)
    x_train, x_test, y_train, _, test_ids = load_csv_data(data_path)
    feature_names = load_feature_names(data_path / "x_train.csv")
    print("labelled rows:", len(y_train), "test rows:", len(test_ids))

    # The reviewed rules are fixed, so they are applied in the same way to both sets.
    print("Cleaning reviewed features...", flush=True)
    x_train = clean_known_features(x_train, feature_names)
    x_test = clean_known_features(x_test, feature_names)

    # Bins are learned from the labelled rows only, then applied to the test rows.
    print("Binning features...", flush=True)
    edges = fit_bins(x_train)
    bins_train = apply_bins(x_train, edges)
    bins_test = apply_bins(x_test, edges)

    # Train on all labelled rows (0/1 labels); no validation is needed here.
    print(f"Training {N_TREES} boosted trees of depth {MAX_DEPTH}...", flush=True)
    y_train_01 = (y_train == 1).astype(float)
    trees, base_score, _, _ = fit_boosting(
        bins_train, y_train_01, None, None,
        n_trees=N_TREES, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
        lambda_=LAMBDA, min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE, seed=SEED,
    )

    # Predict probabilities and convert them to -1/+1 labels.
    probabilities = sigmoid(
        predict_boosting(trees, base_score, LEARNING_RATE, bins_test)
    )
    predictions = np.where(probabilities >= THRESHOLD, 1, -1)
    assert len(predictions) == len(test_ids)

    create_csv_submission(test_ids, predictions, output_path)
    print(f"Predicted +1 for {np.mean(predictions == 1):.1%} of test rows.")
    print("Saved submission:", output_path, flush=True)


if __name__ == "__main__":
    main()
