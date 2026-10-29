"""Baseline submission without trees: Ridge regression on one-hot features.

The settings were chosen on the stratified validation split in train.py
(results/feature_comparison.csv, row "reviewed + one-hot"). Here the model is
retrained on all labelled rows with those fixed settings and predicts x_test.
Ridge has no random steps, so running this script again writes the same file.
Our best submission is reproduced by run.py; this script gives the baseline.
"""
from pathlib import Path

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import ridge_regression
from preprocessing import load_feature_names, prepare_train_test

# Settings chosen on validation (validation F1 = 0.4275).
LAMBDA = 1e-5
# Ridge gives linear scores for -1/+1 targets, not probabilities.
THRESHOLD = -0.60


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "dataset"
    output_path = root / "submission_ridge.csv"

    # Load all labelled rows and the test rows; keep the test IDs in order.
    print("Loading CSV files...", flush=True)
    x_train, x_test, y_train, _, test_ids = load_csv_data(data_path)
    feature_names = load_feature_names(data_path / "x_train.csv")
    print("labelled rows:", len(y_train), "test rows:", len(test_ids))

    # Rules, one-hot, filling and scaling are learned from the labelled rows only.
    print("Preparing features (reviewed rules + one-hot)...", flush=True)
    tx_train, tx_test = prepare_train_test(
        x_train,
        x_test,
        feature_names,
        use_feature_rules=True,
        use_one_hot=True,
    )
    print("training shape (including bias):", tx_train.shape)
    print("test shape (including bias):", tx_test.shape)

    # Fit on the -1/+1 labels of all labelled rows.
    print("Training ridge regression...", flush=True)
    w, train_loss = ridge_regression(y_train, tx_train, LAMBDA)
    print("training loss (MSE):", train_loss)

    # Convert test scores to -1/+1 labels.
    predictions = np.where(tx_test @ w >= THRESHOLD, 1, -1)
    assert len(predictions) == len(test_ids)

    create_csv_submission(test_ids, predictions, output_path)
    print(f"Predicted +1 for {np.mean(predictions == 1):.1%} of test rows.")
    print("Saved submission:", output_path, flush=True)


if __name__ == "__main__":
    main()
