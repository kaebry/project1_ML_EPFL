from pathlib import Path

import numpy as np

from helpers import load_csv_data
from implementations import logistic_regression, sigmoid
from preprocessing import (
    add_bias,
    apply_imputer,
    apply_scaler,
    fit_imputer,
    fit_scaler,
    stratified_split,
)


def classification_metrics(y_true, y_pred):
    assert y_true.ndim == 1
    assert y_true.shape == y_pred.shape
    assert len(y_true) > 0

    # Count the four prediction outcomes.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    assert tp + tn + fp + fn == len(y_true)

    accuracy = (tp + tn) / len(y_true)

    # Use 0 when the denominator is zero.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # Check the metrics on a small example.
    example_y = np.array([1, 1, -1, -1])
    example_pred = np.array([1, -1, 1, -1])

    example_metrics = classification_metrics(example_y, example_pred)

    for name in ("tp", "tn", "fp", "fn"):
        assert example_metrics[name] == 1

    for name in ("accuracy", "precision", "recall", "f1"):
        assert np.isclose(example_metrics[name], 0.5)

    # Load a small data sample.
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, _x_test, y_train, _train_ids, _test_ids = load_csv_data(
        data_path, sub_sample=True
    )

    # Split the data into training and validation sets.
    train_indices, val_indices = stratified_split(y_train)

    x_fit = x_train[train_indices]
    y_fit = y_train[train_indices]
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]

    # Check that the split uses each row once.
    assert np.intersect1d(train_indices, val_indices).size == 0

    all_indices = np.concatenate((train_indices, val_indices))
    np.testing.assert_array_equal(
        np.sort(all_indices),
        np.arange(len(y_train)),
    )

    # Check that the same seed gives the same split.
    train_again, val_again = stratified_split(y_train)
    np.testing.assert_array_equal(train_indices, train_again)
    np.testing.assert_array_equal(val_indices, val_again)

    print("training shape:", x_fit.shape)
    print("validation shape:", x_val.shape)
    print("training positive fraction:", np.mean(y_fit == 1))
    print("validation positive fraction:", np.mean(y_val == 1))

    # Learn medians from training data, then clean both sets.
    keep_columns, medians = fit_imputer(x_fit)

    x_fit_clean = apply_imputer(x_fit, keep_columns, medians)
    x_val_clean = apply_imputer(x_val, keep_columns, medians)

    assert x_fit_clean.shape == (len(y_fit), len(medians))
    assert x_val_clean.shape == (len(y_val), len(medians))
    assert np.isfinite(x_fit_clean).all()
    assert np.isfinite(x_val_clean).all()

    print("removed columns:", np.sum(~keep_columns))
    print("clean training shape:", x_fit_clean.shape)
    print("clean validation shape:", x_val_clean.shape)
    print("training NaNs remaining:", np.isnan(x_fit_clean).sum())
    print("validation NaNs remaining:", np.isnan(x_val_clean).sum())

    # Learn scaling from training data, then scale both sets.
    means, stds = fit_scaler(x_fit_clean)

    x_fit_scaled = apply_scaler(x_fit_clean, means, stds)
    x_val_scaled = apply_scaler(x_val_clean, means, stds)

    assert x_fit_scaled.shape == x_fit_clean.shape
    assert x_val_scaled.shape == x_val_clean.shape
    assert np.isfinite(x_fit_scaled).all()
    assert np.isfinite(x_val_scaled).all()

    # Check that training means are close to zero.
    np.testing.assert_allclose(
        np.mean(x_fit_scaled, axis=0),
        0.0,
        atol=1e-6,
    )

    # Check that nonconstant training columns have standard deviation 1.
    variable_columns = np.std(x_fit_clean, axis=0) > 0
    np.testing.assert_allclose(
        np.std(x_fit_scaled, axis=0)[variable_columns],
        1.0,
        atol=1e-6,
    )

    print("scaled training shape:", x_fit_scaled.shape)
    print("scaled validation shape:", x_val_scaled.shape)
    print("training means:", np.mean(x_fit_scaled, axis=0)[:5])
    print("training stds:", np.std(x_fit_scaled, axis=0)[:5])

    # Baseline: predict the majority training class for every validation row.
    fit_labels, fit_counts = np.unique(y_fit, return_counts=True)
    baseline_label = fit_labels[np.argmax(fit_counts)]
    baseline_predictions = np.full_like(y_val, baseline_label)

    baseline_metrics = classification_metrics(y_val, baseline_predictions)

    print("baseline label:", baseline_label)
    for name, value in baseline_metrics.items():
        print(f"baseline {name}: {value}")


    # Add the intercept after scaling.
    tx_fit = add_bias(x_fit_scaled)
    tx_val = add_bias(x_val_scaled)

    # Convert training labels from -1/+1 to 0/1.
    y_fit_01 = (y_fit == 1).astype(int)

    # One zero weight per input column.
    initial_w = np.zeros(tx_fit.shape[1])

    # Train the first logistic regression model.
    w, train_loss = logistic_regression(
        y_fit_01,
        tx_fit,
        initial_w,
        max_iters=1000,
        gamma=0.01,
    )

    # Calculate validation probabilities.
    val_probabilities = sigmoid(tx_val @ w)

    # Convert probabilities to -1/+1 predictions.
    val_predictions = (val_probabilities >= 0.5).astype(int) * 2 - 1

    assert w.shape == (tx_fit.shape[1],)
    assert np.isfinite(w).all()
    assert np.isfinite(train_loss)
    assert np.isfinite(val_probabilities).all()
    assert np.all(
        (val_probabilities >= 0) & (val_probabilities <= 1)
    )

    print("weights shape:", w.shape)
    print("training loss:", train_loss)

    model_metrics = classification_metrics(y_val, val_predictions)
    for name, value in model_metrics.items():
        print(f"logistic {name}: {value}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
