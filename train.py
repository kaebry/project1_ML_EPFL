from pathlib import Path

import numpy as np

from helpers import load_csv_data
from implementations import logistic_regression, sigmoid
from preprocessing import prepare_data


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


def print_metrics(name, metrics):
    for metric, value in metrics.items():
        print(f"{name} {metric}: {value}", flush=True)


def print_progress(current, total, loss):
    # Update one line in the terminal.
    fraction = current / total if total > 0 else 1.0
    filled = int(25 * fraction)
    bar = "#" * filled + "-" * (25 - filled)
    end = "\n" if current == total else ""
    print(
        f"\rTraining [{bar}] {fraction:6.1%} "
        f"({current}/{total}) loss={loss:.4f}",
        end=end,
        flush=True,
    )


def evaluate_baseline(y_fit, y_val):
    # Predict the majority training class for every validation row.
    labels, counts = np.unique(y_fit, return_counts=True)
    baseline_label = labels[np.argmax(counts)]
    predictions = np.full_like(y_val, baseline_label)

    metrics = classification_metrics(y_val, predictions)
    print("baseline label:", baseline_label)
    print_metrics("baseline", metrics)
    return metrics


def train_logistic(
    tx_fit, y_fit, tx_val, y_val, max_iters=1000, gamma=0.01, threshold=0.5
):
    if max_iters < 0:
        raise ValueError("max_iters cannot be negative.")

    # Convert training labels from -1/+1 to 0/1.
    y_fit_01 = (y_fit == 1).astype(int)

    # One zero weight per input column.
    initial_w = np.zeros(tx_fit.shape[1])

    # Calculate the initial loss before training.
    print("\nTraining logistic regression...", flush=True)
    w, train_loss = logistic_regression(
        y_fit_01, tx_fit, initial_w, 0, gamma
    )
    print_progress(0, max_iters, train_loss)

    # Continue from the current weights and update progress every 20 steps.
    for start in range(0, max_iters, 20):
        steps = min(20, max_iters - start)
        w, train_loss = logistic_regression(y_fit_01, tx_fit, w, steps, gamma)
        print_progress(start + steps, max_iters, train_loss)

    assert w.shape == (tx_fit.shape[1],)
    assert np.isfinite(w).all()
    assert np.isfinite(train_loss)
    assert 0 <= threshold <= 1

    # Calculate training and validation probabilities.
    print("\nEvaluating logistic regression...", flush=True)
    train_probabilities = sigmoid(tx_fit @ w)
    val_probabilities = sigmoid(tx_val @ w)

    assert np.isfinite(train_probabilities).all()
    assert np.isfinite(val_probabilities).all()

    # Convert probabilities to -1/+1 predictions.
    train_predictions = (train_probabilities >= threshold).astype(int) * 2 - 1
    val_predictions = (val_probabilities >= threshold).astype(int) * 2 - 1

    train_metrics = classification_metrics(y_fit, train_predictions)
    val_metrics = classification_metrics(y_val, val_predictions)

    print("weights shape:", w.shape)
    print("training loss:", train_loss)
    print_metrics("logistic training", train_metrics)
    print_metrics("logistic validation", val_metrics)

    return w, train_loss


def main():
    # Use False here when we are ready for the full labelled dataset.
    sub_sample = False

    # Load the data once.
    print("Loading CSV files...", flush=True)
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=sub_sample
    )
    print("sub_sample:", sub_sample)
    print("labelled rows:", len(y_train))

    # Prepare the same training and validation data for all models.
    print("\nPreparing training and validation data...", flush=True)
    tx_fit, y_fit, tx_val, y_val = prepare_data(x_train, y_train)
    print("training shape (including bias):", tx_fit.shape)
    print("validation shape (including bias):", tx_val.shape)

    print("\nEvaluating baseline...", flush=True)
    evaluate_baseline(y_fit, y_val)
    train_logistic(tx_fit, y_fit, tx_val, y_val)
    print("\nFinished.", flush=True)


if __name__ == "__main__":
    main()
