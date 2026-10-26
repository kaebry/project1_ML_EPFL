import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers import load_csv_data
from implementations import logistic_loss, logistic_regression, sigmoid
from preprocessing import load_feature_names, prepare_data


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


def evaluate_thresholds(y_true, probabilities):
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Expected finite probabilities between 0 and 1.")

    results = []
    # Compare 0.01, 0.02, ..., 0.99 using the same probabilities.
    for percent in range(1, 100):
        threshold = percent / 100
        predictions = np.where(probabilities >= threshold, 1, -1)
        results.append({
            "threshold": threshold,
            **classification_metrics(y_true, predictions),
        })

    return results


def save_threshold_results(results):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "logistic_threshold_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "threshold", "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        writer.writerows(results)

    default = next(row for row in results if row["threshold"] == 0.5)
    best = max(results, key=lambda row: row["f1"])
    thresholds = [row["threshold"] for row in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(thresholds, [row[metric] for row in results], label=metric)
    ax.axvline(0.5, color="gray", linestyle=":", label="Default: 0.50")
    ax.axvline(
        best["threshold"], color="tab:green", linestyle="--",
        label=f"Best tested: {best['threshold']:.2f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Validation score")
    ax.set_title("Logistic regression threshold comparison")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plot_path = figures_dir / "logistic_thresholds.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\nValidation threshold comparison:", flush=True)
    print("Choice        Threshold  Precision  Recall     F1         Accuracy")
    for name, row in [("Default", default), ("Best tested", best)]:
        print(
            f"{name:<13} {row['threshold']:<10.2f} "
            f"{row['precision']:<10.4f} {row['recall']:<10.4f} "
            f"{row['f1']:<10.4f} {row['accuracy']:.4f}",
            flush=True,
        )
    print("Saved threshold comparison:", csv_path, flush=True)
    print("Saved threshold plot:", plot_path, flush=True)


def save_logistic_history(history, comparison):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Save the losses and the validation metrics for the compared iterations.
    history_path = results_dir / "logistic_learning_curve.csv"
    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "training_loss", "validation_loss"])
        writer.writerows(history)

    comparison_path = results_dir / "logistic_iteration_comparison.csv"
    with comparison_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "iteration", "gamma", "threshold", "training_loss", "validation_loss",
            "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        writer.writerows(comparison)

    iterations, training_losses, validation_losses = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, training_losses, label="Training", color="tab:blue")
    ax.plot(
        iterations, validation_losses, label="Validation",
        color="tab:orange", linestyle="--",
    )
    if iterations[-1] > 1000:
        ax.axvline(1000, color="gray", linestyle=":", label="Previous: 1000 steps")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean logistic loss")
    ax.set_title("Logistic regression learning curves")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plot_path = figures_dir / "logistic_loss.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Saved learning curve:", history_path, flush=True)
    print("Saved iteration comparison:", comparison_path, flush=True)
    print("Saved loss plot:", plot_path, flush=True)


def train_logistic(
    tx_fit, y_fit, tx_val, y_val, max_iters=1000, gamma=0.01, threshold=0.5
):
    if max_iters < 0:
        raise ValueError("max_iters cannot be negative.")

    # Convert training labels from -1/+1 to 0/1.
    y_fit_01 = (y_fit == 1).astype(int)
    y_val_01 = (y_val == 1).astype(int)

    # One zero weight per input column.
    initial_w = np.zeros(tx_fit.shape[1])

    # Calculate the initial loss before training.
    print("\nTraining logistic regression...", flush=True)
    w, train_loss = logistic_regression(
        y_fit_01, tx_fit, initial_w, 0, gamma
    )
    val_loss = logistic_loss(y_val_01, tx_val, w)
    history = [(0, train_loss, val_loss)]
    comparison = []
    print_progress(0, max_iters, train_loss)

    # Continue from the current weights and update progress every 20 steps.
    for start in range(0, max_iters, 20):
        steps = min(20, max_iters - start)
        w, train_loss = logistic_regression(y_fit_01, tx_fit, w, steps, gamma)
        completed = start + steps
        print_progress(completed, max_iters, train_loss)

        # Validation measures progress; it does not update the weights.
        if completed % 100 == 0 or completed == max_iters:
            val_loss = logistic_loss(y_val_01, tx_val, w)
            assert np.isfinite(val_loss)
            history.append((completed, train_loss, val_loss))
            print(
                f"\nStep {completed}: training loss={train_loss:.5f}, "
                f"validation loss={val_loss:.5f}",
                flush=True,
            )

        # Keep the 1000-step result for comparison with the final result.
        if completed == 1000 and max_iters > 1000:
            predictions = np.where(sigmoid(tx_val @ w) >= threshold, 1, -1)
            comparison.append({
                "iteration": completed, "gamma": gamma, "threshold": threshold,
                "training_loss": train_loss, "validation_loss": val_loss,
                **classification_metrics(y_val, predictions),
            })

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
    print("validation loss:", val_loss)
    print_metrics("logistic training", train_metrics)
    print_metrics("logistic validation", val_metrics)

    comparison.append({
        "iteration": max_iters, "gamma": gamma, "threshold": threshold,
        "training_loss": train_loss, "validation_loss": val_loss,
        **val_metrics,
    })
    save_logistic_history(history, comparison)

    # Tune the decision threshold on validation after training is complete.
    threshold_results = evaluate_thresholds(y_val, val_probabilities)
    save_threshold_results(threshold_results)

    return w, train_loss


def main():
    # Use False here when we are ready for the full labelled dataset.
    sub_sample = False
    # Set False to use the original preprocessing.
    use_feature_rules = True

    # Load the data once.
    print("Loading CSV files...", flush=True)
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=sub_sample
    )
    feature_names = load_feature_names(data_path / "x_train.csv")
    print("sub_sample:", sub_sample)
    print("use_feature_rules:", use_feature_rules)
    print("labelled rows:", len(y_train))

    # Prepare the same training and validation data for all models.
    print("\nPreparing training and validation data...", flush=True)
    tx_fit, y_fit, tx_val, y_val = prepare_data(
        x_train, y_train,
        feature_names=feature_names, use_feature_rules=use_feature_rules,
    )
    print("training shape (including bias):", tx_fit.shape)
    print("validation shape (including bias):", tx_val.shape)

    print("\nEvaluating baseline...", flush=True)
    evaluate_baseline(y_fit, y_val)
    train_logistic(
        tx_fit, y_fit, tx_val, y_val,
        max_iters=5000, gamma=0.01, threshold=0.5,
    )
    print("\nFinished.", flush=True)


if __name__ == "__main__":
    main()
