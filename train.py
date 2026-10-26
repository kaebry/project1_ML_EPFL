import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boosting import apply_bins, fit_bins, fit_boosting, predict_boosting
from helpers import load_csv_data
from implementations import (
    logistic_loss, logistic_regression, ridge_regression, sigmoid,
)
from preprocessing import (
    cleaned_feature_names, load_feature_names, prepare_data, split_data,
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


def evaluate_thresholds(y_true, scores, thresholds):
    # Scores can be probabilities or unrestricted linear scores.
    if not np.isfinite(scores).all():
        raise ValueError("Expected finite scores.")

    results = []
    # Compare all thresholds using the same scores, without retraining.
    for threshold in thresholds:
        predictions = np.where(scores >= threshold, 1, -1)
        results.append({
            "threshold": threshold,
            **classification_metrics(y_true, predictions),
        })

    return results


def save_threshold_results(results, name, title, default_threshold):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # The name keeps each model's files separate, e.g. "logistic" or "ridge".
    csv_path = results_dir / f"{name}_threshold_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "threshold", "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        writer.writerows(results)

    default = next(
        row for row in results if row["threshold"] == default_threshold
    )
    best = max(results, key=lambda row: row["f1"])
    thresholds = [row["threshold"] for row in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(thresholds, [row[metric] for row in results], label=metric)
    ax.axvline(
        default_threshold, color="gray", linestyle=":",
        label=f"Default: {default_threshold:.2f}",
    )
    ax.axvline(
        best["threshold"], color="tab:green", linestyle="--",
        label=f"Best tested: {best['threshold']:.2f}",
    )
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Validation score")
    ax.set_title(f"{title} threshold comparison")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plot_path = figures_dir / f"{name}_thresholds.png"
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
    assert np.all((val_probabilities >= 0) & (val_probabilities <= 1))
    # Probability thresholds 0.01, 0.02, ..., 0.99.
    thresholds = [percent / 100 for percent in range(1, 100)]
    threshold_results = evaluate_thresholds(y_val, val_probabilities, thresholds)
    save_threshold_results(
        threshold_results, "logistic", "Logistic regression", 0.5
    )

    return w, train_loss


def save_ridge_results(rows):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # One row per lambda, with metrics at that lambda's best threshold.
    csv_path = results_dir / "ridge_lambda_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "lambda", "training_loss", "validation_loss", "default_f1",
            "threshold", "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        writer.writerows(rows)

    lambdas = [row["lambda"] for row in rows]
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: loss without the penalty term, to see over- or underfitting.
    loss_ax.plot(
        lambdas, [row["training_loss"] for row in rows],
        marker="o", label="Training", color="tab:blue",
    )
    loss_ax.plot(
        lambdas, [row["validation_loss"] for row in rows],
        marker="o", label="Validation", color="tab:orange", linestyle="--",
    )
    loss_ax.set_xscale("log")
    loss_ax.set_xlabel("lambda")
    loss_ax.set_ylabel("MSE with factor 1/2 (-1/+1 targets)")
    loss_ax.set_title("Ridge regression loss")
    loss_ax.grid(alpha=0.2)
    loss_ax.legend()

    # Right: validation F1 at score 0 and at the best tested threshold.
    f1_ax.plot(
        lambdas, [row["default_f1"] for row in rows],
        marker="o", label="Threshold 0", color="gray",
    )
    f1_ax.plot(
        lambdas, [row["f1"] for row in rows],
        marker="o", label="Best tested threshold", color="tab:green",
    )
    # Label only the selected lambda, so the small differences stay readable.
    best = max(rows, key=lambda row: row["f1"])
    f1_ax.annotate(
        f"{best['f1']:.4f}", (best["lambda"], best["f1"]),
        textcoords="offset points", xytext=(0, 8), ha="center",
    )
    f1_ax.set_xscale("log")
    f1_ax.set_ylim(0, max(row["f1"] for row in rows) * 1.15)
    f1_ax.set_xlabel("lambda")
    f1_ax.set_ylabel("Validation F1")
    f1_ax.set_title("Ridge regression validation F1")
    f1_ax.grid(alpha=0.2)
    f1_ax.legend()

    fig.tight_layout()
    plot_path = figures_dir / "ridge_lambda.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Saved lambda comparison:", csv_path, flush=True)
    print("Saved lambda plot:", plot_path, flush=True)


def tune_ridge(tx_fit, y_fit, tx_val, y_val, lambdas):
    """Fit ridge for each lambda and tune its score threshold on validation."""
    if len(lambdas) == 0 or min(lambdas) <= 0:
        raise ValueError("Expected at least one positive lambda.")

    # Ridge gives linear scores, not probabilities, so we tune a score
    # threshold: -1.00, -0.99, ..., 1.00 for the -1/+1 targets.
    thresholds = [step / 100 for step in range(-100, 101)]

    print("\nTraining ridge regression...", flush=True)
    rows = []
    weights = []
    all_threshold_results = []
    for lambda_ in lambdas:
        # Fit directly on the -1/+1 labels; validation is not used here.
        w, train_loss = ridge_regression(y_fit, tx_fit, lambda_)
        assert w.shape == (tx_fit.shape[1],)
        assert np.isfinite(w).all()

        val_scores = tx_val @ w
        val_loss = 0.5 * np.mean((y_val - val_scores) ** 2)

        # Compare the natural threshold 0 with the best tested threshold.
        threshold_results = evaluate_thresholds(y_val, val_scores, thresholds)
        default = next(row for row in threshold_results if row["threshold"] == 0)
        best = max(threshold_results, key=lambda row: row["f1"])

        rows.append({
            "lambda": lambda_, "training_loss": train_loss,
            "validation_loss": val_loss, "default_f1": default["f1"],
            **best,
        })
        weights.append(w)
        all_threshold_results.append(threshold_results)
        print(
            f"lambda={lambda_:g}: training loss={train_loss:.5f}, "
            f"validation loss={val_loss:.5f}, F1 at 0={default['f1']:.4f}, "
            f"best threshold={best['threshold']:.2f}, best F1={best['f1']:.4f}",
            flush=True,
        )

    # Choose the lambda with the highest tuned validation F1.
    best_index = int(np.argmax([row["f1"] for row in rows]))
    return rows, weights, all_threshold_results, best_index


def train_ridge(tx_fit, y_fit, tx_val, y_val, lambdas):
    rows, weights, all_threshold_results, best_index = tune_ridge(
        tx_fit, y_fit, tx_val, y_val, lambdas
    )
    best_lambda = rows[best_index]["lambda"]
    print(
        f"\nSelected lambda: {best_lambda:g} "
        f"(validation F1={rows[best_index]['f1']:.4f}, "
        f"threshold={rows[best_index]['threshold']:.2f})",
        flush=True,
    )

    save_ridge_results(rows)
    save_threshold_results(
        all_threshold_results[best_index], "ridge",
        f"Ridge regression (lambda={best_lambda:g})", 0.0,
    )

    return weights[best_index], rows[best_index]["training_loss"]


def save_feature_comparison(rows):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # One row per feature set, with its best lambda and threshold.
    csv_path = results_dir / "feature_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "feature_set", "columns", "lambda", "training_loss",
            "validation_loss", "default_f1", "threshold", "tp", "tn", "fp",
            "fn", "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        writer.writerows(rows)

    labels = [f"{row['feature_set']}\n({row['columns']} columns)" for row in rows]
    f1_values = [row["f1"] for row in rows]

    # Bars start at zero; the exact values are written on top of each bar.
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, f1_values, color="tab:blue", width=0.6)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in f1_values], padding=3)
    ax.set_ylim(0, max(f1_values) * 1.15)
    ax.set_ylabel("Validation F1 (best lambda and threshold)")
    ax.set_title("Ridge regression: effect of feature preprocessing")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plot_path = figures_dir / "feature_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\nFeature comparison (ridge, best lambda and threshold each):")
    print("Feature set           Columns  Lambda   Threshold  Precision  Recall     F1")
    for row in rows:
        print(
            f"{row['feature_set']:<21} {row['columns']:<8} {row['lambda']:<8g} "
            f"{row['threshold']:<10.2f} {row['precision']:<10.4f} "
            f"{row['recall']:<10.4f} {row['f1']:.4f}",
            flush=True,
        )
    print("Saved feature comparison:", csv_path, flush=True)
    print("Saved feature comparison plot:", plot_path, flush=True)


def compare_feature_sets(x, y, feature_names, lambdas):
    # Same split, model and lambdas; only the preprocessing changes.
    feature_sets = [
        # (name, use_feature_rules, use_one_hot)
        ("original", False, False),
        ("reviewed", True, False),
        ("original + one-hot", False, True),
        ("reviewed + one-hot", True, True),
    ]

    rows = []
    for name, use_feature_rules, use_one_hot in feature_sets:
        print(f"\nPreparing feature set: {name}", flush=True)
        tx_fit, y_fit, tx_val, y_val = prepare_data(
            x, y, feature_names=feature_names,
            use_feature_rules=use_feature_rules, use_one_hot=use_one_hot,
        )
        print("training shape (including bias):", tx_fit.shape)

        ridge_rows, _, _, best_index = tune_ridge(
            tx_fit, y_fit, tx_val, y_val, lambdas
        )
        rows.append({
            "feature_set": name, "columns": tx_fit.shape[1],
            **ridge_rows[best_index],
        })

        # Free the large arrays before preparing the next feature set.
        del tx_fit, tx_val

    save_feature_comparison(rows)
    return rows


def save_boosting_results(rows, histories):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # One row per depth, at its best number of trees and threshold.
    comparison_path = results_dir / "boosting_depth_comparison.csv"
    with comparison_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "depth", "n_trees", "training_loss", "validation_loss",
            "threshold", "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Losses after every tree, for every depth.
    curve_path = results_dir / "boosting_learning_curve.csv"
    with curve_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "trees", "training_loss", "validation_loss"])
        for depth, history in histories.items():
            writer.writerows((depth, *record) for record in history)

    # Ordered depths use one blue from light to dark; dashed = training.
    blues = ["#6baed6", "#2171b5", "#08306b", "#041a3a"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, row in zip(blues, rows):
        trees, training_losses, validation_losses = zip(*histories[row["depth"]])
        ax.plot(
            trees, validation_losses, color=color,
            label=f"Depth {row['depth']}, validation",
        )
        ax.plot(
            trees, training_losses, color=color, linestyle="--",
            label=f"Depth {row['depth']}, training",
        )
        # Mark the number of trees with the lowest validation loss.
        ax.plot(
            row["n_trees"], row["validation_loss"], marker="o",
            markersize=8, color=color,
        )
    # Leave out the first trees, whose large losses would hide the differences.
    all_losses = [
        loss for history in histories.values()
        for _, train_loss, val_loss in history[10:]
        for loss in (train_loss, val_loss)
    ]
    if all_losses:
        ax.set_ylim(min(all_losses) * 0.995, max(all_losses) * 1.005)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Mean logistic loss")
    ax.set_title("Boosted trees learning curves (dot: lowest validation loss)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plot_path = figures_dir / "boosting_loss.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\nBoosted trees comparison (best number of trees and threshold each):")
    print("Depth  Trees  Validation loss  Threshold  Precision  Recall     F1")
    for row in rows:
        print(
            f"{row['depth']:<6} {row['n_trees']:<6} "
            f"{row['validation_loss']:<16.5f} {row['threshold']:<10.2f} "
            f"{row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:.4f}",
            flush=True,
        )
    print("Saved depth comparison:", comparison_path, flush=True)
    print("Saved learning curves:", curve_path, flush=True)
    print("Saved loss plot:", plot_path, flush=True)


def save_feature_importance(trees, feature_names, top=15):
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Importance = total loss reduction (gain) of all splits on a feature.
    importance = np.zeros(len(feature_names))
    for tree in trees:
        internal = tree["feature"] >= 0
        np.add.at(importance, tree["feature"][internal], tree["gain"][internal])
    total = importance.sum()
    share = importance / total if total > 0 else importance
    order = np.argsort(importance)[::-1]

    csv_path = results_dir / "boosting_feature_importance.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "total_gain", "share"])
        for index in order:
            if importance[index] > 0:
                writer.writerow([feature_names[index], importance[index], share[index]])

    # Show only features the trees used, with the largest bar at the top.
    shown = np.array([i for i in order[:top] if importance[i] > 0][::-1], dtype=int)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        [feature_names[index] for index in shown], 100 * share[shown],
        color="tab:blue", height=0.6,
    )
    ax.bar_label(bars, labels=[f"{100 * share[i]:.1f}%" for i in shown], padding=3)
    ax.set_xlim(0, 100 * share[shown].max() * 1.15 if len(shown) > 0 else 1)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title(f"Boosted trees: top {len(shown)} features")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plot_path = figures_dir / "boosting_feature_importance.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Saved feature importance:", csv_path, flush=True)
    print("Saved feature importance plot:", plot_path, flush=True)


def train_boosting(
    x_fit, y_fit, x_val, y_val, feature_names, depths,
    n_trees=400, learning_rate=0.1,
):
    # Trees split on raw values, so no filling, scaling or one-hot is needed.
    print("\nBinning features for boosted trees...", flush=True)
    edges = fit_bins(x_fit)
    bins_fit = apply_bins(x_fit, edges)
    bins_val = apply_bins(x_val, edges)

    # Convert labels from -1/+1 to 0/1 for the logistic loss.
    y_fit_01 = (y_fit == 1).astype(float)
    y_val_01 = (y_val == 1).astype(float)
    # Probability thresholds 0.01, 0.02, ..., 0.99.
    thresholds = [percent / 100 for percent in range(1, 100)]

    rows = []
    histories = {}
    models = []
    for depth in depths:
        print(
            f"\nTraining boosted trees (depth {depth}, up to {n_trees} trees)...",
            flush=True,
        )
        trees, base_score, history, best_n_trees = fit_boosting(
            bins_fit, y_fit_01, bins_val, y_val_01,
            n_trees=n_trees, learning_rate=learning_rate, max_depth=depth,
        )
        histories[depth] = history

        # Keep the number of trees with the lowest validation loss.
        trees = trees[:best_n_trees]
        val_probabilities = sigmoid(
            predict_boosting(trees, base_score, learning_rate, bins_val)
        )
        threshold_results = evaluate_thresholds(
            y_val, val_probabilities, thresholds
        )
        best = max(threshold_results, key=lambda row: row["f1"])

        rows.append({
            "depth": depth, "n_trees": best_n_trees,
            "training_loss": history[best_n_trees][1],
            "validation_loss": history[best_n_trees][2],
            **best,
        })
        models.append((trees, base_score, threshold_results))
        print(
            f"depth={depth}: best number of trees={best_n_trees}, "
            f"validation loss={history[best_n_trees][2]:.5f}, "
            f"best threshold={best['threshold']:.2f}, best F1={best['f1']:.4f}",
            flush=True,
        )

    # Choose the depth with the highest tuned validation F1.
    best_index = int(np.argmax([row["f1"] for row in rows]))
    best_row = rows[best_index]
    trees, base_score, threshold_results = models[best_index]
    print(
        f"\nSelected depth: {best_row['depth']} with {best_row['n_trees']} trees "
        f"(validation F1={best_row['f1']:.4f}, "
        f"threshold={best_row['threshold']:.2f})",
        flush=True,
    )

    save_boosting_results(rows, histories)
    save_threshold_results(
        threshold_results, "boosting",
        f"Boosted trees (depth {best_row['depth']}, {best_row['n_trees']} trees)",
        0.5,
    )
    save_feature_importance(trees, feature_names)

    return trees, base_score, best_row


def main():
    # Use False here when we are ready for the full labelled dataset.
    sub_sample = False
    # Set False to use the original preprocessing.
    use_feature_rules = True
    # Set True to one-hot encode columns with at most 10 distinct values.
    use_one_hot = False
    # Choose which experiments to run.
    run_feature_comparison = False
    run_logistic = False
    run_ridge = False
    run_boosting = True
    ridge_lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    boosting_depths = [1, 3, 5]

    # Load the data once.
    print("Loading CSV files...", flush=True)
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=sub_sample
    )
    feature_names = load_feature_names(data_path / "x_train.csv")
    print("sub_sample:", sub_sample)
    print("use_feature_rules:", use_feature_rules)
    print("use_one_hot:", use_one_hot)
    print("run_feature_comparison:", run_feature_comparison)
    print("run_logistic:", run_logistic)
    print("run_ridge:", run_ridge)
    print("run_boosting:", run_boosting)
    print("labelled rows:", len(y_train))

    # Compare feature sets first, before the main arrays use memory.
    if run_feature_comparison:
        compare_feature_sets(x_train, y_train, feature_names, ridge_lambdas)

    # Prepare the same training and validation data for all models.
    print("\nPreparing training and validation data...", flush=True)
    tx_fit, y_fit, tx_val, y_val = prepare_data(
        x_train, y_train,
        feature_names=feature_names, use_feature_rules=use_feature_rules,
        use_one_hot=use_one_hot,
    )
    print("training shape (including bias):", tx_fit.shape)
    print("validation shape (including bias):", tx_val.shape)

    print("\nEvaluating baseline...", flush=True)
    evaluate_baseline(y_fit, y_val)
    if run_logistic:
        train_logistic(
            tx_fit, y_fit, tx_val, y_val,
            max_iters=5000, gamma=0.01, threshold=0.5,
        )
    if run_ridge:
        train_ridge(tx_fit, y_fit, tx_val, y_val, lambdas=ridge_lambdas)
    if run_boosting:
        # Same split, but raw values with missing entries kept as NaN.
        print("\nSplitting raw data for boosted trees...", flush=True)
        x_fit_raw, y_fit_raw, x_val_raw, y_val_raw = split_data(
            x_train, y_train,
            feature_names=feature_names, use_feature_rules=use_feature_rules,
        )
        tree_feature_names = (
            cleaned_feature_names(feature_names) if use_feature_rules
            else feature_names
        )
        train_boosting(
            x_fit_raw, y_fit_raw, x_val_raw, y_val_raw, tree_feature_names,
            depths=boosting_depths, n_trees=400, learning_rate=0.1,
        )
    print("\nFinished.", flush=True)


if __name__ == "__main__":
    main()
