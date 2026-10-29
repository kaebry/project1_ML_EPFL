"""Extra tuning experiment for the boosted trees.

This file is kept separate from the final pipeline: train.py, run.py and the
submission files are not changed by it. On the same stratified validation
split as train.py, it compares:

1. the current model: depth 3, learning rate 0.1 (up to 400 trees);
2. depth 4, learning rate 0.1 (up to 400 trees);
3. depth 3, learning rate 0.05 (up to 800 trees);
4. depth 4, learning rate 0.05 (up to 800 trees);
5. the average of 3 seeds of the option with the lowest validation loss;
6. a blend of that average with Ridge + one-hot (the run_ridge.py model).

The number of trees is always chosen by the lowest validation loss. The F1
after every 50 trees is only recorded as a diagnostic. Results are written to
results/tuning/ and figures/tuning/.
"""
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boosting import (
    apply_bins,
    boosting_loss,
    fit_bins,
    fit_boosting,
    predict_boosting,
    predict_tree,
)
from helpers import load_csv_data
from implementations import ridge_regression, sigmoid
from preprocessing import load_feature_names, prepare_data, split_data
from train import evaluate_thresholds

# (name, depth, learning rate, maximum number of trees)
CONFIGS = [
    ("depth 3, lr 0.1", 3, 0.1, 400),
    ("depth 4, lr 0.1", 4, 0.1, 400),
    ("depth 3, lr 0.05", 3, 0.05, 800),
    ("depth 4, lr 0.05", 4, 0.05, 800),
]
SEEDS = [42, 43, 44]
RIDGE_LAMBDA = 1e-5
BLEND_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Probability thresholds 0.01, ..., 0.99 for the trees.
PROBABILITY_THRESHOLDS = [percent / 100 for percent in range(1, 100)]
# Thresholds -2.00, ..., 6.00 for the blended (standardized) scores.
BLEND_THRESHOLDS = [step / 100 for step in range(-200, 601)]


def best_f1(y_val, scores, thresholds):
    # The threshold with the highest validation F1, with its metrics.
    results = evaluate_thresholds(y_val, scores, thresholds)
    return max(results, key=lambda row: row["f1"])


def logit(probabilities):
    # Log-odds; clipping avoids log(0).
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def checkpoint_curve(name, trees, base_score, learning_rate, bins_val, y_val, step=50):
    """Validation loss and best-threshold F1 every `step` trees (diagnostic)."""
    y_val_01 = (y_val == 1).astype(float)
    scores = np.full(bins_val.shape[1], base_score)
    rows = []
    for n, tree in enumerate(trees, start=1):
        scores += learning_rate * predict_tree(tree, bins_val)
        if n % step == 0 or n == len(trees):
            best = best_f1(y_val, sigmoid(scores), PROBABILITY_THRESHOLDS)
            rows.append(
                {
                    "option": name,
                    "trees": n,
                    "validation_loss": boosting_loss(y_val_01, scores),
                    "f1": best["f1"],
                }
            )
    return rows


def ridge_scores(x, y, feature_names):
    """Ridge + one-hot on the same split: training and validation scores."""
    print("\nRidge + one-hot on the validation split...", flush=True)
    tx_fit, y_fit, tx_val, y_val = prepare_data(
        x,
        y,
        feature_names=feature_names,
        use_feature_rules=True,
        use_one_hot=True,
    )
    w, _ = ridge_regression(y_fit, tx_fit, RIDGE_LAMBDA)
    return tx_fit @ w, tx_val @ w, y_val


def blend_scores(tree_fit, tree_val, ridge_fit, ridge_val, weight):
    """Weighted sum of both models' validation scores on a common scale.

    Each model's scores are standardized with the mean and standard deviation
    of its scores on the training rows, so the same rule works for test rows.
    """
    z_tree = (tree_val - tree_fit.mean()) / tree_fit.std()
    z_ridge = (ridge_val - ridge_fit.mean()) / ridge_fit.std()
    return (1 - weight) * z_tree + weight * z_ridge


def save_tuning_results(rows, curves, blend_rows, configs):
    root = Path(__file__).resolve().parent
    results_dir = root / "results" / "tuning"
    figures_dir = root / "figures" / "tuning"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    comparison_path = results_dir / "tuning_comparison.csv"
    with comparison_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "option",
                "depth",
                "learning_rate",
                "seeds",
                "n_trees",
                "validation_loss",
                *metric_names,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    curve_path = results_dir / "tuning_checkpoints.csv"
    with curve_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["option", "trees", "validation_loss", "f1"]
        )
        writer.writeheader()
        for curve in curves.values():
            writer.writerows(curve)

    blend_path = results_dir / "tuning_blend.csv"
    with blend_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ridge_weight", *metric_names])
        writer.writeheader()
        writer.writerows(blend_rows)

    # Diagnostic curves: color = depth, line style = learning rate.
    depth_colors = {3: "#2171b5", 4: "#d95f02"}
    styles = {0.1: "-", 0.05: "--"}
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, depth, learning_rate, _ in configs:
        curve = curves[name]
        trees = [row["trees"] for row in curve]
        color = depth_colors.get(depth, "gray")
        style = styles.get(learning_rate, ":")
        loss_ax.plot(
            trees,
            [row["validation_loss"] for row in curve],
            color=color,
            linestyle=style,
            label=name,
        )
        f1_ax.plot(
            trees,
            [row["f1"] for row in curve],
            color=color,
            linestyle=style,
            label=name,
        )
    loss_ax.set_xlabel("Number of trees")
    loss_ax.set_ylabel("Validation logistic loss")
    loss_ax.set_title("Validation loss (used to choose the trees)")
    f1_ax.set_xlabel("Number of trees")
    f1_ax.set_ylabel("Validation F1 (best threshold)")
    f1_ax.set_title("Validation F1 (diagnostic only)")
    # Leave out the first checkpoint so the differences stay visible.
    later = [row for curve in curves.values() for row in curve if row["trees"] > 50]
    if later:
        losses = [row["validation_loss"] for row in later]
        f1s = [row["f1"] for row in later]
        loss_ax.set_ylim(min(losses) * 0.998, max(losses) * 1.002)
        f1_ax.set_ylim(min(f1s) - 0.005, max(f1s) + 0.005)
    for ax in (loss_ax, f1_ax):
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    curve_plot = figures_dir / "tuning_checkpoints.png"
    fig.savefig(curve_plot, dpi=150)
    plt.close(fig)

    # One bar per option, largest at the top; bars start at zero.
    ordered = sorted(rows, key=lambda row: row["f1"])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [row["option"] for row in ordered],
        [row["f1"] for row in ordered],
        color="#2171b5",
        height=0.6,
    )
    ax.bar_label(bars, labels=[f"{row['f1']:.4f}" for row in ordered], padding=3)
    ax.set_xlim(0, max(row["f1"] for row in rows) * 1.15)
    ax.set_xlabel("Validation F1 (best threshold)")
    ax.set_title("Boosted trees tuning on the validation split")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    bar_plot = figures_dir / "tuning_comparison.png"
    fig.savefig(bar_plot, dpi=150)
    plt.close(fig)

    print("\nTuning comparison (validation split):")
    print(
        f"{'Option':<40} {'Trees':<13} {'Val. loss':<10} {'Thr.':<6} "
        f"{'Precision':<10} {'Recall':<10} F1"
    )
    for row in rows:
        loss = row["validation_loss"]
        loss_text = f"{loss:.5f}" if loss != "" else "-"
        print(
            f"{row['option']:<40} {str(row['n_trees']):<13} {loss_text:<10} "
            f"{row['threshold']:<6.2f} {row['precision']:<10.4f} "
            f"{row['recall']:<10.4f} {row['f1']:.4f}",
            flush=True,
        )
    for path in (comparison_path, curve_path, blend_path, curve_plot, bar_plot):
        print("Saved:", path, flush=True)


def run_tuning(x, y, feature_names, configs, seeds):
    # Ridge first, so its large one-hot matrices are freed before the trees.
    ridge_fit, ridge_val, y_val_ridge = ridge_scores(x, y, feature_names)

    # The same split as train.py; trees use raw values with NaN kept.
    print("\nSplitting raw data for boosted trees...", flush=True)
    x_fit, y_fit, x_val, y_val = split_data(
        x,
        y,
        feature_names=feature_names,
        use_feature_rules=True,
    )
    assert np.array_equal(y_val, y_val_ridge)
    edges = fit_bins(x_fit)
    bins_fit = apply_bins(x_fit, edges)
    bins_val = apply_bins(x_val, edges)
    y_fit_01 = (y_fit == 1).astype(float)
    y_val_01 = (y_val == 1).astype(float)

    def train(depth, learning_rate, n_trees, seed):
        start = time.time()
        trees, base_score, history, best_n_trees = fit_boosting(
            bins_fit,
            y_fit_01,
            bins_val,
            y_val_01,
            n_trees=n_trees,
            learning_rate=learning_rate,
            max_depth=depth,
            seed=seed,
        )
        print(
            f"  ({time.time() - start:.0f} s, best number of trees " f"{best_n_trees})",
            flush=True,
        )
        return trees, base_score, history, best_n_trees

    rows = []
    curves = {}
    models = {}
    for name, depth, learning_rate, n_trees in configs:
        print(
            f"\nTraining {name}, up to {n_trees} trees (seed {seeds[0]})...", flush=True
        )
        trees, base_score, history, best_n_trees = train(
            depth, learning_rate, n_trees, seeds[0]
        )
        curves[name] = checkpoint_curve(
            name, trees, base_score, learning_rate, bins_val, y_val
        )
        trees = trees[:best_n_trees]
        probabilities = sigmoid(
            predict_boosting(trees, base_score, learning_rate, bins_val)
        )
        rows.append(
            {
                "option": name,
                "depth": depth,
                "learning_rate": learning_rate,
                "seeds": str(seeds[0]),
                "n_trees": best_n_trees,
                "validation_loss": history[best_n_trees][2],
                **best_f1(y_val, probabilities, PROBABILITY_THRESHOLDS),
            }
        )
        models[name] = [(trees, base_score)]

    # The option with the lowest validation loss gets two more seeds.
    best_row = min(rows, key=lambda row: row["validation_loss"])
    name, depth, learning_rate = (
        best_row["option"],
        best_row["depth"],
        best_row["learning_rate"],
    )
    n_trees = next(c[3] for c in configs if c[0] == name)
    tree_counts = [best_row["n_trees"]]
    for seed in seeds[1:]:
        print(f"\nTraining {name} again with seed {seed}...", flush=True)
        trees, base_score, _, best_n_trees = train(depth, learning_rate, n_trees, seed)
        models[name].append((trees[:best_n_trees], base_score))
        tree_counts.append(best_n_trees)

    # Average the probabilities of the seeds, on validation and training rows.
    def average_probability(bins):
        return np.mean(
            [
                sigmoid(predict_boosting(trees, base_score, learning_rate, bins))
                for trees, base_score in models[name]
            ],
            axis=0,
        )

    val_average = average_probability(bins_val)
    fit_average = average_probability(bins_fit)
    rows.append(
        {
            "option": f"{len(seeds)}-seed average of {name}",
            "depth": depth,
            "learning_rate": learning_rate,
            "seeds": "/".join(str(seed) for seed in seeds),
            "n_trees": "/".join(str(count) for count in tree_counts),
            "validation_loss": boosting_loss(y_val_01, logit(val_average)),
            **best_f1(y_val, val_average, PROBABILITY_THRESHOLDS),
        }
    )

    # Blend the seed average with Ridge + one-hot on a common scale.
    print("\nBlending the trees with Ridge + one-hot...", flush=True)
    blend_rows = []
    for weight in BLEND_WEIGHTS:
        blended = blend_scores(
            logit(fit_average), logit(val_average), ridge_fit, ridge_val, weight
        )
        blend_rows.append(
            {
                "ridge_weight": weight,
                **best_f1(y_val, blended, BLEND_THRESHOLDS),
            }
        )
        print(
            f"  Ridge weight {weight:.1f}: best F1={blend_rows[-1]['f1']:.4f}",
            flush=True,
        )
    best_blend = max(blend_rows, key=lambda row: row["f1"])
    rows.append(
        {
            "option": f"blend with Ridge (weight {best_blend['ridge_weight']:.1f})",
            "depth": depth,
            "learning_rate": learning_rate,
            "seeds": "/".join(str(seed) for seed in seeds),
            "n_trees": "/".join(str(count) for count in tree_counts),
            "validation_loss": "",
            **{
                key: value for key, value in best_blend.items() if key != "ridge_weight"
            },
        }
    )

    save_tuning_results(rows, curves, blend_rows, configs)
    return rows, curves, blend_rows


def main():
    print("Loading CSV files...", flush=True)
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, _, y_train, _, _ = load_csv_data(data_path)
    feature_names = load_feature_names(data_path / "x_train.csv")
    print("labelled rows:", len(y_train))
    run_tuning(x_train, y_train, feature_names, CONFIGS, SEEDS)
    print("\nFinished.", flush=True)


if __name__ == "__main__":
    main()
