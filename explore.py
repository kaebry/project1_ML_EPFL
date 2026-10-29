from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers import load_csv_data
from preprocessing import load_feature_names, stratified_split


def print_feature_summary(x, feature_names):
    # Describe observed values without changing the data.
    print("\nFeature summary (raw training data):")
    print("Missing percentages count NaN values only.")

    for index, name in enumerate(feature_names):
        column = x[:, index]
        missing = np.isnan(column)
        values, counts = np.unique(column[~missing], return_counts=True)

        print(f"\n[{index}] {name}")
        print(f"  missing: {np.mean(missing):.2%}")
        print("  distinct values:", len(values))

        # An entirely missing column has no minimum or maximum.
        if len(values) == 0:
            print("  min: N/A, max: N/A")
            continue

        print(f"  min: {values[0]}, max: {values[-1]}")

        # Show exact counts when the list is short.
        if len(values) <= 20:
            value_counts = ", ".join(
                f"{value:g}: {count}" for value, count in zip(values, counts)
            )
            print("  value counts:", value_counts)


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved plot:", path, flush=True)


def plot_class_balance(y, output_dir):
    labels, counts = np.unique(y, return_counts=True)
    percentages = 100 * counts / len(y)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [f"Label {label:+d}" for label in labels],
        percentages,
        color=["tab:blue", "tab:orange"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,} rows\n{percentage:.2f}%"
            for count, percentage in zip(counts, percentages)
        ],
        padding=4,
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Training rows (%)")
    ax.set_title("Class balance (training split)")
    save_plot(fig, output_dir / "class_balance.png")


def plot_missing_values(x, feature_names, output_dir):
    missing_percentages = 100 * np.mean(np.isnan(x), axis=0)
    top_indices = np.argsort(missing_percentages)[-20:]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        [feature_names[index] for index in top_indices],
        missing_percentages[top_indices],
        color="tab:blue",
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Training rows with NaN (%)")
    ax.set_title("20 features with the most missing values (NaN only)")
    save_plot(fig, output_dir / "missing_values.png")


def plot_numeric_by_class(values, y, bins, title, xlabel, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label, color in [(-1, "tab:blue"), (1, "tab:orange")]:
        class_values = values[y == label]
        observed = class_values[~np.isnan(class_values)]
        if len(observed) == 0:
            continue

        missing_percentage = 100 * np.mean(np.isnan(class_values))
        # Normalize each class separately so the larger class does not dominate.
        weights = np.full(len(observed), 100 / len(observed))
        ax.hist(
            observed,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"Label {label:+d} (NaN: {missing_percentage:.1f}%)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observed values within each class (%) per bin")
    ax.set_title(title)
    ax.legend()
    save_plot(fig, output_path)


def plot_categories_by_class(values, y, codes, category_labels, title, output_path):
    # Keep unavailable responses visible as a separate category.
    positions = np.arange(len(codes) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for label, offset, color in [(-1, -0.2, "tab:blue"), (1, 0.2, "tab:orange")]:
        class_values = values[y == label]
        if len(class_values) == 0:
            continue

        counts = [np.sum(class_values == code) for code in codes]
        counts.append(np.sum(~np.isin(class_values, codes)))
        percentages = 100 * np.array(counts) / len(class_values)
        ax.bar(
            positions + offset,
            percentages,
            width=0.4,
            color=color,
            label=f"Label {label:+d}",
        )

    ax.set_xticks(positions, category_labels + ["Unavailable"])
    ax.set_ylabel("Rows within each class (%)")
    ax.set_title(title)
    ax.legend()
    save_plot(fig, output_path)


def main():
    # Load the full dataset.
    print("Loading CSV files...", flush=True)
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=False
    )

    # Match feature names to the data columns.
    feature_names = load_feature_names(data_path / "x_train.csv")
    assert len(feature_names) == x_train.shape[1]
    assert feature_names[0] == "_STATE"

    print("feature count:", len(feature_names))
    for index, name in enumerate(feature_names[:10]):
        print(index, name)

    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)

    # Inspect the same training split used by train.py.
    train_indices, val_indices = stratified_split(
        y_train, validation_fraction=0.20, seed=42
    )
    x_fit = x_train[train_indices]
    y_fit = y_train[train_indices]

    print("\nInspecting raw training features...", flush=True)
    print("training rows inspected:", len(y_fit))
    print("validation rows held aside:", len(val_indices))

    # Check the class balance.
    labels, counts = np.unique(y_fit, return_counts=True)
    print("labels:", labels)
    print("counts:", counts)
    print("positive fraction:", np.mean(y_fit == 1))

    # Check for missing values.
    missing_per_column = np.mean(np.isnan(x_fit), axis=0)

    print("overall missing fraction:", np.mean(np.isnan(x_fit)))
    print("columns entirely missing:", np.sum(missing_per_column == 1))
    print("columns over 50% missing:", np.sum(missing_per_column > 0.5))

    print_feature_summary(x_fit, feature_names)

    # Save exploration plots using only the training split.
    print("\nCreating plots...", flush=True)
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plot_class_balance(y_fit, output_dir)
    plot_missing_values(x_fit, feature_names, output_dir)

    plot_numeric_by_class(
        x_fit[:, feature_names.index("_AGE80")],
        y_fit,
        bins=np.arange(17.5, 81.5),
        title="Age distribution (training split)",
        xlabel="Age (years; 80 means 80+)",
        output_path=output_dir / "age_distribution.png",
    )
    plot_numeric_by_class(
        x_fit[:, feature_names.index("_BMI5")],
        y_fit,
        bins=np.arange(10, 101, 2),
        title="BMI distribution (training split)",
        xlabel="BMI (kg/m²)",
        output_path=output_dir / "bmi_distribution.png",
    )

    plot_categories_by_class(
        x_fit[:, feature_names.index("GENHLTH")],
        y_fit,
        codes=[1, 2, 3, 4, 5],
        category_labels=["Excellent", "Very good", "Good", "Fair", "Poor"],
        title="Self-rated health (training split)",
        output_path=output_dir / "general_health.png",
    )
    plot_categories_by_class(
        x_fit[:, feature_names.index("SEX")],
        y_fit,
        codes=[1, 2],
        category_labels=["Male", "Female"],
        title="Recorded sex (training split)",
        output_path=output_dir / "sex_distribution.png",
    )


if __name__ == "__main__":
    main()
