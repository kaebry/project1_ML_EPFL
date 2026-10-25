import csv
from pathlib import Path

import numpy as np

from helpers import load_csv_data
from preprocessing import stratified_split


def load_feature_names(csv_path):
    # Read the header and remove the Id column.
    with open(csv_path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

    return header[1:]


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


if __name__ == "__main__":
    main()
