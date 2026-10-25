import csv
from pathlib import Path

import numpy as np

from helpers import load_csv_data


def load_feature_names(csv_path):
    # Read the header and remove the Id column.
    with open(csv_path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

    return header[1:]


def main():
    # Load a small data sample.
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=True
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

    # Check the class balance.
    labels, counts = np.unique(y_train, return_counts=True)
    print("labels:", labels)
    print("counts:", counts)
    print("positive fraction:", np.mean(y_train == 1))

    # Check for missing values.
    missing_per_column = np.mean(np.isnan(x_train), axis=0)

    print("overall missing fraction:", np.mean(np.isnan(x_train)))
    print("columns entirely missing:", np.sum(missing_per_column == 1))
    print("columns over 50% missing:", np.sum(missing_per_column > 0.5))


if __name__ == "__main__":
    main()
