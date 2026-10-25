from pathlib import Path

import numpy as np

from helpers import load_csv_data


def main():
    # Load a small data sample.
    data_path = Path(__file__).resolve().parent / "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=True
    )

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
