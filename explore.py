import numpy as np
from helpers import load_csv_data

# sub_sample is true this is why it does not load the whole train
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "dataset", sub_sample=True
)

# Check the sizes.
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)

# Count each class.
labels, counts = np.unique(y_train, return_counts=True)
print("labels:", labels)
print("counts:", counts)
print("positive fraction:", np.mean(y_train == 1))

# Inspect missing values.
missing_per_column = np.mean(np.isnan(x_train), axis=0)

print("overall missing fraction:", np.mean(np.isnan(x_train)))
print("columns entirely missing:", np.sum(missing_per_column == 1))
print("columns over 50% missing:", np.sum(missing_per_column > 0.5))

