import numpy as np
from helpers import load_csv_data


def stratified_split(y, validation_fraction=0.20, seed=42):
    rng = np.random.default_rng(seed)

    negative_indices = np.where(y == -1)[0]
    positive_indices = np.where(y == 1)[0]

    rng.shuffle(negative_indices)
    rng.shuffle(positive_indices)

    n_negative_val = int(validation_fraction * len(negative_indices))
    n_positive_val = int(validation_fraction * len(positive_indices))

    val_indices = np.concatenate((
        negative_indices[:n_negative_val],
        positive_indices[:n_positive_val],
    ))

    train_indices = np.concatenate((
        negative_indices[n_negative_val:],
        positive_indices[n_positive_val:],
    ))

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def fit_imputer(x):
    # Keep columns with at least one observed training value.
    keep_columns = ~np.all(np.isnan(x), axis=0)

    if not np.any(keep_columns):
        raise ValueError("All feature columns are missing.")

    medians = np.nanmedian(x[:, keep_columns], axis=0)

    return keep_columns, medians


def apply_imputer(x, keep_columns, medians):
    # Use the columns and medians learned from training.
    x_clean = x[:, keep_columns].copy()
    return np.where(np.isnan(x_clean), medians, x_clean)


# Load a training sample for development.
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "dataset", sub_sample=True
)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)

labels, counts = np.unique(y_train, return_counts=True)
print("labels:", labels)
print("counts:", counts)
print("positive fraction:", np.mean(y_train == 1))

missing_per_column = np.mean(np.isnan(x_train), axis=0)

print("overall missing fraction:", np.mean(np.isnan(x_train)))
print("columns entirely missing:", np.sum(missing_per_column == 1))
print("columns over 50% missing:", np.sum(missing_per_column > 0.5))


# Split before learning any preprocessing statistics.
train_indices, val_indices = stratified_split(y_train)

x_fit = x_train[train_indices]
y_fit = y_train[train_indices]
x_val = x_train[val_indices]
y_val = y_train[val_indices]

assert np.intersect1d(train_indices, val_indices).size == 0

all_indices = np.concatenate((train_indices, val_indices))
np.testing.assert_array_equal(
    np.sort(all_indices),
    np.arange(len(y_train)),
)

train_again, val_again = stratified_split(y_train)
np.testing.assert_array_equal(train_indices, train_again)
np.testing.assert_array_equal(val_indices, val_again)

print("training shape:", x_fit.shape)
print("validation shape:", x_val.shape)
print("training positive fraction:", np.mean(y_fit == 1))
print("validation positive fraction:", np.mean(y_val == 1))


# Learn medians from training and apply them to both sets.
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
print("All checks passed.")