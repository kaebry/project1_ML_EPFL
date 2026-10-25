import numpy as np


def stratified_split(y, validation_fraction=0.20, seed=42):
    # Make the split repeatable.
    rng = np.random.default_rng(seed)

    # Get the indices for each class.
    negative_indices = np.where(y == -1)[0]
    positive_indices = np.where(y == 1)[0]

    # Shuffle each class separately.
    rng.shuffle(negative_indices)
    rng.shuffle(positive_indices)

    # Choose validation rows from each class.
    n_negative_val = int(validation_fraction * len(negative_indices))
    n_positive_val = int(validation_fraction * len(positive_indices))

    # Combine the validation and training indices.
    val_indices = np.concatenate((
        negative_indices[:n_negative_val],
        positive_indices[:n_positive_val],
    ))

    train_indices = np.concatenate((
        negative_indices[n_negative_val:],
        positive_indices[n_positive_val:],
    ))

    # Shuffle the final groups.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def fit_imputer(x):
    # Keep columns with at least one value.
    keep_columns = ~np.all(np.isnan(x), axis=0)

    if not np.any(keep_columns):
        raise ValueError("All feature columns are missing.")

    # Get the median of each kept column.
    medians = np.nanmedian(x[:, keep_columns], axis=0)

    return keep_columns, medians


def apply_imputer(x, keep_columns, medians):
    # Drop empty columns and fill missing values.
    x_clean = x[:, keep_columns].copy()
    return np.where(np.isnan(x_clean), medians, x_clean)


def fit_scaler(x):
    # One mean and standard deviation per column.
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)

    # Replace zero standard deviations with 1.
    stds[stds == 0] = 1.0

    return means, stds


def apply_scaler(x, means, stds):
    return (x - means) / stds


def add_bias(x):
    # Add a column of ones for the intercept.
    return np.column_stack((np.ones(x.shape[0]), x))
