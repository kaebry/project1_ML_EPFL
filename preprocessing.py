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


def prepare_data(x, y, validation_fraction=0.20, seed=42):
    """Return prepared training and validation arrays."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected x with shape (N, D) and y with shape (N,).")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Labels must be -1 or +1.")

    # Split the data into training and validation sets.
    train_indices, val_indices = stratified_split(y, validation_fraction, seed)
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("The split must contain training and validation rows.")

    x_fit, y_fit = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]

    # Check that the split uses each row once.
    assert np.intersect1d(train_indices, val_indices).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train_indices, val_indices))),
        np.arange(len(y)),
    )

    # Learn medians from training data, then clean both sets.
    keep_columns, medians = fit_imputer(x_fit)
    x_fit_clean = apply_imputer(x_fit, keep_columns, medians)
    x_val_clean = apply_imputer(x_val, keep_columns, medians)

    # Learn scaling from training data, then scale both sets.
    means, stds = fit_scaler(x_fit_clean)
    x_fit_scaled = apply_scaler(x_fit_clean, means, stds)
    x_val_scaled = apply_scaler(x_val_clean, means, stds)
    assert np.isfinite(x_fit_scaled).all()
    assert np.isfinite(x_val_scaled).all()

    # Add the intercept after scaling.
    tx_fit = add_bias(x_fit_scaled)
    tx_val = add_bias(x_val_scaled)

    return tx_fit, y_fit, tx_val, y_val
