import csv

import numpy as np


def load_feature_names(csv_path):
    # Read the header and remove the Id column.
    with open(csv_path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

    return header[1:]


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


def clean_known_features(x, feature_names):
    """Drop survey IDs, decode two features and append their missing indicators."""
    if len(feature_names) != x.shape[1]:
        raise ValueError("Feature names must match the input columns.")

    keep_columns = [name not in ("SEQNO", "_PSU") for name in feature_names]
    kept_names = [name for name in feature_names if name not in ("SEQNO", "_PSU")]
    x_clean = x[:, keep_columns]

    health_index = kept_names.index("GENHLTH")
    days_index = kept_names.index("PHYSHLTH")
    health = x_clean[:, health_index]
    days = x_clean[:, days_index]

    # Count both NaN and coded unavailable responses as missing.
    health_missing = np.isnan(health) | np.isin(health, [7, 9])
    days_missing = np.isnan(days) | np.isin(days, [77, 99])
    x_clean[health_missing, health_index] = np.nan
    x_clean[days_missing, days_index] = np.nan
    x_clean[x_clean[:, days_index] == 88, days_index] = 0

    # The last two columns flag missing GENHLTH and PHYSHLTH responses.
    return np.column_stack((x_clean, health_missing, days_missing))


def fit_one_hot(x, max_categories=10):
    """Find columns with few distinct training values and store those values."""
    categories = {}
    for column in range(x.shape[1]):
        missing = np.isnan(x[:, column])
        values = np.unique(x[~missing, column])

        # Small value sets are treated as categories, e.g. codes 1-5, 7 and 9.
        if 0 < len(values) <= max_categories:
            categories[column] = (values, bool(np.any(missing)))

    return categories


def apply_one_hot(x, categories):
    """Keep the other columns, then add one 0/1 column per training category."""
    numeric_columns = [
        column for column in range(x.shape[1]) if column not in categories
    ]
    n_indicators = sum(
        len(values) + has_missing for values, has_missing in categories.values()
    )

    # Fill one new array so the large matrix is not copied again.
    x_encoded = np.empty((len(x), len(numeric_columns) + n_indicators))
    x_encoded[:, :len(numeric_columns)] = x[:, numeric_columns]

    position = len(numeric_columns)
    for column, (values, has_missing) in categories.items():
        for value in values:
            x_encoded[:, position] = x[:, column] == value
            position += 1
        # Missing values get their own column when training had them.
        # Values never seen in training give zeros in all of this column's indicators.
        if has_missing:
            x_encoded[:, position] = np.isnan(x[:, column])
            position += 1

    return x_encoded


def fit_imputer(x):
    # Keep columns with at least one value.
    keep_columns = ~np.all(np.isnan(x), axis=0)

    if not np.any(keep_columns):
        raise ValueError("All feature columns are missing.")

    # Get the median of each kept column.
    medians = np.nanmedian(x[:, keep_columns], axis=0)

    return keep_columns, medians


def apply_imputer(x, keep_columns, medians):
    # Drop empty columns; boolean indexing returns a copy, so x is unchanged.
    x_clean = x[:, keep_columns]

    # Fill each column's missing values with its training median.
    for column in range(x_clean.shape[1]):
        values = x_clean[:, column]
        values[np.isnan(values)] = medians[column]

    return x_clean


def fit_scaler(x):
    # One mean and standard deviation per column.
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)

    # Replace zero standard deviations with 1.
    stds[stds == 0] = 1.0

    return means, stds


def apply_scaler(x, means, stds):
    # Create one new array and divide it in place to save memory.
    x_scaled = x - means
    x_scaled /= stds
    return x_scaled


def add_bias(x):
    # Add a column of ones for the intercept.
    return np.column_stack((np.ones(x.shape[0]), x))


def prepare_data(
    x, y, validation_fraction=0.20, seed=42,
    feature_names=None, use_feature_rules=False,
    use_one_hot=False, max_categories=10,
):
    """Return prepared training and validation arrays."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected x with shape (N, D) and y with shape (N,).")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Labels must be -1 or +1.")

    # Split the data into training and validation sets.
    print("  Splitting rows...", flush=True)
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

    if use_feature_rules:
        if feature_names is None:
            raise ValueError("Feature names are required for feature rules.")
        print("  Cleaning reviewed features...", flush=True)
        x_fit = clean_known_features(x_fit, feature_names)
        x_val = clean_known_features(x_val, feature_names)

    # Each step below reuses the names x_fit and x_val, so the previous
    # large arrays can be freed before the next step.
    if use_one_hot:
        # Learn the categories from training data, then encode both sets.
        categories = fit_one_hot(x_fit, max_categories)
        print(
            f"  One-hot encoding {len(categories)} columns "
            f"with at most {max_categories} values...",
            flush=True,
        )
        x_fit = apply_one_hot(x_fit, categories)
        x_val = apply_one_hot(x_val, categories)

    # Learn medians from training data, then clean both sets.
    print("  Filling missing values...", flush=True)
    keep_columns, medians = fit_imputer(x_fit)
    x_fit = apply_imputer(x_fit, keep_columns, medians)
    x_val = apply_imputer(x_val, keep_columns, medians)

    # Learn scaling from training data, then scale both sets.
    print("  Standardizing features...", flush=True)
    means, stds = fit_scaler(x_fit)
    x_fit = apply_scaler(x_fit, means, stds)
    x_val = apply_scaler(x_val, means, stds)
    assert np.isfinite(x_fit).all()
    assert np.isfinite(x_val).all()

    # Add the intercept after scaling.
    print("  Adding bias...", flush=True)
    tx_fit = add_bias(x_fit)
    tx_val = add_bias(x_val)

    return tx_fit, y_fit, tx_val, y_val
