"""Gradient-boosted decision trees for the logistic loss, using only NumPy.

The idea follows XGBoost: each feature is grouped into bins, each tree is
grown level by level from sums of gradients and hessians, and each leaf gets
a Newton step. Labels are 0/1, as in logistic regression.
"""
import numpy as np

from implementations import sigmoid

# Values use bins 0..63; missing values use their own bin, 64.
MAX_BINS = 64
MISSING_BIN = MAX_BINS


def fit_bins(x):
    """Learn bin edges for each feature from the training rows."""
    edges = []
    for column in range(x.shape[1]):
        observed = x[~np.isnan(x[:, column]), column]
        unique_values = np.unique(observed)

        if len(unique_values) < MAX_BINS:
            # Few distinct values: each value gets its own bin.
            edges.append(unique_values)
        else:
            # Many values: quantile edges give bins with similar row counts.
            levels = np.arange(MAX_BINS - 1) / (MAX_BINS - 1)
            edges.append(np.unique(np.quantile(observed, levels)))

    return edges


def apply_bins(x, edges):
    """Replace each value by its bin number; returns shape (features, rows)."""
    # Feature-major layout, so bins[f] holds feature f for all rows.
    bins = np.empty((x.shape[1], len(x)), dtype=np.uint8)
    for column, column_edges in enumerate(edges):
        values = x[:, column]
        # Bin number = how many edges are <= the value.
        bins[column] = np.searchsorted(column_edges, values, side="right")
        bins[column, np.isnan(values)] = MISSING_BIN

    return bins


def split_score(g_sum, h_sum, lambda_):
    # Score of one side of a split; larger means a larger loss reduction.
    return g_sum ** 2 / (h_sum + lambda_)


def leaf_value(g_sum, h_sum, lambda_):
    # Newton step for the leaf, with an L2 penalty lambda_ on the leaf value.
    return -g_sum / (h_sum + lambda_)


def find_best_splits(g_hist, h_hist, lambda_, min_child_weight):
    """Best split of each node from histograms of shape (nodes, features, bins).

    A split sends value bins 0..t left and the other value bins right.
    Missing values go to whichever side gives the larger gain.
    """
    n_nodes, n_features, _ = g_hist.shape
    g_total = g_hist[:, 0, :].sum(axis=1)
    h_total = h_hist[:, 0, :].sum(axis=1)

    # Left sums for every threshold t: first without, then with missing values.
    g_left = np.cumsum(g_hist[:, :, :MISSING_BIN], axis=2)
    h_left = np.cumsum(h_hist[:, :, :MISSING_BIN], axis=2)
    g_left = np.stack((g_left, g_left + g_hist[:, :, MISSING_BIN:]))
    h_left = np.stack((h_left, h_left + h_hist[:, :, MISSING_BIN:]))
    g_right = g_total[:, None, None] - g_left
    h_right = h_total[:, None, None] - h_left

    gain = (
        split_score(g_left, h_left, lambda_)
        + split_score(g_right, h_right, lambda_)
        - split_score(g_total, h_total, lambda_)[:, None, None]
    )
    # Both children need enough hessian weight (roughly, enough rows).
    valid = (h_left >= min_child_weight) & (h_right >= min_child_weight)
    gain = np.where(valid, gain, -np.inf)

    # For each node, pick the best (missing side, feature, threshold).
    gain = np.moveaxis(gain, 1, 0).reshape(n_nodes, -1)
    best = np.argmax(gain, axis=1)
    missing_left, feature, threshold = np.unravel_index(
        best, (2, n_features, MISSING_BIN)
    )
    return gain[np.arange(n_nodes), best], feature, threshold, missing_left == 1


def build_tree(bins, g, h, max_depth, lambda_, min_child_weight):
    """Grow one tree level by level; bins has shape (features, rows)."""
    n_features, n_rows = bins.shape
    n_bins = MISSING_BIN + 1
    tree = {
        "feature": [], "threshold": [], "missing_left": [],
        "left": [], "right": [], "value": [], "gain": [],
    }

    def add_leaf(g_sum, h_sum):
        tree["feature"].append(-1)  # -1 marks a leaf
        tree["threshold"].append(0)
        tree["missing_left"].append(False)
        tree["left"].append(-1)
        tree["right"].append(-1)
        tree["gain"].append(0.0)
        tree["value"].append(leaf_value(g_sum, h_sum, lambda_))
        return len(tree["value"]) - 1

    # Every row starts in the root.
    node_of_row = np.zeros(n_rows, dtype=np.int64)
    level_nodes = [add_leaf(g.sum(), h.sum())]

    for _ in range(max_depth):
        # One histogram slot per node of this level; other rows use a spare slot.
        n_slots = len(level_nodes)
        slot_of_node = np.full(len(tree["value"]), n_slots)
        slot_of_node[level_nodes] = np.arange(n_slots)
        offset = slot_of_node[node_of_row] * n_bins

        # Sum gradients and hessians per (node, feature, bin).
        size = (n_slots + 1) * n_bins
        g_hist = np.empty((n_slots + 1, n_features, n_bins))
        h_hist = np.empty((n_slots + 1, n_features, n_bins))
        for column in range(n_features):
            index = offset + bins[column]
            g_hist[:, column] = np.bincount(
                index, weights=g, minlength=size
            ).reshape(n_slots + 1, n_bins)
            h_hist[:, column] = np.bincount(
                index, weights=h, minlength=size
            ).reshape(n_slots + 1, n_bins)

        gains, features, thresholds, missing_lefts = find_best_splits(
            g_hist[:n_slots], h_hist[:n_slots], lambda_, min_child_weight
        )

        next_level = []
        for slot, node in enumerate(level_nodes):
            # The node stays a leaf when no split reduces the loss.
            if gains[slot] <= 0:
                continue

            column = int(features[slot])
            threshold = int(thresholds[slot])
            missing_left = bool(missing_lefts[slot])
            rows = np.flatnonzero(node_of_row == node)
            row_bins = bins[column, rows]
            goes_left = (row_bins <= threshold) | (
                missing_left & (row_bins == MISSING_BIN)
            )
            left_rows, right_rows = rows[goes_left], rows[~goes_left]

            left = add_leaf(g[left_rows].sum(), h[left_rows].sum())
            right = add_leaf(g[right_rows].sum(), h[right_rows].sum())
            node_of_row[left_rows] = left
            node_of_row[right_rows] = right

            tree["feature"][node] = column
            tree["threshold"][node] = threshold
            tree["missing_left"][node] = missing_left
            tree["left"][node] = left
            tree["right"][node] = right
            tree["gain"][node] = float(gains[slot])
            next_level += [left, right]

        level_nodes = next_level
        if not level_nodes:
            break

    # Arrays make prediction fast.
    return {key: np.array(values) for key, values in tree.items()}


def predict_tree(tree, bins):
    """Leaf value for each row; bins has shape (features, rows)."""
    rows = np.arange(bins.shape[1])
    node = np.zeros(bins.shape[1], dtype=np.int64)

    # Move rows down one level at a time; rows in a leaf stay there.
    while True:
        feature = tree["feature"][node]
        internal = feature >= 0
        if not np.any(internal):
            break
        row_bins = bins[np.maximum(feature, 0), rows]
        goes_left = (row_bins <= tree["threshold"][node]) | (
            tree["missing_left"][node] & (row_bins == MISSING_BIN)
        )
        child = np.where(goes_left, tree["left"][node], tree["right"][node])
        node = np.where(internal, child, node)

    return tree["value"][node]


def boosting_loss(y, scores):
    # Mean logistic loss for labels in {0, 1} and scores (log-odds).
    return np.mean(np.logaddexp(0, scores) - y * scores)


def predict_boosting(trees, base_score, learning_rate, bins):
    """Log-odds scores of the boosted model; sigmoid gives probabilities."""
    scores = np.full(bins.shape[1], base_score)
    for tree in trees:
        scores += learning_rate * predict_tree(tree, bins)
    return scores


def fit_boosting(
    bins_fit, y_fit, bins_val, y_val, n_trees, learning_rate, max_depth,
    lambda_=1.0, min_child_weight=1.0, subsample=0.5, seed=42,
):
    """Train boosted trees on 0/1 labels.

    Validation rows never change a tree; their loss is recorded after each
    tree so the best number of trees can be chosen afterwards.
    """
    rng = np.random.default_rng(seed)

    # Start every row from the log-odds of the positive rate.
    positive_rate = np.mean(y_fit)
    base_score = np.log(positive_rate / (1 - positive_rate))
    scores_fit = np.full(len(y_fit), base_score)
    scores_val = np.full(len(y_val), base_score)

    trees = []
    history = [(0, boosting_loss(y_fit, scores_fit), boosting_loss(y_val, scores_val))]
    for n in range(1, n_trees + 1):
        # Gradient and hessian of the logistic loss at the current scores.
        probabilities = sigmoid(scores_fit)
        g = probabilities - y_fit
        h = probabilities * (1 - probabilities)

        # Each tree is grown on a random part of the training rows.
        rows = rng.random(len(y_fit)) < subsample
        tree = build_tree(
            bins_fit[:, rows], g[rows], h[rows],
            max_depth, lambda_, min_child_weight,
        )
        trees.append(tree)

        scores_fit += learning_rate * predict_tree(tree, bins_fit)
        scores_val += learning_rate * predict_tree(tree, bins_val)
        history.append(
            (n, boosting_loss(y_fit, scores_fit), boosting_loss(y_val, scores_val))
        )
        print(
            f"\rTree {n}/{n_trees}: training loss={history[-1][1]:.5f}, "
            f"validation loss={history[-1][2]:.5f}",
            end="\n" if n == n_trees else "",
            flush=True,
        )

    # The number of trees with the lowest validation loss.
    best_n_trees = int(np.argmin([val_loss for _, _, val_loss in history]))
    return trees, base_score, history, best_n_trees
