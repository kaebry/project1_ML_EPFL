import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent on the MSE loss.

    At each iteration, w is updated using the gradient of the MSE loss
    computed on the full dataset.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient descent steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weights after the last iteration.
        loss: float, the MSE loss (with factor 1/2) of w on the full dataset.
    """
    # 1. start w from initial_w
    w = initial_w

    # 2. repeat max_iters times:
    #       - error from the CURRENT w
    #       - gradient from the error
    #       - update w with the gradient and gamma
    for _ in range(max_iters):
        e = y - tx @ w
        gradient = (-tx.T / len(y)) @ e
        w = w - gradient * gamma

    # loss for the final w
    loss = 0.5 * np.mean((y - tx @ w) ** 2)

    return (w, loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent (batch size 1) on the MSE loss.

    At each iteration, one data point is sampled uniformly at random and w is
    updated using the gradient of the MSE loss on that single point.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of SGD steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weights after the last iteration.
        loss: float, the MSE loss (with factor 1/2) of w on the full dataset.
    """
    # 1. start w from initial_w
    w = initial_w

    # 2. repeat max_iters times:
    #       - error from the CURRENT w of one randomly selected data row
    #       - gradient from the error of this one row
    #       - update w with the gradient and gamma
    for _ in range(max_iters):
        rand_index = np.random.randint(0, len(y))
        e = y[rand_index] - tx[rand_index] @ w
        gradient = -tx[rand_index] * e
        w = w - gradient * gamma

    # loss for the final w
    loss = 0.5 * np.mean((y - tx @ w) ** 2)

    return (w, loss)


def least_squares(y, tx):
    """
    Least squares regression using the normal equations.

    Computes the weights that minimize the MSE loss in closed form, by solving
    the linear system (tx^T tx) w = tx^T y.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: float, the MSE loss (with factor 1/2) of w on the full dataset.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = 0.5 * np.mean((y - tx @ w) ** 2)

    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using the normal equations.

    Computes the weights minimizing the MSE loss plus the penalty
    lambda_ * ||w||^2, by solving (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        lambda_: float, the regularization parameter.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: float, the MSE loss (with factor 1/2) of w on the full dataset,
            without the penalty term.
    """
    w = np.linalg.solve(
        tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T @ y
    )
    loss = 0.5 * np.mean((y - tx @ w) ** 2)

    return (w, loss)


def sigmoid(z):
    """
    Logistic function 1 / (1 + exp(-z)), applied elementwise.

    Args:
        z: numpy array of any shape (or a scalar), the logits.

    Returns:
        numpy array of the same shape as z, with values in [0, 1].
    """
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, tx, w):
    """
    Logistic loss: the negative log-likelihood of the labels, averaged over the dataset.

        loss(w) = 1/N * sum_n [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        w: numpy array of shape (D,), the weights.

    Returns:
        float, the logistic loss of w on the full dataset.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_gradient(y, tx, w):
    """
    Gradient of the logistic loss with respect to w, on the full dataset.

        grad(w) = 1/N * tx^T (sigmoid(tx w) - y)

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        w: numpy array of shape (D,), the weights.

    Returns:
        numpy array of shape (D,), the gradient of the logistic loss at w.
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y in {0, 1}).

    At each iteration, w is updated using the gradient of the logistic loss
    (negative log-likelihood) computed on the full dataset.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient descent steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weights after the last iteration.
        loss: float, the logistic loss of w on the full dataset.
    """
    # 1. start w from initial_w
    w = initial_w

    # 2. repeat max_iters times:
    #       - gradient of the logistic loss at the CURRENT w
    #       - update w with the gradient and gamma
    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w)
        w = w - gradient * gamma

    # loss for the final w
    loss = logistic_loss(y, tx, w)

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent (y in {0, 1}).

    Minimizes the logistic loss plus the penalty lambda_ * ||w||^2. At each
    iteration, w is updated using the gradient of this penalized objective
    computed on the full dataset: the penalty adds 2 * lambda_ * w to the
    gradient of the logistic loss. All entries of w are penalized, including
    the bias weight if tx has a bias column.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the data matrix (with bias column if used).
        lambda_: float, the regularization parameter.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient descent steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weights after the last iteration.
        loss: float, the logistic loss of w on the full dataset,
            without the penalty term.
    """
    # 1. start w from initial_w
    w = initial_w

    # 2. repeat max_iters times:
    #       - gradient of the logistic loss at the CURRENT w
    #       - add the gradient of the penalty, 2 * lambda_ * w
    #       - update w with the penalized gradient and gamma
    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gradient * gamma

    # loss for the final w (without the penalty term)
    loss = logistic_loss(y, tx, w)

    return (w, loss)
