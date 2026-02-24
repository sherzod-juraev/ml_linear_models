from numpy.random import default_rng
import numpy as np


class AdalineGD:
    """Adaptive Linear Neuron classifier using batch gradient descent.

    Parameters
    ----------
    eta : float, default=0.01
        Learning rate (between 0.0 and 1.0)
    n_iter : int, default=100
        Maximum number of passes over the training dataset.
    eps : float, default=1e-5
        Convergence tolerance for early stopping.
    random_state : int or None, default=None
        Seed for random weight initialization.

    Attributes
    ----------
    w_ : np.ndarray
        Weights after fitting.
    b_ : float
        Bias after fitting.
    """

    def __init__(
            self,
            eta: float = 1e-2,
            n_iter: int = 100,
            eps: float = 1e-5,
            random_state: int | None = None
    ):

        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state


    def fit(self, X: np.ndarray, y: np.ndarray, /) -> 'AdalineGD':
        """Fit training data using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training vectors.
        y : np.ndarray, shape (n_samples,)
            Target values (-1 or 1 for classification).

        Returns
        -------
        self : AdalineGD
        """

        if not np.all(np.isin(np.unique(y), [-1, 1])):
            raise ValueError('Target values must be -1 or 1 for AdalineGD')
        rng = default_rng(self.random_state)
        self.w_ = rng.uniform(-1e-2, 1e-2, size=X.shape[1])
        self.b_ = rng.uniform(-1e-2, 1e-2)
        J_last, J_old = None, None
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            err = y - net_input
            self.w_ += self.eta * X.T.dot(err)
            self.b_ += self.eta * err.sum()
            J_last = np.sum(np.power(err, 2)) / 2
            if i != 0 and np.abs(J_last - J_old) < self.eps:
                break
            J_old = J_last
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Compute linear combination of inputs and weights.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Net input values for all samples.
        """

        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted labels (-1 or 1)
        """
        net_input = self.net_input(X)
        return np.where(net_input >= 0, 1, -1)