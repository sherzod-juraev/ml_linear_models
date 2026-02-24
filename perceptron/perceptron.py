from numpy.random import default_rng
import numpy as np


class Perceptron:
    """Perceptron classifier for binary classification (-1 and 1).

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for weight updates.
    n_iter : int, default=100
        Maximum number of iterations over the training dataset.
    random_state : int or None, default=None
        Random seed for weight initialization.

    Attributes
    ----------
    w_ : np.ndarray
        Weights after fitting.
    b_ : float
        Bias after fitting."""

    def __init__(
            self,
            learning_rate: float = .1,
            n_iter: int = 100,
            random_state: int | None = None
    ):

        self.eta = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state


    def initialize_weights_bias(self, size: int):
        """Initialize weights and bias with small random numbers.

        Parameters
        ----------
        size : int
            Number of features.
        """

        rng = default_rng(self.random_state)
        self.w_ = rng.uniform(-1e-2, 1e-2, size=size)
        self.b_ = rng.uniform(-1e-2, 1e-2)

    def net_input(self, X: np.ndarray, /) -> np.ndarray:
        """Calculate the linear combination of inputs and weights.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Net input values, shape (n_samples,)
        """

        z = X @ self.w_ + self.b_
        return z

    def fit(self, X: np.ndarray, y: np.ndarray, /) -> 'Perceptron':
        """
        Fit the Perceptron model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Target labels, must be -1 or 1, shape (n_samples,).

        Returns
        -------
        self : Perceptron
        """

        if not np.all(np.isin(np.unique(y), [1, -1])):
            raise ValueError('Target values must be -1 or 1 for Perceptron')
        self.initialize_weights_bias(X.shape[1])
        for i in range(self.n_iter):
            pred = self.predict(X)
            err = y - pred
            ind = err == 0
            if not np.all(ind):
                self.w_ += self.eta * X.T.dot(err)
                self.b_ += self.eta * err.sum()
            else:
                break
        return self

    def predict(self, X: np.ndarray, /) -> np.ndarray:
        """Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels (-1 or 1), shape (n_samples,)
        """

        z = self.net_input(X)
        pred = np.where(z >= 0, 1, -1)
        return pred