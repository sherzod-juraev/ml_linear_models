from numpy.random import default_rng
import numpy as np


class AdalineSGD:
    """ Adaptive Linear Neuron classifier

    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)

    n_iter: int
        Passes over the training dataset.

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

    def fit(self, X: np.ndarray, y: np.ndarray, /) -> 'AdalineSGD':
        """Fit training data

        Parameters
        -----------

        X:{array-like},
            shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.

        y:-array-like,
            shape = [n_examples]
            Target values.
        """

        if not np.all(np.isin(np.unique(y), [-1, 1])):
            raise ValueError('Target values must be -1 or 1 for AdalineSGD')
        rng = default_rng(self.random_state)
        self.w_ = rng.uniform(-1e-2, 1e-2, size=X.shape[1])
        self.b_ = rng.uniform(-1e-2, 1e-2)
        n_sample = X.shape[0]
        J_last, J_old = None, None
        for i in range(self.n_iter):
            J_old = J_last
            J_last = 0
            for j in range(n_sample):
                net_input = self.net_input(X[j])
                error = y[j] - net_input
                self.w_ += self.eta * error * X[j]
                self.b_ += self.eta * error
                J_last += (error ** 2) / 2
            if i != 0 and np.abs(J_last - J_old) <= self.eps:
                break
        return self


    def net_input(self, xi: np.ndarray):
        """Calculate net input
        Parameters
        ----------

        X: ndarray
            shape = [n_feature]
        """

        net_input = xi @ self.w_ + self.b_
        return net_input

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        result = np.where(self.net_input(X) > 0, 1, -1)
        return result