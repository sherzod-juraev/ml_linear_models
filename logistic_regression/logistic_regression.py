from numpy.random import default_rng
import numpy as np
from ._validator import validation_params, validation_fit, validation_predict


class LogisticRegressionGD:
    """
    Logistic Regression classifier using gradient descent.

    Parameters
    ----------
    eta : float, default=0.01
        Learning rate (between 0.0 and 1.0).
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

    @validation_params
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
        self.fitted_ = False

    @validation_fit
    def fit(self, X: np.ndarray, y: np.ndarray, /) -> 'LogisticRegressionGD':
        """Fit training data using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training vectors.
        y : np.ndarray, shape (n_samples,)
            Target values (0 or 1).

        Returns
        -------
        self : LogisticRegressionGD
        """

        rng = default_rng(self.random_state)
        self.w_ = rng.uniform(-1e-2, 1e-2, size=X.shape[1])
        self.b_ = rng.uniform(-1e-2, 1e-2)
        n = X.shape[0]
        J_last, J_old = None, None
        for _ in range(self.n_iter):
            z = self.net_input(X)
            sigmoid = self.sigmoid(z)
            errors = y - sigmoid
            self.w_ += self.eta * (X.T @ errors) / n
            self.b_ += self.eta * errors.sum() / n
            J_last = - (1 / n) * np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
            if J_old is not None and np.abs(J_last - J_old) <= self.eps:
                break
            J_old = J_last
        self.fitted_ = True
        return self


    def net_input(self, X: np.ndarray, /) -> np.ndarray:
        """Compute linear combination of inputs and weights.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Net input values for all samples.
        """

        net_input = X @ self.w_ + self.b_
        return net_input

    def sigmoid(self, z: np.ndarray, /) -> np.ndarray:
        """Compute logistic sigmoid activation.

        Parameters
        ----------
        z : np.ndarray
            Net input values.

        Returns
        -------
        np.ndarray
            Sigmoid-transformed values.
        """

        sigmoid = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        return sigmoid

    @validation_predict
    def predict(self, X: np.ndarray, /) -> np.ndarray:
        """Return class labels after thresholding at 0.5.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        if not self.fitted_:
            raise Exception('LogisticRegression not fitted yet')
        sigmoid = self.sigmoid(self.net_input(X))
        result = np.where(sigmoid >= .5, 1, 0)
        return result