from functools import wraps
import numpy as np


def validation_params(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        if not isinstance(self.eta, float):
            raise TypeError(f'eta must be float, got {type(self.eta).__name__}')
        if self.n_iter <= 0:
            raise Exception(f"n_iter must be positive, got {self.n_iter}")
        if not isinstance(self.n_iter, int):
            raise TypeError(f"n_iter must be int, got {type(self.n_iter).__name__}")
        if not isinstance(self.eps, float | int):
            raise TypeError(f"eps must be float or int, got {type(self.eps).__name__}")
        if not isinstance(self.random_state, int | None):
            raise TypeError(f"random_state must be int or None, got {type(self.random_state).__name__}")
        return res
    return wrapper


def validation_fit(func):
    @wraps(func)
    def wrapper(self, X, y, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy.ndarray, got {type(X).__name__}")
        if X.ndim != 2:
            raise Exception(f"X.ndim must be 2, got {X.ndim}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be numpy.ndarray, got {type(y).__name__}")
        if y.ndim != 1:
            raise Exception(f"y.ndim must be 1, got {y.ndim}")
        if not np.all(np.isin(np.unique(y), [1,-1])):
            raise ValueError(f"Target values must be 1 or -1 fot AdalineGD")
        return func(self, X, y, *args, **kwargs)
    return wrapper


def validation_predict(func):
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy.ndarray, got {type(X).__name__}")
        return func(self, X, *args, **kwargs)
    return wrapper