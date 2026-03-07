from functools import wraps
import numpy as np


def validation_fit(func):
    @wraps(func)
    def wrapper(self, X, y, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy.ndarray, got {type(X).__name__}")
        if X.ndim != 1:
            raise ValueError(f"X.ndim must be 1, got {X.ndim}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be numpy.ndarray, got {type(y).__name__}")
        if y.ndim != 1:
            raise ValueError(f"y.ndim must be 1, got {y.ndim}")
        if X.shape[0] != y.shape[0]:
            raise Exception(f"X.shape must be equal to y.shape")
        return func(self, X, y, *args, **kwargs)
    return wrapper


def validation_predict(func):
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy.ndarray, got {type(X).__name__}")
        if X.ndim != 1:
            raise ValueError(f"X.ndim must be 1, got {X.ndim}")
        return func(self, X, *args, **kwargs)
    return wrapper