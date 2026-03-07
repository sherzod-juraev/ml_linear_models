import numpy as np
from typing import Any
from ._validator import validation_fit, validation_predict


class OLS:

    def __init__(self):
        pass

    @validation_fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLS':
        x_mk = np.mean(X)
        y_mk = np.mean(y)
        x_diff = X - x_mk
        y_diff = y - y_mk
        self.m = np.sum(x_diff * y_diff) / np.sum(np.power(x_diff, 2))
        self.c = y_mk - self.m * x_mk
        return self

    @validation_predict
    def predict(self, X: np.ndarray) -> Any:

        y = self.m * X + self.c
        return y