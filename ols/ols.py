import numpy as np
from typing import Any
from ._validator import validation_fit, validation_predict


class OLS:
    """
    Ordinary Least Squares (OLS) linear regression.

    This model fits a simple linear regression line to the input data using
    the ordinary least squares method. It estimates the slope (m) and
    intercept (c) of the linear function:

        y = m * x + c

    where:
        m : slope of the regression line
        c : intercept of the regression line
    """
    def __init__(self):
        self.fitted_ = False

    @validation_fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLS':
        """
        Fit the OLS model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        y : np.ndarray
            Target values.

        Returns
        -------
        OLS
            Returns the fitted model instance.
        """
        x_mk = np.mean(X)
        y_mk = np.mean(y)
        x_diff = X - x_mk
        y_diff = y - y_mk
        self.m = np.sum(x_diff * y_diff) / np.sum(np.power(x_diff, 2))
        self.c = y_mk - self.m * x_mk
        self.fitted_ = True
        return self

    @validation_predict
    def predict(self, X: np.ndarray) -> Any:
        """
        Predict target values using the fitted OLS model.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if not self.fitted_:
            raise Exception('OLS not fitted yet')
        y = self.m * X + self.c
        return y