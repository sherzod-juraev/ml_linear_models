# LogisticRegressionGD

A simple Logistic Regression classifier
implemented in Python using gradient descent.
Designed for educational purposes and basic experiments.

## Usage

```python
from ml_linear_models.logistic_regression import LogisticRegressionGD
import numpy as np

# Sample dataset
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

model = LogisticRegressionGD(eta=0.01, n_iter=100, eps=1e-5, random_state=42)
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:", predictions)
print("Weights:", model.w_)
print("Bias:", model.b_)
```

## Notes
- Targets must be 0 or 1.
- Supports random weight initialization for reproducibility using random_state.
- Uses batch gradient descent with early stopping based on convergence tolerance eps.