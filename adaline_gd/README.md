# AdalineGD

Adaptive Linear Neuron (Adaline) classifier implemented from scratch in Python using NumPy.  
Uses batch gradient descent with early stopping.

---

## Usage

```python
from ml_linear_models.adaline_gd import AdalineGD
import numpy as np

# Sample dataset
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, -1, 1])

clf = AdalineGD(eta=0.01, n_iter=100, eps=1e-5, random_state=42)
clf.fit(X, y)

predictions = clf.predict(X)
print(f"Predictions: {predictions}")
print(f"Weights: {clf.w_}")
print(f"Bias: {clf.b_}")
```

