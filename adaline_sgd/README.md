# AdalineSGD

Adaptive Linear Neuron classifier using **stochastic gradient descent (SGD)** using Numpy.  
Suitable for online learning or large datasets.

---

## Usage

```python
import numpy as np
from ml_linear_models.adaline_sgd import AdalineSGD

# Sample dataset
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, -1, 1])

clf = AdalineSGD(eta=0.01, n_iter=100, eps=1e-5, random_state=42)
clf.fit(X, y)

predictions = clf.predict(X)

print("Predictions:", predictions)
print("Weights:", clf.w_)
print("Bias:", clf.b_)
```