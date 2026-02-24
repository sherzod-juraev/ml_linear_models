# Perceptron Classifier

A simple **Perceptron** implementation in Python using NumPy

This is a minimal, educational implementation API (`fit` and `predict`) with batch weight updates.

---

## Features

- Binary classification (-1 / 1 labels)
- Batch weight updates
- Random weight initialization with `random_state` for reproducibility
- Minimal and readable implementation
- Suitable for small linearly separable datasets

---

## Usage

```python
import numpy as np
from ml_linear_models.perceptron import Perceptron

# Sample dataset
X = np.array([
    [2, 3],
    [1, 5],
    [2, 1],
    [5, 2],
    [6, 1],
    [7, 3]
])
y = np.array([-1, -1, -1, 1, 1, 1])

clf = Perceptron(learning_rate=0.1, n_iter=10_000, random_state=42)
clf.fit(X, y)

predictions = clf.predict(X)
print("Predictions:", predictions)
print("Weights:", clf.w_)
print("Bias:", clf.b_)
```

## Notes
- Input y must only contain -1 and 1. Otherwise, a ValueError will be raised.
- Works best with linearly separable datasets.