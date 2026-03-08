# Perceptron Classifier

A simple NumPy implementation of the Perceptron algorithm for binary classification.

The Perceptron is one of the earliest and simplest supervised machine learning algorithms used to classify linearly separable data.

It learns a decision boundary defined by:

```markdown
w · x + b = 0
```

where:

- w — weight vector
- b — bias
- x — input feature vector

The predicted class is determined by the sign of the linear combination.

```markdown
prediction = sign(w · x + b)
```

Output labels are:

```markdown
-1  or  1
```

## Algorithm

For each training sample the model:

1. Computes the prediction
```markdown
z = X @ w + b
```
2. Converts it to class labels
```markdown
pred = sign(z)
```
3. Updates the parameters if prediction is wrong
```markdown
w = w + η * Xᵀ * error
b = b + η * Σ(error)
```
where:

- η — learning rate
- error = y − pred

Training stops when:

- all samples are classified correctly, or
- the maximum number of iterations is reached.

## Features

- Pure NumPy implementation
- Random weight initialization
- Early stopping if no classification errors remain
- Vectorized weight updates
- Binary classification (-1, 1)

## Usage

Example:
```python
import numpy as np
from perceptron import Perceptron

X = np.array([
    [2, 1],
    [1, -1],
    [-1, -2],
    [-2, 1]
])

y = np.array([1, 1, -1, -1])

model = Perceptron(
    learning_rate=0.1,
    n_iter=100,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```
Output example:
```markdown
[ 1  1 -1 -1 ]
```

## Parameters

| Parameter       | Description                            |
|:----------------|----------------------------------------|
| `learning_rate` | Step size for updating weights         |
| `n_iter`        | Maximum number if training iterations  |
| `random_state`  | Random seed for reproduciblity         |

## Attributes

| Attribute    | Description                            |
|:-------------|----------------------------------------|
| `w_`         | Weight vector learned training during  |
| `b_`         | Bias term                              |
| 'fitted_'    | Indicates if the model has been trained |

## Mathematical Formula

The decision function:
```
f(x) = w · x + b
```
Prediction rule:
```
ŷ = 1  if f(x) ≥ 0
ŷ = -1 if f(x) < 0
```

## Dependencies
- NumPy

## License
MIT license