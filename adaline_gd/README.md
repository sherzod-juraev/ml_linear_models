# AdalineGD (Adaptive Linear Neuron - Batch Gradient Descent)

A simple **NumPy implementation of the Adaptive Linear Neuron (Adaline)** algorithm using **batch gradient descent.**

Adaline is a linear classifier that learns weights by **minimizing the squared error cost function.**

Unlike the Perceptron, Adaline updates its weights using the **continuous output of the linear activation function**, not the discrete class prediction.

## Algorithm

The linear model is defined as:
```markdown
z = w · x + b
```
where:

- w — weight vector
- b — bias
- x — input feature vector

Prediction rule:
```markdown
ŷ = 1  if z ≥ 0
ŷ = -1 if z < 0
```

### Cost function
Adaline minimizes the Sum of Squared Errors (SSE):
```markdown
J(w) = 1/2 Σ (y - z)²
```
where:
```markdown
z = Xw + b
```
The weights are updated using **gradient descent.**

### Weight Update Rule

Weight Update Rule
```markdown
error = y - net_input
```
Update weights
```markdown
w = w + η * Xᵀ · error
b = b + η * Σ(error)
```
where:
- η — learning rate

## Features:

- Pure NumPy implementation
- Batch gradient descent
- Early stopping using convergence tolerance
- Random weight initialization
- Binary classification (-1, 1)
- Vectorized updates for efficiency

## Usage

Example:
```python
import numpy as np
from adaline_gd import AdalineGD

X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [2, 1],
    [3, 2]
])

y = np.array([1, 1, 1, -1, -1])

model = AdalineGD(
    eta=0.01,
    n_iter=100,
    eps=1e-5,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```
Example output:
```markdown
[ 1  1  1 -1 -1 ]
```

## Parameters:

| Parameter        | Description                               |
|:-----------------|-------------------------------------------|
| `eta`            | Learning rate                             |
| `n_iter`         | Maximum number of training iterations     |
| `eps`            | Convergence tolerance for early stopping  |
| `random_state`   | Random seed for reproducibility           |

## Attributes

| Attribute  | Description                             |
|:-----------|-----------------------------------------|
| `w_`       | Learned weight vector                   |
| `b_`       | Bias term                               |
| `fitted_`  | Indicates if the model has been trained |

## Difference from Perceptron

| Perceptron                          | Adaline                         |
|:------------------------------------|---------------------------------|
| Use step activation                 | Use linear activation           |
| Updates using classification error  | Updates using continuous error  |
| Hard threshold learning             | Gradient descent optimization   |

## Dependencies

- NumPy

## License
MIT license