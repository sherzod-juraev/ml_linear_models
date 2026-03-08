# AdalineSGD (Adaptive Linear Neuron — Stochastic Gradient Descent)

A simple **NumPy implementation of the Adaptive Linear Neuron (Adaline) using stochastic gradient descent (SGD).**

Adaline is a linear classifier that learns weights by minimizing the **squared error cost function.**
In this implementation, the parameters are updated **after each training sample**, which is the key idea behind stochastic gradient descent.

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

### Cost Function

Adaline minimizes the **Sum of Squared Errors (SSE):**
```markdown
J(w) = 1/2 Σ (y - z)²
```
where:
```markdown
z = Xw + b
```

### Weight Update Rule (SGD)
For each training sample:
```markdown
error = yᵢ − zᵢ
```
Update weights:
```markdown
w = w + η * error * xᵢ
b = b + η * error
```
where:

- η — learning rate

The parameters are updated **after every sample**, which makes SGD faster on large datasets compared to batch gradient descent.

## Features

- Pure NumPy implementation
- Stochastic Gradient Descent (SGD)
- Random weight initialization
- Early stopping using convergence tolerance
- Binary classification (-1, 1)
- Lightweight and easy to integrate into ML pipelines

Usage

Example:
```python
import numpy as np
from adaline_sgd import AdalineSGD

X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [2, 1],
    [3, 2]
])

y = np.array([1, 1, 1, -1, -1])

model = AdalineSGD(
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

## Parameters

| Parameter          | Description                           |
|:-------------------|---------------------------------------|
| `eta`	             | Learning rate                         |
| `n_iter`	          | Maximum number of training iterations |
| `eps`	             | Convergence tolerance for early stopping |
| `random_state`	    | Random seed for reproducibility       |

## Attributes

| Attribute     | 	Description                                  |
|:--------------|-----------------------------------------------|
| `w_`	         | Learned weight vector                         |
| `b_`	         | Bias term                                     |
| `fitted_`	    | Indicates whether the model has been trained  |

## Difference from AdalineGD

| AdalineGD                                     | AdalineSGD                            |
| --------------------------------------------- | ------------------------------------- |
| Updates weights using **all samples at once** | Updates weights **after each sample** |
| Batch gradient descent                        | Stochastic gradient descent           |
| Slower on large datasets                      | Faster for large datasets             |

## Dependencies

- NumPy

## License
MIT license