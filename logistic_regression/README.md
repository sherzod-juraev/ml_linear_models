# Logistic Regression (Gradient Descent)

A simple **NumPy implementation of Logistic Regression** trained using **batch gradient descent.**

Logistic Regression is a widely used **binary classification algorithm** that models the probability of a class using the **logistic (sigmoid) function.**

## Algorithm

The linear model is defined as:
```markdown
z = w · x + b
```
where:

- w — weight vector
- b — bias
- x — input feature vector

The probability of class 1 is computed using the **sigmoid function:**
```markdown
σ(z) = 1 / (1 + e⁻ᶻ)    
```

### Prediction Rule

The predicted class label is determined by thresholding the probability:
```markdown
ŷ = 1  if σ(z) ≥ 0.5
ŷ = 0  if σ(z) < 0.5
```

### Cost Function

Logistic Regression minimizes the **Binary Cross-Entropy (Log Loss):**
```markdown
J(w) = -1/n Σ [ y log(p) + (1 - y) log(1 - p) ]
```
where:
```markdown
p = σ(Xw + b)
```

### Gradient Descent Update
At each iteration:
```markdown
error = y - p
```
Update parameters:
```markdown
w = w + eta * (Xᵀ · error) / n
b = b + eta * Σ(error) / n
```

where:

- eta — learning rate
- n — number of samples

Training stops when:

- convergence tolerance is reached, or
- the maximum number of iterations is completed.

## Features

- Pure **NumPy implementation**
- **Batch gradient descent optimization**
- **Sigmoid activation function**
- **Binary classification (0, 1)**
- Early stopping using convergence tolerance
- Random weight initialization

## Usage

Example:
```python
import numpy as np
from logistic_regression import LogisticRegressionGD

X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [2, 1],
    [3, 2]
])

y = np.array([1, 1, 1, 0, 0])

model = LogisticRegressionGD(
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
[1 1 1 0 0]
```

## Parameters

| Parameter      | Description                              |
| -------------- | ---------------------------------------- |
| `eta`          | Learning rate                            |
| `n_iter`       | Maximum number of training iterations    |
| `eps`          | Convergence tolerance for early stopping |
| `random_state` | Random seed for reproducibility          |

## Attributes

| Attribute | Description                                  |
| --------- | -------------------------------------------- |
| `w_`      | Learned weight vector                        |
| `b_`      | Bias term                                    |
| `fitted_` | Indicates whether the model has been trained |

## Dependencies

- NumPy

## License
MIT license