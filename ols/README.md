# Ordinary Least Squares (OLS)

Simple implementation of Ordinary Least Squares (OLS) Linear Regression using NumPy.

OLS estimates the relationship between a dependent variable
y and an independent variable X by fitting a linear equation:

```markdown
y = m * x + c
```

where:

m — slope of the regression line

c — intercept

The parameters are computed using the least squares formula:

```markdown
m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
c = ȳ - m * x̄
```

## Features

- Simple and lightweight implementation
- Uses vectorized NumPy operations
- Compatible with custom validation decorators
- Easy to integrate into small ML libraries

## Usage

Example:

```python
import numpy as np
from ols import OLS

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = OLS()
model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```

## API

### `fit(X, y)`

Fits the OLS model to the training data.

Parameters:

- X : numpy array of input features
- y : numpy array of target values

Returns:

- fitted model instance

### `predict(X)`

Predicts target values using the trained model.

Parameters:

- X : numpy array of input features

Returns:

- predicted values

## Dependencies
- Numpy

## License
MIT License
