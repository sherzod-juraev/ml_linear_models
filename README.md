# ML Linear Models

A collection of **linear machine learning models** implemented in Python.
This repository is for educational purposes, experimentation,
and learning the basics of linear models

## Included Models

| Model                | Description                                                  | Link                                                               |
|:---------------------|:-------------------------------------------------------------|:-------------------------------------------------------------------|
| Perceptron           | Simple binary classifier using the perceptron learning rule. | [More info →](./perceptron/README.md#perceptron-classifier)        |
| AdalineGD            | Adaptive Linear Neuron using batch gradient descent.         | [More info →](./adaline_gd/README.md#adalinegd)                    |
| AdalineSGD           | Adaptive Linear Neuron using stochastic gradient descent.    | [More info →](./adaline_sgd/README.md#adalinesgd)                  |
| LogisticRegressionGD | Logistic Regression classifier using batch gradient descent. | [More info →](./logistic_regression/README.md#logisticregressiongd) | 

## Usage

Each model has its own folder and README with usage examples.
You can import a model like this:

```python
from ml_linear_models.perceptron import Perceptron
from ml_linear_models.adaline_gd import AdalineGD
from ml_linear_models.adaline_sgd import AdalineSGD
from ml_linear_models.logistic_regression import LogisticRegressionGD
```

## Notes
- All models are implemented from scratch for learning purposes.
- Random weight initialization is supported via random_state.
- Suitable for binary classification tasks.