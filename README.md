# ML Linear Models

A collection of **machine learning algorithms implemented from scratch in Python using NumPy.**

This project focuses on **understanding the core mechanics of linear models** by implementing them without relying on external machine learning libraries such as scikit-learn.

The goal of this repository is to provide:

- clear implementations of classical linear algorithms
- educational reference code
- a lightweight foundation for building a custom ML library

## Implemented Models

| Model                 | Description                                                       | Link                                                                                                   |
|:----------------------| ----------------------------------------------------------------- |--------------------------------------------------------------------------------------------------------|
| Perceptron            | Binary linear classifier based on the perceptron learning rule.   | [More info →](./perceptron/README.md#perceptron-classifier)                                            |
| AdalineGD             | Adaptive Linear Neuron trained using batch gradient descent.      | [More info →](./adaline_gd/README.md#adalinegd-adaptive-linear-neuron---batch-gradient-descent)        |
| AdalineSGD            | Adaptive Linear Neuron trained using stochastic gradient descent. | [More info →](./adaline_sgd/README.md#adalinesgd-adaptive-linear-neuron--stochastic-gradient-descent)  |
| LogisticRegressionGD  | Logistic Regression classifier trained using gradient descent.    | [More info →](./logistic_regression/README.md#logistic-regression-gradient-descent)                    |
| OLS                   | Ordinary Least Squares linear regression model.                   | [More info →](./ols/README.md#ordinary-least-squares-ols)                                              |

## Project Structure

```markdown
ml_linear_models/
│
├── perceptron/
│   └── __init__.py
│   └── _validator.py
│   └── perceptron.py
│   └── README.md
│
├── adaline_gd/
│   └── __init__.py
│   └── _validator.py
│   └── adaline_gd.py
│   └── README.md
│
├── adaline_sgd/
│   └── __init__.py
│   └── _validator.py
│   └── adaline_sgd.py
│   └── README.md
│
├── logistic_regression/
│   └── __init__.py
│   └── _validator.py
│   └── logistic_regression.py
│   └── README.md
│
├── ols/
│   └── __init__.py
│   └── _validator.py
│   └── ols.py
│   └── README.md
│   
└── .gitignore
└── LICENSE
└── requirements.txt
```
Each algorithm has its **own folder and documentation** explaining:

- the algorithm
- mathematical formulation
- usage examples

## Dependencies

- NumPy

## Purpose of This Repository

This project was created to:

- learn how classical machine learning algorithms work internally
- understand optimization methods such as gradient descent
- explore how machine learning libraries are structured
- All algorithms are implemented from scratch for educational purposes.

## License
MIT license