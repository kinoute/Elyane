# A Keras-Like Deep Neural Network Abstraction

An OOP Deep Neural Network using a similar syntax as Keras with many hyper-parameters, optimizers and activation functions available.

## Installation

Just clone the repository to your computer to get started:

```sh
git clone git@github.com:kinoute/Elyane.git
cd Elyane
```

The only dependencies to make this neural network work are "numpy" and "pdoc3". They can be installed directly from the main directory if you don't have them already:

```sh
pip3 install -r requirements.txt
```

Three examples are available: `mnist.py`, `fashion_mnist.py` and `xor.py`.

## Features

### Batches

* Batch Gradient Descent
* Mini-batch gradient descent
* SGD

### Layers

* Full-Connected Layer
* Softmax Layer
* Dropout Layer

### Optimizers

* Momentum
* RMSprop
* Adam Optimization
* Amsgrad optimization

### Losses

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Cross Entropy
* Multi-Class Cross Entropy

### Regularizers

* L1 Regularization
* L2 Regularization
* Dropout

### Activation functions

* Sigmoid
* TanH
* Relu
* Leaky Relu
* Softmax

### Tools

* One hot encoding
* Normalization of images


