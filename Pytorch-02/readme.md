# Day 2: Learning PyTorch - Autograd and Backpropagation

## Introduction
This repository documents my journey in learning PyTorch. On Day 2, I explored Autograd, PyTorch's automatic differentiation engine, and applied it to computing gradients and backpropagation in neural networks.

## What is PyTorch?
PyTorch is an open-source machine learning library used for deep learning and tensor computations. It provides dynamic computation graphs, making it easy to build and train neural networks.

## Topics Covered

### Understanding Derivatives and Gradients
We first calculate derivatives manually before using PyTorch's Autograd.

```python
import torch
import math

# Derivative of y = x^2
def dy_dx(x):
    return 2*x
print(dy_dx(3))  # Output: 6

# Derivative of z = x^2 * cos(x^2)
def dz_dx(x):
    return 2 * x * math.cos(x**2)
print(dz_dx(2))  # Output: -3.3072
```
## Autograd: Automatic Differentiation in PyTorch
```python
# Compute the derivative of y = x^2 using Autograd
x = torch.tensor(3.0, requires_grad=True)  # Enable gradient tracking
y = x**2  # y = x^2
y.backward()  # Compute gradient
print(x.grad)  # Output: 6.0
```

## Computing Gradients for Composite Functions
``` python
x = torch.tensor(3.0, requires_grad=True)
y = x**2
z = torch.sin(y)
z.backward()
print(x.grad)  # Computes dz/dx
```
## Backpropagation in a Simple Neural Network
``` python
# Inputs
x = torch.tensor(6.7)
y = torch.tensor(0.0)

# Parameters
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Forward pass
z = w * x + b
y_pred = torch.sigmoid(z)

# Loss function (Binary Cross Entropy)
def binary_cross_entropy(prediction, target):
    epsilon = 1e-8  # To avoid log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)

loss = binary_cross_entropy(y_pred, y)

# Backward pass
loss.backward()
print(w.grad)  # Gradient of loss w.r.t weight
print(b.grad)  # Gradient of loss w.r.t bias
```
## Computing Gradients for Vector Inputs
``` python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x**2).mean()
y.backward()
print(x.grad)  # Gradient of loss w.r.t input
```
## Clearing Gradients
* Gradients accumulate in PyTorch, so we must clear them before computing new gradients.
``` python
x.grad.zero_()
```
## Disabling Gradient Tracking
``` python
# Option 1: Disabling gradients manually
x.requires_grad = False

# Option 2: Using detach()
z = x.detach()

# Option 3: Using `torch.no_grad()`
with torch.no_grad():
    y = x**2
```
## Summary
On Day 2, I learned about PyTorch's Autograd system, how to compute gradients automatically, and how backpropagation works in a simple neural network. This knowledge is essential for training deep learning models efficiently.