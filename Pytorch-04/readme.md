# Learning PyTorch - Day 4: torch.nn Module

## Overview of `torch.nn` Module

The `torch.nn` module is an essential part of PyTorch, providing the core functionality to build neural networks. It contains a variety of classes and functions that help in creating layers, defining activation functions, and building the architecture of neural networks. The module includes tools for defining loss functions, optimizers, and a variety of utilities for model training and evaluation. It is used to create models that can be trained on data, and also provides built-in support for automatic differentiation.

### Key Components of `torch.nn`:
1. **`nn.Module`**: The base class for all neural network models in PyTorch. You need to subclass this to create custom models.
2. **Layers**: Predefined layers like `nn.Linear`, `nn.Conv2d`, and `nn.LSTM` allow building complex architectures.
3. **Activation Functions**: Functions such as `nn.ReLU()`, `nn.Sigmoid()`, and `nn.Tanh()` to introduce non-linearity in the model.
4. **Loss Functions**: Includes predefined functions like `nn.CrossEntropyLoss` and `nn.MSELoss` for training the models.
5. **Optimizers**: Used to update the model’s parameters, such as `torch.optim.SGD` or `torch.optim.Adam`.

---

## Notebook Summary: Day 4 Learnings

In this notebook, I worked with the `torch.nn` module to improve my understanding of neural networks and how to define them using PyTorch.

### What I Learned:
- **Creating a Simple Neural Network**: I created a basic model with a single linear layer and a sigmoid activation function.
- **Using `nn.Sequential`**: I learned how to define a neural network using the `nn.Sequential` container, which simplifies the process of stacking layers in a model.
- **Forward Pass**: I understood the importance of defining a `forward()` method, where the operations on input data (like matrix multiplications, activations, etc.) occur.
- **Model Summary**: I explored the `torchinfo.summary()` function to view details about my model, including layer configurations and parameter counts.

### New Concepts Introduced:
- **Sequential Model**: The use of `nn.Sequential` to define a sequence of layers in a more compact way.
- **Activation Functions**: I added the ReLU activation function after the first layer, and the sigmoid activation after the final layer for binary classification.

### Improvement Over Previous Notebook:
- In my previous notebook, I only worked with a simple linear model. This time, I introduced a more complex network with a hidden layer and non-linear activation functions (ReLU and Sigmoid). 
- Additionally, I used `torchinfo.summary()` to inspect the model, which provides a structured overview of the model’s architecture.

By using these tools, I was able to build a slightly more sophisticated neural network, improving upon my prior understanding and model-building workflow.
