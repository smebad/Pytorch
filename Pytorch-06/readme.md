# Day 6: Building ANN with PyTorch

## Introduction
In this notebook, I built an Artificial Neural Network (ANN) using PyTorch to classify a subset of the Fashion MNIST dataset. Due to CPU limitations, I used only 6000 images for training and testing. This exercise allowed me to learn how to construct, train, and evaluate a simple Multi-Layer Perceptron (MLP) model using PyTorch's `nn` module along with the Dataset and DataLoader classes to handle data efficiently.

## What I Learned
- **Data Handling with Dataset and DataLoader:**  
  I learned how to create a custom dataset class by inheriting from `torch.utils.data.Dataset`. This class encapsulates the features and labels, allowing for easy indexing and batching. The DataLoader class was then used to batch and shuffle the data automatically, which is crucial for efficient training.

- **Model Architecture Using `nn.Module` and `nn.Sequential`:**  
  I defined a neural network (MLP) using PyTorch's `nn.Module`. The model was constructed with an input layer, hidden layers with ReLU activations, and an output layer with a Sigmoid activation function. This approach greatly simplifies building and managing neural networks compared to manual implementations.

- **Training Pipeline:**  
  I implemented a training loop that:
  - Performs forward propagation through the network.
  - Calculates the loss using `nn.BCELoss` (Binary Cross-Entropy Loss) for a binary classification task.
  - Computes gradients via backpropagation.
  - Updates model parameters using the SGD optimizer.
  - Resets gradients after each update.
  
- **Model Evaluation:**  
  I evaluated the model on a test set by iterating over a test DataLoader, thresholding the output to generate binary predictions, and calculating overall accuracy.

- **Practical Considerations for CPU Training:**  
  To work within the constraints of a CPU (Intel Core i7), I reduced the dataset size (using only 6000 images) and set appropriate hyperparameters (e.g., batch size, block size). I also learned that while training on a CPU is feasible, GPUs can significantly accelerate the training process, especially for larger datasets and deeper models.

## Code Overview
1. **Data Preprocessing:**  
   - The Fashion MNIST dataset is loaded from a CSV file (`fmnist_small.csv`).
   - A 4x4 grid of the first 16 images is visualized to inspect the data.
   - Features and labels are extracted, scaled (dividing by 255 to normalize between 0 and 1), and then split into training and testing sets.
   - The data is converted into PyTorch tensors for further processing.

2. **Custom Dataset and DataLoader:**  
   - A `CustomDataset` class is defined to handle the features and labels.
   - DataLoader objects are created for both training and testing to provide batches of data during the training loop.

3. **Model Definition:**  
   - An MLP model is defined using a class that inherits from `nn.Module`.  
   - The model is built using `nn.Sequential` with:
     - An input layer transforming the input features to 128 neurons.
     - A ReLU activation function.
     - A hidden layer reducing the number of neurons to 64.
     - Another ReLU activation.
     - A final output layer mapping to 10 classes.
   - The Sigmoid or Softmax (depending on the task) activation function is applied to generate probabilities.

4. **Training Loop:**  
   - The model is trained for 25 epochs using SGD with a learning rate of 0.1.
   - For each epoch, the training loop processes batches from the DataLoader, computes the loss using `nn.BCELoss`, performs backpropagation, and updates the model parameters.
   - After training, the model is evaluated on the test set by computing the accuracy.

5. **Model Evaluation:**  
   - The test set is passed through the model in evaluation mode (`model.eval()`).
   - Predictions are thresholded (using 0.5) to produce binary outputs.
   - Accuracy is calculated by comparing the predicted labels to the true labels.

## Conclusion
This notebook represents an important step in my deep learning journey. I learned how to:
- Efficiently load and preprocess data using PyTorch's Dataset and DataLoader classes.
- Build and train a simple ANN (MLP) using PyTorch's `nn` module.
- Implement a complete training pipeline with forward passes, backpropagation, and parameter updates.
- Evaluate the model's performance on a test set.
- Understand the trade-offs of running deep learning models on a CPU versus a GPU.

I look forward to further expanding my knowledge and tackling more advanced deep learning topics in future projects.
