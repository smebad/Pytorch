# Lecture 7: Optimizing NN using Dropouts, BatchNorm, and Regularization

## Introduction
This notebook is part of my deep learning journey where I focus on optimizing neural networks using PyTorch. In this lecture, I built and trained an Artificial Neural Network (ANN) for a classification task using a subset of the Fashion MNIST dataset (6000 images) to suit my CPU constraints. I experimented with various optimization techniques such as dropout, batch normalization, and L2 regularization to improve model generalization and reduce overfitting.

## What I Learned
- **Dropout:**  
  A regularization technique that randomly "drops" (sets to zero) a fraction of the neurons during training to prevent over-reliance on any single feature. This helps reduce overfitting.

- **Batch Normalization:**  
  A technique to normalize the inputs of each layer so that they have a consistent distribution. This improves training stability, accelerates convergence, and can have a regularizing effect.

- **L2 Regularization (Weight Decay):**  
  Adding a penalty on the magnitude of weights to the loss function, which discourages overly complex models and helps prevent overfitting.

- **Data Handling with PyTorch Dataset and DataLoader:**  
  I created custom Dataset classes to wrap my data and used DataLoader for efficient mini-batch training and shuffling.

- **Model Training Pipeline:**  
  I built a multi-layer perceptron (MLP) using the `torch.nn` module. I then defined a training loop that includes forward passes, loss calculation using cross-entropy, backward propagation, and parameter updates with an SGD optimizer (which includes L2 regularization).

- **Evaluation:**  
  The model was evaluated on both the test and training data to measure accuracy. I observed that with only 6000 images, the model may still overfit; however, the techniques applied help in mitigating this issue. In future experiments, I plan to test on the full dataset using a GPU (via Google Colab) for improved performance.

## Code Overview
1. **Data Preparation:**  
   - Loaded the Fashion MNIST subset from a CSV file.
   - Visualized the first 16 images in a 4x4 grid to inspect the data.
   - Performed a train-test split, scaled the features (by dividing pixel values by 255), and created custom Dataset objects for both training and testing data.
   - DataLoader objects were used to batch and shuffle the data during training.

2. **Model Definition:**  
   - Constructed an MLP model using PyTorch’s `nn.Module` and `nn.Sequential` for simplicity.
   - The network includes:
     - A first linear layer mapping input features to 128 neurons.
     - Batch normalization and ReLU activation.
     - Dropout with a probability of 0.3.
     - A second linear layer reducing to 64 neurons, again followed by batch normalization, ReLU, and dropout.
     - A final linear layer that outputs logits for 10 classes.
  
3. **Training Pipeline:**  
   - The model is trained for 100 epochs using SGD with a learning rate of 0.1 and L2 regularization (weight decay of 1e-4).
   - In each epoch, the model processes mini-batches from the DataLoader, computes the cross-entropy loss, backpropagates the gradients, and updates the parameters.
   - Loss is printed after each epoch to monitor training progress.

4. **Evaluation:**  
   - After training, the model is evaluated on the test dataset.
   - The predictions are compared to the true labels to compute the overall accuracy.

## Conclusion
In this notebook, I learned how to optimize neural networks by integrating dropout, batch normalization, and L2 regularization into my training pipeline. Although I used a smaller dataset (6000 images) due to CPU limitations, this exercise taught me valuable techniques for improving model performance and generalization. I also learned how to use PyTorch’s Dataset and DataLoader classes to manage data efficiently. In the future, I plan to experiment with the full dataset on a GPU (using Google Colab) to further enhance model performance and scalability.

This experience has deepened my understanding of the challenges and solutions in training deep learning models, and it sets the stage for tackling more complex tasks in advanced deep learning topics.
