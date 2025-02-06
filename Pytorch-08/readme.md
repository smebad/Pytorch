# Day 8: Hyperparameter Tuning Using Optuna

## Introduction
In this notebook, I advanced my deep learning journey by exploring hyperparameter tuning. Hyperparameter tuning is the process of optimizing the settings (hyperparameters) that govern the training of a model—such as the number of hidden layers, neurons per layer, learning rate, dropout rate, and optimizer type—to improve performance and generalization. In this project, I applied these concepts to train an Artificial Neural Network (ANN) on a reduced version of the Fashion MNIST dataset (using a small subset of images due to CPU limitations). I used **Optuna**, an automated hyperparameter optimization framework, to search for the best hyperparameters for my model.

## What I Learned
- **Hyperparameter Tuning:**  
  The process of systematically searching for the optimal hyperparameters that lead to the best model performance. This is essential in deep learning since model performance can vary significantly with different hyperparameter choices.

- **Optuna:**  
  Optuna is a flexible, automatic hyperparameter optimization framework. It allows me to define an objective function that trains and evaluates a model, and then it intelligently searches the hyperparameter space to maximize (or minimize) a given metric (in this case, accuracy). Optuna's features include:
  - **Efficient Sampling:** It uses advanced algorithms to sample hyperparameters.
  - **Pruning:** It can stop unpromising trials early, saving time.
  - **Easy Integration:** Works seamlessly with PyTorch and other machine learning frameworks.

- **Optimizations for Neural Networks:**  
  I implemented improvements using dropout (to prevent overfitting), batch normalization (to stabilize and speed up training), and L2 regularization (weight decay) to further control overfitting. These techniques help improve the generalization of the model, especially when training on a smaller dataset.

- **Data Handling:**  
  I reinforced my knowledge of creating custom Dataset classes and using DataLoader for efficient mini-batch training. This helps in handling and shuffling data during training.

## Code Overview
1. **Data Preparation & Visualization:**  
   - Loaded a small Fashion MNIST dataset from a CSV file (`fmnist_small.csv`).
   - Visualized the first 16 images in a 4x4 grid.
   - Split the dataset into training and testing sets, and scaled the pixel values to the range [0, 1].

2. **Custom Dataset and DataLoader:**  
   - Created a `CustomDataset` class to wrap features and labels.
   - Created DataLoader objects for both training and testing to batch and shuffle data.

3. **Model Architecture:**  
   - Defined a neural network (`MyNN`) using PyTorch’s `nn.Module` with several hidden layers, ReLU activations, batch normalization, and dropout layers.
   - This architecture is designed for classification, mapping the 784 input features (28x28 images) to 10 output classes.

4. **Objective Function for Optuna:**  
   - Created an objective function that uses Optuna to suggest hyperparameters such as:
     - Number of hidden layers
     - Neurons per layer
     - Number of training epochs
     - Learning rate (on a logarithmic scale)
     - Dropout rate
     - Batch size
     - Choice of optimizer and weight decay
   - The objective function trains the model using these hyperparameters and returns the test accuracy.

5. **Hyperparameter Optimization with Optuna:**  
   - Installed Optuna (`pip install optuna`).
   - Created an Optuna study to maximize the model accuracy.
   - Ran the optimization for a defined number of trials to identify the best hyperparameters.

## Why This Matters
- **Improved Model Performance:**  
  By tuning hyperparameters, I can significantly improve model performance and reduce overfitting. This is especially important when training on a limited dataset (6000 images instead of the full 60,000).
  
- **Efficient Resource Use:**  
  Since I am currently training on a CPU, I had to reduce the dataset size. However, I plan to run the full dataset on a GPU using Google Colab for faster training and better performance.
  
- **Automation with Optuna:**  
  Optuna automates the tedious process of hyperparameter tuning, allowing me to focus more on model design and understanding the underlying principles.

## Conclusion
This notebook has deepened my understanding of hyperparameter tuning and model optimization techniques in deep learning. I learned how to:
- Use dropout, batch normalization, and L2 regularization to optimize neural network performance.
- Create a flexible training pipeline using PyTorch's Dataset and DataLoader classes.
- Automate hyperparameter search using Optuna to improve model accuracy.
- Understand the trade-offs of training on a smaller dataset due to CPU limitations and the benefits of using GPUs for large-scale training.

I look forward to further expanding this work, especially by training on larger datasets with GPU acceleration in the future.
