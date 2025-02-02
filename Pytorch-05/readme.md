# Day 5: PyTorch Dataset and DataLoader Classes

## Introduction
In this notebook, I deepened my exploration of PyTorch by learning how to efficiently load and manage data using the **Dataset** and **DataLoader** classes. Building on previous notebooks, I improved my data pipeline by creating a custom dataset and utilizing the DataLoader to batch and shuffle data. This streamlined process not only makes training more efficient but also prepares me for scaling to larger datasets. Although my experiments run on a CPU (an Intel Core i7), I understand that GPUs can perform these tasks much faster.

## What is PyTorch?
PyTorch is an open-source machine learning library that provides powerful tools for deep learning and tensor computations. It features a dynamic computation graph, automatic differentiation (Autograd), and flexible data handling with classes like Dataset and DataLoader, which are essential for creating efficient training pipelines.

## Key Concepts Learned
- **Custom Dataset Class**:  
  - I defined a class that inherits from `torch.utils.data.Dataset`.  
  - Implemented `__len__()` to return the number of samples and `__getitem__()` to retrieve individual samples (features and labels).  

- **DataLoader Class**:  
  - I used the `DataLoader` to batch, shuffle, and iterate over the dataset.  
  - This makes it easier to feed data into the model during training.

- **Neural Network Training**:  
  - Built a simple neural network using `torch.nn.Module` for binary classification.  
  - Leveraged the `nn.Linear` layer, activation functions like `nn.Sigmoid`, and loss functions such as `nn.BCELoss`.

- **Data Preprocessing**:  
  - Applied feature scaling with `StandardScaler` and label encoding with `LabelEncoder` (from scikit-learn) to prepare the data.
  - Converted the preprocessed NumPy arrays into PyTorch tensors.

- **Training Pipeline**:  
  - Improved upon previous manual training implementations by using the Dataset and DataLoader classes to automate data loading and batching.
  - Implemented a training loop that includes forward passes, loss computation, backpropagation, and weight updates using an optimizer (SGD).

## Code Overview
1. **Data Preparation**:  
   - The dataset is loaded from a remote CSV file (Breast Cancer Detection data).  
   - Unnecessary columns are dropped, and the data is split into training and test sets.  
   - Features are scaled and labels are encoded, then converted into PyTorch tensors.

2. **Custom Dataset and DataLoader**:  
   - A `CustomDataset` class is defined to encapsulate features and labels.
   - DataLoader objects are created to efficiently iterate over the training and test datasets in mini-batches.

3. **Model Definition**:  
   - A simple neural network is defined using `nn.Module` with one linear layer and a sigmoid activation function.
   - This model predicts binary outcomes for the classification task.

4. **Training Loop**:  
   - The training loop processes mini-batches from the DataLoader, calculates the binary cross-entropy loss, performs backpropagation, and updates the model parameters.
   - After training, the model is evaluated on the test set, and accuracy is computed.

5. **Evaluation**:  
   - The model's predictions are thresholded to yield binary outputs.
   - Overall accuracy is calculated and printed.

## Conclusion
This notebook demonstrates how PyTorch’s **Dataset** and **DataLoader** classes can make data management more efficient and the training process more streamlined. By integrating these classes into my pipeline, I improved my previous approach and built a more scalable and modular system for training a neural network for binary classification.

While training on a CPU (Core i7) is feasible with these optimizations, using a GPU would significantly speed up the training process, especially for larger datasets and more complex models. This learning experience has deepened my understanding of PyTorch, particularly in the areas of data handling, model construction with `nn.Module`, and training using a custom pipeline.

I look forward to further exploring advanced topics in PyTorch in future notebooks.
