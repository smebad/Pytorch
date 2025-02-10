# Day 9: Building CNN in PyTorch

## Introduction
This notebook is part of my Pytorch learning journey. In Day 9, I focused on building a Convolutional Neural Network (CNN) using PyTorch to classify images from the Fashion MNIST dataset (using a small subset due to CPU limitations). In this lecture, I learned how to construct a CNN model, how to preprocess and handle image data, and how to design an efficient training pipeline using PyTorch’s Dataset and DataLoader classes.

## What is a CNN?
A Convolutional Neural Network (CNN) is a specialized type of neural network designed for processing grid-like data such as images. Key characteristics of CNNs include:
- **Convolutional Layers:** Automatically extract spatial features using learnable filters.
- **Pooling Layers:** Reduce the spatial dimensions of the feature maps, thereby lowering computation and providing some translation invariance.
- **Activation Functions:** Introduce non-linearity into the model (e.g., ReLU).
- **Regularization Techniques:** Use methods like dropout and batch normalization to improve generalization and reduce overfitting.

## Key Concepts and Techniques Learned
- **Data Handling and Preprocessing:**  
  - Loaded a Fashion MNIST subset from a CSV file.
  - Visualized the first 16 images in a 4x4 grid to inspect the dataset.
  - Performed a train-test split and normalized the pixel values to be in the range [0, 1].
  - Reshaped the image data into a 4D tensor format `(batch_size, channels, height, width)` for CNN processing.

- **Custom Dataset and DataLoader:**  
  - Created a custom `Dataset` class to wrap the features and labels.
  - Utilized PyTorch’s `DataLoader` to batch, shuffle, and iterate over the dataset efficiently during training.

- **Building the CNN Model:**  
  - Implemented the CNN using PyTorch’s `nn.Module` and `nn.Sequential` for clean and modular model design.
  - The model consists of:
    - **Feature Extraction:** Two convolutional blocks (each with a convolutional layer, ReLU activation, batch normalization, and max pooling).
    - **Classification:** A classifier block that flattens the output and passes it through fully connected layers with dropout for regularization, ending with a linear layer that outputs logits for 10 classes.
  
- **Training Pipeline:**  
  - Defined a training loop that processes mini-batches from the DataLoader, computes the cross-entropy loss, performs backpropagation, and updates model parameters using SGD.
  - Monitored the training loss for each epoch.

- **Model Evaluation:**  
  - Evaluated the model on the test set by calculating its accuracy.
  - Learned the importance of setting the model to evaluation mode during testing (`model.eval()`) to disable dropout and other training-specific behaviors.

- **Resource Considerations:**  
  - Due to CPU limitations, I trained on a smaller subset of the Fashion MNIST dataset (6000 images instead of 60,000). This allowed me to experiment with CNN architectures and training techniques.
  - I understand that using a GPU (e.g., via Google Colab) will significantly speed up training and allow for scaling to larger datasets.

## Code Overview
1. **Data Preparation and Visualization:**  
   - Loaded the dataset, normalized pixel values, and visualized the first 16 images in a 4x4 grid.
   - Performed a train-test split.

2. **Custom Dataset and DataLoader:**  
   - Defined a `CustomDataset` class that reshapes image data into the appropriate 4D tensor format.
   - Created DataLoader objects for training and testing.

3. **Model Architecture:**  
   - Built a CNN model (`MyCNN`) that uses convolutional layers, ReLU activations, batch normalization, max pooling, and dropout, followed by a classifier with fully connected layers.
  
4. **Training and Evaluation:**  
   - Implemented a training loop with forward and backward passes, loss computation using cross-entropy loss, and parameter updates with SGD.
   - Evaluated the model’s accuracy on the test set.

## Conclusion
In this lecture, I deepened my understanding of building and training Convolutional Neural Networks using PyTorch. I learned how to:
- Construct a CNN model for image classification.
- Efficiently handle and preprocess image data with custom Dataset and DataLoader classes.
- Integrate regularization techniques such as dropout and batch normalization to improve model generalization.
- Set up and run a complete training pipeline, from data loading to model evaluation.

While I trained on a limited dataset due to CPU constraints, I plan to extend these experiments to the full Fashion MNIST dataset using GPU acceleration on platforms like Google Colab. This lecture has solidified my foundational knowledge of CNNs and prepared me for more advanced deep learning challenges.
