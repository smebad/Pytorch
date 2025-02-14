# Day 10 - Transfer Learning using PyTorch

## Overview
This notebook explores **Transfer Learning** using PyTorch. Transfer learning allows us to leverage pre-trained models like **VGG16** and **ResNet** to perform classification tasks with limited computational resources and smaller datasets. The key advantage of transfer learning is that it helps in achieving high accuracy even with a relatively small amount of labeled data by utilizing the pre-trained feature extraction capabilities of deep neural networks.

## What I Learned
- How to use **pre-trained models** (VGG16 and ResNet18) for transfer learning.
- The process of **freezing** feature extraction layers and modifying the classifier layer to fit a new dataset.
- Transformations needed to prepare dataset images for pre-trained models:
  - **Resizing** images to the appropriate input size.
  - **Converting grayscale images to RGB** (since models like VGG16 expect 3-channel images).
  - **Normalization** based on ImageNet statistics.
- How to define a **custom PyTorch dataset** and **data loaders**.
- Training and evaluation of a modified model on a small dataset.

## Dataset Used
- The dataset used was **Fashion-MNIST (small version)** loaded from `fmnist_small.csv`.
- Images were **28x28 grayscale** and needed conversion to **3-channel RGB**.
- Labels were extracted and split into training and testing sets.

## Implementation Steps
1. **Data Preprocessing:**
   - Loaded dataset and split into **train/test**.
   - Created a **custom PyTorch Dataset class** to apply required transformations.
   - Used `torchvision.transforms` to apply **resizing, conversion to tensor, and normalization**.
   - Defined **DataLoaders** for batch processing.

2. **Loading Pre-Trained Models:**
   - Initially tried to use **VGG16**, but due to computational limitations, switched to **ResNet18**.
   - Loaded **pre-trained ResNet18** and **froze feature extraction layers**.
   - Replaced the **fully connected layer (fc layer)** with a custom classifier for our dataset.

3. **Training the Model:**
   - Defined **CrossEntropyLoss** as the loss function.
   - Used **Adam optimizer** to update only the classifier parameters.
   - Implemented a **training loop** for 10 epochs, tracking loss after each epoch.

4. **Model Evaluation:**
   - Set the model to **evaluation mode**.
   - Used **forward pass** on test data to get predictions.
   - Compared predicted labels with actual labels to calculate accuracy.

## Challenges & Limitations
- Due to **hardware constraints**, I could not fully train the VGG16 model.
- Although the concepts of **transfer learning** were well understood, the lack of computational power restricted actual fine-tuning and large-scale training.
- The final accuracy was not computed as expected due to an issue with model evaluation (attempting to use VGG16 instead of the trained model).

## Key Takeaways
- Transfer learning is powerful and can be implemented **even with limited resources** by selecting lightweight models like **ResNet18**.
- Understanding how to **freeze layers** and **modify classifier layers** is crucial for adapting pre-trained models to new tasks.
- **Data transformations** play a significant role in preparing images for deep learning models.

## Future Improvements
- Running the training process on a **GPU-enabled environment** to fine-tune larger models like VGG16.
- Exploring **different optimizers and learning rates** to improve performance.
- Fixing evaluation code to ensure the correct model is used during inference.

---

### Notes:
- The implementation is based on **PyTorch** and uses **Torchvision models**.
- The notebook is structured for ease of understanding and can be extended to train on larger datasets with proper hardware support.
- The computed accuracy should be re-evaluated after fixing the evaluation step.

