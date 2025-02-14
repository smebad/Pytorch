# Using RNN in PyTorch

## Overview
This project explores how to implement a Recurrent Neural Network (RNN) in PyTorch for processing and generating responses to questions. The dataset used consists of 100 unique question-answer pairs. The goal was to train an RNN model to predict answers based on given questions.

## What I Learned
- How to preprocess textual data for an RNN model.
- Tokenization and vocabulary creation for handling textual inputs.
- Implementing an RNN model in PyTorch with embedding layers.
- Training an RNN with a custom dataset and debugging training loop issues.
- Using PyTorch’s DataLoader for efficient batch processing.
- Implementing a prediction function for inference.

## What is an RNN?
A Recurrent Neural Network (RNN) is a type of neural network designed to handle sequential data. Unlike traditional neural networks, RNNs have loops that allow information to persist, making them suitable for tasks like text generation and sequence prediction.

Key components used in this project:
- **Embedding Layer**: Converts words into dense vector representations.
- **RNN Layer**: Captures sequential patterns in data.
- **Linear Layer**: Maps the RNN output to the vocabulary space for prediction.

## Implementation Steps
1. **Data Preprocessing:**
   - Loaded the dataset (`100_Unique_QA_Dataset.csv`).
   - Tokenized text by converting it to lowercase and removing unnecessary symbols.
   - Built a vocabulary for unique words and assigned numerical indices.
   - Converted words into numerical sequences for model input.

2. **Dataset and DataLoader:**
   - Created a custom dataset class (`CustomDataset`) to process the question-answer pairs.
   - Implemented a DataLoader to feed data efficiently during training.

3. **Model Architecture:**
   - Implemented an RNN model with:
     - An embedding layer (word to vector representation).
     - An RNN layer (hidden state propagation through time steps).
     - A linear layer (output prediction).
   
4. **Training the Model:**
   - Defined loss function (`CrossEntropyLoss`) and optimizer (`Adam`).
   - Implemented a training loop that:
     - Processes batches of question-answer pairs.
     - Performs forward propagation.
     - Computes loss and updates weights via backpropagation.
   
5. **Debugging Training Issues:**
   - Initially encountered shape mismatch errors in the training loop.
   - Debugged by manually checking tensor shapes through each layer.
   - Ensured correct input dimensions to the RNN and Linear layers.
   - Fixed indexing issues when passing answers to the loss function.

6. **Making Predictions:**
   - Implemented a `predict()` function to infer answers from trained model.
   - Used softmax to get probabilities and return the most likely response.
   - Implemented a threshold mechanism to filter low-confidence predictions.

## Results
After training for 20 epochs, the model successfully learned to generate accurate answers for many questions in the dataset. Debugging and refining the training loop significantly improved model performance.

This notebook of Day 11 has provided me valuable hands-on experience with RNNs and debugging deep learning models in PyTorch.

