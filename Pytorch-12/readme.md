# Pytorch Day 12 - Next Word Predictor using LSTM in PyTorch

## Overview
This notebook explores the implementation of a next-word predictor using a Long Short-Term Memory (LSTM) model in PyTorch. A manually written dataset of machine learning-related questions and answers is used to train the model, which learns to predict the next word in a sentence.

## What is LSTM?
LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to handle sequential data while mitigating the vanishing gradient problem. Unlike traditional RNNs, LSTMs use special gating mechanisms to retain long-term dependencies in the data, making them well-suited for tasks like text generation and sequence prediction.

## Concepts Covered
In this notebook, we covered the following key concepts:
- **Tokenization**: Splitting text into individual words using `nltk.word_tokenize`.
- **Vocabulary Building**: Assigning an index to each unique word in the dataset.
- **Text-to-Index Conversion**: Mapping words to numerical representations.
- **Sequence Padding**: Ensuring uniform sequence length by padding shorter sequences.
- **Custom Dataset and DataLoader**: Creating a PyTorch dataset class and loading data efficiently using a DataLoader.
- **LSTM Model Architecture**: Implementing an LSTM-based model with embedding layers and a fully connected layer.
- **Loss Function and Optimizer**: Using `CrossEntropyLoss` and the Adam optimizer.
- **Training the Model**: Iterating through multiple epochs to minimize loss.
- **Generating Predictions**: Predicting the next word in a sentence based on learned patterns.

## Training the Model
1. **Data Preprocessing**:
   - Tokenized the text using NLTK.
   - Constructed a vocabulary and replaced words with their respective indices.
   - Created training sequences and applied padding to standardize input lengths.

2. **Model Training**:
   - Defined an LSTM model with an embedding layer, LSTM layer, and a fully connected layer.
   - Used CrossEntropyLoss as the loss function.
   - Trained the model for 50 epochs with the Adam optimizer.

3. **Prediction Process**:
   - The trained model takes an input sentence.
   - The words are tokenized and converted into numerical form.
   - The LSTM model predicts the next word based on learned patterns.

## What I Learned
- Understanding the working of LSTMs and how they handle sequential data.
- Building a vocabulary and efficiently mapping words to indices.
- Creating and managing custom datasets and data loaders in PyTorch.
- Training a model using a manually written dataset.
- Using padding techniques to handle varying sequence lengths.
- Generating predictions by feeding sequences into an LSTM model.

This notebook of Day 12 has provided me valuable insights into NLP tasks and deep learning techniques using PyTorch.

