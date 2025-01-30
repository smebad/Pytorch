# Day 3: PyTorch Pipeline Training  

## Introduction  
In this notebook, I explored **manual neural network training in PyTorch** by building a simple model for **breast cancer detection** using logistic regression. I implemented a **custom training pipeline** from scratch, including forward propagation, loss calculation, backpropagation, and weight updates using gradient descent.  

---

## Concepts Learned  

### 🔹 Manual Neural Network Pipeline  
Instead of using built-in PyTorch models, I manually implemented a neural network pipeline. The key steps included:  
1. **Data Preprocessing** – Cleaning and normalizing the dataset.  
2. **Model Definition** – Implementing a simple neural network using PyTorch tensors.  
3. **Forward Pass** – Computing predictions using matrix multiplication and the sigmoid function.  
4. **Loss Calculation** – Using **binary cross-entropy loss** to measure model performance.  
5. **Backward Pass** – Computing gradients using PyTorch’s automatic differentiation (`requires_grad=True`).  
6. **Weight Updates** – Adjusting model weights and biases using gradient descent.  
7. **Model Evaluation** – Calculating accuracy on the test set.  

### 🔹 What is Pipeline Training?  
Pipeline training refers to **structuring the training process** into well-defined steps. Each step is executed iteratively for multiple epochs to gradually improve model performance. In this notebook, the pipeline follows:  
1. **Forward Propagation** – Compute predictions.  
2. **Loss Computation** – Measure how far predictions are from actual values.  
3. **Backward Propagation** – Compute gradients.  
4. **Parameter Updates** – Adjust model weights using gradients.  

---

## Code Breakdown  

### 1️⃣ **Loading and Preprocessing Data**  
- Used the **Breast Cancer Dataset**.  
- Dropped unnecessary columns.  
- Performed **train-test split** (80% training, 20% testing).  
- **Standardized** the features using `StandardScaler()`.  
- Converted categorical labels (M/B) into numerical values using `LabelEncoder()`.  
- Converted data into **PyTorch tensors** for efficient computation.  

### 2️⃣ **Defining a Simple Neural Network**  
Created a **custom neural network** class `NN()` with:  
- **Weights (`self.weights`)** – Initialized randomly.  
- **Bias (`self.bias`)** – Initialized to zero.  
- **Forward Pass (`forward()`)** – Computes predictions using matrix multiplication.  
- **Loss Function (`loss_function()`)** – Uses **binary cross-entropy loss** to evaluate performance.  

### 3️⃣ **Training Pipeline**  
Implemented **manual training loop**:
- Used **100 epochs** for training.  
- **Forward Pass:** Generated predictions.  
- **Loss Computation:** Used binary cross-entropy.  
- **Backward Pass:** Computed gradients.  
- **Weight Update:** Used **gradient descent** to optimize model parameters.  
- **Gradient Reset:** Cleared gradients after each update.  

### 4️⃣ **Evaluating the Model**  
- Used the trained model to predict on test data.  
- Converted predictions to binary labels (0 or 1).  
- **Calculated accuracy** by comparing predictions with actual labels.  

---

## Results  

After training, the model was evaluated on the test set:  

✔ **Final Accuracy:** *Measured using PyTorch’s `.mean()` function*  
✔ **Optimized Weights & Biases:** Updated using gradient descent  

---

## Key Takeaways  

✅ Learned how to **implement a neural network from scratch** without `nn.Module`.  
✅ Understood **forward propagation, backpropagation, and gradient descent** manually.  
✅ Used **PyTorch tensors for computations and autograd for differentiation**.  
✅ Built a **custom training loop**, instead of using PyTorch’s `torch.nn` or `torch.optim`.  
✅ Gained insights into **manual weight updates and gradient tracking** in PyTorch.  

---

## Next Steps  

🔹 Experiment with **different learning rates** and observe performance.  
🔹 Replace manual training with `torch.optim.SGD()` for optimization.  
🔹 Extend the model with **more layers** for better accuracy.  
🔹 Use **built-in PyTorch models** (`nn.Module`) for efficiency.  

This hands-on approach gave me **a deeper understanding of how neural networks work internally** in PyTorch! 🚀  
