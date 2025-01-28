# Day 1: PyTorch 

Welcome to **Day 1** of my journey to learn and explore **PyTorch**, an open-source machine learning framework. PyTorch is widely used for its dynamic computation graph, flexibility, and extensive support for deep learning tasks. 

In this notebook, I explored various fundamental concepts and operations in PyTorch, focusing on **tensors**, which are the building blocks of any PyTorch model. Below, I summarize what I learned, the concepts used, and how the code is organized.

---

## What is PyTorch?

PyTorch is a powerful library for **deep learning** and **tensor computation**. It allows for easy creation, manipulation, and operation of multi-dimensional arrays (tensors). It also supports GPU computation for high performance, making it suitable for machine learning research and production.

---

## Key Concepts I Learned Today

### 1. **Basic Tensor Operations**
   - **Creating Tensors**: Using functions like `torch.empty`, `torch.zeros`, `torch.ones`, `torch.rand`, and `torch.tensor`.
   - **Data Type Manipulation**: Assigning and changing tensor data types using `dtype` or `to()` function.
   - **Tensor Shapes**: Understanding tensor dimensions and using shape-changing functions like `reshape`, `flatten`, `squeeze`, and `unsqueeze`.

### 2. **Mathematical Operations**
   - **Scalar Operations**: Operations like addition, subtraction, multiplication, division, and power applied to tensors.
   - **Element-wise Operations**: Operations between tensors of the same shape (e.g., addition, subtraction, multiplication).
   - **Reduction Operations**: Aggregating data using `sum`, `mean`, `min`, `max`, `argmin`, `argmax`, `prod`, `std`, and `var`.
   - **Matrix Operations**: Using `matmul`, `dot`, `transpose`, `det`, and `inverse` for matrix and vector computations.

### 3. **Comparison Operations**
   - Comparing tensors using operations like `>`, `<`, `==`, `>=`, `<=`, and `!=`.

### 4. **Special Functions**
   - Logarithmic, exponential, square root, sigmoid, softmax, and ReLU operations.

### 5. **Inplace Operations**
   - Using operations with `_` suffix (e.g., `add_()`, `relu_()`) to modify tensors in-place.

### 6. **Tensor Copying**
   - Properly copying tensors using `clone()` to avoid issues with memory sharing.

### 7. **GPU Support**
   - Checking GPU availability using `torch.cuda.is_available()`.
   - Moving tensors to GPU using `to()` for faster computations.

### 8. **Speed Comparison**
   - Comparing the performance of matrix operations between CPU and GPU (tested in Google Colab).

### 9. **Tensor-Numpy Interoperability**
   - Converting between Numpy arrays and PyTorch tensors using `torch.from_numpy()` and `.numpy()`.

---

## Code Walkthrough

### Checking PyTorch Installation
- Displayed the PyTorch version and verified GPU availability.

### Tensor Creation and Manipulation
- Created tensors using functions like `empty`, `zeros`, `ones`, `rand`, `tensor`, `arange`, `linspace`, and more.
- Learned to manipulate tensor shapes using functions like `reshape`, `flatten`, `permute`, `squeeze`, and `unsqueeze`.

### Mathematical Operations
- Applied scalar, element-wise, reduction, and matrix operations to tensors.
- Used PyTorch's special functions like `log`, `exp`, `sqrt`, `sigmoid`, and `softmax`.

### Comparison and Special Operations
- Compared tensors element-wise and explored the use of `clamp` to restrict values within a range.

### Performance Evaluation
- Measured the speed of matrix multiplication on the CPU and GPU (tested in Google Colab).

### Reshaping Tensors
- Experimented with reshaping and permuting tensors to adapt them to different dimensional requirements.

---

## Reflections

This notebook introduced me to PyTorch's core capabilities, specifically **tensor operations**, which are fundamental to building and training machine learning models. I feel confident about creating and manipulating tensors, performing mathematical operations, and working with GPU computation in PyTorch.

This is just the beginning of my PyTorch journey. Future notebooks will explore topics like building neural networks, working with datasets, implementing backpropagation, and training deep learning models.

Stay tuned for more updates!
