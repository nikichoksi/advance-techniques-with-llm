# Advanced Techniques with LLM

This repository contains projects and implementations exploring advanced techniques with Large Language Models and deep learning fundamentals.

## Projects

### 1. Neural Network Backpropagation Fundamentals
**File:** `Neural_Network_Backpropagation_Fundamentals.ipynb`

This project demonstrates the foundational concepts of neural networks, backpropagation, and gradient descent optimization.

#### Concepts Covered

**1. Manual Backpropagation**
- Implementing gradient computation by hand using the chain rule
- Understanding how gradients flow through neural networks
- Deriving gradients for custom activation and loss functions

**2. Activation Functions**
- **LeakyReLU**: Piecewise activation function that prevents "dying neurons"
  - Returns `x` for positive values
  - Returns `0.01 * x` for negative values
- **Sigmoid**: Smooth activation function that outputs values between 0 and 1
  - Formula: `σ(x) = 1 / (1 + e^(-x))`
  - Used for binary classification and probability outputs

**3. Loss Functions**
- **L2 Loss (Mean Squared Error)**: `L = (prediction - target)²`
  - Penalizes large errors more heavily
  - Smooth gradients
- **L1 Loss (Mean Absolute Error)**: `L = |prediction - target|`
  - Robust to outliers
  - Piecewise differentiable

**4. Gradient Descent Optimization**
- Manual weight updates using computed gradients
- Learning rate selection and its impact on convergence
- Iterative optimization to minimize loss

**5. PyTorch Fundamentals**
- **Autograd**: Automatic differentiation engine
- **Tensors**: Multi-dimensional arrays with gradient tracking (`requires_grad=True`)
- **Backward Pass**: Computing gradients with `.backward()`
- **Manual Optimization**: Weight updates without using `torch.optim`

**6. Neural Network Architecture**
- Single neuron optimization
- Multi-layer Perceptrons (MLPs)
- Forward and backward propagation in deep networks

#### Project Structure

**Problem 1: LeakyReLU with L2 Loss**
- **1a**: Manual backpropagation for a single neuron (pure Python/NumPy)
- **1b**: Same optimization using PyTorch autograd
- **1c**: 2-layer neural network with LeakyReLU activation

**Problem 2: Sigmoid with L1 Loss**
- **2a**: Manual backpropagation with 3 inputs (pure Python/NumPy)
- **2b**: Same optimization using PyTorch autograd
- **2c**: 2-layer neural network with Sigmoid activation

#### Key Learning Outcomes

1. **Understanding Backpropagation**: How gradients are computed and propagated through networks
2. **Chain Rule Application**: Breaking down complex derivatives into simpler components
3. **PyTorch Mechanics**: How automatic differentiation works under the hood
4. **Optimization Process**: How neural networks learn through iterative gradient descent
5. **Activation Functions**: Their role in introducing non-linearity and affecting gradient flow

#### Technical Skills Demonstrated

- Calculus: Derivative computation and chain rule
- Python programming with NumPy and PyTorch
- Neural network implementation from scratch
- Gradient descent optimization
- Loss function design and implementation
- Model architecture design

## Course Information

**Course:** INFO 7374 - Advanced Techniques with LLM
**Institution:** Northeastern University

## Technologies Used

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Setup and Usage

1. Clone the repository:
```bash
git clone https://github.com/nikichoksi/advance-techniques-with-llm.git
cd advance-techniques-with-llm
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib jupyter
```

3. Run the notebooks:
```bash
jupyter notebook
```

## About

This repository showcases practical implementations of fundamental concepts in deep learning and neural networks, providing a strong foundation for understanding how Large Language Models work at their core.
