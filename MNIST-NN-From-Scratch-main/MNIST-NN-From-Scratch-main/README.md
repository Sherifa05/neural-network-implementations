# Neural Network from Scratch: MNIST Classifier 🧠

A complete implementation of a two-layer neural network for handwritten digit recognition (MNIST) built from scratch using only NumPy, demonstrating fundamental deep learning concepts.

## Overview

This project implements a neural network from the ground up to classify handwritten digits from the MNIST dataset. The implementation includes forward propagation, backward propagation (backpropagation), and three different optimization algorithms: Standard Gradient Descent, SGD, and Adam.

## Neural Network Architecture
 - Input Layer: 784 neurons (28×28 pixels)
 - Hidden Layer: 128 neurons with ReLU activation
 - Output Layer: 10 neurons (digits 0-9) with Softmax

## Features

### Activation Functions
- **ReLU**: Used in hidden layer for non-linearity and sparse activations
- **Softmax**: Used in output layer for multi-class classification

### Optimization Algorithms Implemented

| Algorithm | Key Characteristics |
|-----------|---------------------|
| **Standard Gradient Descent** | Fixed learning rate, simple update rule |
| **SGD** | Stochastic gradient descent, faster convergence |
| **Adam** | Adaptive learning rate with momentum, fastest convergence |

### Performance Comparison

| Optimizer | Accuracy (550 iterations) |
|-----------|---------------------------|
| Standard GD | ~90.9% |
| SGD | ~90.8% |
| **Adam** | **~98.1%** |

## Implementation Details

### Data Preprocessing
- **Shuffling**: Randomizes data before splitting
- **Normalization**: Divides pixel values by 255 → range [0, 1]
- **Split**: 1,000 samples for validation, remaining for training

### Core Functions

#### Forward Propagation
```python
Z1 = W1·X + b1
A1 = ReLU(Z1)
Z2 = W2·A1 + b2
A2 = Softmax(Z2)
```

### Backward Propagation
- Computes gradients using chain rule
- Uses one-hot encoding for labels
- Calculates weight and bias updates

### Parameter Updates
- Standard/SGD: θ = θ - α·∇θJ(θ)
- Adam: Adaptive moment estimation with bias correction

## Results

### Training Progress
- Initial Accuracy: ~10% (random)
- Final Accuracy: ~98% with Adam
- Convergence: Adam achieves highest accuracy

### Sample Predictions
The model correctly predicts:
- Digit 5 as 5 ✓
- Digit 2 as 2 ✓
- Digit 0 as 0 ✓

### Technologies
- Python
- NumPy
- Pandas
- Matplotlib

## Implementation Notes
- Weight Initialization: Small random values (×0.01) to break symmetry
- Bias Initialization: Zeros
- Learning Rate: 0.1 for GD/SGD, 0.001 for Adam
- Hidden Layer Size: 128 neurons (balanced between capacity and efficiency)

## Future Improvements
- Add dropout for regularization
- Implement mini-batch gradient descent
- Add learning rate scheduling
- Include batch normalization
- Extend to deeper architectures
- Visualize learned weights and feature maps
