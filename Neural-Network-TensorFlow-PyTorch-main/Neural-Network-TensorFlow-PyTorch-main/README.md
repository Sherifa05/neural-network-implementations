# Neural Network Implementations: TensorFlow vs PyTorch 🧠

A comparative implementation of neural networks for regression tasks using two major deep learning frameworks - TensorFlow/Keras and PyTorch. Both implementations use the California Housing dataset to predict median house values.

## Overview

This repository contains two separate notebooks demonstrating neural network implementations for regression tasks using:
- **TensorFlow/Keras**: High-level API with Sequential model
- **PyTorch**: Low-level API with custom neural network class

Both models share the same architecture and are evaluated on the same dataset for fair comparison.

## Projects

### 1. TensorFlow NN (`Tensorflow_NN.ipynb`)
A simple feedforward neural network built with TensorFlow/Keras demonstrating:
- Sequential model API
- Dense layers with ReLU activation
- Model compilation and training
- Loss history visualization

### 2. PyTorch NN (`PyTorch_NN.ipynb`)
A feedforward neural network built with PyTorch demonstrating:
- Custom `nn.Module` class
- Manual training loop
- DataLoader for batch processing
- Loss history visualization

## Dataset

**California Housing Dataset**
- **Samples**: 20,640
- **Features**: 8 (MedInc, HouseAge, AveRooms, etc.)
- **Target**: Median house value (scaled)
- **Source**: `sklearn.datasets.fetch_california_housing`

## Neural Network Architecture
- Input Layer: 8 features
- Hidden Layer 1: 5 neurons, ReLU
- Hidden Layer 2: 3 neurons, ReLU
- Output Layer: 1 neuron (linear)


### Key Parameters
| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss Function** | MSE |
| **Epochs** | 100 |
| **Batch Size** | 32 |
| **Train/Test Split** | 67/33 |

## Results Comparison

| Framework | Test Loss (MSE) | Test MAE |
|-----------|-----------------|----------|
| **TensorFlow** | 0.3706 | 0.4279 |
| **PyTorch** | 0.3855 | 0.4352 |

*Both implementations achieve similar performance, as expected given identical architectures.*

## Technologies

### TensorFlow Implementation
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

### PyTorch Implementation
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

## Repository Structure

```text
neural-network-implementations/
│
├── Tensorflow_NN.ipynb          # TensorFlow/Keras implementation
├── PyTorch_NN.ipynb             # PyTorch implementation
└── README.md                    # Project documentation
```

## Key Concepts Demonstrated

### TensorFlow/Keras
- Sequential API: Layer-by-layer model building
- `model.compile()`: Configures optimizer, loss, and metrics
- `model.fit()`: Single-line training with validation split
- history object: Stores training metrics per epoch

### PyTorch
- `nn.Module`: Custom model class with `__init__` and `forward`
- `DataLoader`: Batch processing with shuffling
- Manual Training Loop: Explicit forward pass, loss calculation, backpropagation
- Gradient Management: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`


## Implementation Notes

### Common Elements
- Standardization: Features scaled using StandardScaler
- ReLU Activation: Used in hidden layers
- Adam Optimizer: Adaptive learning rate
- MSE Loss: Standard for regression tasks

## Future Improvements
- Add dropout for regularization
- Implement early stopping
- Hyperparameter tuning (grid search)
- Add BatchNormalization layers
- Experiment with different architectures
- Compare GPU vs CPU performance
- Add TensorBoard logging (TensorFlow)
- Implement model saving/loading




