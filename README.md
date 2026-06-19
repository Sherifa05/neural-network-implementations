# Deep Learning & Neural Network Portfolio 🧠

A comprehensive collection of deep learning and neural network projects demonstrating implementations across computer vision, regression modeling, and framework comparisons. Includes both from-scratch implementations and industry-standard framework usage.

## Projects Overview

### 1. MNIST Neural Network from Scratch 🧠
**File:** `mnist-nn-from-scratch.ipynb`

A complete two-layer neural network for handwritten digit recognition built from the ground up using only NumPy. This project demonstrates fundamental deep learning concepts without relying on high-level frameworks.

- **Architecture**: 784 → 128 → 10 (ReLU + Softmax)
- **Optimizers**: Standard GD, SGD, Adam
- **Best Accuracy**: 98.1% (with Adam)
- **Key Technologies**: NumPy, Pandas, Matplotlib
- **Key Concepts**: Forward propagation, backpropagation, one-hot encoding

[View Project Details →](MNIST-NN-From-Scratch-main/MNIST-NN-From-Scratch-main/README.md)

---

### 2. Spotify Track Popularity Prediction 🎵
**File:** `nnwithbn.ipynb`

A deep learning regression model to predict track popularity using Spotify audio features and playlist metadata. Features include comprehensive data preprocessing, feature engineering, and a neural network with batch normalization.

- **Architecture**: Dense layers with Batch Normalization
- **Features**: Audio features (danceability, energy, loudness, etc.)
- **Data Processing**: Date extraction, year imputation, correlation analysis
- **Validation MAE**: ~11.3
- **Key Technologies**: TensorFlow/Keras, Scikit-Learn, Pandas

[View Project Details →](NN-using-BN-main/NN-using-BN-main/README.md)

---

### 3. Neural Network Implementations: TensorFlow vs PyTorch 🧠
**Files:** `Tensorflow_NN.ipynb`, `PyTorch_NN.ipynb`

Comparative implementation of neural networks for regression using both major deep learning frameworks. Both implementations use the California Housing dataset with identical architecture for fair comparison.

- **Dataset**: California Housing
- **Architecture**: 8 → 5 → 3 → 1
- **TensorFlow Test MAE**: 0.4279
- **PyTorch Test MAE**: 0.4352
- **Key Technologies**: TensorFlow/Keras, PyTorch
- **Key Concepts**: Framework comparison, DataLoader, Sequential API

[View Project Details →](Neural-Network-TensorFlow-PyTorch-main/Neural-Network-TensorFlow-PyTorch-main/README.md)

---

## Technologies

### Core Libraries
- **Python**: 3.11+
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Scikit-Learn**: Data preprocessing and metrics

### Deep Learning Frameworks
- **TensorFlow/Keras**: High-level neural network API
- **PyTorch**: Low-level neural network framework
- **Custom NumPy**: From-scratch implementations

### Specialized Libraries
- **Seaborn**: Statistical visualization
- **Joblib**: Model persistence

## Repository Structure

```text
deep-learning-portfolio/
│
├── mnist-nn-from-scratch/
│   └── mnist-nn-from-scratch.ipynb
│   └── README.md
│
├── spotify-popularity-prediction/
│   └── nnwithbn.ipynb
│   └── README.md
│
├── neural-network-implementations/
│   ├── Tensorflow_NN.ipynb
│   └── PyTorch_NN.ipynb
│   └── README.md
│
└── README.md
```

## Project Comparison

| Project | Domain | Framework | Key Metric | Complexity | Key Learning |
|---------|--------|-----------|------------|------------|--------------|
| MNIST from Scratch | Computer Vision | NumPy | 98.1% Accuracy | Advanced | Backpropagation, Optimizers |
| Spotify Prediction | Regression | TensorFlow | MAE 11.3 | Intermediate | Batch Norm, Feature Engineering |
| TF vs PyTorch | Regression | TF/PyTorch | MAE 0.43 | Beginner | Framework Differences |

## Key Concepts Demonstrated

### Neural Network Fundamentals
- **Forward Propagation**: Computing predictions through network layers
- **Backward Propagation**: Computing gradients via chain rule
- **Activation Functions**: ReLU and Softmax in practice
- **Parameter Initialization**: Proper weight initialization strategies
- **One-Hot Encoding**: Converting labels for multi-class classification

### Optimization Algorithms
- **Standard Gradient Descent**: Fixed learning rate, simple update rule
- **Stochastic Gradient Descent**: Faster convergence with mini-batches
- **Adam**: Adaptive learning rate with momentum

### Deep Learning Best Practices
- **Batch Normalization**: Faster convergence and stability
- **Data Preprocessing**: Standardization, normalization
- **Feature Engineering**: Date extraction, correlation analysis
- **Model Evaluation**: Loss curves, metrics, confusion matrices

### Framework Proficiency
- **TensorFlow/Keras**: Sequential API, model.fit(), callbacks
- **PyTorch**: Custom modules, DataLoader, manual training loops
- **NumPy**: From-scratch implementations

## Implementation Highlights

### From Scratch Implementations
- **MNIST NN**: Complete backpropagation implementation
- **Three Optimizers**: GD, SGD, Adam with performance comparison
- **Activation Functions**: Custom ReLU and Softmax implementations

### TensorFlow Project
- **Batch Normalization**: Applied after each hidden layer
- **Feature Engineering**: Release date extraction, year imputation
- **Correlation Analysis**: Feature selection and interpretation
- **Preprocessing Pipeline**: ColumnTransformer with StandardScaler and OneHotEncoder

### PyTorch Implementation
- **Custom nn.Module**: Class-based model definition
- **DataLoader**: Batch processing with shuffling
- **Manual Training Loop**: Full control over training process
- **Evaluation**: MSE and MAE calculation

## Learning Outcomes

1. **Neural Network Fundamentals**: Deep understanding of backpropagation and optimization
2. **Framework Proficiency**: Ability to work with both TensorFlow and PyTorch
3. **Data Preprocessing**: Handle real-world data challenges
4. **Feature Engineering**: Extract meaningful features from raw data
5. **Model Optimization**: Compare different optimizers and architectures
6. **Evaluation Techniques**: Proper model assessment using appropriate metrics
7. **Batch Normalization**: Understanding its impact on training convergence
