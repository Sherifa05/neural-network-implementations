# Spotify Track Popularity Prediction 🎵

A deep learning regression model built with TensorFlow/Keras to predict track popularity based on Spotify audio features and playlist metadata.

## Overview

This project builds a neural network regression model to predict track popularity scores using Spotify's audio features and playlist information. The model leverages batch normalization and a multi-layer architecture to achieve accurate popularity predictions.

## Features

### Data Processing
- **Date Feature Extraction**: Converts release dates to year, month, and day components
- **Year Imputation**: Fills missing release years using:
  - Regex extraction from track names (e.g., "(2020)")
  - Artist median release year
  - Global median release year (2016)
- **Duplicate Analysis**: Identifies duplicate track entries with inconsistent metadata
- **Correlation Analysis**: Feature correlation with target variable

### Preprocessing Pipeline
- **Numerical Features**: Standard scaling (StandardScaler)
- **Categorical Features**: One-hot encoding (OneHotEncoder)
- **ColumnTransformer**: Combined preprocessing for mixed data types

### Model Architecture
- Input Layer: X features (after preprocessing)
- Dense 128 ReLU activation
- BatchNormalization
- Dense 64 ReLU activation
- BatchNormalization
- Dense 32 ReLU activation
- BatchNormalization
- Dense 1 Linear activation (regression output)


### Model Parameters
- **Optimizer**: Adam
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Epochs**: 50
- **Batch Size**: 32
- **Train/Validation Split**: 80/20

## Results

| Metric | Value |
|--------|-------|
| **Training Loss** | ~22.1 MSE |
| **Validation Loss** | ~309.3 MSE |
| **Training MAE** | ~3.6 |
| **Validation MAE** | ~11.3 |

*Note: Validation loss suggests potential overfitting; regularization could improve generalization.*

## Technologies

- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

## Data Sources

### Training Data
- 26,266 tracks with 24 features
- Target: `track_popularity` (0-100 scale)

### Test Data
- 6,567 tracks with 23 features (no target column)

### Features
**Numerical Features:**
- `track_popularity` (target)
- `danceability`, `energy`, `key`, `loudness`, `mode`
- `speechiness`, `acousticness`, `instrumentalness`
- `liveness`, `valence`, `tempo`, `duration_ms`
- `release_year`

**Categorical Features:**
- `track_id`, `track_name`, `track_artist`
- `track_album_id`, `track_album_name`
- `playlist_name`, `playlist_id`
- `playlist_genre`, `playlist_subgenre`

### Dataset Statistics
- Total tracks: 26,266
- Unique artists: 9,417
- Unique playlist genres: 6
- Unique playlist subgenres: 24
- Release years range: 1957-2020

## Implementation Notes
- Missing Values: Release dates imputed using multiple strategies
- Categorical Cardinality: High-cardinality features one-hot encoded
- Batch Normalization: Applied after each hidden layer for faster convergence
- Reproducibility: Fixed random_state=42 for splits

## Future Improvements
- Add dropout layers to reduce overfitting
- Implement early stopping
- Hyperparameter tuning (GridSearchCV)
- Feature engineering (audio feature interactions)
- Try ensemble methods (XGBoost, Random Forest)
- Add learning rate scheduling
- Experiment with different architectures
