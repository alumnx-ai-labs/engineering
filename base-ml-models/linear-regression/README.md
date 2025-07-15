<!-- /base-ml-models/linear-regression/README.md -->
# Simple Linear Regression from Scratch

This repository contains a Jupyter notebook implementation of simple linear regression built from scratch using Python. The implementation includes data preprocessing, a custom linear regression class, and visualization of results.

## Overview

The project demonstrates:
- Data loading and preprocessing
- Feature normalization (z-score standardization)
- Custom implementation of simple linear regression using gradient descent
- Model training and evaluation
- Visualization of data and model fit
- Loss function tracking

## Dataset

The implementation uses a salary dataset (`Salary_dataset.csv`) with the following features:
- **YearsExperience**: Number of years of work experience
- **Salary**: Corresponding salary amount

## Key Components

### 1. Data Preprocessing
- Loading data using pandas
- Feature normalization using z-score standardization
- Data visualization with matplotlib

### 2. SimpleLinearRegression Class
A custom implementation featuring:
- **Initialization parameters:**
  - `learningRate`: Controls step size for weight updates (default: 0.01)
  - `maxIter`: Maximum number of iterations (default: 5000)
  - `threshold`: Convergence threshold (default: 1e-6)

- **Key methods:**
  - `fit()`: Train the model using gradient descent
  - `predict()`: Make predictions on new data
  - `loss_err()`: Calculate mean squared error loss
  - `plot()`: Visualize data and fitted line

### 3. Model Training
- Uses gradient descent optimization
- Implements convergence checking based on loss threshold
- Tracks loss history for analysis

## Mathematical Foundation

The model implements the linear equation:
```
y = weight * x + bias
```

**Gradient Descent Updates:**
- Weight update: `weight += learning_rate * (1/n) * Σ(errors * x)`
- Bias update: `bias += learning_rate * (1/n) * Σ(errors)`

**Loss Function:**
- Mean Squared Error: `MSE = (1/2n) * Σ(y_actual - y_predicted)²`

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the `Salary_dataset.csv` file in the same directory
2. Open the Jupyter notebook:
```bash
jupyter notebook simple-linear-regession.ipynb
```
3. Run all cells to see the complete implementation

## Key Features

- **Feature Normalization**: Implements z-score standardization for better convergence
- **Convergence Control**: Automatic stopping when improvement falls below threshold
- **Visualization**: Plots original data, normalized data, model fit, and loss curve
- **Configurable Parameters**: Easily adjustable learning rate, iterations, and threshold

## Results

The notebook demonstrates:
- Successful model training with convergence
- Visualization of the fitted linear relationship
- Loss function decreasing over iterations
- Model performance on normalized data

## File Structure

```
├── simple-linear-regession.ipynb    # Main implementation notebook
├── Salary_dataset.csv               # Dataset file
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Dependencies

- **NumPy**: Numerical computations and array operations
- **Pandas**: Data loading and manipulation
- **Matplotlib**: Data visualization and plotting

## Learning Objectives

This implementation helps understand:
- Gradient descent optimization
- Feature normalization importance
- Linear regression mathematics
- Model convergence and training dynamics
- Python implementation of ML algorithms from scratch

## Notes

- The model uses normalized features for better numerical stability
- Learning rate of 0.01-0.1 typically provides good convergence
- Threshold of 1e-6 ensures sufficient convergence precision
- The implementation is educational and demonstrates core ML concepts