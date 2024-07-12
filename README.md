# Project Overview

This project involves data preprocessing and training multiple machine learning models on a dataset. The dataset is loaded from a pickle file, and several preprocessing steps are performed before training and evaluating various classification models. The models used include Logistic Regression, Decision Tree, Random Forest, and AdaBoost.

## Dependencies

Ensure you have the following Python libraries installed:

- pandas
- scikit-learn
- matplotlib

You can install these libraries using pip:

```sh
pip install pandas scikit-learn matplotlib
```

# Instructions

## Data Loading and Exploration

### Load the Dataset

Load the dataset and explore its structure, data types, summary statistics, and handle missing values.

```python
# Example Python code for data loading and exploration
import pandas as pd

# Load dataset
data = pd.read_pickle("path_to_your_dataset.pickle")

# Display dataset structure
print("Dataset structure:")
print(data.head())

# Check data types
print("\nData types:")
print(data.dtypes)

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())
