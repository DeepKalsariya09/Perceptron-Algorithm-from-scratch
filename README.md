# Perceptron-Algorithm-from-scratch
This repository contains a Python implementation of the Perceptron algorithm from scratch. It is a fundamental binary classifier. It represents the simplest form of a neural network, consisting of a single neuron.

## Files
perceptron.py

README.md

## Requirements
Python 3.x

NumPy

Matplotlib

Scikit-learn

## Usage

Clone the repository:
```bash
git clone https://github.com/yourusername/perceptron-from-scratch.git
cd perceptron-from-scratch
```

Install the required dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

Run the script:
```bash
python perceptron.py
```
## How It Works
1. Generate Dataset: A synthetic dataset is generated using make_classification from sklearn.datasets with 100 samples, 5 features, and 2 classes.
2. Normalize Input Vector: The feature vectors are normalized to unit length.
3. Split Dataset: The dataset is split into training and testing sets, and visualized.
4. Initialize Weight Vector: The weight vector is initialized randomly and normalized.
5. Implement Perceptron Algorithm: The algorithm iterates over each training sample. If the prediction 
ℎ
𝑖
h 
i
​
  does not match the actual label, the weight vector 
𝑤
w is updated.
6. Plot Data and Decision Boundary: The data and decision boundary are plotted for both training and testing datasets.
7. Evaluate Performance: The number of misclassifications and overall accuracy are calculated. A confusion matrix is displayed to show detailed performance metrics.
