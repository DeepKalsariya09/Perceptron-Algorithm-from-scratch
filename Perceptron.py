# Perceptron Algorithm for Binary Classification Problem
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay,confusion_matrix

# Generate Dataset
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=0)

# Normalize Input Vector
norm_X = np.linalg.norm(X)
normalized_x = X / norm_X

# changing output value "0" to "-1"
y[y == 0] = -1

# Split Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(normalized_x, y, train_size=0.8, test_size=0.2, shuffle=False)

# Display overall dataset
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', edgecolors='k', label='Class  (1)')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='o', edgecolors='k', label='Class  (-1)')
plt.title('Overall Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and testing datasets
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', marker='x', label='Training')
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='o', label='Testing')
plt.title('Overall Dataset with Training and Testing Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Initialize Weight Vector
w = np.random.rand(X_train.shape[1])
w = w / np.linalg.norm(w)
mistakes = 0
iterations = 0
m = len(X_train)  # No of training data points

# Perceptron Algorithm:
for i in range(m):
    while True:
        iterations += 1
        # Activation Function/ Prediction
        hi = np.sign(np.dot(w, X_train[i]))
        if hi != y[i]:
            # update weight vector on mistake
            w = w + (y_train[i] * X_train[i])
            mistakes += 1
        else:
            break

print(f"Final Weight Vector: {w}")
print(f"Number of Iterations: {iterations}")
print(f"Number of Mistakes: {mistakes}\n")

# Test the Algorithm
predictions = np.sign(np.dot(X_test, w))
# Calculate misclassification errors
misclassification_errors = np.sum(predictions != y_test)
print("Number of misclassification errors:", misclassification_errors)
# Display accuracy score
accuracy = 1 - misclassification_errors / len(y_test)
print("Accuracy:", accuracy * 100, "%\n")

# Plot decision boundary
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

y_values = -(w[0] * x_values) / w[1]

plt.figure(figsize=(6, 6))
plt.ylim(-0.15, 0.15)
plt.xlim(-0.15, 0.15)
plt.plot(x_values, y_values, label='Decision Boundary')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, marker='o', label='Actual Test Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap=plt.cm.Paired, marker='x', s=80, label='Predicted Test Data')
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Create and display confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=["False", "True"])
conf_matrix_display.plot()
plt.show()
    