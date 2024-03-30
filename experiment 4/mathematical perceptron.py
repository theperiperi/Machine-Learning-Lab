import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.predictions = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.predictions = np.zeros(n_samples)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)

                # Update weights and bias
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

                # Save predictions
                self.predictions[idx] = y_predicted

    def _activation(self, x):
        return np.where(x >= 0, 1, 0)

    def accuracy(self, y_true):
        return np.mean(self.predictions == y_true)


def generate_linearly_separable_data():
    # Generate linearly separable data
    np.random.seed(0)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)  # Linear boundary
    return X, y


def generate_non_linearly_separable_data():
    # Generate non-linearly separable data
    np.random.seed(0)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.where(X[:, 0]**2 + X[:, 1]**2 > 1, 1, 0)  # Non-linear boundary (circle)
    return X, y


def generate_higher_dim_linearly_separable_data():
    # Generate linearly separable data in higher dimensions
    np.random.seed(0)
    X = np.random.randn(100, 3)  # 100 samples, 3 features (3D)
    # Generate a linear boundary in 3D
    y = np.where(X[:, 0] + X[:, 1] + X[:, 2] > 0, 1, 0)
    return X, y


def plot_data_with_decision_boundary(X, y, model, title):
    # Plot the dataset and decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.show()


# 1. Linearly Separable 2D Data
X_linear, y_linear = generate_linearly_separable_data()
perceptron_linear = Perceptron()
perceptron_linear.fit(X_linear, y_linear)
plot_data_with_decision_boundary(X_linear, y_linear, perceptron_linear, title='Linearly Separable Data with Perceptron')
accuracy_linear = perceptron_linear.accuracy(y_linear)
print("Accuracy on Linearly Separable Data:", accuracy_linear)

# 2. Non-linearly Separable 2D Data
X_non_linear, y_non_linear = generate_non_linearly_separable_data()
perceptron_non_linear = Perceptron()
perceptron_non_linear.fit(X_non_linear, y_non_linear)
plot_data_with_decision_boundary(X_non_linear, y_non_linear, perceptron_non_linear,
                                 title='Non-linearly Separable Data with Perceptron')
accuracy_non_linear = perceptron_non_linear.accuracy(y_non_linear)
print("Accuracy on Non-linearly Separable Data:", accuracy_non_linear)

# 3. Higher-dimensional Linearly Separable Data (3D)
X_higher_dim, y_higher_dim = generate_higher_dim_linearly_separable_data()
perceptron_higher_dim = Perceptron()
perceptron_higher_dim.fit(X_higher_dim, y_higher_dim)

# Since we cannot visualize 3D decision boundary directly, we can print the model's weights
print("\nWeights for Higher-dimensional Linearly Separable Data:")
print("Feature 1 Weight:", perceptron_higher_dim.weights[0])
print("Feature 2 Weight:", perceptron_higher_dim.weights[1])
print("Feature 3 Weight:", perceptron_higher_dim.weights[2])

# Optionally, we can predict and print the accuracy for higher-dimensional data
accuracy_higher_dim = perceptron_higher_dim.accuracy(y_higher_dim)
print("Accuracy on Higher-dimensional Data:", accuracy_higher_dim)
