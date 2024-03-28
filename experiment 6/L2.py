import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Normalize features (optional)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# One-hot encode the target labels
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Set up neural network parameters
input_size = X.shape[1]
hidden_size1 = 5  # Adjust the number of nodes in the first hidden layer as needed
hidden_size2 = 3  # Adjust the number of nodes in the second hidden layer as needed
output_size = num_classes

# Initialize weights and biases for the neural network
mw1 = np.random.rand(input_size, hidden_size1)
mw2 = np.random.rand(hidden_size1, hidden_size2)
nw = np.random.rand(hidden_size2, output_size)
bh1 = np.random.rand(hidden_size1)
bh2 = np.random.rand(hidden_size2)
bo = np.random.rand(output_size)

mew = 0.01  # Learning rate

# Activation function (sigmoid)
def act(num):
    return 1 / (1 + np.exp(-num))

# Training the neural network
for epoch in range(1000):  # Adjust the number of epochs as needed
    for i in range(len(X_train)):
        alpha = 0.1
        # Forward pass
        z1 = X_train[i].dot(mw1) + bh1
        z1_act = act(z1) + (alpha * act(z1) * (1 - act(z1)))
        z2 = z1_act.dot(mw2) + bh2
        z2_act = act(z2) + (alpha * act(z2) * (1 - act(z2)))
        y_pred = z2_act.dot(nw) + bo
        y_act = act(y_pred) + (alpha * act(y_pred) * (1 - act(y_pred)))

        # Backpropagation
        err = y_train[i] - y_act
        efnih2 = err * y_act * (1 - y_act)
        efnhj2 = efnih2.dot(nw.T) * z2_act * (1 - z2_act)
        efnhj1 = efnhj2.dot(mw2.T) * z1_act * (1 - z1_act)

        # Weight and bias updates
        alpha = 2
        mw1 += mew * X_train[i][:, np.newaxis] @ efnhj1[np.newaxis, :]
        mw2 += mew * z1_act[:, np.newaxis] @ efnhj2[np.newaxis, :]
        nw += mew * z2_act[:, np.newaxis] @ efnih2[np.newaxis, :]
        bh1 += mew * efnhj1
        bh2 += mew * efnhj2
        bo += mew * efnih2

# Testing the trained network
correct_predictions = 0
for i in range(len(X_test)):
    z1 = X_test[i].dot(mw1) + bh1
    z1_act = act(z1)
    z2 = z1_act.dot(mw2) + bh2
    z2_act = act(z2)
    y_pred = z2_act.dot(nw) + bo
    y_act = act(y_pred)

    predicted_class = np.argmax(y_act)
    true_class = np.argmax(y_test[i])

    if predicted_class == true_class:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f"Test Accuracy: {accuracy}")
