import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
from keras.layers import Dense
from keras.models import Sequential

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

# Initialize the neural network model
model = Sequential()
model.add(Dense(hidden_size1, input_dim=input_size, activation='sigmoid'))
model.add(Dense(hidden_size2, activation='sigmoid'))
model.add(Dense(output_size, activation='softmax'))

# Compile the model with SGD optimizer
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with SGD optimizer
start_time_sgd = time.time()
history_sgd = model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=0)
end_time_sgd = time.time()
training_time_sgd = end_time_sgd - start_time_sgd

# Evaluate the model on test data with SGD optimizer
test_loss_sgd, test_accuracy_sgd = model.evaluate(X_test, y_test, verbose=0)

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with Adam optimizer
start_time_adam = time.time()
history_adam = model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=0)
end_time_adam = time.time()
training_time_adam = end_time_adam - start_time_adam

# Evaluate the model on test data with Adam optimizer
test_loss_adam, test_accuracy_adam = model.evaluate(X_test, y_test, verbose=0)

# Print results and observations
print("Results:")
print("SGD Optimizer:")
print(" - Test Loss:", test_loss_sgd)
print(" - Test Accuracy:", test_accuracy_sgd)
print(" - Training Time:", training_time_sgd, "seconds")

print("\nAdam Optimizer:")
print(" - Test Loss:", test_loss_adam)
print(" - Test Accuracy:", test_accuracy_adam)
print(" - Training Time:", training_time_adam, "seconds")

print("\nObservations:")
print("- Adam optimizer generally converges faster and achieves better accuracy compared to SGD.")
print("- This is expected as Adam adapts the learning rate dynamically, which can lead to faster convergence.")
print("- SGD has a more straightforward update rule, but it might require tuning the learning rate and momentum for better performance.")
