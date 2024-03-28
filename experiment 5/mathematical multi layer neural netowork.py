import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the network
        
        # Input to hidden layer
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        
        # Hidden to output layer
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        
        return self.output
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        
        # Calculate the error
        output_error = y - self.output
        
        # Calculate the gradient at the output layer
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Calculate the hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        
        # Calculate the gradient at the hidden layer
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update the weights and biases
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_hidden_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)
        self.bias_input_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Backpropagation and update weights
            self.backward(X, y, learning_rate)
            
            # Compute and print loss
            loss = np.mean(np.square(y - output))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        # Make predictions
        hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        predicted_output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return predicted_output
    
    def accuracy(self, X, y):
        # Calculate accuracy
        predictions = self.predict(X)
        correct = np.sum(predictions.round() == y)
        total = y.shape[0]
        accuracy = correct / total * 100
        return accuracy


# Create a simple dataset (XOR)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize and train the neural network
input_size = 2
hidden_size = 4
output_size = 1
epochs = 10000
learning_rate = 0.1

model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X, y, epochs, learning_rate)

# Make predictions
predictions = model.predict(X)
print("Predictions:")
print(predictions)

# Calculate and print accuracy
accuracy = model.accuracy(X, y)
print(f"Accuracy: {accuracy:.2f}%")
