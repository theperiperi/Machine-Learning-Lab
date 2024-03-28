from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.var = {}

    def fit(self, data, y):
        self.classes = np.unique(y)
        for current_class in self.classes:
            data_current_class = data[y == current_class]
            self.class_priors[current_class] = len(data_current_class) / len(data)
            self.var[current_class] = np.var(data_current_class, axis=0)
            self.mean[current_class] = np.mean(data_current_class, axis=0)

    def predict(self, data):
        predictions = []
        for coordinate in data:
            posteriors = []
            for current_class in self.classes:
                prior = np.log(self.class_priors[current_class])
                likelihood = np.sum(np.log(self.gaussian_pdf(coordinate, self.mean[current_class], self.var[current_class])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        return predictions

    def gaussian_pdf(self, data, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((data - mean) ** 2) / (2 * var))

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = np.sum(predictions == y_test)
        total = len(y_test)
        return correct / total

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Naive Bayes model
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Calculate and print accuracy on the test set
accuracy = nb.accuracy(X_test, y_test)
print("Accuracy on test data:", accuracy)

# Make predictions on the test set
predictions = nb.predict(X_test)
print("Predictions:", predictions)
