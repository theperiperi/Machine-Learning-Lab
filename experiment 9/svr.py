# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVR: kernal= linear, rbf, sigmoid and poly. 
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # You can adjust parameters like kernel, C, epsilon, etc.

# Train the SVR model
svr_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svr_model.predict(X_test)

# Calculate Mean Squared Error (MSE) as a measure of performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for rbf:", mse)
