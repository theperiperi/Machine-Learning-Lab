import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Iris dataset (used for regression in this experiment)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVR models with different kernels
svr_linear = SVR(kernel='linear')
svr_rbf = SVR(kernel='rbf')
svr_sigmoid = SVR(kernel='sigmoid')
svr_poly = SVR(kernel='poly')

# Define hyperparameter grids for SVR models
param_grid_linear = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.01, 0.001]
}

param_grid_rbf = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.01, 0.001],
    'gamma': ['scale', 'auto']
}

param_grid_sigmoid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.01, 0.001],
    'gamma': ['scale', 'auto']
}

param_grid_poly = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.01, 0.001],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}

# Perform GridSearchCV for each SVR model
grid_search_linear = GridSearchCV(svr_linear, param_grid_linear, cv=5, scoring='neg_mean_squared_error')
grid_search_rbf = GridSearchCV(svr_rbf, param_grid_rbf, cv=5, scoring='neg_mean_squared_error')
grid_search_sigmoid = GridSearchCV(svr_sigmoid, param_grid_sigmoid, cv=5, scoring='neg_mean_squared_error')
grid_search_poly = GridSearchCV(svr_poly, param_grid_poly, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV to the training data
grid_search_linear.fit(X_train, y_train)
grid_search_rbf.fit(X_train, y_train)
grid_search_sigmoid.fit(X_train, y_train)
grid_search_poly.fit(X_train, y_train)

# Get the best SVR models
best_svr_linear = grid_search_linear.best_estimator_
best_svr_rbf = grid_search_rbf.best_estimator_
best_svr_sigmoid = grid_search_sigmoid.best_estimator_
best_svr_poly = grid_search_poly.best_estimator_

# Make predictions on the testing data
y_pred_linear = best_svr_linear.predict(X_test)
y_pred_rbf = best_svr_rbf.predict(X_test)
y_pred_sigmoid = best_svr_sigmoid.predict(X_test)
y_pred_poly = best_svr_poly.predict(X_test)

# Calculate Mean Squared Error (MSE) for each SVR model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_rbf = mean_squared_error(y_test, y_pred_rbf)
mse_sigmoid = mean_squared_error(y_test, y_pred_sigmoid)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print("Mean Squared Error for Linear SVR:", mse_linear)
print("Mean Squared Error for RBF SVR:", mse_rbf)
print("Mean Squared Error for Sigmoid SVR:", mse_sigmoid)
print("Mean Squared Error for Polynomial SVR:", mse_poly)

# Linear Regression for comparison
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_lr = linear_reg.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print("Mean Squared Error for Linear Regression:", mse_lr)
