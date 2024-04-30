import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def huber_loss(residual, delta=1.35):
    return np.where(np.abs(residual) < delta, 
                    0.5 * residual**2, 
                    delta * (np.abs(residual) - 0.5 * delta))

def weighted_least_squares(X, y, weights):
    # Minimize the weighted least squares objective
    X_w = X * np.sqrt(weights[:, np.newaxis])
    y_w = y * np.sqrt(weights)
    beta = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
    return beta

def robust_regression(X, y, delta=1.35, max_iter=10):
    n = len(y)
    weights = np.ones(n)
    mse_history = []

    for _ in range(max_iter):
        beta = weighted_least_squares(X, y, weights)
        residuals = y - X.dot(beta)
        weights = delta / np.maximum(delta, np.abs(residuals))
        mse = np.mean(residuals**2)
        mse_history.append(mse)

    return beta, mse_history, X.dot(beta)

# Data preparation
data_path = 'Life_Expectancy_Data_Cleaned.csv'
data = pd.read_csv(data_path)
data_subset = data[["Life expectancy", "HIV/AIDS", "Income composition of resources", "Adult Mortality", "BMI", "Schooling"]].copy()
data_subset.fillna(0, inplace=True)

# Prepare the data matrix X and target vector y
X = data_subset[["HIV/AIDS", "Income composition of resources", "Adult Mortality", "BMI", "Schooling"]].values
y = data_subset["Life expectancy"].values

# Adding a constant to the matrix X to account for the intercept
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Run the robust regression
beta, mse_history, predictions = robust_regression(X, y)

# Print the estimated coefficients
print("Intercept and coefficients:", beta)

# Plotting the MSE over iterations
plt.figure(figsize=(10, 5))
plt.plot(mse_history, marker='o')
plt.title('MSE over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Plotting Actual vs Predicted Life Expectancy
plt.figure(figsize=(10, 5))
plt.scatter(y, predictions, alpha=0.6)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r')  # Line for perfect predictions
plt.title('Actual vs Predicted Life Expectancy')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.grid(True)
plt.show()
