import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def recursive_least_squares(X, y, delta=1.0, lam=0.99):
    n_samples, n_features = X.shape
    P = delta * np.eye(n_features)
    theta = np.zeros(n_features)
    predictions = np.zeros(n_samples)

    for i in range(n_samples):
        x_i = X[i, :]
        y_i = y[i]
        # Compute the prediction error
        prediction = x_i.dot(theta)
        predictions[i] = prediction
        error = y_i - prediction
        
        # Compute the gain vector
        K = (P @ x_i) / (lam + x_i.T @ P @ x_i)
        
        # Update the coefficient vector
        theta += K * error
        
        # Update the inverse covariance matrix
        P = (P - np.outer(K, x_i.T @ P)) / lam

    return theta, predictions

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

# Run the Recursive Least Squares
theta, predictions = recursive_least_squares(X, y)

# Print the estimated coefficients
print("Intercept and coefficients:", theta)

# Plotting Actual vs Predicted Life Expectancy
plt.figure(figsize=(10, 5))
plt.scatter(y, predictions, alpha=0.6)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r')  # Line for perfect predictions
plt.title('Actual vs Predicted Life Expectancy')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.grid(True)
plt.show()
